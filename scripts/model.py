"""Image encoder + masked mean text encoder + one-output linear regressor."""

import torch
from torch import nn
import timm
from transformers import AutoConfig, AutoModel

from scripts.config import Config


class CalorieModel(nn.Module):
    def __init__(self, cfg: Config, text_config: dict | None = None,
                 initialize_pretrained: bool | None = None):
        super().__init__()
        self.cfg = cfg
        pretrained = cfg.pretrained if initialize_pretrained is None else initialize_pretrained
        self.image_encoder = timm.create_model(cfg.image_model, pretrained=pretrained, num_classes=0)
        if text_config is not None:
            values = text_config.copy()
            model_type = values.pop("model_type")
            text_cfg = AutoConfig.for_model(model_type, **values)
            self.text_encoder = AutoModel.from_config(text_cfg, attn_implementation="eager")
        elif pretrained:
            self.text_encoder = AutoModel.from_pretrained(cfg.text_model, attn_implementation="eager")
        else:
            text_cfg = AutoConfig.from_pretrained(cfg.text_model)
            self.text_encoder = AutoModel.from_config(text_cfg, attn_implementation="eager")
        dim = cfg.projection_dim
        self.image_projection = nn.Sequential(nn.Linear(self.image_encoder.num_features, dim),
                                              nn.LayerNorm(dim), nn.GELU(), nn.Dropout(cfg.dropout))
        self.text_projection = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size, dim),
                                             nn.LayerNorm(dim), nn.GELU(), nn.Dropout(cfg.dropout))
        self.mass_projection = nn.Sequential(nn.Linear(1, 32), nn.GELU()) if cfg.use_mass else None
        self.regressor = nn.Sequential(nn.Linear(2 * dim + (32 if cfg.use_mass else 0), dim),
                                       nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(dim, 1))
        # No sigmoid, softmax or other activation after the final Linear.

    def configure_trainable(self, warmup: bool) -> None:
        for p in self.image_encoder.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        if warmup:
            return
        if self.cfg.image_trainable == "all":
            self.image_encoder.requires_grad_(True)
        elif self.cfg.image_trainable == "last_stage":
            if not hasattr(self.image_encoder, "layer4"):
                raise ValueError("last_stage currently supports ResNet encoders with layer4")
            self.image_encoder.layer4.requires_grad_(True)
        count = self.cfg.text_trainable_layers
        if count:
            if not hasattr(self.text_encoder, "transformer"):
                raise ValueError("Partial text fine-tuning is implemented for DistilBERT")
            layers = self.text_encoder.transformer.layer
            if count > len(layers):
                raise ValueError("text_trainable_layers exceeds the encoder layer count")
            for layer in layers[-count:]:
                layer.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            # Frozen encoders must not change BatchNorm state or produce dropout noise.
            if not any(p.requires_grad for p in self.image_encoder.parameters()):
                self.image_encoder.eval()
            if self.cfg.freeze_batchnorm:
                for module in self.image_encoder.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
            self.text_encoder.eval()
            if any(p.requires_grad for p in self.text_encoder.parameters()):
                for layer in self.text_encoder.transformer.layer:
                    if any(p.requires_grad for p in layer.parameters()):
                        layer.train()
        return self

    def forward(self, image, input_ids, attention_mask, mass=None):
        image_vector = self.image_encoder(image)
        hidden = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        text_vector = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        features = [self.image_projection(image_vector), self.text_projection(text_vector)]
        if self.mass_projection is not None:
            if mass is None:
                raise ValueError("This model requires portion mass")
            features.append(self.mass_projection(mass))
        return self.regressor(torch.cat(features, dim=1)).squeeze(-1)
