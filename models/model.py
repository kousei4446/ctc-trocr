import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import VisionEncoderDecoderModel


class TrOCREncoderCTC(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        upsample: int = 1,
        freeze_encoder: bool = False,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        base = VisionEncoderDecoderModel.from_pretrained(model_name)
        base.config.eos_token_id = 1
        base.config.pad_token_id = 2
        base.config.decoder_start_token_id = 2
        self.encoder = base.encoder
        self.hidden = self.encoder.config.hidden_size
        self.patch_size = getattr(self.encoder.config, "patch_size", 16)

        self.upsample = max(1, upsample)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(self.hidden, num_classes)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
            if hasattr(self.encoder, "config"):
                self.encoder.config.use_cache = False

    def _maybe_resize_pos_embed(self, pixel_values: torch.Tensor):
        embeddings = getattr(self.encoder, "embeddings", None)
        if embeddings is None:
            return

        patch = getattr(embeddings, "patch_embeddings", None)
        pos_embed = getattr(embeddings, "position_embeddings", None)
        if patch is None or pos_embed is None:
            return

        image_size = getattr(patch, "image_size", None)
        if image_size is None:
            return

        if isinstance(image_size, int):
            orig_h = orig_w = image_size
        else:
            orig_h, orig_w = image_size

        patch_size = getattr(patch, "patch_size", None) or self.patch_size
        if isinstance(patch_size, tuple):
            patch_h, patch_w = patch_size
        else:
            patch_h = patch_w = int(patch_size)

        orig_gh = orig_h // patch_h
        orig_gw = orig_w // patch_w

        num_extra = pos_embed.shape[1] - (orig_gh * orig_gw)
        if num_extra < 0:
            return

        H_img = int(pixel_values.shape[-2])
        W_img = int(pixel_values.shape[-1])
        new_gh = H_img // patch_h
        new_gw = W_img // patch_w

        if (new_gh == orig_gh) and (new_gw == orig_gw):
            return

        extra = pos_embed[:, :num_extra]
        pos = pos_embed[:, num_extra:]
        if pos.shape[1] != orig_gh * orig_gw:
            side = int(math.sqrt(pos.shape[1]))
            orig_gh = side
            orig_gw = side

        pos = pos.reshape(1, orig_gh, orig_gw, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(new_gh, new_gw), mode="bicubic", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, new_gh * new_gw, -1)
        new_pos = torch.cat([extra, pos], dim=1)

        embeddings.position_embeddings = nn.Parameter(new_pos)
        patch.image_size = (H_img, W_img)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, T, C)
        T = (W/patch_size) * upsample
        """
        self._maybe_resize_pos_embed(pixel_values)
        enc = self.encoder(pixel_values=pixel_values)

        hs = enc.last_hidden_state  # (B, N, H)
        B, N, Hh = hs.shape

        H_img = pixel_values.shape[-2]
        W_img = pixel_values.shape[-1]
        gh = H_img // self.patch_size
        gw = W_img // self.patch_size

        # Remove cls/distill tokens when present.
        if N == 2 + gh * gw:
            hs = hs[:, 2:, :]
            N = hs.shape[1]

        if N != gh * gw:
            raise RuntimeError(
                f"Unexpected token length N={N}, expected gh*gw={gh*gw} "
                f"(img={H_img}x{W_img}, patch={self.patch_size})"
            )

        # (B, gh, gw, H)
        hs2d = hs.view(B, gh, gw, Hh)
        # Pool across height to 1D sequence (B, gw, H)
        hs1d = hs2d.mean(dim=1)

        if self.upsample > 1:
            hs1d = hs1d.repeat_interleave(self.upsample, dim=1)

        hs1d = self.drop(hs1d)
        logits = self.classifier(hs1d)  # (B, T, C)
        return logits
