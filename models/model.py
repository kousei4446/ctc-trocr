import torch
import torch.nn as nn

from transformers import VisionEncoderDecoderModel


class TrOCREncoderCTC(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        upsample: int = 1,
        freeze_encoder: bool = False,
        dropout: float = 0.0,
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

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, T, C)
        T = (W/patch_size) * upsample
        """
        # DeiT/ViT 系: interpolate_pos_encoding=True で任意解像度を許ぁE
        try:
            enc = self.encoder(pixel_values=pixel_values, interpolate_pos_encoding=True)
        except TypeError:
            # もし環墁E�E実裁E��対応してなければ、ここで落ちる�Eで注意（その場合�E幁E384に揁E��る忁E��あり！E
            enc = self.encoder(pixel_values=pixel_values)

        hs = enc.last_hidden_state  # (B, N, H)
        B, N, Hh = hs.shape

        H_img = pixel_values.shape[-2]
        W_img = pixel_values.shape[-1]
        gh = H_img // self.patch_size
        gw = W_img // self.patch_size

        # CLS token を除去�E�E == 1 + gh*gw のとき！E
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
        # 高さ方向を pool して 1D化: (B, gw, H) 
        hs1d = hs2d.mean(dim=1)

        # 時系列長を増や
        if self.upsample > 1:
            hs1d = hs1d.repeat_interleave(self.upsample, dim=1)

        hs1d = self.drop(hs1d)
        logits = self.classifier(hs1d)  # (B, T, C)
        return logits
