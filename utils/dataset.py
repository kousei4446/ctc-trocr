from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from PIL import Image
from transformers import TrOCRProcessor

from utils.image import find_image_file, resize_keep_aspect_and_pad_right
from utils.text import encode_text, read_labels


class IAMLineDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        labels_path: Path,
        processor: TrOCRProcessor,
        vocab: Dict[str, int],
        img_h: int,
        max_w: int,
        patch_size: int,
        upsample: int,
        drop_too_long: bool = True,
    ):
        self.images_dir = images_dir
        self.items_all = read_labels(labels_path)
        self.processor = processor
        self.vocab = vocab
        self.img_h = img_h
        self.max_w = max_w
        self.patch_size = patch_size
        self.upsample = max(1, upsample)

        # CTCで最低限忁E��な時系列長�E�幁E��向パチE��数 ÁEupsample�E�E
        self.max_T = (max_w // patch_size) * self.upsample

        if drop_too_long:
            kept = []
            for img_id, text in self.items_all:
                if len(text) <= self.max_T:
                    kept.append((img_id, text))
            self.items = kept
        else:
            self.items = self.items_all

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_id, text = self.items[idx]
        img_path = find_image_file(self.images_dir, img_id)
        img = Image.open(img_path)

        img = resize_keep_aspect_and_pad_right(img, target_h=self.img_h, max_w=self.max_w)

        # IMPORTANT: processor の resize を無効化して、こちらで作ったサイズを保つ
        pixel_values = self.processor.image_processor(
            images=img,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        ).pixel_values[0]  # (3,H,W)

        target_ids = encode_text(text, self.vocab)
        return {
            "pixel_values": pixel_values,
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_text": text,
            "img_id": img_id,
        }


@dataclass
class Batch:
    pixel_values: torch.Tensor      # (B,3,H,W)
    targets: torch.Tensor           # (B,Smax)
    target_lengths: torch.Tensor    # (B,)
    target_texts: List[str]


def collate_fn(samples: List[dict], pad_value: int = 0) -> Batch:
    pixel_values = torch.stack([s["pixel_values"] for s in samples], dim=0)

    lengths = torch.tensor([len(s["target_ids"]) for s in samples], dtype=torch.long)
    smax = int(lengths.max().item()) if len(samples) else 0
    targets = torch.full((len(samples), smax), fill_value=pad_value, dtype=torch.long)
    for i, s in enumerate(samples):
        t = s["target_ids"]
        if len(t) > 0:
            targets[i, : len(t)] = t

    target_texts = [s["target_text"] for s in samples]
    return Batch(pixel_values=pixel_values, targets=targets, target_lengths=lengths, target_texts=target_texts)
