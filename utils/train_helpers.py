from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

from utils.image import resize_keep_aspect_and_pad_right
from utils.text import build_id2char, cer_from_pairs, ctc_greedy_decode


@torch.no_grad()
def evaluate(model, loader, device, blank_id, id2char) -> float:
    model.eval()
    preds_all, gts_all = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        pixel_values = batch.pixel_values.to(device)
        logits = model(pixel_values)
        preds = ctc_greedy_decode(logits, blank_id, id2char)
        preds_all.extend(preds)
        gts_all.extend(batch.target_texts)
    return cer_from_pairs(preds_all, gts_all)


@torch.no_grad()
def infer_one(model, processor, vocab, device, image_path: Path, img_h: int, max_w: int) -> str:
    model.eval()
    id2char = build_id2char(vocab)
    blank_id = vocab["<blank>"]

    img = Image.open(image_path)
    img = resize_keep_aspect_and_pad_right(img, target_h=img_h, max_w=max_w)
    pixel_values = processor.image_processor(
        images=img,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).pixel_values.to(device)

    logits = model(pixel_values)
    pred = ctc_greedy_decode(logits, blank_id, id2char)[0]
    return pred


def resolve_split_dir(root: Path, name_candidates: List[str]) -> Path:
    for n in name_candidates:
        p = root / n
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find any of these under {root}: {name_candidates}")
