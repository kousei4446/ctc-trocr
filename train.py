# 実行例:
# python trocr_encoder_ctc_iam_full.py --mode train --data_root data\IAM_Aachen --out_dir out_ctc_all --epochs 800 --batch_size 6 --lr 1e-4 --max_w 2048

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

from utils.dataset import IAMLineDataset, collate_fn
from utils.logger import TensorBoardLogger
from utils.model import TrOCREncoderCTC
from utils.text import build_char_vocab, build_id2char, read_labels
from utils.train_helpers import evaluate, infer_one, resolve_split_dir


# -------------------------
# Repro
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML is required to load config.yaml") from e

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("config.yaml must be a mapping")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=None)
    ap.add_argument("--config", dest="config_flag", type=str, default=None)
    cli = ap.parse_args()

    config_path = cli.config_flag or cli.config or "config.yaml"
    cfg = load_config(Path(config_path))
    defaults = {
        "mode": "train",
        "data_root": None,
        "model_name": "microsoft/trocr-small-handwritten",
        "out_dir": "out_ctc_best",
        "log_dir": None,
        "epochs": 20,
        "batch_size": 6,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "freeze_encoder": False,
        "dropout": 0.0,
        "img_h": 384,
        "max_w": 2048,
        "upsample": 1,
        "num_workers": 0,
        "no_fp16": False,
        "seed": 42,
        "infer_image": None,
        "ckpt": None,
    }
    defaults.update(cfg)
    args = SimpleNamespace(**defaults)

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else out_dir / "tb"
    logger = TensorBoardLogger(log_dir)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = (not args.no_fp16) and (device.type == "cuda")

    processor = TrOCRProcessor.from_pretrained(args.model_name)

    if args.mode == "train":
        if args.data_root is None:
            raise ValueError("--data_root is required for train")
        data_root = Path(args.data_root)

        train_dir = resolve_split_dir(data_root, ["train", "tarain"])
        val_dir = resolve_split_dir(data_root, ["VAL", "val"])

        train_labels = train_dir / "labels.txt"
        val_labels = val_dir / "labels.txt"

        train_items = read_labels(train_labels)
        vocab = build_char_vocab(train_items)
        id2char = build_id2char(vocab)
        blank_id = vocab["<blank>"]

        # patch_size はモチE�E��E�から取る�E�E�E�基本16�E�E�E�E
        tmp = VisionEncoderDecoderModel.from_pretrained(args.model_name).encoder
        patch_size = getattr(tmp.config, "patch_size", 16)

        # save vocab
        with (out_dir / "vocab.json").open("w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        train_ds = IAMLineDataset(
            images_dir=train_dir / "images",
            labels_path=train_labels,
            processor=processor,
            vocab=vocab,
            img_h=args.img_h,
            max_w=args.max_w,
            patch_size=patch_size,
            upsample=args.upsample,
            drop_too_long=True,
        )
        val_ds = IAMLineDataset(
            images_dir=val_dir / "images",
            labels_path=val_labels,
            processor=processor,
            vocab=vocab,
            img_h=args.img_h,
            max_w=args.max_w,
            patch_size=patch_size,
            upsample=args.upsample,
            drop_too_long=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=lambda s: collate_fn(s, pad_value=0),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=lambda s: collate_fn(s, pad_value=0),
        )

        model = TrOCREncoderCTC(
            model_name=args.model_name,
            num_classes=len(vocab),
            upsample=args.upsample,
            freeze_encoder=args.freeze_encoder,
            dropout=args.dropout,
        ).to(device)

        print(f"[info] image: H={args.img_h}, W={args.max_w}  patch={model.patch_size}")
        print(f"[info] CTC time steps T = (W/patch)*upsample = {(args.max_w//model.patch_size) * max(1,args.upsample)}")
        print(f"[info] train samples: {len(train_ds)} (dropped too-long automatically)")
        print(f"[info] val samples:   {len(val_ds)}")

        ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=fp16)

        best = 1e9
        for epoch in range(1, args.epochs + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}")

            ema = None
            step = (epoch - 1) * len(train_loader)
            for batch in pbar:
                pixel_values = batch.pixel_values.to(device, non_blocking=True)
                targets = batch.targets.to(device, non_blocking=True)
                target_lengths = batch.target_lengths.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=fp16):
                    logits = model(pixel_values)                  # (B,T,C)
                    log_probs = F.log_softmax(logits, dim=-1)     # (B,T,C)

                    B, T, C = log_probs.shape
                    input_lengths = torch.full((B,), T, dtype=torch.long, device=device)

                    loss = ctc_loss(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)

                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                if args.grad_clip is not None and args.grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optim)
                scaler.update()

                v = float(loss.item())
                ema = v if ema is None else (0.95 * ema + 0.05 * v)
                pbar.set_postfix(loss=ema)
                logger.add_scalar("train/loss", v, step)
                logger.add_scalar("train/loss_ema", ema, step)
                step += 1

            val_cer = evaluate(model, val_loader, device, blank_id, id2char)
            print(f"[epoch {epoch}] val CER = {val_cer:.4f}")
            logger.add_scalar("val/cer", val_cer, epoch)

            if val_cer < best:
                best = val_cer
                ckpt = {
                    "model_state": model.state_dict(),
                    "model_name": args.model_name,
                    "vocab": vocab,
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                     
                     
                     
                     
                     
                     
                     
                     
                     
                     
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    "img_h": args.img_h,
                    "max_w": args.max_w,
                    "upsample": args.upsample,
                    "patch_size": model.patch_size,
                }
                torch.save(ckpt, out_dir / "best.pt")
                print(f"  -> saved: {out_dir / 'best.pt'} (best CER={best:.4f})")

    else:  # infer
        if args.ckpt is None or args.infer_image is None:
            raise ValueError("--ckpt and --infer_image are required for infer")

        ckpt = torch.load(args.ckpt, map_location="cpu")
        vocab = ckpt["vocab"]
        model_name = ckpt.get("model_name", args.model_name)
        img_h = int(ckpt.get("img_h", args.img_h))
        max_w = int(ckpt.get("max_w", args.max_w))
        upsample = int(ckpt.get("upsample", args.upsample))

        processor = TrOCRProcessor.from_pretrained(model_name)
        model = TrOCREncoderCTC(model_name=model_name, num_classes=len(vocab), upsample=upsample).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)

        pred = infer_one(model, processor, vocab, device, Path(args.infer_image), img_h=img_h, max_w=max_w)
        print(pred)

    logger.close()


if __name__ == "__main__":
    main()
