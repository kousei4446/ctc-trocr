from pathlib import Path

from PIL import Image


def find_image_file(images_dir: Path, img_id: str) -> Path:
    # if img_id already includes extension
    p = images_dir / img_id
    if p.exists():
        return p

    for ext in [".jpg"]:
        p = images_dir / f"{img_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found: id={img_id} under {images_dir}")


def resize_keep_aspect_and_pad_right(
    img: Image.Image,
    target_h: int,
    max_w: int,
    pad_value: int = 255,
) -> Image.Image:
    """
    - 高さ target_h に合わせてアスペクト比維持でリサイズ
    - 幁E�� max_w を趁E��るなら縮小して max_w に収めめE
    - 幁E�� max_w 未満なら右側を白でパディングして max_w にする
    """
    img = img.convert("RGB")
    w, h = img.size
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image size")

    scale = target_h / h
    new_w = int(round(w * scale))
    new_w = max(1, new_w)

    # clamp to max_w
    if new_w > max_w:
        scale2 = max_w / new_w
        new_w = max_w
        new_h = int(round(target_h * scale2))
        new_h = max(1, new_h)
        # もう一度 target_h に合わせ直す（最終的に target_h にする�E�E
        # ここは「縦が縮む」�Eで、後で上下パチE��ングして target_h に戻ぁE
        img_rs = img.resize((new_w, new_h), resample=Image.BICUBIC)
        canvas = Image.new("RGB", (max_w, target_h), (pad_value, pad_value, pad_value))
        # 上寁E���E�行画像なら上寁E��/中央寁E��は好み。中央寁E��にするなめEy=(target_h-new_h)//2�E�E
        canvas.paste(img_rs, (0, 0))
        return canvas

    # normal: resize to (new_w, target_h)
    img_rs = img.resize((new_w, target_h), resample=Image.BICUBIC)

    if new_w < max_w:
        canvas = Image.new("RGB", (max_w, target_h), (pad_value, pad_value, pad_value))
        canvas.paste(img_rs, (0, 0))
        return canvas

    return img_rs
