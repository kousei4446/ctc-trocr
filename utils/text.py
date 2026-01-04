from typing import Dict, List, Tuple

import torch


def read_labels(labels_path) -> List[Tuple[str, str]]:
    items = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                img_id, text = parts[0], ""
            else:
                img_id, text = parts[0], parts[1]
            items.append((img_id, text))
    return items


def build_char_vocab(train_items: List[Tuple[str, str]]) -> Dict[str, int]:
    chars = set()
    for _, text in train_items:
        for ch in text:
            chars.add(ch)

    vocab = {"<blank>": 0, "<unk>": 1}
    for i, ch in enumerate(sorted(chars), start=2):
        vocab[ch] = i
    return vocab


def encode_text(text: str, vocab: Dict[str, int]) -> List[int]:
    unk = vocab["<unk>"]
    return [vocab.get(ch, unk) for ch in text]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def cer_from_pairs(preds: List[str], gts: List[str]) -> float:
    total_edits = 0
    total_chars = 0
    for p, g in zip(preds, gts):
        total_edits += levenshtein(p, g)
        total_chars += max(1, len(g))
    return total_edits / total_chars


def build_id2char(vocab: Dict[str, int]) -> List[str]:
    id2char = [""] * (max(vocab.values()) + 1)
    for ch, i in vocab.items():
        id2char[i] = ch
    return id2char


def ctc_greedy_decode(logits: torch.Tensor, blank_id: int, id2char: List[str]) -> List[str]:
    pred_ids = logits.argmax(dim=-1)  # (B,T)
    out = []
    for seq in pred_ids.tolist():
        s = []
        prev = None
        for t in seq:
            if t == prev:
                continue
            prev = t
            if t == blank_id:
                continue
            ch = id2char[t] if 0 <= t < len(id2char) else ""
            if ch in ("<blank>",):
                continue
            if ch == "<unk>":
                ch = "?"
            if ch.startswith("<") and ch.endswith(">"):
                continue
            s.append(ch)
        out.append("".join(s))
    return out
