"""Fine-tune TrOCR on your own handwritten exam lines.

Data layout:
  training/datasets/handwriting/
    images/xxx.png          one text LINE per image (cut from original pages, untouched)
    labels.tsv              filename<TAB>exact transcript (verbatim, including symbols)

  python training/handwriting/finetune_trocr.py --epochs 10
Then set ocr.handwritten_engine: trocr and point TrOCREngine at models/trocr-finetuned.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "training/datasets/handwriting")
    ap.add_argument("--base", default="microsoft/trocr-base-handwritten")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    a = ap.parse_args()

    import jiwer
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    rows = list(csv.reader(open(a.data / "labels.tsv", encoding="utf-8"), delimiter="\t"))
    random.Random(0).shuffle(rows)
    n_val = max(1, len(rows) // 10)
    val, train = rows[:n_val], rows[n_val:]
    proc = TrOCRProcessor.from_pretrained(a.base)
    model = VisionEncoderDecoderModel.from_pretrained(a.base)
    model.config.decoder_start_token_id = proc.tokenizer.cls_token_id
    model.config.pad_token_id = proc.tokenizer.pad_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    class Lines(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            name, text = self.rows[i]
            img = Image.open(a.data / "images" / name).convert("RGB")
            pv = proc(images=img, return_tensors="pt").pixel_values[0]
            labels = proc.tokenizer(text, padding="max_length", max_length=128, truncation=True).input_ids
            labels = [t if t != proc.tokenizer.pad_token_id else -100 for t in labels]
            return {"pixel_values": pv, "labels": torch.tensor(labels)}

    dl = DataLoader(Lines(train), batch_size=a.batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    best_cer = 1e9
    out = ROOT / "models/trocr-finetuned"
    for epoch in range(a.epochs):
        model.train()
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        model.eval()
        preds, refs = [], []
        for name, text in val:
            img = Image.open(a.data / "images" / name).convert("RGB")
            pv = proc(images=img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                ids = model.generate(pv, max_new_tokens=128)
            preds.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
            refs.append(text)
        cer = jiwer.cer(refs, preds)
        print(f"epoch {epoch + 1}: loss {loss.item():.4f}  val CER {cer:.4f}")
        if cer < best_cer:
            best_cer = cer
            model.save_pretrained(out)
            proc.save_pretrained(out)
    print(f"best val CER {best_cer:.4f}, saved to {out}")


if __name__ == "__main__":
    main()
