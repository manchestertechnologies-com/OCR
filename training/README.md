# Training the ML models

The pipeline works without any training (heuristic layout and Tesseract). These scripts make
it stronger, and your reviewers' corrections become the training data.

```
upload papers -> review/correct in the web UI -> export labels -> train -> drop weights in models/
```

| Model | Script | Data it needs | Output |
|---|---|---|---|
| Layout detector (YOLOv8) | `layout/export_yolo_dataset.py` then `layout/train_yolo.py` | Reviewed projects (boxes) | `models/layout_yolov8.pt` |
| Question-type classifier | `classifier/train_type_classifier.py` | Reviewed projects (text + type) | `models/qtype_clf.joblib` |
| Handwriting OCR (TrOCR fine-tune) | `handwriting/finetune_trocr.py` | Line images + transcripts | `models/trocr-finetuned/` |
| Quality thresholds | `eval/calibrate_quality.py` | Pages sorted into `good/` and `blurry/` | Numbers for `configs/default.yaml` |
| Accuracy report | `eval/evaluate.py` | Ground-truth JSON per paper | CER, WER, crop IoU, review stats |

Use a Python 3.11 environment for training (`requirements-ml.txt`). A GPU helps a lot for YOLO
and TrOCR. On CPU, train YOLO with `--model yolov8n.pt --imgsz 960`.

## Recommended order

1. Process 20–50 real papers per exam and review them carefully. Fix boxes, types and text.
2. `python training/eval/calibrate_quality.py` to set the blur thresholds.
3. `python training/classifier/train_type_classifier.py`. This needs only around 300 labelled questions.
4. `python training/layout/export_yolo_dataset.py` and then `python training/layout/train_yolo.py`. Set
   `layout.engine: yolo` in `configs/default.yaml`.
5. For handwriting, collect line crops with their transcripts and fine-tune TrOCR. Set
   `ocr.handwritten_engine: trocr`.
6. Re-run `eval/evaluate.py` after each step and keep the change only if the numbers improve.
