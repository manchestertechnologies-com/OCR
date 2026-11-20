# Exam Paper OCR

Upload a JEE, NEET or CET question paper (PDF, scan or photo; printed or handwritten).
**You crop** each question, passage, diagram or table on the original page; the app uses your
box exactly as drawn. It reads everything inside it (text, formulas, tables) with an offline
OCR model, cuts out each drawing inside it as its own locked image, and labels the question
type. A reviewer can then fix text and crops without ever changing the locked originals.

> **Team guidance:** add your work to this repository and push to the `main` branch only.
> Don't push to other branches.

## Quick start (Windows)

```powershell
# Python 3.11+ (3.11 recommended if you will add the ML extras) and Tesseract 5 installed
python -m venv C:\Users\DELL\.venvs\examocr
C:\Users\DELL\.venvs\examocr\Scripts\pip install -r requirements.txt

# Web app: upload, review and export
C:\Users\DELL\.venvs\examocr\Scripts\python -m examocr.cli serve
#   open http://127.0.0.1:8000

# Or from the command line
C:\Users\DELL\.venvs\examocr\Scripts\python -m examocr.cli process paper.pdf --exam NEET

# Tests (includes an end-to-end run on a generated sample page)
C:\Users\DELL\.venvs\examocr\Scripts\python -m pytest tests
```

For bilingual CET papers, install the Tesseract language packs and set `ocr.tesseract_lang`
(for example `eng+mar` or `eng+kan`) in `configs/default.yaml`.

### The OCR models (offline, one-time download)

Text, handwriting, formulas and tables are read by **GLM-OCR** (`zai-org/GLM-OCR`, 0.9B, MIT
licence), a document-OCR model that runs on this computer's CPU. Drawings inside your box are
found by **PP-DocLayoutV3** (`PaddlePaddle/PP-DocLayoutV3_safetensors`, Apache-2.0). No
internet is needed after the first download (~2.8 GB in total; ~4 GB RAM while running).

```powershell
C:\Users\DELL\.venvs\examocr\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
C:\Users\DELL\.venvs\examocr\Scripts\pip install "transformers>=5.2"
C:\Users\DELL\.venvs\examocr\Scripts\python -m examocr.cli download-models
```

- Formulas are written as plain text: `K₂MnO₄ + MnO₂ + O₂`, `NBS/AIBN →(Heat; CCl₄)`, `M/3`,
  `CH₃-C≡CH`. Unknown LaTeX is kept as written, never guessed.
- Drawings (ring structures, hand-drawn structural formulas, graphs, apparatus) are cut out as
  locked images. Reagents written over a reaction arrow stay text. A row of drawings that each
  sit next to their own option label ("A) [drawing] B) [drawing]") become **image options**.
- Speed on a laptop CPU (4 cores): about 30-90 s per question. The models load once at server
  start (~40 s). Without torch/transformers the app falls back to Tesseract (printed only).
- At upload, **Paper is** *Detect* works for most papers. Choose *Handwritten* or *Printed* if
  detection picks wrong.
- Settings in `configs/default.yaml`: `ocr.printed_engine` / `ocr.handwritten_engine`
  (`glm` | `tesseract` | `trocr`), `glm_letter_px` (letters are scaled to this height; higher is
  slower), `layout.figure_model` (`""` = ink rules only).
- Review still matters: messy writing can be misread (a handwritten `5]` read as `3)`). Every
  low-confidence or suspicious crop is flagged with the reason.

## Using the web app

1. **Upload.** Choose the exam. The pages load in a few seconds. Nothing is cropped by the app.
2. **Crop.** Drag a box around a question. **Your box is used exactly as drawn**: the app never
   moves, shrinks or extends it. If writing touches your box edge, or the box runs into the next
   question, the app tells you, so you can redraw it (M). For each box the app:
   - reads the text and formulas and finds the question number on the first line
   - finds every **diagram and table inside the box** and saves each as its own locked image.
     Diagram labels are kept out of the question text, and a diagram that touches your box
     edge is flagged so you can widen the box.
   - recognises **image options** (four graphs as options 1–4)
   - splits the **options** and **statements**
   - works out the **question type** from keywords (tolerant of OCR misreads), the shape of the
     options ("Both A and R…", "P-ii, Q-i", "B, A, D, C") and what the box contains. It shows
     its reasons and confidence.
3. **Keyboard shortcuts:**
   - **Q:** new question
   - **P:** next part of the selected question, when it continues in the next column or on the next page
   - **D / T / I:** add a missed diagram, table or image option
   - **A:** passage. Pick it in the "Passage" box and the next questions link to it.
   - **S / H:** section heading or header
   - **M:** redraw the box. The app reads the new box and keeps the old crop for undo.
   - **Esc:** select mode
   - **← / →:** change page
4. **Review.** The original crop is on the left (locked) and the editable copy is on the right.
   You can fix text, type, number, subject or passage. Your fixes become training data.
   Undo and Reset to original always work.
5. **Run round-trip check.** It confirms every crop matches the original pixel for pixel, and
   outlines in red any ink no crop covers, so a missed question shows up.
6. **Export ZIP.** You get `questions.json`, `crops.json` and every PNG. Export is blocked if
   any original has changed.

The command line (`process --mode auto`) can still auto-crop a whole paper for batch work; the
web app does not.

## Folder structure

```
new ocr/
├── configs/
│   ├── default.yaml          thresholds, OCR engines, layout engine
│   └── exams/                jee.yaml, neet.yaml, cet.yaml: subjects, option labels, keywords
├── examocr/                  the Python package
│   ├── ingest.py             PDF/image → original pages (native resolution, ≥300 DPI, never upscaled)
│   ├── quality.py            blur (variance of Laplacian), per-region blur, contrast, edge-cut check
│   ├── preprocess.py         working copy for detection only: shadow removal, denoise, deskew + inverse map
│   ├── ink.py                printed vs handwritten classification per line
│   ├── layout/
│   │   ├── docmodel.py       PP-DocLayoutV3: drawings / tables inside the user's box
│   │   ├── heuristic.py      column/zone detection, text lines, diagram + table detection (ink rules)
│   │   └── yolo.py           optional trained YOLOv8 layout detector
│   ├── ocr/
│   │   ├── glmocr.py         GLM-OCR: printed + handwritten text, formulas, tables (default)
│   │   ├── latex.py          the model's Markdown/LaTeX -> plain text (H₂O, M/3, →)
│   │   ├── tesseract.py      fast engine (dictionary correction OFF); finds printed word boxes
│   │   ├── paddle.py         optional PaddleOCR (printed text, Indic scripts)
│   │   ├── trocr.py          optional older TrOCR (handwriting, line by line)
│   │   ├── handseg.py        handwritten line / piece segmentation
│   │   └── router.py         sends each region to the right engine, falls back safely
│   ├── segment.py            question markers, sections, passages, multi-column/page continuations
│   ├── parse.py              options and statements (always exact substrings of the OCR text)
│   ├── classify.py           question type: fuzzy keyword rules, then trained classifier
│   ├── crop.py               lossless PNG crops cut from the original, SHA-256, read-only files
│   ├── store.py              projects, locked originals + editable layer, history, undo, re-crop
│   ├── verify.py             hash check + round-trip pixel check + lost-ink detection
│   ├── pipeline.py           runs everything end to end
│   ├── api/app.py            FastAPI server
│   └── cli.py                command line
├── web/                      upload + review interface (HTML/CSS/JS, no build step)
├── training/                 the ML side: see training/README.md
│   ├── layout/               export reviewed boxes → YOLO dataset, train YOLOv8
│   ├── classifier/           train the question-type classifier
│   ├── handwriting/          fine-tune TrOCR on your handwriting samples
│   └── eval/                 CER/WER/crop-accuracy report, blur threshold calibration
├── tests/                    fidelity, rules and end-to-end tests
├── tools/make_sample_paper.py   synthetic test page generator
├── models/                   trained weights go here
└── data/                     uploads and processed projects (data/projects/<id>/...)
```

## Fidelity guarantees (from the project plan)

- Crops are plain array slices of the original page, saved as PNG with no resizing and no
  filters. The round-trip test asserts they are bit-identical.
- The deskewed and denoised working copy is used only to find boxes. Boxes are mapped back
  through the inverse transform before cutting.
- Originals are written once, made read-only and hashed. Writing over one raises an error.
- The user's box is never changed. The app only reads what is inside it and warns when writing
  is cut at the box edge.
- OCR output is never "corrected" afterwards. With Tesseract, low-confidence words become `[?]`;
  with GLM-OCR, a crop below the confidence threshold or a reading that repeats itself is flagged
  for review.
- Diagram boxes grow to follow dashed lines, arrowheads and small labels. Any diagram whose
  ink touches its crop edge is flagged as possibly cut off.
- Anything uncertain sets `needs_review` and lists the reasons. Nothing is fixed silently.
