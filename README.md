# Exam Paper OCR

Upload a JEE, NEET or CET question paper (PDF, scan or photo). The app crops every
question, passage, diagram and table from the **original page**, reads it with OCR
verbatim, and labels its type. A reviewer can then fix crops and text without ever
changing the locked originals.

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

## Using the web app

1. **Upload.** Choose the exam and keep **"I will crop the questions"** selected. The pages
   load in a few seconds. Nothing is auto-cropped.
2. **Crop.** Drag a box around a question; a loose box is fine. For each box the app:
   - reads the text verbatim and finds the question number on the first line
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
   - **M:** move the box. The app re-analyzes it and keeps the old crop.
   - **Esc:** select mode
   - **← / →:** change page
4. **Review.** The original crop is on the left (locked) and the editable copy is on the right.
   You can fix text, type, number, subject or passage. Your fixes become training data.
   Undo and Reset to original always work.
5. **Run round-trip check.** It confirms every crop matches the original pixel for pixel, and
   outlines in red any ink no crop covers, so a missed question shows up.
6. **Export ZIP.** You get `questions.json`, `crops.json` and every PNG. Export is blocked if
   any original has changed.

If you want the app to propose all the crops first, choose "Auto-crop" at upload. Those crops
go through the same analyzer, and you can then fix them the same way.

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
│   │   ├── heuristic.py      column/zone detection, text lines, diagram + table detection
│   │   └── yolo.py           optional trained YOLOv8 layout detector
│   ├── ocr/
│   │   ├── tesseract.py      default engine (dictionary correction OFF, so text stays verbatim)
│   │   ├── paddle.py         optional PaddleOCR (printed text, Indic scripts)
│   │   ├── trocr.py          optional TrOCR (handwriting, line by line)
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
- OCR runs with Tesseract's dictionaries off, so words are not "corrected". Low-confidence
  words become `[?]`, never a guess.
- Diagram boxes grow to follow dashed lines, arrowheads and small labels. Any diagram whose
  ink touches its crop edge is flagged as possibly cut off.
- Anything uncertain sets `needs_review` and lists the reasons. Nothing is fixed silently.
