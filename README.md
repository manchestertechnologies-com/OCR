# Exam Paper OCR

Upload a JEE, NEET or CET question paper (PDF, scan or photo; printed or handwritten).
**You crop** each question, passage, diagram or table on the original page; the app uses your
box exactly as drawn. **Mathpix** (online OCR, needs an API key) reads everything inside it
(text, formulas, tables, chemical structures); the app cuts out each drawing inside it as its
own locked image and labels the question type. No OCR model runs on this computer. A reviewer
can then fix text and crops without ever changing the locked originals.

> **Team guidance:** add your work to this repository and push to the `main` branch only.
> Don't push to other branches.

## Quick start (Windows)

```powershell
# Python 3.11+, and Tesseract 5 (used only by the tests and command-line auto-crop)
python -m venv C:\Users\DELL\.venvs\examocr
C:\Users\DELL\.venvs\examocr\Scripts\pip install -r requirements.txt
C:\Users\DELL\.venvs\examocr\Scripts\python -m examocr.cli download-models   # browser libraries, once

# The Mathpix key (console.mathpix.com -> your organization -> API Keys), once:
setx MATHPIX_APP_ID "your-app-id"
setx MATHPIX_APP_KEY "your-app-key"

# Web app: upload, review and export
C:\Users\DELL\.venvs\examocr\Scripts\python -m examocr.cli serve
#   open http://127.0.0.1:8000

# Or from the command line
C:\Users\DELL\.venvs\examocr\Scripts\python -m examocr.cli process paper.pdf --exam NEET

# Tests: offline and free (Mathpix is played by Tesseract); -m mathpix calls the real service
C:\Users\DELL\.venvs\examocr\Scripts\python -m pytest tests -m "not mathpix"
```

### Reading with Mathpix (online, needs an API key)

Every box you draw is read by **Mathpix** (`v3/text`): about 1-3 s per question, ≈ ₹0.17
($0.002) per question. No OCR or layout model runs on this computer.

- Only the boxes you draw are sent to Mathpix, never whole pages. The key is never logged or
  sent to the browser.
- Text, `$…$` formulas, every drawing in the box (diagrams, graphs, structures - each placed
  where Mathpix read it) and chemical structures (a SMILES suggestion for Ketcher) come back in
  one request.
- With no key, no internet, a Mathpix error or the monthly limit reached, the box is **not
  read**: it stays on the page outlined in red, with the reason, and a **Read again** button.
  Nothing is guessed on this computer.
- Usage and cost per month are shown on the page. The limit is `mathpix.monthly_limit_usd` in
  `configs/default.yaml` (default $20 ≈ ₹1,700).
- Set the key once in PowerShell (never in code or chat) - see Quick start. Keys are made at
  console.mathpix.com (Pay as you go); a new key can take a few minutes to start working.
- Trial on papers you already cropped: Mathpix's text next to what was read before.
  ```powershell
  C:\Users\DELL\.venvs\examocr\Scripts\python tools\mathpix_trial.py            # list papers
  C:\Users\DELL\.venvs\examocr\Scripts\python tools\mathpix_trial.py <paper id> # read its boxes
  ```
- Word uploads (`.docx`, `.doc`, `.odt`, `.rtf`) need **LibreOffice** (free, libreoffice.org),
  which converts them to PDF.

Full plan, budget and guarantees: `docs/Mathpix-Fast-Reading.pdf` / `.docx`. Rebuild them with
`python tools/build_docs.py docs/Mathpix-Fast-Reading.md`.

- Formulas are kept as LaTeX (`$\frac{M}{3}$`, `$\ce{K2MnO4}$`) and shown rendered (see below).
  A plain-text copy (`K₂MnO₄ + MnO₂ + O₂`, `M/3`, `sec⁻¹ mol⁻¹ L`) is used for options, type
  detection and search. No backslash ever appears in the plain text.
- Drawings (ring structures, hand-drawn structural formulas, graphs, apparatus) are cut out as
  locked images. Reagents written over a reaction arrow stay text. A row of drawings that each
  sit next to their own option label ("A) [drawing] B) [drawing]") become **image options**.
- Review still matters: no OCR is 100% right. Printed papers come back near-perfect; messy
  handwriting and hand-drawn structures can be misread ("reaction" read as "reation"). Every
  low-confidence or suspicious crop is flagged with the reason.

### Formulas and chemical structures (free, offline)

| Task | Tool | Where in the app |
|---|---|---|
| Maths rendering | **MathJax 3** (Apache-2.0) | the OCR text and the live preview under the editable copy; exported HTML |
| Chemistry notation | **mhchem** (part of MathJax) | `$\ce{2KMnO4 ->[\Delta] K2MnO4 + MnO2 + O2}$` renders as a reaction |
| Word export | **latex2mathml + mathml2omml** | every formula becomes a native, editable Word equation |
| 2D structure editing | **Ketcher 3.21** (Apache-2.0) | "Draw / Edit structure" on a figure: redraw it, Save |
| Structure checks + images | **RDKit** (BSD) | validates the molecule, draws a clean image used in the copy and exports |
| Organic structures | **SmilesDrawer 2.4** (MIT) | draws a SMILES as you type it, and the recognised structure |
| Reading a drawing | **Mathpix** (online) | "Recognise drawing": a SMILES suggestion to check; printed drawings work, hand-drawn often don't |
| 3D molecules | **3Dmol.js 3.0** (BSD) | "3D" on a figure with a structure: rotate and zoom |
| OCR | **Mathpix** (online) | (above). No OCR model runs on this computer |

All browser libraries are downloaded once into `web/vendor/` by `download-models` and served by
the app, so editing, rendering and exporting work offline. Reading a box needs the internet
(Mathpix).

**Speed** (this 4-core laptop, measured in Chrome): drawing a box ~30-130 ms; rendering a
question's formulas ~50 ms (then ~2 ms per keystroke); SmilesDrawer ~2 ms; opening the
structure editor ~0.1-0.2 s (Ketcher is loaded in the background a few seconds after a paper
opens); saving a structure ~0.4 s (Ketcher ~0.35 s + RDKit ~0.05 s); 3D view ~0.4 s.
**Reading** a question (Mathpix) takes about 1-3 s and runs in the background.

## Using the web app

1. **Upload** a PDF, Word file, scan or photo and choose the exam. The pages load in a few
   seconds. Nothing is cropped by the app.
2. **Crop.** Drag a box around a question. **Your box is used exactly as drawn**: the app never
   moves, shrinks or extends it. The box is saved at once and shows "reading…" while the text is
   read in the background - keep drawing the next boxes; a counter shows how many are waiting.
   If writing touches your box edge, or the box runs into the next question, the app tells you,
   so you can redraw it (M). For each box the app:
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
│   │   ├── docmodel.py       PP-DocLayoutV3 (no longer used)
│   │   ├── heuristic.py      column/zone detection, text lines, diagram + table detection (ink rules)
│   │   └── yolo.py           optional trained YOLOv8 layout detector
│   ├── ocr/
│   │   ├── mathpix.py        Mathpix online OCR: reads every box - text, formulas, diagrams, SMILES
│   │   ├── latex.py          Markdown/LaTeX -> plain text (H₂O, M/3, →)
│   │   ├── tesseract.py      Tesseract: the tests' offline stand-in for Mathpix; auto-crop layout
│   │   ├── glmocr.py         GLM-OCR (no longer used; its pure-OpenCV helpers still are)
│   │   ├── paddle.py         PaddleOCR (not used)
│   │   ├── trocr.py          TrOCR (not used)
│   │   ├── handseg.py        line segmentation and binarisation (used to place drawings)
│   │   └── router.py         engine lookup for the offline stand-in; RDKit warm-up at start
│   ├── segment.py            question markers, sections, passages, multi-column/page continuations
│   ├── parse.py              options and statements (always exact substrings of the OCR text)
│   ├── classify.py           question type: fuzzy keyword rules, then trained classifier
│   ├── crop.py               lossless PNG crops cut from the original, SHA-256, read-only files
│   ├── store.py              projects, locked originals + editable layer, history, undo, re-crop
│   ├── worker.py             background reader: boxes are read one by one after they are drawn
│   ├── usage.py              Mathpix images and cost per month, monthly limit
│   ├── clean.py              clean copy of a drawing (ink on white, no paper / shadows)
│   ├── chem.py               RDKit (check, draw, 3D); "Recognise drawing" goes to Mathpix
│   ├── mathdoc.py            formulas -> native Word equations (LaTeX -> MathML -> OMML)
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
- OCR output is never "corrected" afterwards. A crop Mathpix reads with low confidence is
  flagged for review; a crop Mathpix could not read is marked "not read", never guessed.
- Diagram boxes grow to follow dashed lines, arrowheads and small labels. Any diagram whose
  ink touches its crop edge is flagged as possibly cut off.
- Anything uncertain sets `needs_review` and lists the reasons. Nothing is fixed silently.
