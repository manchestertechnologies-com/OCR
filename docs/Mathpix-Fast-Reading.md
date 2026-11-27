# Mathpix Reading for the Exam Paper OCR App

**Implementation · Budget · Benefits · Guarantees · Working process**

Manchester Technologies · Exam Paper OCR · 25 November 2026, updated 27 November 2026 · github.com/manchestertechnologies-com/OCR

## 1. Summary

The app reads each question you crop. It used to read them **on the computer** with GLM-OCR, which is private but takes **40–90 seconds per question** on a laptop CPU.

Since 27 November 2026 **every** cropped question is read by **Mathpix**, a paid online OCR service made for maths and science. Mathpix reads the text, formulas, equations, diagrams and chemical structures in about **1–3 seconds**.

- There is nothing to choose at upload: every paper is read by Mathpix.
- No OCR model runs on the computer any more. The local models were removed, which freed 7.6 GB of disk and about 4 GB of memory.
- **PDF, Word (DOC, DOCX, ODT, RTF) and image** files can be uploaded.
- Status: **built, tested, and tried with the real key on your own papers** (section 8).

## 2. Benefits

| | Before: GLM-OCR on this computer | Now: Mathpix |
|---|---|---|
| Time to read one question | 40–90 s (in the background) | about 1–3 s (1.7–3.5 s measured) |
| GPU needed for speed | yes (≈ ₹21,600/month cloud, or ₹50k–1L to buy) | no |
| Memory used by the app | about 4 GB | about 65 MB |
| Printed questions | good | near-perfect in the trial |
| Handwritten maths and chemistry | good on some boxes | mixed on phone photos: some boxes much better, some worse (section 8) |
| Formulas rendered in the app, editable in Word | yes | yes |
| Diagrams and graphs cut out as clean images | yes | yes, each placed where Mathpix read it |
| Chemical structure → molecule (SMILES) | not on handwriting | a suggestion to accept or fix in Ketcher |
| Word file upload | yes | yes |
| Works without internet | yes | no: the box is marked "not read", with a Read again button |
| Papers stay on the computer | yes | no: the boxes you draw are sent to Mathpix |
| Cost | free | about ₹0.17 per question |

## 3. What is guaranteed, and what is not

### Guaranteed (the app's part, built and tested)

1. **Every box you draw is sent to Mathpix**, and its text, formulas, equations, diagrams and structure suggestions come into the app automatically: the preview, the Word export and Ketcher.
2. A box appears **instantly** (about 0.06–0.13 s measured). The app adds about **0.5 s** of its own work to Mathpix's time.
3. **A box is never lost or guessed.** With no key, no internet, a Mathpix error or the monthly limit reached, the box stays on the page outlined in red, with the reason and a **Read again** button. Nothing is read on the computer instead.
4. **Spending never goes above your monthly limit.** Every image is counted, and the cost is shown on the page.
5. **The key stays safe:** kept on your computer only, never in the code, on GitHub or in the browser.
6. Only the **boxes you draw** are sent, never whole pages.

### Not guaranteed (outside anyone's control)

1. **100% accuracy.** No OCR reads messy or faint handwriting perfectly. On your handwritten phone photo Mathpix misread words ("reation" for "reaction", "Math the itemer" for "Match the items"). The text still needs checking.
2. **Hand-drawn chemical structures.** The SMILES is often wrong for hand-drawn structures, so it is only a suggestion you check in Ketcher.
3. **Every box in 1–2 s.** Mathpix's speed depends on its servers and your internet: 1.7–3.5 s in the trial.
4. **Reading without internet.** There is no on-computer reader any more: without internet, boxes wait as "not read" until Read again.

## 4. Working process

### What you do

1. **Upload** a PDF, Word file or image and choose the exam.
2. **Draw a box** around a question. It appears at once, marked "reading… n s".
3. About **1–3 seconds later** its text, formulas and diagrams appear. Keep drawing the next boxes.
4. **Check and fix** the text. Formulas show rendered in the preview.
5. For a structure, click **Draw structure**: Mathpix's suggestion is already in the editor. Accept or fix it, then save.
6. **Export ZIP**: an HTML file (rendered formulas), a Word file (editable equations) and a JSON file.

### What the app does for each box

1. It saves the locked crop of the box: about 0.06 s.
2. The background reader sends **only that crop** to Mathpix, in grey to keep the upload small: about 0.3–0.6 MB.
3. Mathpix returns text with `$...$` formulas, the outline of each drawing, and SMILES for structures.
4. The text becomes a plain copy (for options, question type, search) and a LaTeX copy (for MathJax on screen and Word equations).
5. Each drawing is cut from the locked original page and a clean copy is made. It is placed in the text where Mathpix read it: Mathpix lists text and drawings in reading order. Drawings beside option labels, as in "(a) [graph] (b) [graph]", become image options.
6. The question number, options and type are worked out as before. Mathpix's structures become suggestions in Ketcher.
7. The page checks every 0.3 s, so the text shows as soon as it is ready.
8. **If Mathpix cannot be used**, the box is marked "not read" with the reason. Read again sends the same box once the problem is fixed.

## 5. What was built

| Part | File | Job |
|---|---|---|
| Mathpix reader | examocr/ocr/mathpix.py | Sends one crop; returns text, formulas, drawing outlines in reading order and SMILES; handles time-limits, wrong keys, no internet |
| Analysis | examocr/analyze.py | Turns Mathpix's answer into the app's normal result; places each drawing where Mathpix read it |
| Not read / Read again | examocr/store.py, web/app.js | A box Mathpix cannot read keeps its reason; Read again sends the same box again |
| Structure suggestion | examocr/store.py | "Recognise drawing" sends the structure to Mathpix (no MolScribe model) |
| Usage and limit | examocr/usage.py | Counts images and cost per month; stops reading at the limit |
| Word upload | examocr/ingest.py | DOC / DOCX / ODT / RTF → PDF via LibreOffice → the normal PDF path |
| No local models | examocr/ocr/router.py, configs/default.yaml | No OCR or layout model is loaded at start |
| Screens | web/index.html, app.js, styles.css | Mathpix line at upload, key and usage status, "reading… n s", "Read by Mathpix in 1.5 s", red "not read" boxes, usage pill |
| Trial tool | tools/mathpix_trial.py | Reads your already-cropped questions with Mathpix and shows speed and text |
| Tests | tests/test_mathpix.py, tests/test_real_models.py | 14 tests with Mathpix-style answers, plus 1 real-Mathpix test |

## 6. Budget and cost

### One-time

| Item | Cost |
|---|---|
| Mathpix API setup fee (paid; comes with $29 of credit) | $19.99 (about ₹1,700) |
| LibreOffice, for Word files | free |

### Monthly (one question = one Mathpix image, $0.002 each)

| Questions per month | Mathpix cost |
|---|---|
| 1,000 | about $2 (≈ ₹170) |
| 5,000 | about $10 (≈ ₹850) |
| 10,000 | about $20 (≈ ₹1,700) |
| 50,000 | about $100 (≈ ₹8,500) |

- Mathpix API price, November 2026; lower above 1 million images.
- Redrawing a box, Read again and "Recognise drawing" each cost one more image.
- Rupee amounts use about ₹85 per $; Mathpix bills in dollars.
- The default monthly limit in the app is $20 (≈ ₹1,700). Change `monthly_limit_usd` in configs/default.yaml.

### Compared with a GPU for speed

| Option | Cost |
|---|---|
| Mathpix, 10,000 questions per month | ≈ ₹1,700 / month |
| Cloud GPU (E2E Networks T4, 24×7) | ≈ ₹21,600 / month |
| Buying a GPU (RTX 5060 Ti 16 GB) | ≈ ₹50,000–1,05,000 once |

## 7. Necessary requirements

1. **A Mathpix API account** at console.mathpix.com (an organization on the Pay as you go plan), giving an **app_id** and **app_key**. Done.
2. **The key on the computer or server, never in chat or code.** In PowerShell, then restart the app (a new key can take a few minutes to start working):
   `setx MATHPIX_APP_ID "your-app-id"`
   `setx MATHPIX_APP_KEY "your-app-key"`
3. **Internet** on the server, reaching api.mathpix.com. Without it nothing is read.
4. **LibreOffice** (free) on the server, for Word uploads: libreoffice.org. Without it, Word uploads show a clear message; PDFs and images work as before.
5. **Agreement** that the cropped question images are sent to Mathpix (a US company). There is no on-computer reader for confidential papers any more.
6. **Server:** a normal server, no GPU. The app needs about 100 MB of memory.

## 8. Testing and results

| Check | Result |
|---|---|
| Mathpix-style answers: text, formulas, diagram positions, graph options, SMILES suggestions | passed (14 tests) |
| No internet / wrong key / no key / limit reached → box marked "not read" with the reason; Read again reads it | passed |
| Usage counter and monthly limit | passed |
| Large crops stay under Mathpix's 5 MB limit | passed |
| Word upload without LibreOffice → clear message | passed |
| All earlier features | passed (85 automatic tests, offline: Tesseract stands in for Mathpix) |
| Real Mathpix on the sample paper (automatic test) | passed, 2.8 s for question 1 |
| Browser check (real Chrome; a stand-in for Mathpix answering in 1.2 s) | box saved in 0.13 s; text on screen 1.9 s after drawing |

### Real Mathpix on your papers (27 November 2026)

| Paper | Boxes | Time per box | Result |
|---|---|---|---|
| Thermodynamics.pdf (printed) | 3 | 1.9–3.0 s | text exact, confidence 1.00. Question 8: all 6 drawings found and placed (2 diagrams, and 4 graphs as options a–d) |
| WhatsApp photo (handwritten chemistry) | 3 | 1.7–3.5 s | readable, with misread words ("reation", "Mark is", "Math the itemer given in columnt"). The old on-computer reader was better on one box and much worse on another. The structure SMILES was wrong |

### Trying more of your papers

```
C:\Users\DELL\.venvs\examocr\Scripts\python tools\mathpix_trial.py
C:\Users\DELL\.venvs\examocr\Scripts\python tools\mathpix_trial.py adb21d791e86
```

The first command lists your papers. The second reads every question you cropped in that paper with Mathpix, and prints the seconds per question and Mathpix's text next to the text read before. Each question costs one image (≈ ₹0.17).
