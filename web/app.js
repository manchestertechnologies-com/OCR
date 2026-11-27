"use strict";
const $ = (s) => document.querySelector(s);
const api = async (url, opts = {}) => {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(msg);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let P = null;        // current project
let page = 1;
let selected = null; // question id
let pollTimer = null;

// ------------------------------------------------------------------ home
// Warn when the server runs older Python code than what is on disk (it loads code once at
// start; this page is re-read every time). A server too old to know /api/version is stale too.
async function checkStale() {
  let stale;
  try { const r = await fetch("/api/version"); stale = !r.ok || (await r.json()).stale; }
  catch { return; }                                     // server unreachable: other errors show
  $("#stale").hidden = !stale;
}

async function loadHome() {
  checkStale();
  $("#home").hidden = false; $("#review").hidden = true;
  const exams = await api("/api/exams");
  $("#exam").innerHTML = exams.map((e) => `<option>${esc(e)}</option>`).join("");
  showMathpixStatus();
  const rows = await api("/api/projects");
  $("#projects tbody").innerHTML = rows.map((p) => `
    <tr><td>${esc(p.source_name)}</td><td>${esc(p.exam)}</td>
    <td><span class="pill ${p.status === "ready" ? "ok" : p.status === "error" ? "bad" : "warn"}">${esc(p.status)}</span></td>
    <td>${p.questions}</td><td>${p.needs_review}</td>
    <td><button class="link" data-open="${esc(p.id)}">Open</button></td></tr>`).join("")
    || `<tr><td colspan="6" class="hint">No papers yet.</td></tr>`;
}
let MX = null;                                    // Mathpix status: key set? usage this month
async function showMathpixStatus() {
  try { MX = await api("/api/mathpix"); } catch { return; }
  const el = $("#mathpix-status");
  if (!el) return;
  $("#rc-price").textContent = `₹${MX.price_inr}`;
  const rs = (v) => "₹" + Number(v).toLocaleString("en-IN", { maximumFractionDigits: 2 });
  const used = `This month: ${MX.images} question${MX.images === 1 ? "" : "s"} · ${rs(MX.inr)}` + (MX.limit_usd ? ` of ${rs(MX.limit_inr)} limit` : "");
  el.innerHTML = MX.configured
    ? `⚡ Mathpix key is set. ${esc(used)}.` + (MX.limit_reached
      ? " <b>This month's limit is reached: boxes can't be read until it is raised (mathpix: monthly_limit_usd in configs/default.yaml).</b>" : "")
    : "⚡ <b>Mathpix key not set: boxes can't be read until it is.</b> In PowerShell: " +
      "<code>setx MATHPIX_APP_ID \"…\"</code> and <code>setx MATHPIX_APP_KEY \"…\"</code>";
  el.innerHTML += MX.word_upload ? "" : " · Word files need <a href=\"https://www.libreoffice.org\" target=\"_blank\">LibreOffice</a> on this computer.";
}
$("#projects").addEventListener("click", (e) => { const id = e.target.dataset.open; if (id) openProject(id); });
$("#nav-home").onclick = () => { clearTimeout(pollTimer); setUploading(false); loadHome(); };

const drop = $("#drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) { $("#file").files = e.dataTransfer.files; showFile(); } });
$("#file").addEventListener("change", showFile);
function showFile() {
  const f = $("#file").files[0];
  drop.classList.remove("missing");
  $("#drop-text").textContent = f ? `${f.name} (${(f.size / 1e6).toFixed(1)} MB)` : "Drop a PDF, Word file, scan or photo here, or click to choose";
}

$("#upload-form").onsubmit = async (e) => {
  e.preventDefault();
  const f = $("#file").files[0];
  if (!f) {
    // The real <input type=file> is hidden, so the browser can't show its own "required" bubble.
    drop.classList.add("missing");
    $("#drop-text").textContent = "Choose a file first: click here or drop a PDF, Word file, scan or photo.";
    return;
  }
  const fd = new FormData(); fd.append("file", f); fd.append("exam", $("#exam").value);
  fd.append("mode", "manual");             // the user crops; the app never auto-crops
  $("#progress").hidden = false; $("#progress-msg").textContent = "Uploading…";
  $("#bar-fill").style.width = "0%";
  setUploading(true);                     // a second click would create a duplicate paper
  try {
    const { id } = await api("/api/projects", { method: "POST", body: fd });
    pollProgress(id);
  } catch (err) { $("#progress-msg").textContent = "Upload failed: " + err.message; setUploading(false); }
};
function setUploading(on) { $("#upload-form button[type=submit]").disabled = on; }

function pollProgress(id) {
  clearTimeout(pollTimer);
  const tick = async () => {
    let p;
    try { p = await api(`/api/projects/${id}`); }
    catch (err) { $("#progress-msg").textContent = "Lost contact with the server: " + err.message; setUploading(false); return; }
    $("#bar-fill").style.width = `${Math.round((p.progress?.fraction ?? 0) * 100)}%`;
    $("#progress-msg").textContent = p.progress?.message ?? p.status;
    if (p.status === "processing") { pollTimer = setTimeout(tick, 1000); return; }   // next poll only after this one
    setUploading(false);
    if (p.status === "error") $("#progress-msg").textContent = "Failed: " + p.error;
    else { $("#progress").hidden = true; openProject(id); }
  };
  pollTimer = setTimeout(tick, 1000);
}

// ------------------------------------------------------------------ review
async function openProject(id, keepPage = false) {
  if (!keepPage) checkStale();
  if (!$("#d-qtype").options.length)
    $("#d-qtype").innerHTML = (await api("/api/question-types")).map((t) => `<option>${esc(t)}</option>`).join("");
  P = await api(`/api/projects/${id}`);
  $("#home").hidden = true; $("#review").hidden = false;
  $("#r-title").textContent = `${P.source_name} · ${P.exam}`;
  showUsagePill();
  const st = $("#r-status"); st.textContent = P.status;
  st.className = "pill " + (P.status === "ready" ? "ok" : P.status === "error" ? "bad" : "warn");
  st.title = P.error || "";               // hover the red pill to see why a paper failed
  $("#btn-export").href = `/api/projects/${id}/export`;
  if (!keepPage) {
    const prof = await api(`/api/exams/${P.exam}`);
    const subs = prof.subjects || [];
    $("#cur-subject").innerHTML = `<option value="">auto</option>` + subs.map((s) => `<option>${esc(s)}</option>`).join("");
    $("#subjects-list").innerHTML = subs.map((s) => `<option value="${esc(s)}">`).join("");
    $("#draw-kind").value = P.mode === "manual" ? "question" : "";
    $("#overlay").classList.toggle("drawing", !!drawMode());
  }
  const ps = P.crops.filter((c) => c.kind === "passage" && !c.superseded);
  const popts = `<option value="">none</option>` + ps.map((c) => `<option value="${esc(c.passage_id)}">${esc(c.passage_id)}: ${esc(c.text.slice(0, 30))}</option>`).join("");
  const cur = $("#cur-passage").value;
  $("#cur-passage").innerHTML = popts; $("#cur-passage").value = ps.some((c) => c.passage_id === cur) ? cur : "";
  $("#d-passage").innerHTML = popts;
  $("#page-select").innerHTML = P.pages.map((pg) =>
    `<option value="${pg.page}">${pg.page} / ${P.pages.length}${pg.quality_flag !== "ok" ? " ⚠ " + pg.quality_flag : ""}</option>`).join("");
  if (!keepPage) page = P.pages[0]?.page ?? 1;
  $("#page-select").value = page;
  showRoundtrip(P.roundtrip);
  renderPage(); renderList();
  if (selected) renderDetail();
  scheduleReadPoll();
  setTimeout(() => preloadKetcher(), 2000);
}

async function showUsagePill() {
  try { MX = await api("/api/mathpix"); } catch { return; }
  let u = $("#mx-usage");
  if (!u) { u = document.createElement("span"); u.id = "mx-usage"; u.className = "pill"; $("#btn-verify").before(u); }
  u.className = "pill " + (MX.configured && !MX.limit_reached ? "fast" : "warn");
  u.textContent = !MX.configured ? "⚡ Mathpix key not set - boxes can't be read"
    : MX.limit_reached ? "⚡ Mathpix limit reached - boxes can't be read"
    : `⚡ Mathpix: ${MX.images} this month · ₹${Number(MX.inr).toLocaleString("en-IN")}` +
      (MX.limit_usd ? ` / ₹${Number(MX.limit_inr).toLocaleString("en-IN")}` : "");
}

// ---- background reading: boxes appear at once; their text arrives when the reader is done
let readTimer = null;
const readingSince = {};                          // crop id -> when this page first saw it reading
function scheduleReadPoll() {
  clearTimeout(readTimer);
  const n = P?.reading_queue ?? 0;
  const busy = n > 0 || (P?.questions ?? []).some((q) => q.status === "reading") ||
    (P?.crops ?? []).some((c) => c.status === "reading" && !c.superseded);
  let b = $("#read-queue");
  if (!b) {
    b = document.createElement("span"); b.id = "read-queue"; b.className = "pill warn";
    $("#btn-verify").before(b);
  }
  b.hidden = !busy;
  b.textContent = n > 1 ? `reading ${n} boxes in the background…` : "reading in the background…";
  if (busy) readTimer = setTimeout(refreshReading, 300);   // Mathpix: the text arrives in ~1-3 s
}
async function refreshReading() {
  if (!P) return;
  const id = P.id, before = Object.fromEntries(P.questions.map((q) => [q.id, q.status]));
  let fresh;
  try { fresh = await api(`/api/projects/${id}`); } catch { scheduleReadPoll(); return; }
  if (!P || P.id !== id) return;
  P = fresh;
  renderPage(); renderList();
  const q = P.questions.find((x) => x.id === selected);
  if (q && before[q.id] !== q.status) renderDetail();      // only when ITS reading finished: never wipe typing
  if (Object.values(before).includes("reading")) showUsagePill();
  scheduleReadPoll();
}
$("#page-select").onchange = (e) => { page = +e.target.value; renderPage(); };

function showRoundtrip(rt) {
  const el = $("#rt-report");
  if (!rt) { el.hidden = true; return; }
  el.hidden = false;
  const lost = rt.pages.reduce((a, p) => a + p.lost_ink_count, 0);
  el.className = "banner" + (rt.ok && !rt.hash_errors.length ? " ok" : "");
  const pix = rt.pages.every((p) => p.pixel_equal) && !rt.hash_errors.length;
  if (P.mode === "manual") {
    el.className = "banner" + (pix ? " ok" : "");
    el.textContent = (pix ? "Every crop matches the original page pixel for pixel. " : "Pixel or hash mismatch found! ") +
      (lost ? `${lost} ink regions on the pages are not inside any crop (outlined in red) - check no question was missed.` : "All ink on the pages is inside a crop.");
    return;
  }
  el.textContent = rt.ok && !rt.hash_errors.length
    ? "Round-trip check passed: crops pasted back match the original pages, nothing lost."
    : `Round-trip check: ${rt.hash_errors.length} hash errors, ${lost} uncropped ink regions` +
      (rt.pages.some((p) => !p.pixel_equal) ? ", pixel mismatch found" : "") + ". Uncropped regions are outlined in red.";
}

function pageInfo() { return P.pages.find((p) => p.page === page); }

function renderPage() {
  const pg = pageInfo(); if (!pg) return;
  const img = $("#page-img");
  img.src = `/api/projects/${P.id}/file/${pg.path}`;
  const ov = $("#overlay"); ov.innerHTML = "";
  const pct = (b) => `left:${(b.x0 / pg.width) * 100}%;top:${(b.y0 / pg.height) * 100}%;width:${((b.x1 - b.x0) / pg.width) * 100}%;height:${((b.y1 - b.y0) / pg.height) * 100}%`;
  const qByCrop = {};
  P.questions.forEach((q) => q.parts.forEach((pt) => (qByCrop[pt.crop_id] = q)));
  for (const c of P.crops) {
    if (c.page !== page || c.superseded) continue;
    const q = qByCrop[c.id] || P.questions.find((x) => x.id === c.parent_id);
    const d = document.createElement("div");
    const reading = (c.kind === "question" && q?.status === "reading") || c.status === "reading";
    const failed = (c.kind === "question" && q?.status === "error") || c.status === "error";   // Mathpix could not read it
    d.className = "box" + (q?.needs_review && c.kind === "question" ? " review" : "") + (q && q.id === selected ? " sel" : "") +
      (reading ? " reading" : "") + (failed ? " failed" : "");
    d.dataset.kind = c.kind; d.style.cssText = pct(c.box);
    const label = failed ? (c.kind === "question" && q ? `Q${q.q_no ?? "?"} · not read` : `${c.kind} · not read`)
      : c.kind === "question" && q ? `Q${q.q_no ?? "?"}${q.parts.length > 1 ? " part " + (q.parts.findIndex((p) => p.crop_id === c.id) + 1) : ""} · ${q.question_type}`
      : c.kind === "passage" ? `passage ${c.passage_id}` : c.kind;
    if (failed) d.title = (c.kind === "question" ? q?.read_error : c.text) || "not read";
    if (reading && !readingSince[c.id]) readingSince[c.id] = Date.now();
    const secs = reading ? ((Date.now() - readingSince[c.id]) / 1000).toFixed(1) : "";
    d.innerHTML = `<span class="tag">${esc(reading ? (c.kind === "question" ? `reading… ${secs} s` : `${c.kind} · reading… ${secs} s`) : label)}</span>`;
    if (q) d.onclick = (e) => { if (!drawMode()) { e.stopPropagation(); select(q.id); } };
    ov.appendChild(d);
    if (q && q.id === selected && c.user_box && c.snap_notes?.length) {   // the box as the user drew it
      const u = document.createElement("div");
      u.className = "userbox"; u.style.cssText = pct(c.user_box);
      u.title = "your box - snapped: " + c.snap_notes.join("; ");
      ov.appendChild(u);
    }
  }
  const rp = P.roundtrip?.pages?.find((p) => p.page === page);
  (rp?.lost_ink_regions ?? []).forEach((b) => {
    const d = document.createElement("div"); d.className = "rubber"; d.style.cssText = pct(b) + ";border-color:#dc2626";
    ov.appendChild(d);
  });
}

function renderList() {
  const only = $("#only-review").checked;
  const qs = P.questions.filter((q) => !only || q.needs_review);
  $("#q-count").textContent = `${qs.length} of ${P.questions.length} questions`;
  $("#q-list").innerHTML = qs.map((q) => `
    <li data-q="${esc(q.id)}" class="${q.id === selected ? "sel" : ""}">
      <b>Q${esc(q.q_no ?? "?")}</b><span>${esc(q.subject ?? "")}</span>
      <span class="snippet">${q.status === "reading" ? "<i>reading…</i>" : q.status === "error" ? '<i class="read-error">not read - open it to read again</i>'
        : esc((q.text_edited ?? baseText(q)).replace(/\s+/g, " ").trim().slice(0, 90))}</span>
      ${q.needs_review ? '<span class="flag" title="needs review">⚠</span>' : ""}${q.edited ? "✎" : ""}
    </li>`).join("");
}
$("#only-review").onchange = renderList;
$("#q-list").onclick = (e) => { const li = e.target.closest("li"); if (li) select(li.dataset.q); };

function select(qid) {
  selected = qid;
  const q = P.questions.find((x) => x.id === qid);
  if (q && q.page !== page) { page = q.page; $("#page-select").value = page; }
  renderPage(); renderList(); renderDetail();
}

const file = (rel) => `/api/projects/${P.id}/file/${rel}?t=${Date.now() % 1e7}`;
// a figure is shown as the clean drawing (dark on white); click it to see the locked photo crop
function figImg(path) {
  const c = P.crops.find((x) => x.path === path);
  const shown = c?.clean_path || path;
  return `<a href="${file(path)}" target="_blank" title="clean drawing - click for the original photo crop">` +
    `<img class="clean" src="${file(shown)}"></a>`;
}

function renderDetail() {
  const q = P.questions.find((x) => x.id === selected);
  $("#q-detail").hidden = !q; if (!q) return;
  $("#d-title").textContent = `Q${q.q_no ?? "?"} · page ${q.page}${q.subject ? " · " + q.subject : ""}`;
  $("#d-type").textContent = q.question_type + (q.multi_answer ? " (multi)" : "");
  const c = $("#d-conf");
  c.className = "pill " + (q.needs_review ? "warn" : "ok");
  c.textContent = `type ${(q.type_confidence * 100).toFixed(0)}% · OCR ${(q.ocr_confidence * 100).toFixed(0)}% · ${q.quality_flag}`;
  const by = q.read_by ? `Read by ${q.read_by}${q.read_seconds != null ? ` in ${q.read_seconds} s` : ""}. ` : "";
  $("#d-signals").textContent = by + (q.type_signals.length ? "Why: " + q.type_signals.join(" · ") : "");
  const snapNotes = q.parts.flatMap((p) => P.crops.find((c) => c.id === p.crop_id)?.snap_notes ?? []);
  $("#d-snap").textContent = snapNotes.length ? "Box snapped (older crop): " + [...new Set(snapNotes)].join(" · ") : "";
  $("#d-passage").value = q.passage_id ?? "";
  $("#d-reasons").innerHTML = q.review_reasons.map((r) => `<li>${esc(r)}</li>`).join("");
  $("#d-crops").innerHTML = q.parts.map((p, i) => `<div class="part"><span>Part ${i + 1} · page ${p.page}</span>` +
      (q.parts.length > 1 ? `<button class="link" data-rmpart="${i}">remove</button>` : "") +
      `<img src="${file(p.crop_path)}" alt="original crop"></div>`).join("") +
    (q.diagram_paths.length ? "<h4>Diagrams</h4>" + q.diagram_paths.map(figImg).join("") : "") +
    (q.option_images.length ? "<h4>Image options</h4>" + q.option_images.map(figImg).join("") : "");
  const ocr = $("#d-ocr");
  if (q.status === "error") {               // Mathpix could not read it (no internet, key, limit ...)
    ocr.innerHTML = `<p class="read-error">Mathpix could not read this box: ${esc(q.read_error || "unknown error")}</p>` +
      `<button type="button" id="b-reread">Read again</button>`;
    $("#b-reread").onclick = () => readAgain(q);
  } else {
    ocr.innerHTML = q.status === "reading" ? "<i>Reading this box in the background…</i>" : esc(q.latex_text || q.text);
    typeset(ocr);
  }
  $("#d-qtype").value = q.question_type;
  $("#d-qno").value = q.q_no ?? "";
  $("#d-subject").value = q.subject ?? "";
  setEditor(q, q.text_edited ?? baseText(q));
  $("#d-diagrams").innerHTML = q.diagram_edited_path ? `<h4>Edited diagram copy</h4><img src="${file(q.diagram_edited_path)}">` : "";
  updateDiff();
  renderPreview();
  renderStructs(q);
  const f = { options: q.options.length ? q.options.join(" | ") : "—",
    statements: q.statements.length ? q.statements.join(" | ") : "—", passage_id: q.passage_id ?? "—",
    source_type: q.source_type, blur_score: q.blur_score.toFixed(1), original_sha256: q.original_sha256.slice(0, 16) + "…",
    edits: q.edit_history.length };
  $("#d-fields").innerHTML = Object.entries(f).map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
}

function wordDiff(a, b) {
  const A = a.split(/(\s+)/), B = b.split(/(\s+)/);
  const n = A.length, m = B.length;
  if (n * m > 4e6) return esc(b);
  const dp = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  let i = 0, j = 0, out = "";
  while (i < n && j < m) {
    if (A[i] === B[j]) { out += esc(A[i]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) out += `<del>${esc(A[i++])}</del>`;
    else out += `<ins>${esc(B[j++])}</ins>`;
  }
  while (i < n) out += `<del>${esc(A[i++])}</del>`;
  while (j < m) out += `<ins>${esc(B[j++])}</ins>`;
  return out;
}
// The editable copy starts from the page layout: same lines, indents and spacing, with
// "[diagram 1]" etc. where each figure sits (older records: plain OCR text).
const baseText = (q) => q.display_latex || q.display_text || q.text;   // formulas as LaTeX $...$
function updateDiff() {
  const q = P.questions.find((x) => x.id === selected); if (!q) return;
  const v = getEditor(), base = baseText(q);
  $("#d-diff").innerHTML = v === base ? "" : "<b>Changes:</b> " + wordDiff(base, v);
}

// ---- editable copy: the text with the real diagram images in it --------------------------
// Stored as plain text with "[diagram 1]" / "[option image 2]" markers; shown with each
// marker replaced by that locked crop image, which can be dragged to another spot.
const FIG_RE = /\[(diagram|table|option image|figure) (\d+)\]/g;
const ed = $("#d-text");
function setEditor(q, text) {
  const figs = Object.fromEntries((q.figures || []).map((f) => [f.label, f.path]));
  ed.innerHTML = esc(text).replace(FIG_RE, (m, kind, n) => {
    const path = figs[`${kind} ${n}`];
    return path ? `<img class="fig" draggable="true" data-label="${kind} ${n}" src="${file(path)}" alt="${m}" title="${m} - drag to move">` : m;
  });
}
function getEditor() {
  let out = "";
  const walk = (node) => {
    for (const n of node.childNodes) {
      if (n.nodeType === Node.TEXT_NODE) out += n.nodeValue;
      else if (n.nodeName === "IMG" && n.dataset.label) out += `[${n.dataset.label}]`;
      else if (n.nodeName === "BR") out += "\n";
      else if (n.nodeName === "DIV" || n.nodeName === "P") { if (out && !out.endsWith("\n")) out += "\n"; walk(n); }
      else walk(n);
    }
  };
  walk(ed);
  return out.replace(/ /g, " ").replace(/\n$/, "");
}
ed.addEventListener("input", updateDiff);
// keep it plain text: Enter = newline, paste = text only (no foreign HTML / images)
ed.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); document.execCommand("insertText", false, "\n"); }
});
ed.addEventListener("paste", (e) => {
  e.preventDefault(); document.execCommand("insertText", false, e.clipboardData.getData("text/plain"));
});
// move a diagram: drag it and drop it where the text cursor shows
let dragged = null;
ed.addEventListener("dragstart", (e) => {
  if (e.target.nodeName !== "IMG" || !e.target.dataset.label) { e.preventDefault(); return; }
  dragged = e.target; dragged.classList.add("dragging");
  e.dataTransfer.effectAllowed = "move"; e.dataTransfer.setData("text/plain", "");
});
ed.addEventListener("dragend", () => { dragged?.classList.remove("dragging"); dragged = null; });
ed.addEventListener("dragover", (e) => { if (dragged) { e.preventDefault(); e.dataTransfer.dropEffect = "move"; } });
ed.addEventListener("drop", (e) => {
  e.preventDefault();
  if (!dragged) return;
  const range = document.caretRangeFromPoint ? document.caretRangeFromPoint(e.clientX, e.clientY)
    : (() => { const p = document.caretPositionFromPoint(e.clientX, e.clientY); if (!p) return null;
        const r = document.createRange(); r.setStart(p.offsetNode, p.offset); return r; })();
  if (!range || !ed.contains(range.startContainer) || dragged.contains(range.startContainer)) return;
  dragged.classList.remove("dragging");
  range.insertNode(dragged);              // moves the image (an element can only be in one place)
  dragged = null;
  updateDiff();
});

// One action at a time: reading a crop takes a minute, and a second box drawn meanwhile only
// waits in a queue on the server (it looked stuck, and the same box got drawn twice).
let working = false;
async function act(fn, msg = "Saving…") {
  if (working) { flashBusy(); return; }
  working = true;
  const t0 = Date.now(), busy = $("#busy");
  const tick = () => { const s = Math.round((Date.now() - t0) / 1000); busy.textContent = s ? `${msg} ${s} s` : msg; };
  tick(); busy.hidden = false;
  const timer = setInterval(tick, 1000);
  try {
    await fn(); await openProject(P.id, true);
    const nextP = $("#cur-passage").dataset.next;           // new passage -> make it current
    if (nextP) { $("#cur-passage").value = nextP; delete $("#cur-passage").dataset.next; }
  } catch (e) { alert(e.message); }
  finally { clearInterval(timer); busy.hidden = true; working = false; }
}
function flashBusy() {
  const b = $("#busy");
  b.classList.remove("flash"); void b.offsetWidth; b.classList.add("flash");
}
const qurl = (suffix) => `/api/projects/${P.id}/questions/${selected}/${suffix}`;
// Read again = the same box drawn again: a new locked crop, read by Mathpix in the background
// (the old crop is kept and can be restored with undo). Only the parts that were not read.
function readAgain(q) {
  const parts = q.parts.map((p, i) => [p, i]).filter(([p]) => p.status === "error");
  return act(async () => {
    for (const [p, i] of parts)
      await api(qurl("recrop") + `?part=${i}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(p.box) });
  }, "Reading again…");
}
$("#b-save").onclick = () => act(() => api(qurl("text"), { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text: getEditor() }) }));
$("#b-labels").onclick = () => act(() => api(qurl("fields"), { method: "POST", headers: { "content-type": "application/json" },
  body: JSON.stringify({ question_type: $("#d-qtype").value, q_no: /^\d+$/.test($("#d-qno").value) ? +$("#d-qno").value : null,
    subject: $("#d-subject").value, passage_id: $("#d-passage").value }) }));
$("#d-crops").addEventListener("click", (e) => {
  const i = e.target.dataset.rmpart;
  if (i !== undefined && confirm("Remove this part from the question? Its crop file is kept."))
    act(() => api(`/api/projects/${P.id}/questions/${selected}/parts/${i}`, { method: "DELETE" }));
});
$("#b-undo").onclick = () => act(() => api(qurl("undo"), { method: "POST" }));
$("#b-reset").onclick = () => { if (confirm("Reset text and diagram to the original?")) act(() => api(qurl("reset"), { method: "POST" })); };
$("#b-delete").onclick = () => {
  if (!confirm("Remove this question from the output? Its original crop files are kept.")) return;
  act(async () => { await api(`/api/projects/${P.id}/questions/${selected}`, { method: "DELETE" }); selected = null; });
};
$("#d-diagram-file").onchange = (e) => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  act(() => api(qurl("diagram"), { method: "POST", body: fd }));
  e.target.value = "";
};
$("#btn-verify").onclick = () => act(() => api(`/api/projects/${P.id}/verify`, { method: "POST" }));

// ------------------------------------------------------------------ manual cropping
const drawMode = () => $("#draw-kind").value;
$("#draw-kind").onchange = () => $("#overlay").classList.toggle("drawing", !!drawMode());
const KEYS = { q: "question", p: "part", d: "diagram", t: "table", i: "option_image", a: "passage", s: "section", h: "header", m: "recrop" };
document.addEventListener("keydown", (e) => {
  if ($("#review").hidden || e.ctrlKey || e.metaKey || e.altKey || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName) || e.target.isContentEditable) return;
  const k = e.key.toLowerCase();
  if (k === "escape") $("#draw-kind").value = "";
  else if (KEYS[k]) $("#draw-kind").value = KEYS[k];
  else if (k === "arrowright" || k === "arrowleft") {
    const n = page + (k === "arrowright" ? 1 : -1);
    if (P.pages.some((p) => p.page === n)) { page = n; $("#page-select").value = n; renderPage(); }
    return;
  } else return;
  $("#overlay").classList.toggle("drawing", !!drawMode());
});
let start = null, rubber = null;
const ov = $("#overlay");
function toPage(e) {
  const r = ov.getBoundingClientRect(), pg = pageInfo();
  return { x: Math.round(((e.clientX - r.left) / r.width) * pg.width), y: Math.round(((e.clientY - r.top) / r.height) * pg.height), px: e.clientX - r.left, py: e.clientY - r.top };
}
ov.addEventListener("pointerdown", (e) => {
  if (!drawMode()) return;
  if (working) { e.preventDefault(); flashBusy(); return; }   // still reading the last box
  e.preventDefault(); ov.setPointerCapture(e.pointerId);
  start = toPage(e); rubber = document.createElement("div"); rubber.className = "rubber"; ov.appendChild(rubber);
});
ov.addEventListener("pointermove", (e) => {
  if (!start) return;
  const p = toPage(e);
  Object.assign(rubber.style, { left: Math.min(start.px, p.px) + "px", top: Math.min(start.py, p.py) + "px", width: Math.abs(p.px - start.px) + "px", height: Math.abs(p.py - start.py) + "px" });
});
ov.addEventListener("pointercancel", () => { start = null; rubber?.remove(); });   // drag interrupted
ov.addEventListener("pointerup", async (e) => {
  if (!start) return;
  const p = toPage(e), s = start; start = null; rubber.remove();
  const box = { x0: Math.min(s.x, p.x), y0: Math.min(s.y, p.y), x1: Math.max(s.x, p.x), y1: Math.max(s.y, p.y) };
  if (box.x1 - box.x0 < 8 || box.y1 - box.y0 < 8) return;
  const mode = drawMode();
  const needsQ = ["part", "diagram", "table", "option_image", "recrop"].includes(mode);
  if (needsQ && !selected) { alert("Select a question first (press Esc and click it)."); return; }
  if (mode === "recrop") {
    const q = P.questions.find((x) => x.id === selected);
    let part = q.parts.findIndex((p) => p.page === page);
    if (part < 0) { alert("Draw the new box on the page where that question is."); return; }
    if (q.parts.filter((p) => p.page === page).length > 1) {
      const cx = (box.x0 + box.x1) / 2, cy = (box.y0 + box.y1) / 2;
      part = q.parts.reduce((best, p, i) => p.page !== page ? best :
        (Math.hypot((p.box.x0 + p.box.x1) / 2 - cx, (p.box.y0 + p.box.y1) / 2 - cy) <
         Math.hypot((q.parts[best].box.x0 + q.parts[best].box.x1) / 2 - cx, (q.parts[best].box.y0 + q.parts[best].box.y1) / 2 - cy) ? i : best), part);
    }
    return act(() => api(qurl("recrop") + `?part=${part}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(box) }));
  }
  act(async () => {
    const r = await api(`/api/projects/${P.id}/crops`, { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ page, box, kind: mode, parent_id: needsQ ? selected : null,
        subject: $("#cur-subject").value || null, passage_id: mode === "question" ? ($("#cur-passage").value || null) : null }) });
    if (r.question) selected = r.question.id;
    if (mode === "passage" && r.crop?.passage_id) $("#cur-passage").dataset.next = r.crop.passage_id;
  });
});

loadHome();


// ---- maths: MathJax + mhchem render $...$ / $$...$$ / \ce{...} (offline copy in /static/vendor)
const figMap = (q) => Object.fromEntries((q?.figures || []).map((f) => [f.label, f.path]));
function mathHTML(text, figs) {
  return esc(text).replace(FIG_RE, (m, kind, n) => {
    const p = figs[`${kind} ${n}`];
    return p ? `<img class="fig" src="${file(p)}" alt="${m}">` : m;
  });
}
async function typeset(el, tries = 0) {
  if (!window.MathJax?.typesetPromise) {                  // still loading (deferred script)
    if (tries < 50) setTimeout(() => typeset(el, tries + 1), 200);
    return;
  }
  const t0 = performance.now();
  try {
    MathJax.typesetClear?.([el]);
    await MathJax.typesetPromise([el]);
    el.dataset.renderMs = (performance.now() - t0).toFixed(1);   // speed check (dev tools)
  } catch (e) { console.warn("MathJax:", e); }
}
let previewTimer = null;
function renderPreview() {
  const q = P?.questions.find((x) => x.id === selected); if (!q) return;
  const el = $("#d-preview");
  el.innerHTML = q.status === "reading" ? "<i>The text appears here when the box has been read.</i>"
    : mathHTML(getEditor(), figMap(q));
  typeset(el);
}
ed.addEventListener("input", () => { clearTimeout(previewTimer); previewTimer = setTimeout(renderPreview, 150); });

// ---- figures & chemical structures: Ketcher (draw), SmilesDrawer (show SMILES), 3Dmol (3D)
function renderStructs(q) {
  const el = $("#d-structs");
  const figs = q.figures || [];
  if (!figs.length) { el.innerHTML = ""; return; }
  el.innerHTML = "<h4>Figures &amp; structures</h4>" + figs.map((f, i) => `
    <div class="struct-card">
      <img src="${file(f.path)}" alt="${esc(f.label)}">
      <div><b>${esc(f.label)}</b>${f.smiles ? `<br><code title="SMILES">${esc(f.smiles)}</code>` : ""}<br>
        <button data-chem="${i}">${f.smiles ? "Edit structure" : "Draw structure"}</button>
        ${f.smiles ? `<button data-mol3d="${i}">3D</button>` : ""}
        <a href="${file(f.original)}" target="_blank">original photo</a></div>
    </div>`).join("");
}
$("#d-structs").addEventListener("click", (e) => {
  const q = P.questions.find((x) => x.id === selected); if (!q) return;
  const b = e.target.closest("button"); if (!b) return;
  if (b.dataset.chem !== undefined) openChem(q.figures[+b.dataset.chem]);
  if (b.dataset.mol3d !== undefined) open3D(q.figures[+b.dataset.mol3d]);
});

let chemFig = null;
const kframe = $("#chem-ketcher");
async function ketcherReady() {
  if (!kframe.getAttribute("src")) kframe.setAttribute("src", "/static/vendor/ketcher/index.html");
  for (let i = 0; i < 300; i++) {                         // first open: the editor loads (~2-5 s)
    const k = kframe.contentWindow?.ketcher;
    if (k) return k;
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error("The structure editor did not load. Run: python -m examocr.cli download-models");
}
let sdTimer = null;
function drawSmiles(smiles) {
  const svg = $("#chem-sd"); svg.innerHTML = "";
  if (!smiles || !window.SmilesDrawer) return;
  const t0 = performance.now();
  try {
    new SmilesDrawer.SmiDrawer({ width: 260, height: 190 }).draw(smiles, svg, "light",
      () => { svg.dataset.renderMs = (performance.now() - t0).toFixed(1); },
      () => { $("#chem-msg").textContent = "This SMILES is not a valid molecule."; });
  } catch (e) { $("#chem-msg").textContent = "This SMILES is not a valid molecule."; }
}
$("#chem-smiles").addEventListener("input", (e) => {
  $("#chem-msg").textContent = "";
  clearTimeout(sdTimer); sdTimer = setTimeout(() => drawSmiles(e.target.value.trim()), 120);
});
async function openChem(f) {
  chemFig = f;
  const crop = P.crops.find((c) => c.id === f.crop_id) || {};
  $("#chem-title").textContent = `${f.label} - structure`;
  $("#chem-ref").src = file(crop.clean_path || f.original);
  const smi = f.smiles || crop.smiles_suggested || "";
  $("#chem-smiles").value = smi;
  $("#chem-msg").textContent = crop.smiles_suggested && !f.smiles
    ? `Suggested by Mathpix - check it, then "Use in editor".` : "";
  $("#chem-clear").hidden = !f.smiles;
  drawSmiles(smi);
  showChem(true);
  try {
    const k = await ketcherReady();
    await k.setMolecule(crop.molfile || smi || "");
  } catch (e) { $("#chem-msg").textContent = e.message; }
}
// The structure panel is always in the page: closed = moved off-screen, never hidden. A hidden
// frame does not start Ketcher; this way it loads in the background while you work.
function showChem(on) {
  $("#chem-dlg").classList.toggle("open", on);
  $("#chem-backdrop").hidden = !on;
}
$("#chem-close").onclick = () => showChem(false);
$("#chem-backdrop").onclick = () => showChem(false);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("#chem-dlg").classList.contains("open")) showChem(false); });
function preloadKetcher() {                       // ~5 s, once, a moment after a paper is opened
  if (!kframe.getAttribute("src")) kframe.setAttribute("src", "/static/vendor/ketcher/index.html");
}
$("#chem-use").onclick = async () => {
  const smi = $("#chem-smiles").value.trim(); if (!smi) return;
  try { await (await ketcherReady()).setMolecule(smi); } catch (e) { $("#chem-msg").textContent = e.message; }
};
$("#chem-recog").onclick = async () => {
  const msg = $("#chem-msg");
  msg.textContent = "Mathpix is reading the drawing…";
  try {
    const r = await api(`/api/projects/${P.id}/crops/${chemFig.crop_id}/recognise`, { method: "POST" });
    $("#chem-smiles").value = r.smiles; drawSmiles(r.smiles);
    msg.textContent = `Read by Mathpix. Check it, then "Use in editor".`;
  } catch (e) { msg.textContent = e.message; }
};
$("#chem-save").onclick = async () => {
  try {
    const k = await ketcherReady();
    // the molfile only: RDKit on the server checks it and makes the SMILES (asking Ketcher
    // for the SMILES too costs another ~0.4 s)
    const molfile = await k.getMolfile();
    if (!/^\s*[1-9]/m.test(molfile.split("\n")[3] || "")) {
      $("#chem-msg").textContent = "The editor is empty - draw the structure first."; return;
    }
    await api(`/api/projects/${P.id}/crops/${chemFig.crop_id}/structure`, {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ molfile }) });
    showChem(false);
    await openProject(P.id, true);
  } catch (e) { $("#chem-msg").textContent = e.message; }
};
$("#chem-clear").onclick = async () => {
  await api(`/api/projects/${P.id}/crops/${chemFig.crop_id}/structure`, { method: "DELETE" });
  showChem(false);
  await openProject(P.id, true);
};

let viewer3d = null;
async function open3D(f) {
  $("#mol3d-title").textContent = `${f.label} - 3D (${f.smiles})`;
  $("#mol3d-dlg").showModal();
  try {
    const t0 = performance.now();
    const r = await api(`/api/chem/3d?smiles=${encodeURIComponent(f.smiles)}`);
    if (!viewer3d) viewer3d = $3Dmol.createViewer($("#mol3d"), { backgroundColor: "white" });
    viewer3d.clear();
    viewer3d.addModel(r.molblock, "mol");
    viewer3d.setStyle({}, { stick: { radius: 0.14 }, sphere: { scale: 0.25 } });
    viewer3d.zoomTo(); viewer3d.render();
    $("#mol3d").dataset.renderMs = (performance.now() - t0).toFixed(1);
  } catch (e) { $("#mol3d").textContent = e.message; }
}
$("#mol3d-close").onclick = () => $("#mol3d-dlg").close();
