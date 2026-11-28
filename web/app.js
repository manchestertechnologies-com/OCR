"use strict";
// GENPREP · Exam Question Builder (frontend). The server (examocr/api) is unchanged: every box
// the user draws is cut from the original page and read by Mathpix in the background.
const $ = (s) => document.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const api = async (url, opts = {}) => {
  let r;
  try { r = await fetch(url, opts); }
  catch { throw new Error("cannot reach the app server - is it running? Start it with: python -m examocr.cli serve"); }
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(msg);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const JSONPOST = (body) => ({ method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

let P = null;        // current project
let page = 1;
let selected = null; // question id
let pollTimer = null;
const PROJ = new Map();   // project id -> project (question bank / test)

const TYPE_NAMES = { mcq: "MCQ (one answer)", multi_correct: "Multiple correct", numerical: "Numerical", match: "Match the columns",
  assertion_reason: "Assertion – Reason", statement: "Statements", paragraph: "Paragraph / passage", diagram: "Diagram based",
  arrange: "Arrange in order", other: "Other" };
const TYPE_SHORT = { mcq: "MCQ", multi_correct: "Multi-correct", numerical: "Numerical", match: "Match", assertion_reason: "A–R",
  statement: "Statements", paragraph: "Paragraph", diagram: "Diagram", arrange: "Arrange", other: "Other" };

// ------------------------------------------------------------------ browser storage
// Drafts, answer keys and the test live in this browser only (the server has no fields for them).
const LS = {
  get(k, d) { try { const v = localStorage.getItem("genprep." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("genprep." + k, JSON.stringify(v)); } catch {} },
};
// A question cropped in this browser stays a draft until "Save question" - then it is listed
// in the Questions panel. Questions made before (or elsewhere) count as saved.
const draftsOf = (pid) => LS.get("drafts", {})[pid] || [];
const isDraft = (pid, qid) => draftsOf(pid).includes(qid);
function setDraft(pid, qid, on) {
  const all = LS.get("drafts", {}), s = new Set(all[pid] || []);
  on ? s.add(qid) : s.delete(qid);
  if (s.size) all[pid] = [...s]; else delete all[pid];
  LS.set("drafts", all);
}
function pruneDrafts() {
  const ids = new Set(P.questions.map((q) => q.id));
  for (const qid of draftsOf(P.id)) if (!ids.has(qid)) setDraft(P.id, qid, false);
}
const answers = () => LS.get("answers", {});
const answerOf = (pid, qid) => answers()[`${pid}/${qid}`] || {};
function setAnswer(pid, qid, a) {
  const all = answers(), k = `${pid}/${qid}`;
  if ((a.keys?.length) || (a.text || "").trim()) all[k] = a; else delete all[k];
  LS.set("answers", all);
}

// ------------------------------------------------------------------ toasts & confirm
function toast(msg, kind = "", ms) {
  const t = document.createElement("div");
  t.className = "toast " + kind; t.textContent = msg; t.title = "click to close";
  t.onclick = () => t.remove();
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), ms ?? (kind === "bad" ? 9000 : 3800));
}
function ask(title, text = "", ok = "OK", danger = false) {
  const d = $("#ask-dlg");
  $("#ask-title").textContent = title; $("#ask-text").textContent = text;
  const y = $("#ask-yes"); y.textContent = ok; y.className = danger ? "danger" : "primary";
  d.returnValue = "";
  d.showModal(); y.focus();
  return new Promise((res) => d.addEventListener("close", () => res(d.returnValue === "yes"), { once: true }));
}

// ------------------------------------------------------------------ views (tabs)
const VIEWS = { papers: "#home", build: "#review", bank: "#bank", test: "#test" };
let view = "papers";
function showView(v) {
  view = v;
  for (const [k, sel] of Object.entries(VIEWS)) $(sel).hidden = k !== v;
  $$(".tab").forEach((t) => { t.classList.toggle("active", t.dataset.tab === v); t.setAttribute("aria-selected", t.dataset.tab === v); });
  if (v === "build") { $("#review-empty").hidden = !!P; $("#review-body").hidden = !P; }
  const h = v === "build" && P ? `#/build/${P.id}` : `#/${v}`;
  if (location.hash !== h) history.replaceState(null, "", h);
}
function go(v) {
  const h = v === "build" && P ? `#/build/${P.id}` : `#/${v}`;
  if (location.hash === h) route(); else location.hash = h;
}
function route() {
  const m = location.hash.match(/^#\/(\w+)(?:\/(\w+))?/) || [];
  const v = m[1], id = m[2];
  if (v === "build" && id && id !== P?.id)
    return openProject(id).catch((e) => { toast("Could not open the paper: " + e.message, "bad"); go("papers"); });
  if (v === "build") { showView("build"); if (P) { lastListHTML = ""; renderList(); } return; }
  if (v === "bank") return loadBank();
  if (v === "test") return loadTest();
  loadHome();
}
window.addEventListener("hashchange", route);
$$(".tab").forEach((t) => (t.onclick = () => go(t.dataset.tab)));
$$("[data-goto]").forEach((b) => (b.onclick = () => go(b.dataset.goto)));
$(".brand").onclick = (e) => { e.preventDefault(); go("papers"); };

// ------------------------------------------------------------------ home (papers)
// Warn when the server runs older Python code than what is on disk (it loads code once at
// start; this page is re-read every time). A server too old to know /api/version is stale too.
async function checkStale() {
  let stale;
  try { const r = await fetch("/api/version"); stale = !r.ok || (await r.json()).stale; }
  catch { return; }                                     // server unreachable: other errors show
  $("#stale").hidden = !stale;
}

let PAPERS = [];
async function loadHome() {
  checkStale();
  showView("papers");
  try {
    const exams = await api("/api/exams");
    const keep = $("#exam").value || LS.get("exam", "");
    $("#exam").innerHTML = exams.map((e) => `<option>${esc(e)}</option>`).join("");
    if (exams.includes(keep)) $("#exam").value = keep;
  } catch (e) { toast(e.message, "bad"); }
  showMathpixStatus();
  try { PAPERS = await api("/api/projects"); } catch (e) { PAPERS = []; toast(e.message, "bad"); }
  renderPapers();
}
const fmtDate = (s) => { const d = new Date(s); return isNaN(d) ? (s || "") : d.toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" }); };
function renderPapers() {
  const s = $("#paper-search").value.trim().toLowerCase();
  const rows = PAPERS.filter((p) => !s || `${p.source_name} ${p.exam}`.toLowerCase().includes(s));
  $("#projects tbody").innerHTML = rows.map((p) => `
    <tr data-open="${esc(p.id)}"><td title="${esc(p.source_name)}">${esc(p.source_name)}<div class="hint">${esc(fmtDate(p.created))}</div></td>
    <td>${esc(p.exam)}</td>
    <td><span class="pill ${p.status === "ready" ? "ok" : p.status === "error" ? "bad" : "warn"}">${esc(p.status)}</span></td>
    <td>${p.questions}</td><td>${p.needs_review ? `<span class="pill warn">${p.needs_review}</span>` : "0"}</td>
    <td><button class="link" data-open="${esc(p.id)}">Open →</button></td></tr>`).join("")
    || `<tr><td colspan="6" class="hint">${PAPERS.length ? "No paper matches the search." : "No papers yet. Upload your first question paper on the left."}</td></tr>`;
  renderHomeStats();
}
function stat(v, label, cls = "") { return `<div class="stat ${cls}"><b>${esc(v)}</b><span>${esc(label)}</span></div>`; }
function renderHomeStats() {
  const q = PAPERS.reduce((a, p) => a + p.questions, 0), r = PAPERS.reduce((a, p) => a + p.needs_review, 0);
  $("#home-stats").innerHTML = stat(PAPERS.length, "papers") + stat(q, "questions") + stat(r, "need review");
}
$("#paper-search").addEventListener("input", renderPapers);

let MX = null;                                    // Mathpix status: key set? usage this month
async function showMathpixStatus() {
  try { MX = await api("/api/mathpix"); } catch { return; }
  renderHomeStats();
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
$("#projects").addEventListener("click", (e) => { const el = e.target.closest("[data-open]"); if (el) openProject(el.dataset.open); });

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
$("#exam").addEventListener("change", () => LS.set("exam", $("#exam").value));

$("#upload-form").onsubmit = async (e) => {
  e.preventDefault();
  const f = $("#file").files[0];
  if (!f) {
    // The real <input type=file> is hidden, so the browser can't show its own "required" bubble.
    drop.classList.add("missing");
    $("#drop-text").textContent = "Choose a file first: click here or drop a PDF, Word file, scan or photo.";
    return;
  }
  $("#progress").hidden = false; $("#progress-msg").textContent = "Uploading…";
  $("#bar-fill").style.width = "0%";
  setUploading(true);                     // a second click would create a duplicate paper
  try {                                   // any problem is shown here - an upload never fails silently
    const fd = new FormData(); fd.append("file", f); fd.append("exam", $("#exam").value);
    fd.append("mode", "manual");          // the user crops; the app never auto-crops
    LS.set("exam", $("#exam").value);
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
    else {
      $("#progress").hidden = true;
      $("#file").value = ""; showFile();
      if (view === "papers") openProject(id);
      else toast(`"${p.source_name}" is ready - open it from Papers.`, "ok");
    }
  };
  pollTimer = setTimeout(tick, 1000);
}

// ------------------------------------------------------------------ build (review)
async function openProject(id, keepPage = false) {
  if (!keepPage) checkStale();
  if (!$("#d-qtype").options.length)
    $("#d-qtype").innerHTML = (await api("/api/question-types")).map((t) => `<option value="${esc(t)}">${esc(TYPE_NAMES[t] || t)}</option>`).join("");
  const fresh = await api(`/api/projects/${id}`);
  const changed = P?.id !== id;
  P = fresh; PROJ.set(id, P);
  if (changed) { selected = null; pending = null; builderMode = "empty"; lastListHTML = ""; lastThumbs = ""; subTab("question"); }
  if (selected && !P.questions.some((q) => q.id === selected)) selected = null;
  pruneDrafts();
  if (!keepPage || view === "build") showView("build");
  $("#r-title").textContent = `${P.source_name} · ${P.exam}`;
  $("#r-title").title = $("#r-title").textContent;
  showUsagePill();
  const st = $("#r-status"); st.textContent = P.status;
  st.className = "pill " + (P.status === "ready" ? "ok" : P.status === "error" ? "bad" : "warn");
  st.title = P.error || "";               // hover the red pill to see why a paper failed
  $("#btn-export").href = `/api/projects/${id}/export`;
  if (!keepPage) {
    let subs = [];
    try { subs = (await api(`/api/exams/${P.exam}`)).subjects || []; } catch {}
    $("#cur-subject").innerHTML = `<option value="">auto</option>` + subs.map((s) => `<option>${esc(s)}</option>`).join("");
    $("#subjects-list").innerHTML = subs.map((s) => `<option value="${esc(s)}">`).join("");
    const all = [...new Set([...subs, ...P.questions.map((q) => q.subject).filter(Boolean)])];
    $("#q-subject-filter").innerHTML = `<option value="">All subjects</option>` + all.map((s) => `<option>${esc(s)}</option>`).join("");
    $("#draw-kind").value = P.mode === "manual" ? "question" : "";
    $("#overlay").classList.toggle("drawing", !!drawMode());
  }
  const ps = P.crops.filter((c) => c.kind === "passage" && !c.superseded);
  const popts = `<option value="">none</option>` + ps.map((c) => `<option value="${esc(c.passage_id)}">${esc(c.passage_id)}: ${esc(c.text.slice(0, 30))}</option>`).join("");
  const cur = $("#cur-passage").value;
  $("#cur-passage").innerHTML = popts; $("#cur-passage").value = ps.some((c) => c.passage_id === cur) ? cur : "";
  $("#d-passage").innerHTML = popts;
  $("#page-select").innerHTML = P.pages.map((pg) =>
    `<option value="${pg.page}">Page ${pg.page} / ${P.pages.length}${pg.quality_flag !== "ok" ? " ⚠ " + pg.quality_flag : ""}</option>`).join("");
  if (!keepPage) page = P.pages[0]?.page ?? 1;
  $("#page-select").value = page;
  if (!keepPage) $("#thumbs").classList.toggle("off", !LS.get("thumbs", P.pages.length > 1));
  showRoundtrip(P.roundtrip);
  renderThumbs(); renderPage(); renderList();
  if (selected) renderDetail(); else showBuilder();
  scheduleReadPoll();
  setTimeout(() => preloadKetcher(), 2000);
}

async function showUsagePill() {               // keeps the Mathpix status (price per read) up to date
  try { MX = await api("/api/mathpix"); } catch { return; }
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
  b.textContent = n > 1 ? `⏳ reading ${n} boxes in the background…` : "⏳ reading in the background…";
  if (busy) readTimer = setTimeout(refreshReading, 300);   // Mathpix: the text arrives in ~1-3 s
}
async function refreshReading() {
  if (!P) return;
  const id = P.id, before = Object.fromEntries(P.questions.map((q) => [q.id, q.status]));
  let fresh;
  try { fresh = await api(`/api/projects/${id}`); } catch { scheduleReadPoll(); return; }
  if (!P || P.id !== id) return;
  P = fresh; PROJ.set(id, P);
  renderPage(); renderList();
  const q = P.questions.find((x) => x.id === selected);
  if (q && before[q.id] !== q.status) {                  // only when ITS reading finished: never wipe typing
    renderDetail();
    if (q.status === "ready") toast(`Q${q.q_no ?? "?"} read${q.read_seconds != null ? ` in ${q.read_seconds} s` : ""}. Check it, then Save question.`, "ok");
    else if (q.status === "error") toast(`Mathpix could not read Q${q.q_no ?? "?"}: ${q.read_error || "unknown error"}`, "bad");
  }
  if (Object.values(before).includes("reading")) showUsagePill();
  scheduleReadPoll();
}
$("#page-select").onchange = (e) => goPage(+e.target.value);
$("#pg-prev").onclick = () => goPage(page - 1);
$("#pg-next").onclick = () => goPage(page + 1);
function goPage(n) {
  if (!P?.pages.some((p) => p.page === n)) return;
  page = n; $("#page-select").value = n; renderPage();
  $("#page-scroll").scrollTop = 0;
}

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
const pctOf = (b, pg) => `left:${(b.x0 / pg.width) * 100}%;top:${(b.y0 / pg.height) * 100}%;width:${((b.x1 - b.x0) / pg.width) * 100}%;height:${((b.y1 - b.y0) / pg.height) * 100}%`;
function qByCropMap() {
  const m = {};
  P.questions.forEach((q) => q.parts.forEach((pt) => (m[pt.crop_id] = q)));
  return m;
}

function renderPage() {
  const pg = pageInfo(); if (!pg) return;
  const img = $("#page-img");
  const src = `/api/projects/${P.id}/file/${pg.path}`;
  if (img.getAttribute("src") !== src) img.src = src;
  const ov = $("#overlay"); ov.innerHTML = "";
  const pct = (b) => pctOf(b, pg);
  const qByCrop = qByCropMap();
  for (const c of P.crops) {
    if (c.page !== page || c.superseded) continue;
    const q = qByCrop[c.id] || P.questions.find((x) => x.id === c.parent_id);
    const d = document.createElement("div");
    const reading = (c.kind === "question" && q?.status === "reading") || c.status === "reading";
    const failed = (c.kind === "question" && q?.status === "error") || c.status === "error";   // Mathpix could not read it
    d.className = "box" + (q?.needs_review && c.kind === "question" ? " review" : "") + (q && q.id === selected ? " sel" : "") +
      (reading ? " reading" : "") + (failed ? " failed" : "") + (q && c.kind === "question" && isDraft(P.id, q.id) ? " draft" : "");
    d.dataset.kind = c.kind; d.style.cssText = pct(c.box);
    const label = failed ? (c.kind === "question" && q ? `Q${q.q_no ?? "?"} · not read` : `${c.kind} · not read`)
      : c.kind === "question" && q ? `Q${q.q_no ?? "?"}${q.parts.length > 1 ? " part " + (q.parts.findIndex((p) => p.crop_id === c.id) + 1) : ""} · ${q.question_type}`
        + (isDraft(P.id, q.id) ? " · draft" : "")
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
  if (pending && pending.page === page) drawPending();
  if (start && rubber) ov.appendChild(rubber);           // a box being drawn survives a refresh
  $("#pg-prev").disabled = !P.pages.some((p) => p.page === page - 1);
  $("#pg-next").disabled = !P.pages.some((p) => p.page === page + 1);
  $$(".thumb").forEach((t) => t.classList.toggle("active", +t.dataset.page === page));
}

// ---- page thumbnails & zoom
let lastThumbs = "";
function renderThumbs() {
  const counts = {};
  P.questions.forEach((q) => q.parts.forEach((pt) => (counts[pt.page] = (counts[pt.page] || 0) + 1)));
  const html = P.pages.map((pg) => `<button class="thumb${pg.page === page ? " active" : ""}" data-page="${pg.page}" title="Page ${pg.page}">` +
    `<img loading="lazy" src="/api/projects/${P.id}/file/${pg.path}" alt=""><span>${pg.page}</span>${counts[pg.page] ? `<i>${counts[pg.page]}</i>` : ""}</button>`).join("");
  if (html !== lastThumbs) { $("#thumbs").innerHTML = html; lastThumbs = html; }
}
$("#thumbs").addEventListener("click", (e) => { const t = e.target.closest(".thumb"); if (t) goPage(+t.dataset.page); });
$("#thumbs-toggle").onclick = () => { const off = $("#thumbs").classList.toggle("off"); LS.set("thumbs", !off); };
let zoom = +LS.get("zoom", 1) || 1;
function setZoom(z) {
  zoom = clamp(Math.round(z * 100) / 100, 0.4, 4);
  LS.set("zoom", zoom);
  $("#page-wrap").style.setProperty("--zoom", zoom);
  $("#zoom-label").textContent = Math.round(zoom * 100) + "%";
}
setZoom(zoom);
$("#zoom-in").onclick = () => setZoom(zoom * 1.25);
$("#zoom-out").onclick = () => setZoom(zoom / 1.25);
$("#zoom-fit").onclick = () => setZoom(1);
$("#page-scroll").addEventListener("wheel", (e) => {
  if (!e.ctrlKey) return;
  e.preventDefault(); setZoom(zoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1));
}, { passive: false });
$("#page-img").addEventListener("load", () => { if (pending) drawPendCanvas(); });

// ---- the Questions panel (bottom): saved questions as cards
const qText = (q) => q.text_edited ?? baseText(q);
const GREEK = { alpha: "α", beta: "β", gamma: "γ", delta: "δ", theta: "θ", lambda: "λ", mu: "μ", pi: "π", sigma: "σ", omega: "ω", Delta: "Δ", Omega: "Ω", circ: "°", degree: "°" };
// a short, readable one-liner of a LaTeX text (cards and lists; the full text is rendered by MathJax)
const plain = (s) => String(s || "").replace(FIG_RE, "▧")
  .replace(/\\(?:frac|dfrac|tfrac)\{([^{}]*)\}\{([^{}]*)\}/g, "($1)/($2)")
  .replace(/\\(?:ce|mathrm|text|mathbf|operatorname|mathit|textbf)\{([^{}]*)\}/g, "$1")
  .replace(/\\(?:left|right|displaystyle|quad|qquad)|\\[,;!]/g, " ")
  .replace(/\\times/g, "×").replace(/\\cdot/g, "·").replace(/\\(?:rightarrow|to|longrightarrow)/g, "→").replace(/\\(?:rightleftharpoons|leftrightarrow)/g, "⇌")
  .replace(/\\(alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|omega|Delta|Omega|circ|degree)\b/g, (m, g) => GREEK[g])
  .replace(/\\([a-zA-Z]+)/g, "$1").replace(/[{}$]/g, "").replace(/\s+/g, " ").trim();
let dockF = "saved", lastListHTML = "", flashCard = null;
function stateOf(q, pid = P.id) {
  if (q.status === "reading") return ["warn", "⏳ reading"];
  if (q.status === "error") return ["bad", "not read"];
  if (isDraft(pid, q.id)) return ["warn", "draft"];
  return ["ok", "✓ saved"];
}
function renderList() {
  const only = $("#only-review").checked, s = $("#q-search").value.trim().toLowerCase(), sub = $("#q-subject-filter").value;
  const drafts = new Set(draftsOf(P.id));
  $("#n-saved").textContent = P.questions.filter((q) => !drafts.has(q.id)).length;
  $("#n-draft").textContent = drafts.size || "";
  $("#q-filter [data-f=draft]").classList.toggle("has", drafts.size > 0);
  const qs = P.questions.filter((q) => (dockF === "all" || (dockF === "draft") === drafts.has(q.id)) && (!only || q.needs_review) &&
    (!sub || q.subject === sub) && (!s || String(q.q_no) === s.replace(/^q/, "") || plain(qText(q)).toLowerCase().includes(s)));
  $("#q-count").textContent = `${qs.length} of ${P.questions.length} questions`;
  const html = qs.map((q) => {
    const [cls, st] = stateOf(q), t = inTest(P.id, q.id);
    return `<div data-q="${esc(q.id)}" class="q-card${q.id === selected ? " sel" : ""}${drafts.has(q.id) ? " draft" : ""}${q.id === flashCard ? " flash" : ""}">
      <div class="qc-head"><b>Q${esc(q.q_no ?? "?")}</b><span class="pill">${esc(TYPE_SHORT[q.question_type] || q.question_type)}</span>
        ${q.subject ? `<span class="pill gold">${esc(q.subject)}</span>` : ""}<span class="spacer"></span>
        ${q.needs_review ? '<span class="flag" title="needs review">⚠</span>' : ""}${q.edited ? '<span title="edited">✎</span>' : ""}</div>
      <div class="snippet">${q.status === "reading" ? "<i>reading…</i>" : q.status === "error" ? '<i class="read-error">not read - open it to read again</i>'
        : esc(plain(qText(q)).slice(0, 220))}</div>
      <div class="qc-foot"><span class="pill ${cls}">${st}</span><span class="spacer"></span>
        <button class="ghost" data-act="edit" title="Edit this question">✎ Edit</button>
        <button class="ghost" data-act="test" title="${t ? "Remove from the test" : "Add to the test"}">${t ? "✓ Test" : "＋ Test"}</button>
        <button class="danger" data-act="rm" title="Remove this question">🗑</button></div>
    </div>`;
  }).join("") || `<div class="empty">${P.questions.length
    ? (dockF === "draft" ? "No drafts. Every question you cropped is saved." : "No question matches.")
    : "No questions yet. Draw a box around a question on the paper, click ✓ Read, then Save question."}</div>`;
  if (html !== lastListHTML) {
    $("#q-list").innerHTML = html; lastListHTML = html;
    if (flashCard) { $(`.q-card[data-q="${flashCard}"]`)?.scrollIntoView({ block: "nearest", inline: "nearest" }); flashCard = null; }
  }
}
$("#only-review").onchange = renderList;
$("#q-search").addEventListener("input", renderList);
$("#q-subject-filter").onchange = renderList;
$("#q-filter").onclick = (e) => {
  const b = e.target.closest("[data-f]"); if (!b) return;
  dockF = b.dataset.f;
  $$("#q-filter [data-f]").forEach((x) => x.classList.toggle("on", x === b));
  renderList();
};
$("#q-list").onclick = async (e) => {
  const card = e.target.closest("[data-q]"); if (!card) return;
  const qid = card.dataset.q, a = e.target.closest("[data-act]")?.dataset.act;
  if (a === "rm") return removeQuestion(qid);
  if (a === "test") { toggleTest(P.id, qid); renderList(); return; }
  await select(qid);
  if (a === "edit") { subTab("question"); $("#d-text").focus(); }
};
$("#dock-toggle").onclick = () => { const c = $("#dock").classList.toggle("collapsed"); LS.set("dock", c); };
if (LS.get("dock", false)) $("#dock").classList.add("collapsed");

const curQ = () => P?.questions.find((x) => x.id === selected);
const dirty = () => {
  const q = curQ();
  return !!q && q.status === "ready" && !$("#q-detail").hidden && getEditor() !== qText(q);
};
async function select(qid) {
  if (qid !== selected && dirty() && !(await ask("Discard your changes?", "The text of this question has changes that are not saved.", "Discard changes", true))) return;
  if (qid !== selected) subTab("question");
  selected = qid; builderMode = "detail";
  const q = P.questions.find((x) => x.id === qid);
  if (q && q.page !== page) { page = q.page; $("#page-select").value = page; }
  renderPage(); renderList(); renderDetail();
  $("#overlay .box.sel")?.scrollIntoView({ block: "nearest", inline: "nearest" });
  $(`.q-card[data-q="${qid}"]`)?.scrollIntoView({ block: "nearest", inline: "nearest" });
}
window.addEventListener("beforeunload", (e) => { if (view === "build" && dirty()) { e.preventDefault(); e.returnValue = ""; } });

const fileOf = (pid, rel) => `/api/projects/${pid}/file/${rel}?t=${Date.now() % 1e7}`;
const file = (rel) => fileOf(P.id, rel);
// a figure is shown as the clean drawing (dark on white); click it to see the locked photo crop
function figImg(path) {
  const c = P.crops.find((x) => x.path === path);
  const shown = c?.clean_path || path;
  return `<a href="${file(path)}" target="_blank" title="clean drawing - click for the original photo crop">` +
    `<img class="clean" src="${file(shown)}"></a>`;
}

// ---- the builder (right): empty / box being adjusted / question
let builderMode = "empty";
function showBuilder() {
  const q = P && curQ();
  const m = pending && builderMode === "pending" ? "pending" : q ? "detail" : "empty";
  $("#b-pending").hidden = m !== "pending"; $("#q-detail").hidden = m !== "detail"; $("#b-empty").hidden = m !== "empty";
}
let curSub = "question";
function subTab(name) {
  curSub = name;
  $$(".subtab").forEach((b) => b.classList.toggle("active", b.dataset.sub === name));
  $$(".subpanel").forEach((p) => (p.hidden = p.dataset.sub !== name));
  if (name === "options") renderOptions();
}
$$(".subtab").forEach((b) => (b.onclick = () => subTab(b.dataset.sub)));

function renderDetail() {
  const q = curQ();
  showBuilder(); if (!q) return;
  const draft = isDraft(P.id, q.id);
  $("#d-title").textContent = `Q${q.q_no ?? "?"} · page ${q.page}${q.subject ? " · " + q.subject : ""}`;
  const [scls, stxt] = stateOf(q);
  $("#d-state").className = "pill " + scls;
  $("#d-state").textContent = scls === "warn" && draft && q.status === "ready" ? "draft · not saved" : stxt;
  $("#d-type").textContent = (TYPE_SHORT[q.question_type] || q.question_type) + (q.multi_answer ? " (multi)" : "");
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
  const al = $("#d-alert"), ocr = $("#d-ocr");
  if (q.status === "error") {               // Mathpix could not read it (no internet, key, limit ...)
    al.innerHTML = `<div class="banner"><p class="read-error">Mathpix could not read this box: ${esc(q.read_error || "unknown error")}</p>` +
      `<button type="button" id="b-reread" class="primary">↻ Read again</button></div>`;
    $("#b-reread").onclick = () => readAgain(q);
    ocr.innerHTML = `<p class="read-error">Not read: ${esc(q.read_error || "unknown error")}</p>`;
  } else {
    al.innerHTML = q.status === "reading" ? `<div class="banner">⏳ Mathpix is reading this box… the text appears here in a second or two.</div>`
      : draft ? `<div class="banner ok">${esc(by)}Check the text, options and images, fix anything that is wrong, then click <b>✓ Save question</b>.</div>` : "";
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
  $("#fig-count").textContent = (q.figures || []).length || "";
  const f = { options: q.options.length ? q.options.join(" | ") : "—",
    statements: q.statements.length ? q.statements.join(" | ") : "—", passage_id: q.passage_id ?? "—",
    source_type: q.source_type, blur_score: q.blur_score.toFixed(1), original_sha256: q.original_sha256.slice(0, 16) + "…",
    edits: q.edit_history.length };
  $("#d-fields").innerHTML = Object.entries(f).map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
  const sv = $("#b-save");
  sv.disabled = q.status === "reading";
  sv.textContent = draft ? "✓ Save question" : "✓ Save changes";
  $("#b-delete").textContent = draft ? "🗑 Discard" : "🗑 Remove";
  renderOptions();
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
  const q = curQ(); if (!q) return;
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
  return out.replace(/ /g, " ").replace(/\n$/, "");
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

// One action at a time: a second box drawn meanwhile only waits in a queue on the server
// (it looked stuck, and the same box got drawn twice). Returns true when it worked.
let working = false;
async function act(fn, msg = "Saving…") {
  if (working) { flashBusy(); return false; }
  working = true;
  const t0 = Date.now(), busy = $("#busy");
  const tick = () => { const s = Math.round((Date.now() - t0) / 1000); busy.textContent = s ? `${msg} ${s} s` : msg; };
  tick(); busy.hidden = false;
  const timer = setInterval(tick, 1000);
  let ok = false;
  try {
    await fn(); await openProject(P.id, true); ok = true;
    const nextP = $("#cur-passage").dataset.next;           // new passage -> make it current
    if (nextP) { $("#cur-passage").value = nextP; delete $("#cur-passage").dataset.next; }
  } catch (e) { toast(e.message, "bad"); }
  finally { clearInterval(timer); busy.hidden = true; working = false; }
  return ok;
}
function flashBusy() {
  const b = $("#busy");
  b.hidden = false; b.textContent = b.textContent || "Busy…";
  b.classList.remove("flash"); void b.offsetWidth; b.classList.add("flash");
  if (!working) setTimeout(() => (b.hidden = !working), 900);
}
const qurl = (suffix) => `/api/projects/${P.id}/questions/${selected}/${suffix}`;
// Read again = the same box drawn again: a new locked crop, read by Mathpix in the background
// (the old crop is kept and can be restored with undo). Only the parts that were not read.
function readAgain(q) {
  const parts = q.parts.map((p, i) => [p, i]).filter(([p]) => p.status === "error");
  return act(async () => {
    for (const [p, i] of parts)
      await api(qurl("recrop") + `?part=${i}`, JSONPOST(p.box));
  }, "Reading again…");
}
const labelBody = () => ({ question_type: $("#d-qtype").value, q_no: /^\d+$/.test($("#d-qno").value.trim()) ? +$("#d-qno").value.trim() : null,
  subject: $("#d-subject").value, passage_id: $("#d-passage").value });
// Save question = the edited text + the labels; a draft then goes into the Questions panel
async function saveQuestion() {
  const q = curQ(); if (!q) return;
  if (q.status === "reading") { toast("Wait a moment: Mathpix is still reading this box."); return; }
  if (q.status === "error" && !(await ask("Save a box that was not read?", "Mathpix could not read this box, so its text is empty. You can click Read again first.", "Save anyway"))) return;
  const qid = q.id, wasDraft = isDraft(P.id, qid), text = getEditor(), f = labelBody();
  const labelsChanged = f.question_type !== q.question_type || (f.q_no !== null && (f.q_no !== q.q_no || q.q_no_inferred)) ||
    (f.subject || null) !== (q.subject || null) || (f.passage_id || null) !== (q.passage_id || null);
  const ok = await act(async () => {
    if (text !== qText(q)) await api(qurl("text"), JSONPOST({ text }));
    if (labelsChanged) await api(qurl("fields"), JSONPOST(f));
    setDraft(P.id, qid, false);
    flashCard = qid;
  });
  if (ok) {
    const n = P.questions.find((x) => x.id === qid)?.q_no ?? "?";
    toast(wasDraft ? `Q${n} saved - it is now in the Questions panel below.` : `Q${n}: changes saved.`, "ok");
  }
}
$("#b-save").onclick = saveQuestion;
$("#b-labels").onclick = async () => {
  if (await act(() => api(qurl("fields"), JSONPOST(labelBody())))) toast("Labels saved.", "ok");
};
$("#d-crops").addEventListener("click", async (e) => {
  const i = e.target.dataset.rmpart;
  if (i !== undefined && await ask("Remove this part from the question?", "Its crop file is kept.", "Remove part", true))
    act(() => api(`/api/projects/${P.id}/questions/${selected}/parts/${i}`, { method: "DELETE" }));
});
$("#b-undo").onclick = () => act(() => api(qurl("undo"), { method: "POST" }));
$("#b-reset").onclick = async () => { if (await ask("Reset to what Mathpix read?", "The text and diagram go back to the original reading.", "Reset", true)) act(() => api(qurl("reset"), { method: "POST" })); };
async function removeQuestion(qid) {
  const q = P.questions.find((x) => x.id === qid); if (!q) return;
  const draft = isDraft(P.id, qid);
  if (!(await ask(draft ? `Discard Q${q.q_no ?? "?"}?` : `Remove Q${q.q_no ?? "?"}?`,
    "It is removed from the questions and the export. Its original crop files are kept.", draft ? "Discard" : "Remove", true))) return;
  const ok = await act(async () => {
    await api(`/api/projects/${P.id}/questions/${qid}`, { method: "DELETE" });
    setDraft(P.id, qid, false); toggleTest(P.id, qid, false);
    if (selected === qid) { selected = null; builderMode = "empty"; }
  }, "Removing…");
  if (ok) toast(`Q${q.q_no ?? "?"} removed.`);
}
$("#b-delete").onclick = () => selected && removeQuestion(selected);
$("#d-diagram-file").onchange = (e) => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  act(() => api(qurl("diagram"), { method: "POST", body: fd }));
  e.target.value = "";
};
$("#btn-verify").onclick = async () => {
  if (await act(() => api(`/api/projects/${P.id}/verify`, { method: "POST" }), "Checking…")) toast("Round-trip check done - see the result above the paper.", "ok");
};
$("#btn-export").addEventListener("click", async (e) => {
  const n = P ? draftsOf(P.id).length : 0;
  if (!n) return;
  e.preventDefault();
  if (await ask(`${n} question${n > 1 ? "s are" : " is"} not saved yet`, "Drafts are included in the export too. Export anyway?", "Export"))
    location.href = $("#btn-export").href;
});

// ------------------------------------------------------------------ manual cropping
// Draw a box -> adjust it (corners, edges, move) -> ✓ (or Enter) sends it. Nothing is read,
// and nothing is charged, before that.
const drawMode = () => $("#draw-kind").value;
const NEEDS_Q = ["part", "diagram", "table", "option_image", "recrop"];
const KEYS = { q: "question", p: "part", d: "diagram", t: "table", i: "option_image", a: "passage", s: "section", h: "header", m: "recrop" };
function onToolChange() {
  $("#overlay").classList.toggle("drawing", !!drawMode());
  if (pending && !pending.adjust && drawMode() && drawMode() !== pending.mode) {
    pending.mode = drawMode(); renderPage(); renderPendingPanel();
  }
}
$("#draw-kind").onchange = onToolChange;
document.addEventListener("keydown", (e) => {
  if ($("#review").hidden || !$("#runner").hidden || e.ctrlKey || e.metaKey || e.altKey || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName) || e.target.isContentEditable) return;
  if ($("#chem-dlg").classList.contains("open") || document.querySelector("dialog[open]")) return;
  const k = e.key.toLowerCase();
  if (pending && k === "enter") { e.preventDefault(); commitPending(); return; }
  if (pending && k === "escape") { cancelPending(); return; }
  if (k === "+" || k === "=") { setZoom(zoom * 1.25); return; }
  if (k === "-") { setZoom(zoom / 1.25); return; }
  if (k === "0") { setZoom(1); return; }
  if (k === "escape") $("#draw-kind").value = "";
  else if (KEYS[k]) $("#draw-kind").value = KEYS[k];
  else if (k === "arrowright" || k === "arrowleft") { goPage(page + (k === "arrowright" ? 1 : -1)); return; }
  else return;
  onToolChange();
});

let pending = null;          // {page, box, mode, part?, adjust?}: drawn, not read yet
const pendInfo = (pd) => {
  const q = curQ(), qn = q ? `Q${q.q_no ?? "?"}` : "the selected question";
  return ({
    question: ["New question", "✓ Read question", true],
    part: [`Next part of ${qn}`, "✓ Add part & read", true],
    diagram: [`Diagram for ${qn}`, "✓ Add diagram", false],
    table: [`Table for ${qn}`, "✓ Add table", false],
    option_image: [`Image option for ${qn}`, "✓ Add image option", false],
    passage: ["New passage", "✓ Read passage", true],
    section: ["Section heading", "✓ Add section heading", true],
    header: ["Header / instructions", "✓ Add header", true],
    recrop: [pd.adjust ? `Adjust the box of ${qn}` : `New box for ${qn}`, "✓ Read again", true],
  })[pd.mode] || ["Box", "✓ Add", false];
};
function setPending(pd) {
  pending = pd; builderMode = "pending";
  renderPage(); showBuilder(); renderPendingPanel();
}
function cancelPending() {
  pending = null; builderMode = selected ? "detail" : "empty";
  renderPage(); showBuilder();
}
function renderPendingPanel() {
  if (!pending) return;
  const [title, btn, read] = pendInfo(pending);
  $("#pend-title").textContent = title;
  $("#pend-ok").innerHTML = `${esc(btn)} <kbd>Enter</kbd>`;
  $("#pend-hint").textContent = read
    ? `Drag the corners or edges on the paper to fit it exactly, or drag the box to move it. Nothing is sent to Mathpix (≈ ₹${MX?.price_inr ?? "0.17"} per read) until you click "${btn.replace("✓ ", "")}".`
    : "Drag the corners or edges on the paper to fit it exactly. It is cut from the page as it is (not sent to Mathpix).";
  drawPendCanvas();
}
function drawPending() {
  const [title, btn] = pendInfo(pending);
  const el = document.createElement("div"); el.className = "pending"; el.id = "pend-box";
  el.innerHTML = `<span class="tag">${esc(title)} · adjust, then ✓</span>` +
    ["nw", "n", "ne", "e", "se", "s", "sw", "w"].map((h) => `<span class="h" data-h="${h}"></span>`).join("");
  const fl = document.createElement("div"); fl.className = "pend-float"; fl.id = "pend-float";
  fl.innerHTML = `<button class="primary" data-pend="ok">${esc(btn)}</button><button data-pend="cancel" title="Cancel (Esc)">✕</button>`;
  $("#overlay").append(el, fl);
  updatePendingEl();
}
function updatePendingEl() {
  const el = $("#pend-box"), fl = $("#pend-float"); if (!el || !pending) return;
  const pg = pageInfo(), b = pending.box;
  el.style.cssText = pctOf(b, pg);
  const low = b.y1 / pg.height > 0.93;
  fl.style.left = (b.x0 / pg.width) * 100 + "%";
  fl.style.top = ((low ? b.y0 : b.y1) / pg.height) * 100 + "%";
  fl.style.transform = low ? "translateY(calc(-100% - 26px))" : "translateY(6px)";
  drawPendCanvas();
}
function drawPendCanvas() {
  const cv = $("#pend-canvas"), img = $("#page-img");
  if (!pending || pending.page !== page || !img.complete || !img.naturalWidth) return;
  const pg = pageInfo(), b = pending.box, s = img.naturalWidth / pg.width;
  const w = (b.x1 - b.x0) * s, h = (b.y1 - b.y0) * s, k = Math.min(1, 1400 / w);
  cv.width = Math.max(1, Math.round(w * k)); cv.height = Math.max(1, Math.round(h * k));
  cv.getContext("2d").drawImage(img, b.x0 * s, b.y0 * s, w, h, 0, 0, cv.width, cv.height);
}
function moveBox(d, p) {
  const pg = pageInfo(), b = { ...d.box }, dx = p.x - d.from.x, dy = p.y - d.from.y;
  if (d.h === "move") {
    const w = b.x1 - b.x0, h = b.y1 - b.y0;
    b.x0 = clamp(b.x0 + dx, 0, pg.width - w); b.y0 = clamp(b.y0 + dy, 0, pg.height - h);
    b.x1 = b.x0 + w; b.y1 = b.y0 + h;
    return b;
  }
  if (d.h.includes("w")) b.x0 += dx;
  if (d.h.includes("e")) b.x1 += dx;
  if (d.h.includes("n")) b.y0 += dy;
  if (d.h.includes("s")) b.y1 += dy;
  const x0 = clamp(Math.min(b.x0, b.x1), 0, pg.width), x1 = clamp(Math.max(b.x0, b.x1), 0, pg.width);
  const y0 = clamp(Math.min(b.y0, b.y1), 0, pg.height), y1 = clamp(Math.max(b.y0, b.y1), 0, pg.height);
  return { x0: Math.round(x0), y0: Math.round(y0), x1: Math.round(Math.max(x1, x0 + 8)), y1: Math.round(Math.max(y1, y0 + 8)) };
}
async function commitPending() {
  if (!pending) return;
  if (working) { flashBusy(); return; }
  const pd = pending, mode = pd.mode, box = pd.box, pg = pd.page;
  const needsQ = NEEDS_Q.includes(mode);
  if (needsQ && !selected) { toast("Select a question first: click its box on the paper or its card below.", "bad"); return; }
  if (mode === "recrop") {
    const q = curQ();
    let part = pd.part ?? q.parts.findIndex((p) => p.page === pg);
    if (part < 0) { toast("Draw the new box on the page where that question is.", "bad"); return; }
    if (pd.part == null && q.parts.filter((p) => p.page === pg).length > 1) {
      const cx = (box.x0 + box.x1) / 2, cy = (box.y0 + box.y1) / 2;
      part = q.parts.reduce((best, p, i) => p.page !== pg ? best :
        (Math.hypot((p.box.x0 + p.box.x1) / 2 - cx, (p.box.y0 + p.box.y1) / 2 - cy) <
         Math.hypot((q.parts[best].box.x0 + q.parts[best].box.x1) / 2 - cx, (q.parts[best].box.y0 + q.parts[best].box.y1) / 2 - cy) ? i : best), part);
    }
    pending = null; builderMode = "detail"; renderPage();
    const ok = await act(() => api(qurl("recrop") + `?part=${part}`, JSONPOST(box)), "Reading again…");
    if (!ok) setPending(pd);
    return;
  }
  pending = null; builderMode = selected || mode === "question" ? "detail" : "empty"; renderPage();
  const ok = await act(async () => {
    const r = await api(`/api/projects/${P.id}/crops`, JSONPOST({ page: pg, box, kind: mode, parent_id: needsQ ? selected : null,
      subject: $("#cur-subject").value || null, passage_id: mode === "question" ? ($("#cur-passage").value || null) : null }));
    if (r.question && mode === "question") { setDraft(P.id, r.question.id, true); subTab("question"); }
    if (r.question) selected = r.question.id;
    if (mode === "passage" && r.crop?.passage_id) $("#cur-passage").dataset.next = r.crop.passage_id;
  }, mode === "question" ? "Sending to Mathpix…" : "Saving…");
  if (!ok) setPending(pd);                  // keep the box so it can be tried again
  else if (mode !== "question") toast(`${pendInfo(pd)[0]} added.`, "ok");
}
$("#pend-ok").onclick = commitPending;
$("#pend-cancel").onclick = cancelPending;
$("#b-adjust").onclick = () => {
  const q = curQ(); if (!q) return;
  let i = q.parts.findIndex((p) => p.page === page);
  if (i < 0) { i = 0; goPage(q.parts[0].page); }
  setPending({ page: q.parts[i].page, box: { ...q.parts[i].box }, mode: "recrop", part: i, adjust: true });
};
$("#b-addpart").onclick = () => { $("#draw-kind").value = "part"; onToolChange(); toast("Draw a box around the next part (another column or page), then ✓."); };
$("#b-adddiag").onclick = () => { $("#draw-kind").value = "diagram"; onToolChange(); toast("Draw a box around the diagram on the paper, then ✓."); };

let start = null, rubber = null, drag = null;
const ov = $("#overlay");
function toPage(e) {
  const r = ov.getBoundingClientRect(), pg = pageInfo();
  return { x: Math.round(((e.clientX - r.left) / r.width) * pg.width), y: Math.round(((e.clientY - r.top) / r.height) * pg.height), px: e.clientX - r.left, py: e.clientY - r.top };
}
function hitQuestion(x, y) {
  const qByCrop = qByCropMap();
  let best = null, area = Infinity;
  for (const c of P.crops) {
    if (c.page !== page || c.superseded) continue;
    const b = c.box;
    if (x < b.x0 || x > b.x1 || y < b.y0 || y > b.y1) continue;
    const q = qByCrop[c.id] || P.questions.find((v) => v.id === c.parent_id);
    const a = (b.x1 - b.x0) * (b.y1 - b.y0);
    if (q && a < area) { area = a; best = q.id; }
  }
  return best;
}
ov.addEventListener("click", (e) => {
  const b = e.target.closest("[data-pend]"); if (!b) return;
  e.stopPropagation();
  b.dataset.pend === "ok" ? commitPending() : cancelPending();
});
ov.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || e.target.closest(".pend-float")) return;
  const sc = $("#page-scroll");                // keys (Enter / Esc / tools) act on the paper now
  if (document.activeElement !== sc) sc.focus({ preventScroll: true });
  const pe = e.target.closest(".pending");
  if (pe && pending) {                     // adjust the box: a corner/edge handle, or move it
    e.preventDefault(); ov.setPointerCapture(e.pointerId);
    drag = { h: e.target.dataset.h || "move", from: toPage(e), box: { ...pending.box } };
    return;
  }
  if (!drawMode()) return;
  if (working) { e.preventDefault(); flashBusy(); return; }   // still sending the last box
  e.preventDefault(); ov.setPointerCapture(e.pointerId);
  start = toPage(e); rubber = document.createElement("div"); rubber.className = "rubber"; ov.appendChild(rubber);
});
ov.addEventListener("pointermove", (e) => {
  if (drag) { pending.box = moveBox(drag, toPage(e)); updatePendingEl(); return; }
  if (!start) return;
  const p = toPage(e);
  Object.assign(rubber.style, { left: Math.min(start.px, p.px) + "px", top: Math.min(start.py, p.py) + "px", width: Math.abs(p.px - start.px) + "px", height: Math.abs(p.py - start.py) + "px" });
});
ov.addEventListener("pointercancel", () => { start = null; rubber?.remove(); drag = null; });   // drag interrupted
ov.addEventListener("pointerup", (e) => {
  if (drag) { drag = null; renderPendingPanel(); return; }
  if (!start) return;
  const p = toPage(e), s = start; start = null; rubber.remove();
  const box = { x0: Math.min(s.x, p.x), y0: Math.min(s.y, p.y), x1: Math.max(s.x, p.x), y1: Math.max(s.y, p.y) };
  if (box.x1 - box.x0 < 8 || box.y1 - box.y0 < 8) {          // a click, not a box: open the question under it
    const hit = hitQuestion(p.x, p.y);
    if (hit) select(hit);
    return;
  }
  const mode = drawMode();
  if (NEEDS_Q.includes(mode) && !selected) { toast("Select a question first: click its box on the paper or its card below.", "bad"); return; }
  if (mode === "recrop" && !curQ().parts.some((pt) => pt.page === page)) { toast("Draw the new box on the page where that question is.", "bad"); return; }
  setPending({ page, box, mode });
});

// ------------------------------------------------------------------ options editor
// The options are found in the editable text ((A) / A) / A. / (1) …), edited here and written
// back into that text - it is the copy that is saved, exported and printed.
const MATH_RE = /\$\$[\s\S]*?\$\$|\$[^$\n]*\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]/g;
const OPT_STYLES = [
  { re: /(\(([A-H])\))/g, idx: (c) => c.charCodeAt(0) - 65, make: (i) => `(${String.fromCharCode(65 + i)})` },
  { re: /(\(([a-h])\))/g, idx: (c) => c.charCodeAt(0) - 97, make: (i) => `(${String.fromCharCode(97 + i)})` },
  { re: /(\(([1-8])\))/g, idx: (c) => +c - 1, make: (i) => `(${i + 1})` },
  { re: /(?:^|\s)(([A-H])[).])(?=\s)/g, idx: (c) => c.charCodeAt(0) - 65, make: (i) => `${String.fromCharCode(65 + i)})` },
  { re: /(?:^|\s)(([a-h])\))(?=\s)/g, idx: (c) => c.charCodeAt(0) - 97, make: (i) => `${String.fromCharCode(97 + i)})` },
  { re: /(?:^|\n)[ \t]*(([1-8])[.)])(?=\s)/g, idx: (c) => +c - 1, make: (i) => `${i + 1})` },
];
function parseOptions(text) {
  const masked = text.replace(MATH_RE, (m) => "\x01".repeat(m.length));
  let best = null, bestStyle = null;
  for (const st of OPT_STYLES) {
    const ms = [...masked.matchAll(st.re)].map((m) => ({ idx: st.idx(m[2]), start: m.index + m[0].length - m[1].length, end: m.index + m[0].length, label: m[1] }));
    let seqBest = null;
    ms.forEach((m, i) => {
      if (m.idx !== 0) return;
      const seq = [m];
      for (let j = i + 1; j < ms.length; j++) if (ms[j].idx === seq.length) seq.push(ms[j]);
      if (seq.length >= 2 && (!seqBest || seq.length > seqBest.length || (seq.length === seqBest.length && seq[0].start > seqBest[0].start))) seqBest = seq;
    });
    if (seqBest && (!best || seqBest.at(-1).start > best.at(-1).start || (seqBest.at(-1).start === best.at(-1).start && seqBest.length > best.length))) { best = seqBest; bestStyle = st; }
  }
  if (!best) return null;
  const stem = text.slice(0, best[0].start).replace(/\s+$/, "");
  if (/\\begin\{/.test(stem) && !/\\end\{/.test(stem.slice(stem.lastIndexOf("\\begin{")))) return null;   // options inside a table
  const opts = best.map((m, i) => ({ label: m.label, text: text.slice(m.end, i + 1 < best.length ? best[i + 1].start : text.length).replace(/\s+/g, " ").trim() }));
  return { stem, opts, style: bestStyle };
}
// an option is previewed under its box only when it has a formula or an image
const needsPrev = (t) => /[$\\]|\[(?:diagram|table|option image|figure) \d+\]/.test(t);
const optKey = (label) => String(label).replace(/[^A-Za-z0-9]/g, "");
let OPT = null;              // {qid, stem, opts, style}
function renderOptions() {
  const q = curQ(), box = $("#opt-editor");
  if (!q) return;
  const parsed = parseOptions(getEditor());
  OPT = parsed ? { qid: q.id, ...parsed } : { qid: q.id, stem: getEditor(), opts: [], style: OPT_STYLES[0], none: true };
  $("#opt-count").textContent = OPT.opts.length || "";
  if (curSub !== "options") return;
  const a = answerOf(P.id, q.id), multi = $("#d-qtype").value === "multi_correct" || q.multi_answer;
  const figs = figMap(q);
  box.innerHTML = `
    <div class="answer-line"><b>Correct answer:</b> <span id="ans-show">${esc((a.keys || []).join(", ") || "not set")}</span>
      <input id="ans-text" type="text" placeholder="or type it, e.g. 2.5" value="${esc(a.text || "")}" size="16">
      <span class="hint">Saved in this browser; used to score the test.</span></div>
    ${OPT.none ? `<p class="hint">No options like (A) (B) (C) (D) were found in the text. Add them below, or type them on the Question tab.</p>` : ""}
    <label class="field block">Question (without the options)<textarea id="opt-stem" class="opt-stem" spellcheck="false">${esc(OPT.stem)}</textarea></label>
    ${OPT.opts.map((o, i) => `<div class="opt-row${(a.keys || []).includes(optKey(o.label)) ? " correct" : ""}" data-i="${i}">
      <label class="ans" title="correct answer"><input type="${multi ? "checkbox" : "radio"}" name="opt-ans" value="${esc(optKey(o.label))}" ${(a.keys || []).includes(optKey(o.label)) ? "checked" : ""}></label>
      <input type="text" class="lab opt-label" value="${esc(o.label)}" title="label">
      <input type="text" class="opt-text" value="${esc(o.text)}" spellcheck="false" placeholder="option text ($x^2$ for formulas)">
      <button class="ghost small" data-o="up" title="move up" ${i ? "" : "disabled"}>▲</button>
      <button class="ghost small" data-o="down" title="move down" ${i < OPT.opts.length - 1 ? "" : "disabled"}>▼</button>
      <button class="danger small" data-o="del" title="remove option">✕</button>
      <div class="opt-prev math">${needsPrev(o.text) ? mathHTML(o.text, figs) : ""}</div></div>`).join("")}`;
  typeset(box);
}
function applyOptions(rerender = false) {
  const q = curQ(); if (!q || !OPT || OPT.qid !== q.id) return;
  const lines = OPT.opts.map((o) => `${o.label} ${o.text}`.trimEnd());
  const text = (OPT.stem ? OPT.stem.replace(/\s+$/, "") + (lines.length ? "\n" : "") : "") + lines.join("\n");
  OPT.none = false;
  setEditor(q, text); updateDiff();
  clearTimeout(previewTimer); previewTimer = setTimeout(renderPreview, 150);
  $("#opt-count").textContent = OPT.opts.length || "";
  if (rerender) renderOptions();
}
let optPrevTimer = null;
$("#opt-editor").addEventListener("input", (e) => {
  const q = curQ(); if (!q || !OPT) return;
  const row = e.target.closest(".opt-row"), i = row ? +row.dataset.i : -1;
  if (e.target.id === "opt-stem") OPT.stem = e.target.value;
  else if (e.target.classList.contains("opt-text")) {
    OPT.opts[i].text = e.target.value;
    clearTimeout(optPrevTimer);
    optPrevTimer = setTimeout(() => { const pv = row.querySelector(".opt-prev"), t = OPT.opts[i].text; pv.innerHTML = needsPrev(t) ? mathHTML(t, figMap(q)) : ""; typeset(pv); }, 300);
  } else if (e.target.classList.contains("opt-label")) OPT.opts[i].label = e.target.value;
  else if (e.target.id === "ans-text") { const a = answerOf(P.id, q.id); setAnswer(P.id, q.id, { ...a, text: e.target.value }); return; }
  else return;
  applyOptions();
});
$("#opt-editor").addEventListener("change", (e) => {
  const q = curQ(); if (!q || e.target.name !== "opt-ans") return;
  const keys = $$("#opt-editor input[name=opt-ans]:checked").map((x) => x.value);
  setAnswer(P.id, q.id, { ...answerOf(P.id, q.id), keys });
  $("#ans-show").textContent = keys.join(", ") || "not set";
  $$("#opt-editor .opt-row").forEach((r) => r.classList.toggle("correct", keys.includes(optKey(OPT.opts[+r.dataset.i].label))));
});
$("#opt-editor").addEventListener("click", (e) => {
  const b = e.target.closest("[data-o]"); if (!b || !OPT) return;
  const i = +b.closest(".opt-row").dataset.i, o = OPT.opts;
  if (b.dataset.o === "del") o.splice(i, 1);
  if (b.dataset.o === "up" && i > 0) [o[i - 1], o[i]] = [o[i], o[i - 1]];
  if (b.dataset.o === "down" && i < o.length - 1) [o[i + 1], o[i]] = [o[i], o[i + 1]];
  applyOptions(true);
});
$("#opt-add").onclick = () => {
  if (!curQ()) return;
  if (!OPT || OPT.qid !== selected) renderOptions();
  OPT.opts.push({ label: OPT.style.make(OPT.opts.length), text: "" });
  applyOptions(true);
  $$("#opt-editor .opt-text").at(-1)?.focus();
};
$("#opt-relabel").onclick = () => {
  if (!OPT?.opts.length) return;
  OPT.opts.forEach((o, i) => (o.label = `(${String.fromCharCode(65 + i)})`));
  OPT.style = OPT_STYLES[0];
  applyOptions(true);
};
ed.addEventListener("input", () => { if (curSub !== "options") OPT = null; });
$("#d-qtype").addEventListener("change", () => { if (curSub === "options") renderOptions(); });


// ---- maths: MathJax + mhchem render $...$ / $$...$$ / \ce{...} (offline copy in /static/vendor)
const figMap = (q) => Object.fromEntries((q?.figures || []).map((f) => [f.label, f.path]));
function mathHTMLp(text, figs, pid) {
  return esc(text).replace(FIG_RE, (m, kind, n) => {
    const p = figs[`${kind} ${n}`];
    return p ? `<img class="fig" src="${fileOf(pid, p)}" alt="${m}">` : m;
  });
}
function mathHTML(text, figs) { return mathHTMLp(text, figs, P.id); }
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
  const q = curQ(); if (!q) return;
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
  if (!figs.length) { el.innerHTML = `<p class="hint">No diagrams or structures in this question. Use <b>▧ Add diagram</b> to cut one from the paper.</p>`; return; }
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
  const q = curQ(); if (!q) return;
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
  $("#chem-dlg").inert = !on;                     // closed: it can't take the keyboard focus
  $("#chem-dlg").classList.toggle("open", on);
  $("#chem-backdrop").hidden = !on;
  if (!on) reclaimFocus();
}
// Ketcher, loading in the closed (off-screen) panel, takes the keyboard focus: the page's keys
// (Enter, Esc, Q/P/D …) then went to the hidden editor. Give the focus back to where it was.
let lastFocus = null;
document.addEventListener("focusin", (e) => { if (e.target !== kframe) lastFocus = e.target; });
window.addEventListener("blur", () => setTimeout(() => {
  if (document.activeElement === kframe && !$("#chem-dlg").classList.contains("open")) reclaimFocus();
}, 0));
function reclaimFocus() {
  if (document.activeElement !== kframe) return;
  const t = lastFocus && document.contains(lastFocus) && !lastFocus.closest?.("#chem-dlg") ? lastFocus : $("#page-scroll");
  try { kframe.blur(); } catch {}
  t.focus?.({ preventScroll: true });
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
    toast("Structure saved.", "ok");
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

// ------------------------------------------------------------------ question bank
let BANK = [], bankShown = 30;
async function getProject(pid, force = false) {
  if (!force && PROJ.has(pid)) return PROJ.get(pid);
  const pr = await api(`/api/projects/${pid}`);
  PROJ.set(pid, pr);
  return pr;
}
async function loadBank() {
  showView("bank");
  $("#bank-list").innerHTML = `<p class="hint">Loading questions…</p>`;
  try {
    const rows = await api("/api/projects");
    const prs = await Promise.all(rows.filter((r) => r.questions).map((r) => getProject(r.id, true).catch(() => null)));
    BANK = prs.filter(Boolean).flatMap((pr) => pr.questions.map((q) => ({ pr, q })));
  } catch (e) { toast(e.message, "bad"); BANK = []; }
  const keep = (sel, opts) => { const v = $(sel).value; $(sel).innerHTML = opts; if ([...$(sel).options].some((o) => o.value === v)) $(sel).value = v; };
  const prs = [...new Map(BANK.map((x) => [x.pr.id, x.pr])).values()];
  keep("#bank-paper", `<option value="">All papers</option>` + prs.map((p) => `<option value="${esc(p.id)}">${esc(p.source_name)}</option>`).join(""));
  keep("#bank-subject", `<option value="">All subjects</option>` + [...new Set(BANK.map((x) => x.q.subject).filter(Boolean))].sort().map((s) => `<option>${esc(s)}</option>`).join(""));
  keep("#bank-type", `<option value="">All types</option>` + Object.entries(TYPE_NAMES).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join(""));
  bankShown = 30;
  renderBank();
}
function bankFiltered() {
  const s = $("#bank-search").value.trim().toLowerCase(), pp = $("#bank-paper").value, sub = $("#bank-subject").value,
    ty = $("#bank-type").value, onlySaved = $("#bank-state").value === "saved";
  return BANK.filter(({ pr, q }) => (!pp || pr.id === pp) && (!sub || q.subject === sub) && (!ty || q.question_type === ty) &&
    (!onlySaved || (q.status === "ready" && !isDraft(pr.id, q.id))) && (!s || plain(qText(q)).toLowerCase().includes(s)));
}
function renderBank() {
  const list = bankFiltered(), shown = list.slice(0, bankShown);
  const nSaved = BANK.filter(({ pr, q }) => q.status === "ready" && !isDraft(pr.id, q.id)).length;
  $("#bank-stats").innerHTML = stat(nSaved, "saved questions") + stat(new Set(BANK.map((x) => x.pr.id)).size, "papers") + stat(T.items.length, "in the test", "gold");
  $("#bank-list").innerHTML = shown.map(({ pr, q }) => {
    const [cls, st] = stateOf(q, pr.id), t = inTest(pr.id, q.id);
    return `<div class="bank-card${t ? " intest" : ""}" data-pid="${esc(pr.id)}" data-qid="${esc(q.id)}">
      <div class="bc-head"><b>Q${esc(q.q_no ?? "?")}</b><span class="pill">${esc(TYPE_SHORT[q.question_type] || q.question_type)}</span>
        ${q.subject ? `<span class="pill gold">${esc(q.subject)}</span>` : ""}<span class="pill ${cls}">${st}</span></div>
      <div class="bc-src" title="${esc(pr.source_name)}">${esc(pr.source_name)} · page ${q.page} · ${esc(pr.exam)}</div>
      <div class="bc-text math">${q.status === "ready" ? mathHTMLp(qText(q), figMap(q), pr.id) : `<i>${q.status === "reading" ? "reading…" : "not read"}</i>`}</div>
      <div class="bc-foot"><button class="ghost small" data-b="edit">✎ Edit</button><span class="spacer"></span>
        <button class="${t ? "dark" : "primary"} small" data-b="test">${t ? "✓ In test" : "＋ Add to test"}</button></div></div>`;
  }).join("") + (list.length > shown.length ? `<div class="more"><button class="ghost" data-b="more">Show more (${list.length - shown.length} left)</button></div>` : "")
    || `<div class="empty-state"><h2>No questions here yet</h2><p>${BANK.length ? "Nothing matches these filters." : "Crop questions in Build questions and click Save question: they appear here."}</p></div>`;
  typeset($("#bank-list"));
}
["#bank-search"].forEach((s) => $(s).addEventListener("input", () => { bankShown = 30; renderBank(); }));
["#bank-paper", "#bank-subject", "#bank-type", "#bank-state"].forEach((s) => $(s).addEventListener("change", () => { bankShown = 30; renderBank(); }));
$("#bank-list").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-b]"); if (!b) return;
  if (b.dataset.b === "more") { bankShown += 30; renderBank(); return; }
  const card = b.closest(".bank-card"), pid = card.dataset.pid, qid = card.dataset.qid;
  if (b.dataset.b === "test") { toggleTest(pid, qid); renderBank(); }
  if (b.dataset.b === "edit") { await openProject(pid); await select(qid); }
});
$("#bank-add-all").onclick = () => {
  const list = bankFiltered().filter(({ q }) => q.status === "ready");
  let n = 0;
  list.forEach(({ pr, q }) => { if (!inTest(pr.id, q.id)) { T.items.push({ pid: pr.id, qid: q.id }); n++; } });
  saveT(); renderBank();
  toast(n ? `${n} question${n > 1 ? "s" : ""} added to the test.` : "These questions are already in the test.", n ? "ok" : "");
};

// ------------------------------------------------------------------ test
const T = Object.assign({ title: "", duration: 60, marks: 4, neg: 1, instr: "", items: [] }, LS.get("test", {}));
function saveT() { LS.set("test", T); testBadge(); }
const inTest = (pid, qid) => T.items.some((it) => it.pid === pid && it.qid === qid);
function toggleTest(pid, qid, on = !inTest(pid, qid)) {
  if (on && !inTest(pid, qid)) T.items.push({ pid, qid });
  if (!on) T.items = T.items.filter((it) => !(it.pid === pid && it.qid === qid));
  saveT();
}
function testBadge() {
  $("#tab-test").innerHTML = `<span class="ico">✎</span>Test${T.items.length ? ` <span class="tab-badge">${T.items.length}</span>` : ""}`;
}
testBadge();
const testItems = () => T.items.map((it) => {
  const pr = PROJ.get(it.pid);
  return { ...it, pr, q: pr?.questions.find((q) => q.id === it.qid) };
});
const marksOf = (x) => (x.marks != null && x.marks !== "" ? +x.marks : +T.marks);
const stripNo = (s) => String(s || "").replace(/^\s*(?:Q(?:uestion)?\.?\s*)?\d{1,3}\s*[.):]\s+/i, "");
async function loadTest() {
  showView("test");
  $("#t-title").value = T.title; $("#t-duration").value = T.duration; $("#t-marks").value = T.marks;
  $("#t-neg").value = T.neg; $("#t-instr").value = T.instr;
  await Promise.all([...new Set(T.items.map((it) => it.pid))].map((pid) => getProject(pid, true).catch(() => null)));
  renderTestList(); renderPaper();
}
function renderTestList() {
  const items = testItems();
  $("#t-count").textContent = items.length ? `(${items.length})` : "";
  $("#t-list").innerHTML = items.map((x, i) => `<li data-i="${i}"><span class="n">${i + 1}</span>
    <div class="tt">${x.q ? `<div>Q${esc(x.q.q_no ?? "?")} · ${esc(TYPE_SHORT[x.q.question_type] || "")}${x.q.subject ? " · " + esc(x.q.subject) : ""} — ${esc(plain(qText(x.q)).slice(0, 80))}</div>
      <small>${esc(x.pr.source_name)}${answerOf(x.pid, x.qid).keys?.length || answerOf(x.pid, x.qid).text ? " · answer set" : " · no answer set"}</small>`
      : `<div class="read-error">This question was removed from its paper.</div>`}</div>
    <div class="tb"><input type="number" step="0.5" data-marks="${i}" value="${x.marks ?? ""}" placeholder="${esc(T.marks)}" title="marks (empty = default)">
      <button class="ghost" data-t="up" ${i ? "" : "disabled"}>▲</button><button class="ghost" data-t="down" ${i < items.length - 1 ? "" : "disabled"}>▼</button>
      <button class="danger" data-t="del" title="remove from the test">✕</button></div></li>`).join("")
    || `<li class="empty">No questions yet. Open the Question bank and click “＋ Add to test”.</li>`;
}
function renderPaper(key = false) {
  const items = testItems().filter((x) => x.q);
  const total = items.reduce((a, x) => a + marksOf(x), 0);
  let html = `<div class="paper-head"><img src="/static/logo.png" alt=""><div><h2>${esc(T.title || "Untitled test")}</h2>
      <div class="hint">Manchester Technologies</div></div></div>
    <div class="paper-meta"><span>Time: ${esc(T.duration)} min</span><span>Questions: ${items.length}</span><span>Maximum marks: ${total}</span></div>
    ${T.instr ? `<div class="paper-instr">${esc(T.instr)}</div>` : ""}`;
  if (key) html += `<h3>Answer key</h3><div class="key">${items.map((x, i) => { const a = answerOf(x.pid, x.qid);
    return `<div><b>${i + 1}.</b> ${esc((a.keys || []).join(", ") || a.text || "—")}</div>`; }).join("")}</div>`;
  else html += items.map((x, i) => `<div class="paper-q"><div class="pn">${i + 1}.</div><div>
      <div class="pt math">${mathHTMLp(stripNo(qText(x.q)), figMap(x.q), x.pid)}</div>
      <div class="pm">[${marksOf(x)} mark${marksOf(x) === 1 ? "" : "s"}${+T.neg ? `, −${esc(T.neg)} for a wrong answer` : ""}]</div></div></div>`).join("")
    || `<p class="hint">No questions yet. Add some from the Question bank.</p>`;
  $("#test-paper").innerHTML = html;
  return typeset($("#test-paper"));
}
let paperTimer = null;
const paperSoon = () => { clearTimeout(paperTimer); paperTimer = setTimeout(() => renderPaper(), 300); };
[["#t-title", "title"], ["#t-duration", "duration"], ["#t-marks", "marks"], ["#t-neg", "neg"], ["#t-instr", "instr"]].forEach(([s, k]) =>
  $(s).addEventListener("input", (e) => { T[k] = e.target.type === "number" ? +e.target.value : e.target.value; saveT(); paperSoon(); if (k === "marks") renderTestList(); }));
$("#t-list").addEventListener("input", (e) => {
  const i = e.target.dataset.marks; if (i === undefined) return;
  T.items[+i].marks = e.target.value === "" ? undefined : +e.target.value; saveT(); paperSoon();
});
$("#t-list").addEventListener("click", (e) => {
  const b = e.target.closest("[data-t]"); if (!b) return;
  const i = +b.closest("li").dataset.i, it = T.items;
  if (b.dataset.t === "del") it.splice(i, 1);
  if (b.dataset.t === "up" && i > 0) [it[i - 1], it[i]] = [it[i], it[i - 1]];
  if (b.dataset.t === "down" && i < it.length - 1) [it[i + 1], it[i]] = [it[i], it[i + 1]];
  saveT(); renderTestList(); renderPaper();
});
$("#t-clear").onclick = async () => {
  if (!T.items.length || !(await ask("Remove every question from the test?", "The questions stay in the Question bank.", "Clear test", true))) return;
  T.items = []; saveT(); renderTestList(); renderPaper();
};
$("#test-print").onclick = async () => { await renderPaper(); window.print(); };
$("#test-print-key").onclick = async () => {
  await renderPaper(true);
  window.addEventListener("afterprint", () => renderPaper(), { once: true });
  window.print();
};

// ---- take the test: one question at a time, timer, palette, score from the answer key
let R = null;
$("#test-start").onclick = () => {
  const items = testItems().filter((x) => x.q && x.q.status === "ready");
  if (!items.length) { toast("Add questions to the test first (Question bank → ＋ Add to test).", "bad"); return; }
  R = { items: items.map((x) => ({ ...x, parsed: parseOptions(qText(x.q)) })), i: 0, ans: {}, mark: {}, end: Date.now() + Math.max(1, +T.duration || 60) * 60000, done: false };
  $("#runner").hidden = false; document.body.style.overflow = "hidden";
  clearInterval(R.timer); R.timer = setInterval(tickRunner, 1000);
  renderRunner();
};
function tickRunner() {
  if (!R || R.done) return;
  const left = Math.max(0, Math.round((R.end - Date.now()) / 1000)), el = $("#run-timer");
  if (el) { el.textContent = `${String(Math.floor(left / 60)).padStart(2, "0")}:${String(left % 60).padStart(2, "0")}`; el.classList.toggle("low", left < 60); }
  if (!left) { toast("Time is up - the test was submitted.", "bad"); finishRunner(); }
}
function renderRunner() {
  if (R.done) return renderResult();
  const x = R.items[R.i], n = R.items.length, opts = x.parsed?.opts || [], given = R.ans[R.i] || [];
  const multi = x.q.question_type === "multi_correct" || x.q.multi_answer, figs = figMap(x.q);
  const answered = Object.values(R.ans).filter((a) => a?.length && a.some((v) => String(v).trim())).length;
  $("#runner").innerHTML = `
    <div class="runner-top"><img src="/static/logo.png" width="34" height="34" alt=""><span>${esc(T.title || "Test")}</span>
      <span class="timer" id="run-timer"></span><button class="primary" data-run="submit">Submit test</button>
      <button class="ghost small" data-run="quit">Exit</button></div>
    <div class="runner-body">
      <div class="runner-q"><div class="card">
        <div class="builder-head"><h3>Question ${R.i + 1} of ${n}</h3><span class="pill gold">+${marksOf(x)}${+T.neg ? ` / −${esc(T.neg)}` : ""}</span>
          ${multi ? '<span class="pill">one or more answers</span>' : ""}</div>
        <div class="qtext math">${mathHTMLp(stripNo(x.parsed ? x.parsed.stem : qText(x.q)), figs, x.pid)}</div>
        ${opts.length ? `<div class="choices">${opts.map((o) => { const k = optKey(o.label);
          return `<button class="choice${given.includes(k) ? " on" : ""}" data-k="${esc(k)}"><span class="cl">${esc(k)}</span><span class="math">${mathHTMLp(o.text, figs, x.pid)}</span></button>`; }).join("")}</div>`
          : `<label class="field block" style="margin-top:14px">Your answer <input id="run-text" type="text" value="${esc(given[0] || "")}" placeholder="Type your answer" autocomplete="off"></label>`}
        <div class="runner-nav"><button data-run="prev" ${R.i ? "" : "disabled"}>◀ Previous</button>
          <button data-run="mark">${R.mark[R.i] ? "★ Marked for review" : "☆ Mark for review"}</button>
          <button data-run="clear">Clear answer</button><span class="spacer"></span>
          <button class="primary" data-run="next">${R.i < n - 1 ? "Save & next ▶" : "Save"}</button></div>
      </div></div>
      <aside class="palette"><b>Questions</b>
        <div class="grid">${R.items.map((_, i) => `<button data-go="${i}" class="${R.ans[i]?.length ? "ans" : ""}${R.mark[i] ? " mark" : ""}${i === R.i ? " cur" : ""}">${i + 1}</button>`).join("")}</div>
        <div class="legend"><span><i class="a"></i>answered</span><span><i class="m"></i>marked for review</span><span><i class="n"></i>not answered</span></div>
        <p class="hint">${answered} of ${n} answered</p>
        <button class="primary" data-run="submit" style="width:100%">Submit test</button></aside>
    </div>`;
  tickRunner();
  typeset($(".runner-q"));
}
function sameSet(a, b) { const A = new Set(a.map(String)), B = new Set(b.map(String)); return A.size === B.size && [...A].every((v) => B.has(v)); }
function isCorrect(given, a) {
  if (a.keys?.length) return sameSet(given, a.keys);
  const g = String(given[0] ?? "").trim().toLowerCase(), k = String(a.text ?? "").trim().toLowerCase();
  if (!k) return null;
  const gn = parseFloat(g), kn = parseFloat(k);
  return !isNaN(gn) && !isNaN(kn) && /^[-+.\d\se]+$/.test(g) && /^[-+.\d\se]+$/.test(k) ? Math.abs(gn - kn) < 1e-6 * Math.max(1, Math.abs(kn)) : g === k;
}
function finishRunner() {
  if (!R || R.done) return;
  R.done = true; clearInterval(R.timer);
  renderResult();
}
function renderResult() {
  let score = 0, max = 0, right = 0, wrong = 0, skip = 0, nokey = 0;
  const rows = R.items.map((x, i) => {
    const given = (R.ans[i] || []).filter((v) => String(v).trim()), a = answerOf(x.pid, x.qid), m = marksOf(x);
    max += m;
    let res;
    if (!given.length) { skip++; res = "not answered"; }
    else {
      const ok = isCorrect(given, a);
      if (ok === null) { nokey++; res = "no answer key"; }
      else if (ok) { right++; score += m; res = "✓ correct"; }
      else { wrong++; score -= +T.neg || 0; res = "✗ wrong"; }
    }
    return `<tr><td>${i + 1}</td><td>${esc(plain(stripNo(x.parsed ? x.parsed.stem : qText(x.q))).slice(0, 90))}</td><td>${esc(given.join(", ") || "—")}</td>
      <td>${esc((a.keys || []).join(", ") || a.text || "—")}</td><td>${res.startsWith("✓") ? `<span class="pill ok">${res}</span>` : res.startsWith("✗") ? `<span class="pill bad">${res}</span>` : `<span class="pill">${res}</span>`}</td></tr>`;
  }).join("");
  $("#runner").innerHTML = `
    <div class="runner-top"><img src="/static/logo.png" width="34" height="34" alt=""><span>${esc(T.title || "Test")} · result</span>
      <span class="spacer"></span><button class="ghost small" data-run="again">↻ Take again</button><button class="primary" data-run="quit">Close</button></div>
    <div class="view" style="overflow:auto">
      <div class="result-big">${stat(`${score} / ${max}`, "score", "gold")}${stat(right, "correct")}${stat(wrong, "wrong")}${stat(skip, "not answered")}${nokey ? stat(nokey, "no answer key") : ""}</div>
      ${nokey ? `<p class="hint">Set the correct answers on each question's Options tab (Build questions) to score them.</p>` : ""}
      <div class="card table-wrap"><table><thead><tr><th>#</th><th>Question</th><th>Your answer</th><th>Correct answer</th><th>Result</th></tr></thead><tbody>${rows}</tbody></table></div>
    </div>`;
}
$("#runner").addEventListener("click", async (e) => {
  if (!R) return;
  const c = e.target.closest("[data-k]"), g = e.target.closest("[data-go]"), b = e.target.closest("[data-run]");
  const x = R.items[R.i];
  if (c && !R.done) {
    const k = c.dataset.k, multi = x.q.question_type === "multi_correct" || x.q.multi_answer, cur = R.ans[R.i] || [];
    R.ans[R.i] = multi ? (cur.includes(k) ? cur.filter((v) => v !== k) : [...cur, k]) : (cur.includes(k) ? [] : [k]);
    renderRunner(); return;
  }
  if (g) { R.i = +g.dataset.go; renderRunner(); return; }
  if (!b) return;
  const a = b.dataset.run;
  if (a === "prev" && R.i > 0) { R.i--; renderRunner(); }
  if (a === "next") { if (R.i < R.items.length - 1) R.i++; renderRunner(); }
  if (a === "mark") { R.mark[R.i] = !R.mark[R.i]; renderRunner(); }
  if (a === "clear") { R.ans[R.i] = []; renderRunner(); }
  if (a === "submit") {
    const left = R.items.length - Object.values(R.ans).filter((v) => v?.length).length;
    if (await ask("Submit the test?", left ? `${left} question${left > 1 ? "s are" : " is"} not answered.` : "All questions are answered.", "Submit")) finishRunner();
  }
  if (a === "again") $("#test-start").click();
  if (a === "quit") {
    if (!R.done && !(await ask("Exit the test?", "Your answers will be lost.", "Exit", true))) return;
    clearInterval(R.timer); R = null; $("#runner").hidden = true; $("#runner").innerHTML = ""; document.body.style.overflow = "";
  }
});
$("#runner").addEventListener("input", (e) => { if (R && e.target.id === "run-text") R.ans[R.i] = e.target.value.trim() ? [e.target.value] : []; });

// ------------------------------------------------------------------ start
route();
