"""Data records. QuestionRecord mirrors the output format in the project plan."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RegionKind(str, Enum):
    header = "header"
    instructions = "instructions"
    section = "section"
    question = "question"
    passage = "passage"
    diagram = "diagram"
    table = "table"
    equation = "equation"
    option_image = "option_image"


class QuestionType(str, Enum):
    mcq = "mcq"
    multi_correct = "multi_correct"
    numerical = "numerical"
    match = "match"
    assertion_reason = "assertion_reason"
    statement = "statement"
    paragraph = "paragraph"
    diagram = "diagram"
    arrange = "arrange"
    other = "other"


class Box(BaseModel):
    """Axis-aligned box in ORIGINAL page pixel coordinates. x1/y1 exclusive."""
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def clip(self, width: int, height: int) -> "Box":
        return Box(x0=max(0, self.x0), y0=max(0, self.y0),
                   x1=min(width, self.x1), y1=min(height, self.y1))

    def pad(self, m: int) -> "Box":
        return Box(x0=self.x0 - m, y0=self.y0 - m, x1=self.x1 + m, y1=self.y1 + m)

    def intersection(self, o: "Box") -> int:
        w = min(self.x1, o.x1) - max(self.x0, o.x0)
        h = min(self.y1, o.y1) - max(self.y0, o.y0)
        return max(0, w) * max(0, h)

    def union(self, o: "Box") -> "Box":
        return Box(x0=min(self.x0, o.x0), y0=min(self.y0, o.y0),
                   x1=max(self.x1, o.x1), y1=max(self.y1, o.y1))


class Word(BaseModel):
    text: str
    conf: float  # 0..1
    box: Box
    line_id: int = 0


class Region(BaseModel):
    kind: RegionKind
    box: Box
    page: int
    source_type: str = "printed"  # printed | handwritten | mixed
    score: float = 1.0


class EditEntry(BaseModel):
    time: str
    field: str
    description: str
    before: Optional[str] = None
    after: Optional[str] = None


class Crop(BaseModel):
    """One locked crop cut from the original page."""
    id: str
    kind: RegionKind
    page: int
    box: Box
    path: str                 # relative to project dir, lossless PNG
    clean_path: Optional[str] = None   # figures: the drawing alone, dark on white (not locked)
    # chemical structure (a reviewer drew it in Ketcher or accepted a recognised one)
    latex_text: str = ""               # passages / headers: text with formulas as LaTeX
    status: str = "ready"              # passages / headers: reading | ready | error
    smiles: Optional[str] = None
    molfile: Optional[str] = None
    structure_path: Optional[str] = None   # RDKit drawing of that molecule (display + export)
    smiles_suggested: Optional[str] = None # MolScribe's reading of the drawing - a suggestion only
    suggest_conf: Optional[float] = None
    sha256: str
    blur_score: float
    quality_flag: str = "ok"
    text: str = ""                     # verbatim OCR (labels only, for diagrams)
    ocr_confidence: float = 0.0
    subject: Optional[str] = None
    passage_id: Optional[str] = None
    parent_id: Optional[str] = None    # question this diagram/table/option image belongs to
    needs_review: bool = False
    superseded: bool = False           # replaced by a re-crop; kept, never deleted
    source_part: Optional[str] = None  # question-part crop this was auto-extracted from
    user_box: Optional[Box] = None     # the box as the user drew it (before snapping)
    snap_notes: list[str] = Field(default_factory=list)   # what snapping changed


class QuestionPart(BaseModel):
    """A question that runs across a column or page break has more than one part."""
    page: int
    box: Box
    crop_id: str
    crop_path: str
    text: str = ""                     # verbatim OCR of this part (diagram areas excluded)
    # the same words laid out as on the page (indents, spacing) with {{crop:<id>}} where each
    # diagram / table / image option sits
    layout_text: str = ""
    # the same with formulas kept as LaTeX ($...$) for MathJax + mhchem; "" = same as above
    latex_text: str = ""
    layout_latex: str = ""
    status: str = "ready"              # reading (background OCR still running) | ready | error
    read_by: str = ""                  # "Mathpix" | "this computer"
    ocr_confidence: float = 0.0
    source_type: str = "printed"
    blur_score: float = 0.0
    quality_flag: str = "ok"
    edge_warnings: list[str] = Field(default_factory=list)


class QuestionRecord(BaseModel):
    id: str
    exam: str
    subject: Optional[str] = None
    q_no: Optional[int] = None
    q_no_inferred: bool = False                # number guessed from the previous question
    page: int
    box: Box
    crop_path: str
    crop_id: str
    parts: list[QuestionPart] = Field(default_factory=list)
    text: str                                  # OCR verbatim, never edited
    # the question as it looks on the page: same lines, indents and spacing, with
    # "[diagram 1]" / "[table 1]" / "[option image 2]" where each figure sits
    display_text: str = ""
    # the same with formulas kept as LaTeX ($...$): the source the editable copy starts from
    latex_text: str = ""
    display_latex: str = ""
    figures: list[dict] = Field(default_factory=list)   # {"label": "diagram 1", "path": crop png}
    text_edited: Optional[str] = None          # the reviewer's copy (LaTeX source, [diagram n])
    # reading happens in the background: "reading" until the OCR result is in
    status: str = "ready"                      # reading | ready | error
    read_by: str = ""                          # "Mathpix" | "this computer" | both (multi-part)
    read_error: Optional[str] = None
    read_seconds: Optional[float] = None
    options: list[str] = Field(default_factory=list)
    statements: list[str] = Field(default_factory=list)
    diagram_path: Optional[str] = None         # locked original diagram crop
    diagram_paths: list[str] = Field(default_factory=list)
    diagram_edited_path: Optional[str] = None
    option_images: list[str] = Field(default_factory=list)
    question_type: QuestionType = QuestionType.mcq
    type_confidence: float = 0.0
    type_signals: list[str] = Field(default_factory=list)   # why this type was chosen
    multi_answer: bool = False
    passage_id: Optional[str] = None
    source_type: str = "printed"
    blur_score: float = 0.0
    quality_flag: str = "ok"
    ocr_confidence: float = 0.0
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    original_sha256: str
    edited: bool = False
    edit_history: list[EditEntry] = Field(default_factory=list)
    superseded_crops: list[str] = Field(default_factory=list)
    labels_reviewed: bool = False              # a reviewer confirmed/corrected type, q_no, subject


class PageInfo(BaseModel):
    page: int
    width: int
    height: int
    path: str
    sha256: str
    blur_score: float
    quality_flag: str
    skew_deg: float = 0.0
    columns: int = 1


class Project(BaseModel):
    id: str
    exam: str
    source_name: str
    created: str
    mode: str = "manual"             # manual: user crops | auto: model crops
    script: str = "auto"             # auto | printed | handwritten (Mathpix detects it itself)
    reader: str = "fast"             # every box is read by Mathpix; "accurate" = older papers only
    status: str = "processing"       # processing | ready | error
    error: Optional[str] = None
    pages: list[PageInfo] = Field(default_factory=list)
    crops: list[Crop] = Field(default_factory=list)
    questions: list[QuestionRecord] = Field(default_factory=list)
    roundtrip: Optional[dict] = None
