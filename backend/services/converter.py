"""Document conversion service — hybrid pipeline.

- XLSX : MarkItDown (handles merged cells correctly, no duplication)
- PDF  : Docling — full layout pipeline, image extraction, optional VLM
- DOCX : Docling — image extraction, no VLM
- PPTX : Docling — image extraction, no VLM
- HTML : Docling — text only

Configuration is read from app settings on construction. If no LLM base_url is
configured, VLM description and post-processing refinement are both disabled
(graceful degradation).
"""

from __future__ import annotations

import re
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    RapidOcrOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.document_converter import (
    DocumentConverter as _DoclingConverter,
    PdfFormatOption,
    WordFormatOption,
    PowerpointFormatOption,
    ExcelFormatOption,
    HTMLFormatOption,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)
from docling_core.types.doc.document import ImageRefMode

from markitdown import MarkItDown
from openai import OpenAI

from .picture_serializer import VaultPictureSerializer


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_title(markdown: str, file_path: str) -> str:
    """Extract title from first H1 heading; fall back to the filename stem."""
    match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return Path(file_path).stem


def _build_pdf_options(settings) -> PdfPipelineOptions:
    """Build the PDF pipeline options (extracted for testability).

    OCR stays enabled but its detector input is hard-capped to prevent the
    native ``std::bad_alloc`` crash. Docling renders OCR regions at 216 DPI and
    RapidOCR's default ``Det.limit_type: min`` / ``limit_side_len: 736`` never
    shrinks a large scanned page, so the detector allocates a feature-map tensor
    proportional to the full page and exhausts the C++ heap — taking down the
    whole sidecar. Forcing ``limit_type: max`` bounds the longest side instead,
    so memory is bounded regardless of source resolution. Region-based OCR
    (``force_full_page_ocr=False``, the default) keeps born-digital PDFs cheap
    while still OCR-ing scanned image pages.
    """
    llm_configured = bool(settings.active_llm.base_url and settings.active_llm.model)

    pdf_options = PdfPipelineOptions(
        # Only generate full-res picture images when VLM will actually use them.
        # images_scale=1.0 (from 2.0) halves linear resolution → 4× less memory
        # per page image, which prevents std::bad_alloc on complex/large pages.
        generate_picture_images=llm_configured,
        images_scale=1.0,
        do_picture_description=llm_configured,
        enable_remote_services=llm_configured,
        accelerator_options=AcceleratorOptions(
            num_threads=2,
            device=AcceleratorDevice.CPU,
        ),
        # Bound the RapidOCR detector input. Without this, large scanned pages
        # crash the native onnxruntime backend with std::bad_alloc.
        ocr_options=RapidOcrOptions(
            rapidocr_params={
                "Det.limit_type": "max",      # cap LONGEST side (default: "min")
                "Det.limit_side_len": 960,    # px ceiling for the detector input
                "Global.max_side_len": 1280,  # overall safety cap
            },
        ),
    )

    if llm_configured:
        # Append the OpenAI-compatible chat completions path to the base URL.
        # Strip trailing slash first to avoid double-slash issues.
        base = settings.active_llm.base_url.rstrip("/")
        url = base + "/chat/completions"
        pdf_options.picture_description_options = PictureDescriptionApiOptions(
            url=url,
            params=dict(
                model=settings.active_llm.model,
                max_completion_tokens=200,
            ),
            prompt="Describe this image in 2-3 concise sentences. Be precise.",
            timeout=60,
        )

    return pdf_options


def _build_docling_converter(settings) -> _DoclingConverter:
    """Build a DocumentConverter configured from app settings.

    If an LLM base_url + model are present, enables remote VLM picture
    description for PDFs (requires enable_remote_services=True per Docling
    docs). Otherwise the pipeline runs fully offline.
    """
    return _DoclingConverter(
        format_options={
            InputFormat.PDF:  PdfFormatOption(pipeline_options=_build_pdf_options(settings)),
            InputFormat.DOCX: WordFormatOption(),
            InputFormat.PPTX: PowerpointFormatOption(),
            InputFormat.XLSX: ExcelFormatOption(),
            InputFormat.HTML: HTMLFormatOption(),
        }
    )


# ── main service ──────────────────────────────────────────────────────────────

class DoclingConverter:
    """Convert uploaded documents to Markdown using Docling.

    Usage::

        converter = DoclingConverter()
        markdown, title = converter.convert_file(
            "/path/to/doc.pdf",
            doc_id="uuid-string",
            assets_dir=Path("/vault/assets"),
        )
    """

    def __init__(self) -> None:
        from ..core.config import get_settings
        self._settings = get_settings()
        self._converter = _build_docling_converter(self._settings)

    def convert_file(
        self,
        file_path: str,
        *,
        doc_id: str,
        assets_dir: Path,
    ) -> tuple[str, str]:
        """Convert a document to Markdown, saving any images to *assets_dir*.

        Args:
            file_path:  Absolute path to the source file.
            doc_id:     Unique document identifier (used as image filename prefix).
            assets_dir: Directory under the vault where PNGs will be saved.

        Returns:
            ``(markdown_content, title)`` where *title* is extracted from the
            first H1 heading or falls back to the filename stem.
        """
        assets_dir.mkdir(parents=True, exist_ok=True)

        # Route Excel files to MarkItDown (handles merged cells correctly)
        ext = Path(file_path).suffix.lower()
        print(f"[converter] convert_file called: path={file_path}, ext={ext}")
        if ext == ".xlsx":
            result = self._convert_excel(file_path)
        else:
            result = self._convert_with_docling(file_path, doc_id=doc_id, assets_dir=assets_dir)

        from ..core.telemetry import track_event_sync
        track_event_sync("document_indexed", {"ext": ext})

        return result

    # ── Excel via MarkItDown ──────────────────────────────────────────────────

    def _convert_excel(self, file_path: str) -> tuple[str, str]:
        """Convert an Excel file using MarkItDown (Microsoft).

        MarkItDown uses pandas + openpyxl internally and handles merged cells
        correctly — values appear once instead of being duplicated across all
        spanned columns.
        """
        print(f"[converter] Using MarkItDown for Excel: {file_path}")
        mid = MarkItDown(enable_plugins=False)
        result = mid.convert(file_path)
        markdown = result.text_content or ""
        markdown = self._post_process_excel(markdown)
        markdown = self._refine(markdown)
        title = _extract_title(markdown, file_path)
        print(f"[converter] Excel conversion done. Title={title}, length={len(markdown)}")
        return markdown, title

    @staticmethod
    def _post_process_excel(md: str) -> str:
        """Clean MarkItDown Excel output.

        - Replace 'NaN' cell values with empty strings
        - Remove 'Unnamed: N' column headers
        - Collapse excessive blank lines
        """
        # Replace NaN values in table cells (| NaN | → |  |)
        md = re.sub(r"(?<=\|)\s*NaN\s*(?=\|)", " ", md)
        # Remove 'Unnamed: N' header labels
        md = re.sub(r"(?<=\|)\s*Unnamed:\s*\d+\s*(?=\|)", " ", md)
        # Collapse excessive blank lines
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md

    # ── Docling pipeline (PDF, DOCX, PPTX, HTML) ─────────────────────────────

    def _convert_with_docling(
        self,
        file_path: str,
        *,
        doc_id: str,
        assets_dir: Path,
    ) -> tuple[str, str]:
        """Convert a document using the Docling pipeline."""
        result = self._converter.convert(file_path)
        if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            raise ValueError(
                f"Docling conversion failed (status={result.status}) for: {file_path}"
            )
        if result.status == ConversionStatus.PARTIAL_SUCCESS:
            print(f"[converter] Warning: partial conversion for {file_path} — some pages may be missing")
        doc = result.document

        serializer = MarkdownDocSerializer(
            doc=doc,
            picture_serializer=VaultPictureSerializer(
                assets_dir=assets_dir,
                doc_id=doc_id,
            ),
            params=MarkdownParams(image_mode=ImageRefMode.PLACEHOLDER),
        )
        markdown = serializer.serialize().text
        markdown = self._post_process(markdown)
        markdown = self._refine(markdown)
        title = _extract_title(markdown, file_path)
        return markdown, title

    # ── deterministic post-processing ────────────────────────────────────────

    @staticmethod
    def _post_process(md: str) -> str:
        """Clean up known Docling output artefacts before optional LLM refinement.

        - ``<!-- formula-not-decoded -->`` → ``$[?]$`` (inline math placeholder)
          Docling emits this when it detects a math formula element but cannot
          recover the LaTeX source (e.g. image-only formulas in scanned PDFs).
        - Collapse runs of 3+ blank lines to 2 (keeps paragraphs tight).
        """
        # Replace HTML comment placeholder with a readable inline math stub
        md = md.replace("<!-- formula-not-decoded -->", "$[?]$")
        # Collapse excessive blank lines
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md

    # ── optional LLM post-processing ─────────────────────────────────────────

    def _refine(self, raw_md: str) -> str:
        """Send Markdown to the configured LLM for OCR noise removal.

        Returns *raw_md* unchanged if:
        - LLM is not configured (no base_url / model)
        - The LLM call raises any exception (graceful fallback)
        """
        s = self._settings
        if not (s.llm.base_url and s.llm.model):
            return raw_md
        try:
            client = OpenAI(
                base_url=s.llm.base_url,
                api_key=s.llm.api_key or "none",
            )
            resp = client.chat.completions.create(
                model=s.llm.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Markdown cleanup assistant. "
                            "Remove OCR noise and garbage characters. "
                            "Strictly preserve all headings, structure, tables, "
                            "and ![image] tags exactly as-is. "
                            "Return only the cleaned Markdown, no commentary."
                        ),
                    },
                    {"role": "user", "content": raw_md},
                ],
                temperature=0,
            )
            return resp.choices[0].message.content or raw_md
        except Exception as exc:
            print(f"[converter] LLM refinement failed (using raw): {exc}")
            return raw_md


# Backwards-compatibility alias — existing code importing DocumentConverter
# will continue to work without changes.
DocumentConverter = DoclingConverter
