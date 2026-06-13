"""Regression test: PDF OCR must run with bounded memory.

Root cause of the std::bad_alloc crash: Docling renders OCR regions at scale=3
(216 DPI) and RapidOCR's default detector uses ``Det.limit_type: min`` with
``limit_side_len: 736``, which does NOT shrink large scanned-page images. The
detector then allocates a feature-map tensor proportional to the full image
size, exhausts the C++ heap, and aborts the whole sidecar process.

The fix caps the detector input via ``rapidocr_params`` so the longest side is
bounded regardless of source resolution. These assertions guard against a
regression where the bound is dropped and the crash returns.
"""

from docling.datamodel.pipeline_options import RapidOcrOptions

from backend.services.converter import _build_pdf_options


class _LLM:
    base_url = ""
    model = ""


class _Settings:
    @property
    def active_llm(self):
        return _LLM()


def test_pdf_ocr_options_bound_detector_memory():
    opts = _build_pdf_options(_Settings())

    # OCR stays enabled (digital PDFs OCR almost nothing; scanned PDFs need it).
    assert opts.do_ocr is True
    # Region-based OCR, not forced full-page — keeps digital PDFs cheap.
    assert opts.ocr_options.force_full_page_ocr is False

    # The detector input must be capped by the LONGEST side, otherwise large
    # scanned pages blow up the native allocation.
    assert isinstance(opts.ocr_options, RapidOcrOptions)
    params = opts.ocr_options.rapidocr_params
    assert params["Det.limit_type"] == "max"
    assert params["Det.limit_side_len"] <= 1280
    assert params["Global.max_side_len"] <= 1536
