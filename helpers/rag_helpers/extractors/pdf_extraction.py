from io import BytesIO

import fitz

from services.core_services import normalize_text


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, dict]:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = []
        page_texts = []
        headings = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            normalized = normalize_text(text)

            if normalized:
                full_text.append(f"[Page {page_num}]\n{normalized}\n")
                page_texts.append({"page": page_num, "text": normalized})

            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            font_size = span.get("size", 0)
                            flags = span.get("flags", 0)
                            text_span = span.get("text", "").strip()

                            if text_span and (flags & 16) and font_size > 10:
                                if len(text_span) < 200:
                                    headings.append(
                                        {
                                            "page": page_num,
                                            "text": normalize_text(text_span),
                                            "size": font_size,
                                        }
                                    )

        page_count = len(doc)
        doc.close()

        return "\n".join(full_text), {
            "page_count": page_count,
            "page_texts": page_texts,
            "headings": headings,
        }

    except Exception as e:
        try:
            from pdfminer.high_level import extract_text

            text = extract_text(BytesIO(pdf_bytes))
            return normalize_text(text), {"page_count": 0, "page_texts": [], "headings": []}
        except Exception as fallback_error:
            return "", {"page_count": 0, "page_texts": [], "headings": []}
