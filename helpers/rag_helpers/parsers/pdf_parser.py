from helpers.rag_helpers import extract_pdf_text
from models import CrawledPage


class PdfParser:
    def parse(self, page: CrawledPage) -> CrawledPage:
        try:
            text, metadata = extract_pdf_text(page.raw_content)

            title = page.title
            if title == "Untitled" or not title:
                if metadata.get("headings"):
                    title = metadata["headings"][0]["text"]
                else:
                    url_str = str(page.url)
                    title = url_str.split("/")[-1].replace(".pdf", "").replace("_", " ").title()

            page.title = title
            page.cleaned_text = text

            return page

        except Exception as e:
            page.cleaned_text = ""
            return page
