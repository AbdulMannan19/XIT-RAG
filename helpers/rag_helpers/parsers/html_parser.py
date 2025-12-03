from bs4 import BeautifulSoup
from readability import Document

from models import CrawledPage
from utils import normalize_text
from helpers.rag_helpers.extractors import extract_title


class HtmlParser:
    def parse(self, page: CrawledPage) -> CrawledPage:
        try:
            html = page.raw_content.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "lxml")

            doc = Document(html)
            main_content = doc.summary()
            main_soup = BeautifulSoup(main_content, "lxml")

            title = extract_title(html)
            if title == "Untitled":
                h1 = soup.find("h1")
                if h1:
                    title = normalize_text(h1.get_text())

            text = normalize_text(main_soup.get_text())

            page.title = title
            page.cleaned_text = text

            return page

        except Exception as e:
            try:
                soup = BeautifulSoup(page.raw_content.decode("utf-8", errors="ignore"), "lxml")
                page.cleaned_text = normalize_text(soup.get_text())
                page.title = extract_title(page.raw_content.decode("utf-8", errors="ignore"))
            except Exception:
                page.cleaned_text = ""
                page.title = "Parse Error"

            return page
