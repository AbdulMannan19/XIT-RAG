"""HTML parsing and extraction."""

import logging
from typing import Optional

from bs4 import BeautifulSoup
from readability import Document

from core.utils import normalize_text
from ingestion.models import ContentType, CrawledPage

logger = logging.getLogger(__name__)


def extract_title(html: str) -> str:
    """Extract page title from HTML."""
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    if title_tag:
        return normalize_text(title_tag.get_text())
    return "Untitled"


def extract_breadcrumbs(html: str) -> list[str]:
    """Extract breadcrumb navigation."""
    soup = BeautifulSoup(html, "lxml")
    breadcrumbs = []

    # Look for common breadcrumb patterns
    nav = soup.find("nav", class_=lambda x: x and "breadcrumb" in x.lower() if x else False)
    if nav:
        links = nav.find_all("a")
        for link in links:
            text = normalize_text(link.get_text())
            if text:
                breadcrumbs.append(text)

    return breadcrumbs


def extract_headings(html: str) -> list[dict[str, str]]:
    """Extract hierarchical headings (h1-h6)."""
    soup = BeautifulSoup(html, "lxml")
    headings = []

    for level in range(1, 7):
        for tag in soup.find_all(f"h{level}"):
            text = normalize_text(tag.get_text())
            if text:
                headings.append({"level": level, "text": text, "id": tag.get("id")})

    return headings


def extract_faq_pairs(html: str) -> list[dict[str, str]]:
    """Extract FAQ question/answer pairs."""
    soup = BeautifulSoup(html, "lxml")
    faqs = []

    # Look for common FAQ patterns
    # Pattern 1: dt/dd pairs
    for dt in soup.find_all("dt"):
        question = normalize_text(dt.get_text())
        dd = dt.find_next_sibling("dd")
        if dd:
            answer = normalize_text(dd.get_text())
            faqs.append({"question": question, "answer": answer})

    # Pattern 2: divs with class containing "faq", "question", "answer"
    faq_containers = soup.find_all(
        "div", class_=lambda x: x and any(kw in x.lower() for kw in ["faq", "question"]) if x else False
    )
    for container in faq_containers:
        question_elem = container.find(["h2", "h3", "strong", "b"])
        answer_elem = container.find(["p", "div"])
        if question_elem and answer_elem:
            question = normalize_text(question_elem.get_text())
            answer = normalize_text(answer_elem.get_text())
            if question and answer:
                faqs.append({"question": question, "answer": answer})

    return faqs


def extract_tables(html: str) -> list[dict]:
    """Extract tables and convert to TSV representation."""
    soup = BeautifulSoup(html, "lxml")
    tables_data = []

    for table in soup.find_all("table"):
        rows = []
        headers = []

        # Extract headers
        thead = table.find("thead")
        if thead:
            header_row = thead.find("tr")
            if header_row:
                headers = [normalize_text(th.get_text()) for th in header_row.find_all(["th", "td"])]

        # Extract rows
        tbody = table.find("tbody") or table
        for tr in tbody.find_all("tr"):
            cells = [normalize_text(td.get_text()) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)

        if rows:
            # Create TSV representation
            tsv_lines = []
            if headers:
                tsv_lines.append("\t".join(headers))
            for row in rows:
                tsv_lines.append("\t".join(row))

            tables_data.append(
                {
                    "headers": headers,
                    "rows": rows,
                    "tsv": "\n".join(tsv_lines),
                    "row_count": len(rows),
                    "col_count": len(headers) if headers else (len(rows[0]) if rows else 0),
                }
            )

    return tables_data


def parse_html(page: CrawledPage) -> CrawledPage:
    """Parse HTML page and extract content."""
    try:
        html = page.raw_content.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")

        # Use readability to extract main content
        doc = Document(html)
        main_content = doc.summary()

        # Parse main content with BeautifulSoup
        main_soup = BeautifulSoup(main_content, "lxml")

        # Extract title
        title = extract_title(html)
        if title == "Untitled":
            h1 = soup.find("h1")
            if h1:
                title = normalize_text(h1.get_text())

        # Extract text from main content
        text = normalize_text(main_soup.get_text())

        # Extract metadata
        breadcrumbs = extract_breadcrumbs(html)
        headings = extract_headings(html)
        faqs = extract_faq_pairs(html)
        tables = extract_tables(html)

        # Update page
        page.title = title
        page.cleaned_text = text

        # Store metadata in a way that can be accessed later
        # For now, we'll add it to the model if needed, or store separately

        return page

    except Exception as e:
        logger.error(f"Error parsing HTML for {page.url}: {e}")
        # Fallback to basic extraction
        try:
            soup = BeautifulSoup(page.raw_content.decode("utf-8", errors="ignore"), "lxml")
            page.cleaned_text = normalize_text(soup.get_text())
            page.title = extract_title(page.raw_content.decode("utf-8", errors="ignore"))
        except Exception:
            page.cleaned_text = ""
            page.title = "Parse Error"

        return page


