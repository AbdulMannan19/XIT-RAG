from bs4 import BeautifulSoup

from services.core_services import normalize_text


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    if title_tag:
        return normalize_text(title_tag.get_text())
    return "Untitled"


def extract_breadcrumbs(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    breadcrumbs = []

    nav = soup.find("nav", class_=lambda x: x and "breadcrumb" in x.lower() if x else False)
    if nav:
        links = nav.find_all("a")
        for link in links:
            text = normalize_text(link.get_text())
            if text:
                breadcrumbs.append(text)

    return breadcrumbs


def extract_headings(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    headings = []

    for level in range(1, 7):
        for tag in soup.find_all(f"h{level}"):
            text = normalize_text(tag.get_text())
            if text:
                headings.append({"level": level, "text": text, "id": tag.get("id")})

    return headings


def extract_faq_pairs(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    faqs = []

    for dt in soup.find_all("dt"):
        question = normalize_text(dt.get_text())
        dd = dt.find_next_sibling("dd")
        if dd:
            answer = normalize_text(dd.get_text())
            faqs.append({"question": question, "answer": answer})

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
    soup = BeautifulSoup(html, "lxml")
    tables_data = []

    for table in soup.find_all("table"):
        rows = []
        headers = []

        thead = table.find("thead")
        if thead:
            header_row = thead.find("tr")
            if header_row:
                headers = [normalize_text(th.get_text()) for th in header_row.find_all(["th", "td"])]

        tbody = table.find("tbody") or table
        for tr in tbody.find_all("tr"):
            cells = [normalize_text(td.get_text()) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)

        if rows:
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
