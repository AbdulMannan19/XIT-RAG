from datetime import datetime
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .crawler_helpers import check_robots_txt, can_fetch_url, apply_rate_limit
from models import ContentType, CrawledPage
from utils import is_irs_domain, normalize_url


class WebCrawler:
    def __init__(
        self,
        base_url: str,
        rate_limit_rps: float = 0.5,
        user_agent: str = "IRS-RAG-Bot/1.0",
    ):
        self.base_url = base_url
        self.rate_limit_rps = rate_limit_rps
        self.user_agent = user_agent
        self.last_request_time = 0.0
        self.robots_parser = None
        self.seen_urls = set()
        self.client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        )

    def _check_robots_txt(self) -> None:
        self.robots_parser = check_robots_txt(self.base_url)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch(self, url: str) -> Optional[CrawledPage]:
        url = normalize_url(url, self.base_url)

        if url in self.seen_urls:
            return None

        if not can_fetch_url(self.robots_parser, self.user_agent, url):
            return None

        if not is_irs_domain(url):
            return None

        self.last_request_time = apply_rate_limit(self.last_request_time, self.rate_limit_rps)

        try:
            response = self.client.get(url)
            response.raise_for_status()

            content_type = ContentType.HTML
            content_type_header = response.headers.get("content-type", "").lower()
            if "application/pdf" in content_type_header or url.lower().endswith(".pdf"):
                content_type = ContentType.PDF

            last_modified = None
            if "last-modified" in response.headers:
                try:
                    last_modified = datetime.strptime(
                        response.headers["last-modified"], "%a, %d %b %Y %H:%M:%S %Z"
                    )
                except Exception:
                    pass

            etag = response.headers.get("etag")
            title = url.split("/")[-1] or "Untitled"

            self.seen_urls.add(url)

            return CrawledPage(
                url=url,
                title=title,
                crawl_timestamp=datetime.utcnow(),
                last_modified=last_modified,
                content_type=content_type,
                raw_content=response.content,
                cleaned_text="",
                content_hash="",
                etag=etag,
                status_code=response.status_code,
            )

        except httpx.HTTPStatusError as e:
            return None
        except Exception as e:
            return None

    def close(self) -> None:
        self.client.close()
