from typing import Iterator, Optional
from xml.etree import ElementTree as ET

import httpx

from services.core_services import is_irs_domain, normalize_url


def parse_sitemap_xml(sitemap_url: str, max_urls: Optional[int] = None) -> Iterator[str]:
    try:
        response = httpx.get(sitemap_url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        if root.tag == "{http://www.sitemaps.org/schemas/sitemap/0.9}sitemapindex":
            for sitemap in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap"):
                loc = sitemap.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
                if loc is not None and loc.text:
                    yield from parse_sitemap_xml(loc.text, max_urls)
        else:
            count = 0
            for url_elem in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
                if max_urls and count >= max_urls:
                    break
                loc = url_elem.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
                if loc is not None and loc.text:
                    url = normalize_url(loc.text)
                    if is_irs_domain(url):
                        yield url
                        count += 1

    except Exception as e:
        pass


def discover_sitemap_locations(base_url: str) -> list[str]:
    sitemap_urls = [
        f"{base_url}/sitemap.xml",
        f"{base_url}/sitemap_index.xml",
        f"{base_url}/sitemap/sitemap.xml",
    ]

    found = []
    for url in sitemap_urls:
        try:
            response = httpx.head(url, timeout=10.0, follow_redirects=True)
            if response.status_code == 200:
                found.append(url)
        except Exception:
            pass

    return found


def parse_robots_for_sitemaps(base_url: str) -> Optional[str]:
    robots_url = f"{base_url}/robots.txt"
    try:
        response = httpx.get(robots_url, timeout=10.0, follow_redirects=True)
        if response.status_code == 200:
            sitemaps = []
            for line in response.text.splitlines():
                line = line.strip()
                if line.lower().startswith("sitemap:"):
                    sitemap_url = line.split(":", 1)[1].strip()
                    if is_irs_domain(sitemap_url):
                        sitemaps.append(sitemap_url)
            return "\n".join(sitemaps) if sitemaps else None
    except Exception as e:
        pass

    return None
