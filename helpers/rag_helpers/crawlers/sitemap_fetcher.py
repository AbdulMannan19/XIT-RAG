from typing import Optional

from .sitemap_helpers import parse_sitemap_xml, discover_sitemap_locations, parse_robots_for_sitemaps
from services.core_services import normalize_url


class SitemapFetcher:
    def get_seed_urls(self, base_url: str, max_urls: Optional[int] = None) -> list[str]:
        urls = set()

        robots_content = parse_robots_for_sitemaps(base_url)
        if robots_content:
            for line in robots_content.splitlines():
                if line.strip() and not line.startswith("#"):
                    urls.add(normalize_url(line.strip()))

        sitemaps = discover_sitemap_locations(base_url)
        for sitemap_url in sitemaps:
            for url in parse_sitemap_xml(sitemap_url, max_urls):
                urls.add(url)
                if max_urls and len(urls) >= max_urls:
                    break

        if not urls:
            urls.add(normalize_url(base_url))

        return sorted(list(urls))
