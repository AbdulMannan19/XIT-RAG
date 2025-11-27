import time
from typing import Optional
from urllib.robotparser import RobotFileParser


def check_robots_txt(base_url: str) -> Optional[RobotFileParser]:
    try:
        robots_url = f"{base_url}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.read()
        return parser
    except Exception as e:
        return None


def can_fetch_url(robots_parser: Optional[RobotFileParser], user_agent: str, url: str) -> bool:
    if robots_parser is None:
        return True
    try:
        return robots_parser.can_fetch(user_agent, url)
    except Exception:
        return True


def apply_rate_limit(last_request_time: float, rate_limit_rps: float) -> float:
    elapsed = time.time() - last_request_time
    min_interval = 1.0 / rate_limit_rps
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    return time.time()
