"""Text cleaning and normalization utilities."""

import re
from typing import Optional

from core.utils import normalize_text


def remove_boilerplate(text: str) -> str:
    """Remove common boilerplate text (navigation, footers, etc.)."""
    # Common IRS.gov boilerplate patterns
    boilerplate_patterns = [
        r"Skip to main content",
        r"Sign in|Sign out",
        r"Search\.\.\.",
        r"Menu",
        r"Home\s*>\s*",
        r"Last updated:.*",
        r"Page Last Reviewed or Updated:.*",
        r"Share this page",
        r"Print this page",
        r"Contact Us",
        r"Privacy Policy",
        r"Terms of Use",
        r"Accessibility",
        r"FOIA",
        r"Taxpayer Advocate",
    ]

    cleaned = text
    for pattern in boilerplate_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    return normalize_text(cleaned)


def clean_html_text(text: str) -> str:
    """Clean text extracted from HTML."""
    # Remove extra whitespace
    text = normalize_text(text)

    # Remove boilerplate
    text = remove_boilerplate(text)

    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_pdf_text(text: str) -> str:
    """Clean text extracted from PDF."""
    # Remove page markers if they're excessive
    text = re.sub(r"\[Page \d+\]\s*\n", "", text)

    # Normalize whitespace
    text = normalize_text(text)

    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_examples(text: str) -> list[str]:
    """Extract example sections from text."""
    examples = []
    # Pattern for "Example" or "Example 1:" etc.
    pattern = r"Example\s*\d*:?\s*\n(.*?)(?=\n\n|\n[A-Z]|$)"
    matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
    for match in matches:
        example_text = normalize_text(match.group(1))
        if len(example_text) > 20:  # Reasonable example length
            examples.append(example_text)

    return examples


def extract_numbered_steps(text: str) -> list[str]:
    """Extract numbered steps/procedures."""
    steps = []
    # Pattern for numbered steps: "1.", "Step 1:", etc.
    pattern = r"(?:Step\s*)?\d+[\.\)]\s*(.*?)(?=\n(?:Step\s*)?\d+[\.\)]|\n\n|$)"
    matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
    for match in matches:
        step_text = normalize_text(match.group(1))
        if len(step_text) > 10:  # Reasonable step length
            steps.append(step_text)

    return steps


def preserve_irs_form_numbers(text: str) -> str:
    """Ensure IRS form numbers are preserved correctly."""
    # Normalize form number formats
    text = re.sub(r"Form\s+(\d+[A-Z]?-\d+)", r"Form \1", text, flags=re.IGNORECASE)
    text = re.sub(r"Form\s+([A-Z]\d+)", r"Form \1", text, flags=re.IGNORECASE)
    return text


