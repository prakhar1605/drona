"""
PDF Parser — extracts clean text from uploaded resumes.
"""

from pypdf import PdfReader


MAX_CHARS = 8000  # ~2000 tokens — safe for GPT-4o-mini context

def extract_text_from_pdf(file) -> str:
    if not file:
        return ""
    reader = PdfReader(file)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    full_text = "\n".join(pages).strip()
    # Truncate to avoid context overflow — keep first N chars (most important resume info)
    if len(full_text) > MAX_CHARS:
        full_text = full_text[:MAX_CHARS] + "\n[...truncated for context limit]"
    return full_text
