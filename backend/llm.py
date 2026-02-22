"""
LLM Module — OpenRouter + Streaming
Handles all LLM calls with token streaming support.
"""

import os
import re
import json
from typing import List, Dict, Generator, Optional
from openai import OpenAI

_client: Optional[OpenAI] = None
MODEL = "openai/gpt-4o-mini"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


def ask_llm(prompt: str, system: str = "You are an honest interview question generator and assessment advisor.") -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content


def ask_llm_stream(
    prompt: str,
    system: str = "You are an honest, concise exam-feedback generator. Produce structured markdown output.",
) -> Generator[str, None, None]:
    client = _get_client()
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.15,
            stream=True,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
            except Exception:
                continue
    except Exception as e:
        yield f"\n⚠️ Streaming error: {e}"


def safe_json(text: str) -> List[Dict]:
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        raise ValueError("No JSON array found in model response.")
    return json.loads(match.group())


def normalize_options(opts: List[str]) -> List[str]:
    return [o.strip() for o in (opts or [])]


def build_quiz_prompt(
    topics: List[str],
    num_q: int,
    difficulty: str,
    role: str,
    audience: str,
    context: str = "",
    weak_topics: List[str] = None,
    history_summary: str = "",
) -> str:
    weak_topics = weak_topics or []

    audience_instr = (
        "Use very simple school-level language." if audience == "School Student"
        else "Use clear, professional interview-style language."
    )

    memory_instr = ""
    if weak_topics:
        memory_instr = (
            f"\nIMPORTANT: The candidate historically struggles with: {', '.join(weak_topics)}. "
            f"Include MORE questions on these weak areas to help them improve."
        )
    if history_summary:
        memory_instr += f"\nCandidate history: {history_summary}"

    return f"""
You must return a JSON array of exactly {num_q} question objects.

Each object must have:
- question (string)
- options (array of exactly 4 strings)
- correct_options (array of correct option text strings)
- topic (string, one of: {topics})
- difficulty (string: "Easy", "Moderate", or "Tough")
- marks (integer: Easy=3, Moderate=5, Tough=10)

Rules:
- Audience: {audience_instr}
- Target role: {role}
- Difficulty preference: {difficulty}
  * If "Easy": ALL questions Easy
  * If "Moderate": ALL questions Moderate
  * If "Tough": ALL questions Tough
  * If "Mixed": spread across Easy/Moderate/Tough
  * If "Adaptive (Auto)": use {difficulty}
- Easy/Moderate: exactly 1 correct option
- Tough: 2-3 correct options allowed
- Cover DIFFERENT subtopics, question styles (conceptual, code-based, scenario)
- Spread questions evenly across topics
{memory_instr}

Resume/Context:
{context if context else "No resume provided."}

Return ONLY a valid JSON array. No commentary, no markdown fences.
"""


def build_feedback_prompt(
    percent: float,
    earned: float,
    total: float,
    num_correct: int,
    total_q: int,
    topic_perf: List[str],
    weak_topics: List[str],
    strengths: List[str],
) -> str:
    return f"""
Create a comprehensive technical interview performance report.

Score: {percent}%
Marks: {earned}/{total}
Correct: {num_correct}/{total_q}
Topic Performance: {', '.join(topic_perf)}
Weak Areas: {weak_topics}
Strong Areas: {strengths}

Generate a professional, actionable assessment:
1. Overall Performance Summary (2-3 key insights)
2. Technical Strengths Demonstrated
3. Critical Improvement Areas
4. 5 Recommended Practice Projects
5. 7-Day Focused Study Plan
6. 3 Best Learning Resources

Keep it interview-focused, honest, and motivating.
"""
