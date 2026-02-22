"""
Adaptive Difficulty Engine
Dynamically adjusts interview complexity based on performance scoring.
"""

from typing import List, Dict


def compute_score(marks_earned: float, marks_total: float) -> float:
    if marks_total == 0:
        return 0.0
    return round((marks_earned / marks_total) * 10, 2)


def get_next_difficulty(answers: List[Dict]) -> str:
    """
    Adaptive difficulty:
      avg score > 8  → Tough
      avg score > 5  → Moderate
      else           → Easy
    Uses rolling window of last 3 answers.
    """
    if not answers:
        return "Moderate"

    recent = answers[-3:]
    scores = [
        compute_score(a.get("marks_earned", 0), a.get("marks_total", 5))
        for a in recent
    ]
    avg = sum(scores) / len(scores)

    if avg > 8:
        return "Tough"
    elif avg > 5:
        return "Moderate"
    else:
        return "Easy"


def get_performance_label(percent: float) -> str:
    if percent >= 85:
        return "🏆 Expert"
    elif percent >= 70:
        return "🌟 Proficient"
    elif percent >= 50:
        return "📈 Developing"
    else:
        return "🔧 Needs Practice"


def topic_weakness_map(answers: List[Dict]) -> Dict[str, float]:
    topic_data: Dict[str, Dict] = {}
    for a in answers:
        t = a.get("topic", "General")
        topic_data.setdefault(t, {"earned": 0, "total": 0})
        topic_data[t]["earned"] += a.get("marks_earned", 0)
        topic_data[t]["total"] += a.get("marks_total", 5)

    return {
        t: round((v["earned"] / v["total"]) * 100, 1) if v["total"] else 0
        for t, v in topic_data.items()
    }


def get_weak_topics(answers: List[Dict], threshold: float = 60.0) -> List[str]:
    wmap = topic_weakness_map(answers)
    return [t for t, pct in wmap.items() if pct < threshold]
