"""
DRONA AI — Autonomous Interview Agent
Upgraded with:
  ✅ ChromaDB interview memory (long-term weakness tracking)
  ✅ Adaptive difficulty engine (performance-based scoring)
  ✅ Real-time streaming LLM responses (token-by-token)
  ✅ Redis Cloud caching (reduced latency)
  ✅ Async-ready modular architecture
"""

import os
import time
import json
import uuid
import streamlit as st
from dotenv import load_dotenv

from backend.llm import (
    ask_llm, ask_llm_stream, safe_json, normalize_options,
    build_quiz_prompt, build_feedback_prompt
)
from backend.adaptive import (
    get_next_difficulty, get_performance_label,
    topic_weakness_map, get_weak_topics
)
from backend.cache import get_cached_questions, cache_questions, is_redis_connected
from backend.memory import (
    store_answer, get_weak_areas, get_session_history_summary, clear_session_memory
)
from backend.pdf_parser import extract_text_from_pdf

load_dotenv()

# Streamlit Cloud secrets support
try:
    import streamlit as _st
    for key, val in _st.secrets.items():
        os.environ.setdefault(key, val)
except Exception:
    pass

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Drona AI — Autonomous Interview Agent",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --accent: #7c6aff;
    --accent2: #ff6a9b;
    --green: #4fffb0;
    --red: #ff4f6a;
    --text: #e8e8f0;
    --muted: #6b6b88;
    --border: #2a2a3d;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }

.main-hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 4px 14px;
    border-radius: 20px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, var(--accent) 60%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0.3rem 0;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    font-family: 'Space Mono', monospace;
    margin-top: 0.5rem;
}
.status-bar {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.status-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
}
.status-chip.online { border-color: var(--green); color: var(--green); }
.status-chip.offline { border-color: var(--red); color: var(--red); }

.q-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin: 1rem 0;
}
.q-meta { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1rem; }
.q-tag {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    color: var(--muted);
}
.q-tag.diff-easy { border-color: var(--green); color: var(--green); }
.q-tag.diff-moderate { border-color: #ffb347; color: #ffb347; }
.q-tag.diff-tough { border-color: var(--accent2); color: var(--accent2); }
.q-text { font-size: 1.15rem; font-weight: 600; color: var(--text); line-height: 1.5; }

.adaptive-indicator {
    background: var(--surface2);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent);
    margin: 0.5rem 0;
    text-align: center;
}
.memory-banner {
    background: linear-gradient(135deg, rgba(124,106,255,0.1), rgba(255,106,155,0.1));
    border: 1px solid rgba(124,106,255,0.3);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: var(--accent);
    margin: 0.5rem 0;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 800; font-family: 'Space Mono', monospace; }
.metric-label { font-size: 0.75rem; color: var(--muted); font-family: 'Space Mono', monospace; text-transform: uppercase; letter-spacing: 1px; }

.topic-row {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
}
.prog-bar-bg { background: var(--surface2); border-radius: 4px; height: 6px; margin-top: 0.4rem; }
.prog-bar-fill { height: 6px; border-radius: 4px; transition: width 0.5s ease; }

.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface2) !important;
    color: var(--text) !important;
}
.stButton > button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border: none !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover { opacity: 0.9 !important; color: white !important; }
.stProgress > div > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; }
div[data-testid="stMarkdownContainer"] p { color: var(--text); }
.stExpander { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
[data-testid="stFileUploader"] { background: var(--surface) !important; border: 2px dashed var(--border) !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "page": "setup",
        "quiz": [],
        "current": 0,
        "answers": [],
        "topics": [],
        "audience": "College Student",
        "is_generating": False,
        "num_q": 10,
        "session_id": str(uuid.uuid4()),
        "current_difficulty": "Moderate",
        "from_cache": False,
        "target_role": "Software Engineer",
        "difficulty": "Moderate",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    _init_state()
    st.session_state.session_id = str(uuid.uuid4())


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
redis_ok = is_redis_connected()
redis_class = "online" if redis_ok else "offline"
redis_label = "⚡ Redis Cache: ON" if redis_ok else "○ Redis Cache: OFF"

st.markdown("""
<div class="main-hero">
    <div class="hero-badge">Autonomous Interview Agent</div>
    <div class="hero-title">DRONA AI</div>
    <div class="hero-sub">adaptive · memory-driven · real-time</div>
</div>
""", unsafe_allow_html=True)




# ─────────────────────────────────────────────
# SETUP PAGE
# ─────────────────────────────────────────────
if st.session_state.page == "setup":

    weak_remembered = get_weak_areas(st.session_state.session_id)
    if weak_remembered:
        st.markdown(
            f'<div class="memory-banner">🧠 <b>Memory Active</b> — '
            f'Weak areas from past sessions: <b>{", ".join(weak_remembered)}</b>. '
            f'More questions will focus here.</div>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("📄 Upload Resume (Optional)")
        uploaded = st.file_uploader("Drop your PDF resume", type=["pdf"], label_visibility="collapsed")
        if uploaded:
            st.success(f"✅ Loaded: {uploaded.name}")

        st.subheader("🏷️ Add Topics")
        c1, c2 = st.columns([3, 1])
        with c1:
            new_topic = st.text_input("Topic", placeholder="e.g. Python, DSA, System Design", label_visibility="collapsed")
        with c2:
            if st.button("Add", use_container_width=True):
                t = new_topic.strip()
                if t and t not in st.session_state.topics:
                    st.session_state.topics.append(t)
                    st.rerun()

        if st.session_state.topics:
            tags_html = "".join(
                f'<span style="display:inline-block;background:#1a1a26;border:1px solid #2a2a3d;'
                f'border-radius:20px;padding:3px 12px;margin:3px;font-size:0.85rem;color:#7c6aff;">{t}</span>'
                for t in st.session_state.topics
            )
            st.markdown(tags_html, unsafe_allow_html=True)
            if st.button("✕ Clear Topics", type="secondary"):
                st.session_state.topics = []
                st.rerun()

    with col2:
        st.subheader("⚙️ Settings")

        st.session_state.audience = st.selectbox(
            "Audience Level",
            ["School Student", "College Student", "Intern / Fresher", "Professional"],
            index=1,
        )
        st.session_state.target_role = st.selectbox(
            "Target Role",
            ["Software Engineer", "Data Scientist", "Data Analyst", "AI / ML Engineer",
             "Web Developer", "DevOps Engineer", "Product Manager", "Learning / Exam Prep"],
        )
        st.session_state.difficulty = st.selectbox(
            "Difficulty",
            ["Easy", "Moderate", "Tough", "Adaptive (Auto)", "Mixed"],
            index=1,
        )
        st.markdown(f"**Questions: {st.session_state.num_q}**")
        st.session_state.num_q = st.slider("Count", 5, 50, st.session_state.num_q, 5, label_visibility="collapsed")

        st.divider()
        if st.button("🗑️ Clear Memory", type="secondary", use_container_width=True):
            clear_session_memory(st.session_state.session_id)
            st.success("Memory cleared!")

    st.divider()

    if st.button("🎯 Launch Interview", type="primary", use_container_width=True):
        if not st.session_state.topics and not uploaded:
            st.error("Add at least one topic or upload a PDF resume.")
        else:
            st.session_state.is_generating = True
            st.rerun()

    # Generation
    if st.session_state.is_generating:
        with st.spinner("🔮 Generating your personalized interview..."):
            try:
                context = extract_text_from_pdf(uploaded) if uploaded else ""
                weak_topics = get_weak_areas(st.session_state.session_id)
                history_summary = get_session_history_summary(st.session_state.session_id)

                resolved_diff = st.session_state.difficulty
                if resolved_diff == "Adaptive (Auto)":
                    resolved_diff = get_next_difficulty(st.session_state.answers)
                st.session_state.current_difficulty = resolved_diff

                # Check Redis cache
                cached = get_cached_questions(
                    st.session_state.topics, resolved_diff,
                    st.session_state.target_role, st.session_state.num_q
                )

                if cached and not weak_topics:
                    quiz = cached
                    st.session_state.from_cache = True
                else:
                    st.session_state.from_cache = False
                    prompt = build_quiz_prompt(
                        topics=st.session_state.topics,
                        num_q=st.session_state.num_q,
                        difficulty=resolved_diff,
                        role=st.session_state.target_role,
                        audience=st.session_state.audience,
                        context=context,
                        weak_topics=weak_topics,
                        history_summary=history_summary,
                    )
                    raw = ask_llm(prompt)
                    quiz = safe_json(raw)

                    for q in quiz:
                        q["options"] = normalize_options(q.get("options", []))
                        q["correct_options"] = [c.strip() for c in q.get("correct_options", [])]
                        if "marks" not in q:
                            d = q.get("difficulty", "Moderate").lower()
                            q["marks"] = 10 if d.startswith("t") else (5 if d.startswith("m") else 3)

                    cache_questions(
                        st.session_state.topics, resolved_diff,
                        st.session_state.target_role, st.session_state.num_q, quiz
                    )

                st.session_state.quiz = quiz
                st.session_state.current = 0
                st.session_state.answers = []
                st.session_state.page = "quiz"
                st.session_state.is_generating = False
                st.rerun()

            except Exception as e:
                err = str(e)
                if "context" in err.lower() or "token" in err.lower():
                    st.error("📄 PDF is too large to process fully. Try uploading a shorter document or add topics manually instead.")
                elif "api" in err.lower() or "key" in err.lower() or "auth" in err.lower():
                    st.error("🔑 API key issue. Please check your OPENROUTER_API_KEY in .env file.")
                else:
                    st.error("⚠️ Something went wrong. Please try again.")
                st.session_state.is_generating = False


# ─────────────────────────────────────────────
# QUIZ PAGE
# ─────────────────────────────────────────────
elif st.session_state.page == "quiz":
    quiz = st.session_state.quiz
    idx = st.session_state.current
    total = len(quiz)

    if idx >= total:
        st.session_state.page = "result"
        st.rerun()

    q = quiz[idx]

    if len(st.session_state.answers) >= 3:
        next_diff = get_next_difficulty(st.session_state.answers)
        diff_emoji = {"Easy": "🟢", "Moderate": "🟡", "Tough": "🔴"}.get(next_diff, "⚪")
        st.markdown(
            f'<div class="adaptive-indicator">⚡ Adaptive Engine → Next difficulty: {diff_emoji} <b>{next_diff}</b></div>',
            unsafe_allow_html=True,
        )

    st.progress((idx + 1) / total, text=f"Question {idx + 1} / {total}")
    if st.session_state.get("from_cache"):
        st.caption("⚡ Served from Redis cache")

    diff = q.get("difficulty", "Moderate")
    diff_class = {"Easy": "diff-easy", "Moderate": "diff-moderate", "Tough": "diff-tough"}.get(diff, "")
    marks = q.get("marks", 5)

    st.markdown(
        f'<div class="q-card">'
        f'<div class="q-meta">'
        f'<span class="q-tag">{q.get("topic", "General")}</span>'
        f'<span class="q-tag {diff_class}">{diff}</span>'
        f'<span class="q-tag">+{marks} pts</span>'
        f'</div>'
        f'<div class="q-text">Q{idx + 1}. {q.get("question")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.write("**Select your answer:**")

    correct_count = len(q.get("correct_options", []))
    is_multi = correct_count > 1

    if is_multi:
        st.caption("*(Select all that apply)*")
        selected = st.multiselect("Options", q.get("options", []), label_visibility="collapsed", key=f"q_{idx}")
    else:
        selected_radio = st.radio("Options", q.get("options", []), label_visibility="collapsed", key=f"q_{idx}")
        selected = [selected_radio] if selected_radio else []

    st.divider()
    c1, c2, c3 = st.columns(3)

    with c1:
        if idx > 0:
            if st.button("⬅ Previous"):
                st.session_state.current -= 1
                st.rerun()

    with c2:
        if st.button("✅ Submit", type="primary"):
            if not selected:
                st.warning("Select at least one option.")
                st.stop()

            chosen_set = set(selected)
            correct_set = set(q.get("correct_options", []))
            marks_total = int(q.get("marks", 5))

            if len(correct_set) == 1:
                earned = marks_total if chosen_set == correct_set else 0
            else:
                matched = len(chosen_set & correct_set)
                earned = round(marks_total * (matched / max(1, len(correct_set))), 2)

            correct_flag = chosen_set == correct_set
            score_pct = round((earned / marks_total) * 100, 1)

            answer_record = {
                "index": idx,
                "question": q.get("question"),
                "chosen": list(chosen_set),
                "correct": list(correct_set),
                "marks_earned": earned,
                "marks_total": marks_total,
                "topic": q.get("topic"),
                "difficulty": diff,
                "correct_flag": correct_flag,
            }
            st.session_state.answers.append(answer_record)

            store_answer(
                session_id=st.session_state.session_id,
                question=q.get("question", ""),
                user_answer=list(chosen_set),
                correct_answer=list(correct_set),
                topic=q.get("topic", "General"),
                difficulty=diff,
                score_percent=score_pct,
            )

            st.session_state.current += 1
            if st.session_state.current >= total:
                st.session_state.page = "result"
            st.rerun()

    with c3:
        if idx < total - 1:
            if st.button("Skip ➡"):
                st.session_state.answers.append({
                    "index": idx,
                    "question": q.get("question"),
                    "chosen": [],
                    "correct": q.get("correct_options", []),
                    "marks_earned": 0,
                    "marks_total": q.get("marks", 5),
                    "topic": q.get("topic"),
                    "difficulty": diff,
                    "correct_flag": False,
                })
                st.session_state.current += 1
                st.rerun()


# ─────────────────────────────────────────────
# RESULT PAGE
# ─────────────────────────────────────────────
elif st.session_state.page == "result":
    answers = st.session_state.answers
    if not answers:
        st.warning("No answers recorded.")
        if st.button("↩ Back to Setup"):
            reset_all()
            st.rerun()
        st.stop()

    total_marks = sum(a["marks_total"] for a in answers)
    earned_marks = sum(a["marks_earned"] for a in answers)
    percent = round((earned_marks / total_marks) * 100, 1) if total_marks else 0
    num_correct = sum(1 for a in answers if a.get("correct_flag"))
    total_q = len(answers)
    perf_label = get_performance_label(percent)

    st.markdown(f"""
    <div style="text-align:center;padding:2rem 0 1rem;">
        <div style="font-size:3.5rem;font-weight:800;
            background:linear-gradient(135deg,#fff,#7c6aff,#ff6a9b);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;">{percent}%</div>
        <div style="font-size:1.3rem;color:#6b6b88;margin-top:0.3rem;">{perf_label}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#7c6aff;">{earned_marks:.0f}<span style="font-size:1rem;color:#6b6b88;">/{total_marks}</span></div><div class="metric-label">Marks</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#4fffb0;">{num_correct}<span style="font-size:1rem;color:#6b6b88;">/{total_q}</span></div><div class="metric-label">Correct</div></div>', unsafe_allow_html=True)
    with c3:
        weak = get_weak_topics(answers)
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#ff6a9b;">{len(weak)}</div><div class="metric-label">Weak Areas</div></div>', unsafe_allow_html=True)

    st.divider()

    st.subheader("📊 Topic Performance")
    topic_scores = {}
    for a in answers:
        t = a.get("topic", "Other")
        topic_scores.setdefault(t, []).append(a)

    for t, qs in topic_scores.items():
        sc = sum(q["marks_earned"] for q in qs)
        tot = sum(q["marks_total"] for q in qs)
        pct = int((sc / tot) * 100) if tot else 0
        bar_color = "#4fffb0" if pct >= 70 else ("#ffb347" if pct >= 50 else "#ff4f6a")
        st.markdown(f"""
        <div class="topic-row">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-weight:600;">{t}</span>
                <span style="font-family:'Space Mono',monospace;font-size:0.85rem;color:#6b6b88;">{sc:.0f}/{tot} pts · {pct}%</span>
            </div>
            <div class="prog-bar-bg">
                <div class="prog-bar-fill" style="width:{pct}%;background:{bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    weak_from_memory = get_weak_areas(st.session_state.session_id)
    if weak_from_memory:
        st.markdown(
            f'<div class="memory-banner">🧠 <b>Memory Updated</b> — '
            f'Stored weak areas for next session: <b>{", ".join(weak_from_memory)}</b></div>',
            unsafe_allow_html=True,
        )

    st.subheader("📋 Detailed Review")
    with st.expander("View all questions & answers", expanded=False):
        for a in answers:
            qn = a.get("index", 0) + 1
            ok = a.get("correct_flag", False)
            icon = "✅" if ok else "❌"
            st.markdown(f"**{icon} Q{qn}.** {a.get('question')}")
            st.caption(f"*{a.get('difficulty')} | {a.get('topic')} | {a.get('marks_earned')}/{a.get('marks_total')} pts*")
            ca, cb = st.columns(2)
            with ca:
                st.write("**Your answer:**")
                chosen = a.get("chosen", [])
                st.write("\n".join(f"• {x}" for x in chosen) if chosen else "*(Skipped)*")
            with cb:
                st.write("**Correct answer:**")
                st.write("\n".join(f"• {x}" for x in a.get("correct", [])))
            st.divider()

    st.subheader("🧠 AI Performance Report")
    if st.button("Generate AI Report ✨", type="primary", use_container_width=True):
        weak_t = get_weak_topics(answers)
        strengths = [t for t, qs in topic_scores.items()
                     if sum(q["marks_earned"] for q in qs) / max(1, sum(q["marks_total"] for q in qs)) >= 0.8]
        topic_perf_strings = [
            f"{t}: {round((sum(q['marks_earned'] for q in qs) / max(1, sum(q['marks_total'] for q in qs))) * 100)}%"
            for t, qs in topic_scores.items()
        ]
        prompt = build_feedback_prompt(
            percent=percent, earned=earned_marks, total=total_marks,
            num_correct=num_correct, total_q=total_q,
            topic_perf=topic_perf_strings, weak_topics=weak_t, strengths=strengths,
        )
        report_box = st.empty()
        buf = ""
        for token in ask_llm_stream(prompt):
            buf += token
            report_box.markdown(buf + "▌")
        report_box.markdown(buf)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 New Interview", use_container_width=True, type="primary"):
            reset_all()
            st.rerun()
    with c2:
        results_data = {
            "score_percent": percent,
            "performance": perf_label,
            "marks_earned": earned_marks,
            "marks_total": total_marks,
            "correct_answers": num_correct,
            "total_questions": total_q,
            "weak_topics": get_weak_topics(answers),
            "topic_performance": {
                t: {
                    "score": sum(q["marks_earned"] for q in qs),
                    "total": sum(q["marks_total"] for q in qs),
                    "percentage": round((sum(q["marks_earned"] for q in qs) / sum(q["marks_total"] for q in qs)) * 100, 1) if sum(q["marks_total"] for q in qs) else 0,
                } for t, qs in topic_scores.items()
            },
            "answers": answers,
        }
        st.download_button(
            "📥 Download Report (JSON)",
            data=json.dumps(results_data, indent=2),
            file_name=f"dronaai_report_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
