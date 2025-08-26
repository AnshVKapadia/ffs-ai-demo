# default_bot.py
"""
Default Chatbot (Tutor + Counselor in one)
- Minimal, function-only module (no Streamlit)
- Cheap default model: gpt-4o-mini
- Short history memory, structured prompts, concise answers
"""

import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ========= Config =========
MODEL_DEFAULT = "gpt-4o-mini"  # Cheap, general-purpose
MAX_TURNS_TO_SEND = 6  # keep convo short to reduce drift/cost


# ========= Helpers =========
def trim_history(msgs: List[Dict], n: int = MAX_TURNS_TO_SEND) -> List[Dict]:
    return msgs[-n:]


def system_instructions() -> str:
    """
    One unified assistant for AP/STEM tutoring and academic counseling.
    Concise, structured, with light step-by-step for math/science.
    """
    return """
You are a friendly, precise academic assistant for high-school and college students.

GENERAL STYLE
- Be concise, structured, and supportive. Prefer bullets and numbered steps.
- If the user asks a question that lacks key details, ask 1–2 focused clarifying questions first.
- If you can reasonably infer details, provide a draft answer and clearly note assumptions.

TUTORING (AP/STEM)
- Subjects: Algebra, Geometry, Precalculus, Calculus (AB/BC), Physics (1/2/C), Chemistry, Biology, AP CS A, intro programming.
- When solving, show core steps succinctly; include essential formulas in $...$.
- Provide a short "Why this works" note and 1–3 practice problems (with brief answers) when helpful.

ACADEMIC COUNSELING
- Topics: course selection, study plans, time management, test prep (SAT/ACT/AP), extracurricular strategy, college application milestones.
- Start by confirming the student’s grade, goals, timeline, and constraints.
- Provide an actionable plan: key milestones, a simple weekly template, and 2–3 reputable resources.

SAFETY / BOUNDARIES
- Do not provide disallowed content, medical or legal advice.
- If you’re uncertain, say so briefly and suggest how to verify or proceed.
""".strip()


def user_prompt_wrap(user_text: str) -> str:
    """
    Keep it simple: answer directly when possible; otherwise ask 1–2 clarifying Qs first.
    """
    return (
        "Respond as a helpful academic assistant. Keep answers structured and concise. "
        "If the question is under-specified, ask 1–2 clarifying questions before proceeding; "
        "otherwise, answer directly. For math/science, show essential steps and key formulas in $...$; "
        "for planning/counseling, propose an actionable plan with milestones and 2–3 reputable resources.\n\n"
        f"User: {user_text}"
    )


def build_api_messages(chat_history: List[Dict], current_user_text: str, keep_history: bool = True) -> List[Dict]:
    """
    Convert your chat history into OpenAI Chat Completions format.
    Each history item: {"role": "user"/"assistant", "content": "..."}.
    """
    msgs: List[Dict] = [{"role": "system", "content": system_instructions()}]
    if keep_history and chat_history:
        msgs.extend(trim_history(chat_history))
    msgs.append({"role": "user", "content": user_prompt_wrap(current_user_text)})
    return msgs


# ========= Core call =========
def create_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI()


def generate_default_response(
    user_input: str,
    chat_history: Optional[List[Dict]] = None,
    keep_history: bool = True,
    client: Optional[OpenAI] = None,
    model: str = MODEL_DEFAULT,
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
) -> Dict:
    """
    Single-call generation for the default (tutor+counselor) chatbot.

    Returns:
      {
        "text": str,
        "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int} | None,
        "updated_chat_history": List[Dict],
        "model_used": str,
      }
    """
    if client is None:
        client = create_client()

    chat_history = chat_history or []
    messages = build_api_messages(chat_history, user_input, keep_history=keep_history)

    kwargs = dict(model=model, messages=messages, temperature=temperature)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens  # optional cap

    completion = client.chat.completions.create(**kwargs)

    choice = completion.choices[0] if completion.choices else None
    text = choice.message.content.strip() if choice and choice.message and choice.message.content else "[No content returned]"

    updated_chat_history = chat_history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": text},
    ]

    usage = None
    if hasattr(completion, "usage") and completion.usage:
        usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }

    return {
        "text": text,
        "usage": usage,
        "updated_chat_history": updated_chat_history,
        "model_used": model,
    }


# ========= Optional CLI smoke test =========
if __name__ == "__main__":
    import sys
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Help me plan a 6-week AP Calc AB review."
    try:
        res = generate_default_response(prompt, chat_history=[], keep_history=False)
        print("\n--- ASSISTANT ---\n")
        print(res["text"])
        if res["usage"]:
            u = res["usage"]
            print(f"\nTokens — prompt: {u['prompt_tokens']} • completion: {u['completion_tokens']} • total: {u['total_tokens']}")
        #print(f"\nModel used: {res['model_used']}")
    except Exception as e:
        print(f"Error: {e}")