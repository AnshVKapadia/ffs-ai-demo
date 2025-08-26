# scholarship_bot.py
"""
Scholarship Finder (functions only)
- Built from your Streamlit app logic with minimal changes
- Uses gpt-4o-mini-search-preview (built-in web search)
- Exposes helpers to build messages, call the model, and post-clean expired items
"""

import os
import re
from datetime import datetime, UTC, date
from typing import List, Dict, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ====== Config ======
MODEL = "gpt-4o-mini-search-preview"
MAX_TURNS_TO_SEND = 6  # keep convo short to reduce token drift and cost


# ====== Helpers from your backend (unchanged where possible) ======
def today_iso() -> str:
    return datetime.now(UTC).date().isoformat()


def flag_expired_lines(output_text: str) -> Tuple[List[str], str]:
    """
    Split on blank lines, look for ISO date [YYYY-MM-DD] in the first line of each chunk.
    If the date is before today, flag and REMOVE that chunk.
    Returns (flagged_chunks, updated_text).

    Note: Slightly adjusted to avoid deleting from the list during iteration.
    """
    pattern = r"\[([0-9]{4}-[0-9]{2}-[0-9]{2})\]"
    today = date.today()
    flagged: List[str] = []

    chunks = [c.strip() for c in re.split(r'(?:\r?\n\s*){2,}', output_text.strip()) if c.strip()]
    kept: List[str] = []

    for scholarship in chunks:
        first_line = scholarship.split('\n', 1)[0]
        match = re.search(pattern, first_line)
        if match:
            try:
                deadline = date.fromisoformat(match.group(1))
                if deadline < today:
                    flagged.append(scholarship)
                    continue  # drop expired
            except Exception:
                pass
        kept.append(scholarship)

    updated_text = (
        "No still-open deadlines found. Try asking again or broadening your query."
        if len(kept) == 0 else
        "\n\n".join(kept)
    )
    return flagged, updated_text


def system_instructions() -> str:
    TODAY = today_iso()
    return f"""
    You are a research assistant that finds current scholarships on the public web.

    CONTEXT:
    - Today is {TODAY}. Do not list scholarships whose deadline is earlier than today.
    - Exception: If the official sponsor page explicitly states that applications reopen annually and the new date is pending, include it and clearly mark: "Next cycle; date TBA".

    GOALS:
    - Aggregators are allowed, but always try to find and prefer the OFFICIAL sponsor page.
    - Never invent awards. If amount or deadline is unclear on the official page, write: "Deadline unclear on official page".
    - Write in clear bullet points for humans (not JSON). Keep each bullet tight.

    WHEN SEARCHING:
    - Add the current or upcoming application year (e.g., “2025 scholarships”) to your search queries.
    - Prioritize pages that appear to have been updated recently or include deadlines clearly in the future.
    - Prefer official domains and reputable sources: site:.org, site:.edu, site:.gov, or sponsor-owned websites.
    - Avoid outdated aggregator lists unless the deadline shown is still valid or updated.
    - Know that a high school "freshman, sophomore, or junior" corresponds to grades 9-11 respectively and is NOT a high school senior.

    OUTPUT FORMAT:
    Start with one short sentence summarizing what you found.

    Then list 3–5 scholarships as bullets. Each bullet MUST include:
    • Name — Amount — Deadline: "December 31, 2025" [2025-12-31]
      (Both formats are required. Do not skip the quoted deadline or the ISO.)
      Link: <direct URL>   Source type: Official | Aggregator
      Cycle currently open? Yes/No. (IF THIS ANSWER IS NO, REMOVE THIS SCHOLARSHIP AND FIND ANOTHER ONE.)
      Why it fits: 1 short sentence (e.g., HS seniors, STEM, nationwide).
      Eligibility: brief bullets of key constraints if present (e.g., class year, GPA, major, region). If none stated, write "Not specified on page."
      Women-only? Yes/No.
      Last verified: {TODAY}

    RULES:
    - If the user mentions "female", "women", or similar, prioritize women-only awards at the top.
    - Otherwise, include the most relevant items; it's fine to mix general and women-only as appropriate to the prompt.
    - Do not list awards with already-passed deadlines unless the new cycle is explicitly open.
    - If you cannot find enough credible items, return fewer and state that you hit your browsing limit.
    - Avoid hyper-local or school-specific awards unless the prompt suggests a specific location or school.
    - Do not include paywalled or login-gated content.
    - Make sure to ONLY include scholarships targeted towards the students, not any other parts of their family (e.g. mothers, fathers, relatives).
    """.strip()


def user_prompt_wrap(user_text: str) -> str:
    return (
        "Find scholarships based on this prompt. "
        "Return 3–5 well-sourced items with links, per the format.\n\n"
        f"Today is {today_iso()}. UNDER ZERO CIRCUMSTANCES WILL YOU list scholarships whose deadline is earlier than today. "
        "If you cannot find any scholarships that are due after today, keep looking. "
        "If you cannot find any scholarships that do not contradict the user's specifications, keep looking.\n\n"
        "Reminder: include both the quoted deadline text and the ISO date like: \"August 31, 2025\" [2025-08-31].\n\n"
        f"Prompt: {user_text}"
    )


def build_api_messages(chat_history: List[Dict]) -> List[Dict]:
    """
    Convert our chat history into OpenAI Chat Completions format.
    We always prepend the current system instructions.
    Then include up to MAX_TURNS_TO_SEND most recent turns.
    Each item in chat_history is a dict: {"role": "user"/"assistant", "content": "..."}.
    """
    msgs: List[Dict] = [{"role": "system", "content": system_instructions()}]
    recent = chat_history[-MAX_TURNS_TO_SEND:]
    msgs.extend(recent)
    return msgs


# ====== Core call helpers ======
def create_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI()


def generate_scholarship_response(
    user_input: str,
    chat_history: Optional[List[Dict]] = None,
    keep_history: bool = True,
    client: Optional[OpenAI] = None,
) -> Dict:
    """
    One-shot call to the search-preview model and cleanup.

    Returns dict:
    {
      "raw_text": str,
      "clean_text": str,
      "flagged_chunks": List[str],
      "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int} | None,
      "updated_chat_history": List[Dict],
    }
    """
    if client is None:
        client = create_client()

    chat_history = chat_history or []

    # Build messages
    msgs = build_api_messages(chat_history)
    msgs.append({"role": "user", "content": user_prompt_wrap(user_input)})

    # Call the model (search-preview models only accept simple args)
    completion = client.chat.completions.create(
        model=MODEL,
        messages=msgs,
    )

    choice = completion.choices[0] if completion.choices else None
    raw_text = choice.message.content.strip() if choice and choice.message and choice.message.content else "[No content returned]"

    # Post-filter expired lines
    flagged, clean_text = flag_expired_lines(raw_text)

    # Update history with cleaned assistant reply
    updated_chat_history = chat_history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": clean_text},
    ]

    usage = None
    if hasattr(completion, "usage") and completion.usage:
        usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }

    return {
        "raw_text": raw_text,
        "clean_text": clean_text,
        "flagged_chunks": flagged,
        "usage": usage,
        "updated_chat_history": updated_chat_history,
    }


# ====== Optional CLI smoke test ======
if __name__ == "__main__":
    import sys
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "women STEM scholarships for high school seniors"
    try:
        res = generate_scholarship_response(prompt, chat_history=[], keep_history=False)
        print("\n--- RAW ---\n")
        print(res["raw_text"])
        print("\n--- CLEAN ---\n")
        print(res["clean_text"])
        if res["flagged_chunks"]:
            print(f"\n(removed {len(res['flagged_chunks'])} expired item(s))")
        if res["usage"]:
            u = res["usage"]
            print(f"\nTokens — prompt: {u['prompt_tokens']} • completion: {u['completion_tokens']} • total: {u['total_tokens']}")
    except Exception as e:
        print(f"Error: {e}")
