"""Answer prompts (chosen from question text only) and judge prompts.

LongMemEval judge templates are the official ones from xiaowu0162/LongMemEval
src/evaluation/evaluate_qa.py (MIT license), used here with Haiku instead of GPT-4o.
The LoCoMo judge is the same prompt as the LoCoMo ledger (eval_cookbook_haiku.py JS),
so LoCoMo numbers stay comparable with the park.
"""
from __future__ import annotations

import re

ANSWER_DEFAULT = (
    "You answer questions about a user's long chat history using only the excerpts given. "
    "Each excerpt starts with its session date in brackets. Today's date is given with the question. "
    "Answer directly and briefly. When the question asks for a count, a duration or a date, work it out from "
    "the excerpts and today's date, and state the result. If facts changed over time, give the latest one. "
    "If the excerpts do not contain the answer, reply exactly: UNKNOWN - followed by a few words on what is missing."
)

ANSWER_PREFERENCE = (
    "You are the user's assistant and know them only from the chat excerpts given. Each excerpt starts with its "
    "session date in brackets. Answer the request helpfully in 3 to 6 sentences, tailored to the user's own "
    "preferences, situation and past choices as shown in the excerpts. Mention the specific personal details "
    "you are using. If the excerpts say nothing relevant about the user, give a short general answer."
)

ANSWER_LOCOMO = (
    "You answer questions about a long chat history between two people using only the excerpts given. "
    "Each excerpt starts with its session date in brackets. Reply with only the fact asked, as briefly as "
    "possible (a few words). For when-questions, use the session date to resolve relative times "
    "(e.g. 'yesterday') into an absolute date. If the excerpts truly lack the answer, give your best guess briefly."
)


def build_answer_user(question: str, question_date: str, excerpts: list[str]) -> str:
    body = "\n".join(excerpts) if excerpts else "(no excerpts)"
    return f"Excerpts:\n{body}\n\nToday's date: {question_date}\nQuestion: {question}\nAnswer:"


# ---------- judges ----------

_LME_STD = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
            "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the correct answer, you should also "
            "answer yes. If the response only contains a subset of the information required by the answer, "
            "answer no. ")
_LME_TAIL = ("\n\nQuestion: {q}\n\nCorrect Answer: {a}\n\nModel Response: {r}\n\n"
             "Is the model response correct? Answer yes or no only.")


def lme_judge_prompt(qtype: str, question: str, gold: str, response: str, abstention: bool) -> str:
    if abstention:
        return ("I will give you an unanswerable question, an explanation, and a response from a model. Please "
                "answer yes if the model correctly identifies the question as unanswerable. The model could say "
                "that the information is incomplete, or some other information is given but the asked information "
                "is not.\n\nQuestion: {q}\n\nExplanation: {a}\n\nModel Response: {r}\n\nDoes the model correctly "
                "identify the question as unanswerable? Answer yes or no only.").format(q=question, a=gold, r=response)
    if qtype in ("single-session-user", "single-session-assistant", "multi-session"):
        return (_LME_STD + _LME_TAIL).format(q=question, a=gold, r=response)
    if qtype == "temporal-reasoning":
        return (_LME_STD + "In addition, do not penalize off-by-one errors for the number of days. If the question "
                "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
                "predicting 19 days when the answer is 18), the model's response is still correct. "
                + _LME_TAIL).format(q=question, a=gold, r=response)
    if qtype == "knowledge-update":
        return ("I will give you a question, a correct answer, and a response from a model. Please answer yes if "
                "the response contains the correct answer. Otherwise, answer no. If the response contains some "
                "previous information along with an updated answer, the response should be considered as correct "
                "as long as the updated answer is the required answer." + _LME_TAIL).format(q=question, a=gold, r=response)
    if qtype == "single-session-preference":
        return ("I will give you a question, a rubric for desired personalized response, and a response from a "
                "model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                "The model does not need to reflect all the points in the rubric. The response is correct as long "
                "as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {q}\n\n"
                "Rubric: {a}\n\nModel Response: {r}\n\nIs the model response correct? Answer yes or no only."
                ).format(q=question, a=gold, r=response)
    raise ValueError(f"unknown LongMemEval type {qtype}")


LOCOMO_JUDGE_SYSTEM = (
    "You grade answers. Given a question, gold answer, and predicted answer, decide if the "
    "prediction is correct: it matches the gold answer's meaning (paraphrase, or same date/time "
    "referent in different format, or gold plus harmless extra detail is fine). Reply exactly "
    "'CORRECT: yes' or 'CORRECT: no'."
)


def locomo_judge_user(question: str, gold: str, pred: str) -> str:
    return f"Question: {question}\nGold answer: {gold}\nPredicted answer: {pred}"


def parse_yes_no(raw: str | None) -> bool | None:
    """Strict verdict parser. Accepts 'yes'/'no' optionally prefixed by 'CORRECT:'.
    Anything else returns None (a parse failure, never counted as correct)."""
    if raw is None:
        return None
    s = raw.strip().lower()
    s = re.sub(r"^\**\s*correct\s*:\s*", "", s)
    m = re.match(r"^\**\s*(yes|no)\b", s)
    if not m:
        return None
    return m.group(1) == "yes"
