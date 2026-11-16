"""Dates (DESIGN.md write path step 3; build step 3): time phrases -> time_mention rows.

Code finds candidate time phrases (TIME_RE); for each, one Jev request names its date parts with Choice
questions (the booster's W2 contract in the booster plan, so the booster can take this decision over
later); code resolves the parts against the turn's own date (in the client's offset) and stores the result.
Confidence is the least certain part used; below REVIEW_BELOW the phrase is kept with no resolved date
(DESIGN.md: below 0.60, keep the phrase with no resolved date). Jev decides, code computes.
"""
from __future__ import annotations

import calendar
import contextvars
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta

log = logging.getLogger("brain.dates")

REVIEW_BELOW = 0.60
MAX_PHRASES_PER_TURN = 12
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

_WD = r"(?:mon|tues?|wed(?:nes)?|thu(?:rs?)?|fri|sat(?:ur)?|sun)(?:day)?"
_MO = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
       r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_NUM = r"(?:\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a couple of|a few|several)"
TIME_RE = re.compile(
    r"\b(?:"
    r"(?:the\s+)?day\s+before\s+yesterday|(?:the\s+)?day\s+after\s+tomorrow|yesterday|today|tonight|tomorrow|"
    r"this\s+(?:morning|afternoon|evening)|last\s+night|"
    rf"{_NUM}\s+(?:days?|weeks?|months?|years?)\s+(?:ago|earlier|before|later|from\s+now)|"
    rf"in\s+{_NUM}\s+(?:days?|weeks?|months?)|"
    rf"(?:last|this|next|past|coming)\s+(?:{_WD}|week(?:end)?|month|year|spring|summer|fall|autumn|winter)|"
    rf"(?:on\s+)?{_WD}|"
    rf"{_MO}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MO}(?:,?\s+\d{{4}})?|"
    rf"(?:in|during|since|until|by|early|mid|late)[-\s]+{_MO}(?:\s+\d{{4}})?|"
    r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?"
    r")\b", re.I)

_RANGE = lambda lo, hi: [str(i) for i in range(lo, hi + 1)]  # noqa: E731


def _choice(instructions: str, options: dict) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": options}


DATE_QUESTIONS = {
    "mode": _choice(
        "How does `phrase` (in `sentence`, written on `writing_date`) point to a time?",
        {"relative_day": "A number of days from the writing date: today, yesterday, tomorrow, 3 days ago, in 2 days, last night.",
         "relative_week": "A week relative to the writing date: last week, this week, next week, two weeks ago, last weekend.",
         "relative_weekday": "A named weekday near the writing date: last Friday, on Monday, this Sunday, next Tuesday.",
         "relative_month": "A month relative to the writing date: last month, this month, three months ago, next month.",
         "absolute": "A calendar date or month by name or number: May 3rd, 3 March 2023, in March, 2023-05-20, 5/3.",
         "none": "Not a point in time: a duration, a frequency, a vague time ('the other day'), a season, or not a date at all."}),
    "offset_days": _choice("For a relative day, how many days from the writing date? Negative is the past.",
                           {**{d: None for d in _RANGE(-45, 45)}, "none": "Not a relative day."}),
    "week_offset": _choice("For a relative week, how many weeks from the writing date's week? Negative is the past.",
                           {**{d: None for d in _RANGE(-12, 12)}, "none": "Not a relative week."}),
    "month_offset": _choice("For a relative month, how many months from the writing date's month? Negative is the past.",
                            {**{d: None for d in _RANGE(-24, 24)}, "none": "Not a relative month."}),
    "weekday": _choice("Which weekday does `phrase` name?", {**{w: None for w in WEEKDAYS}, "none": "No weekday."}),
    "weekday_direction": _choice(
        "For a named weekday, which one is meant, judging from the sentence (tense, 'last', 'next', 'this')?",
        {"past": "The most recent such day before the writing date (last Friday; 'on Friday' about something already done).",
         "upcoming": "The next such day after the writing date (next Tuesday; 'on Monday' about something planned).",
         "this_week": "That day in the writing date's own week (this Sunday)."}),
    "month": _choice("For a calendar date, which month?", {**{m: None for m in MONTHS}, "none": "No month named."}),
    "day": _choice("For a calendar date, which day of the month?", {**{d: None for d in _RANGE(1, 31)},
                                                                   "none": "No day of the month named."}),
    "year": _choice("For a calendar date, which year is stated?", {**{y: None for y in _RANGE(1990, 2040)},
                                                                   "none": "No year stated."}),
}
# exp-date-direction-v1: "which year is stated?" comes back as the writing year for a month named with no year
# ("a workshop I attended in November", written in February, was dated nine months ahead). A year that is not
# written is not a stated year. With the flag on, one more Choice in the same request says, from the wording
# alone, whether the sentence tells of something that happened or of something planned, and code picks the year
# from that in the two cases where the year is really open: the date falls more than a month after the writing
# date (happened: last year; planned: this year), or more than a month before it in the writing year (planned:
# next year). A date within a month of the writing date either way (a clock a day off must not move a plan by a
# year), an unsure answer, and a written year are left as built. Off until its gate is met
# (the experiment log; v1 of the rule also moved dates up to a month ahead and trusted any answer).
DATE_DIRECTION = os.environ.get("BRAIN_DATE_DIRECTION", "0") == "1"
DATE_DIRECTION_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar("date_direction", default=None)
DIRECTION_QUESTION = _choice(
    "By its verb tense and wording alone, does `sentence` tell of what took place at `phrase` as something that "
    "has happened, or as something planned or expected? Do not compare the date with `writing_date`.",
    {"past": "It has happened: past tense or a finished state (a workshop I attended in November; I read it in "
             "December; I got it on March 10th; it expired in December; I submitted it on August 10th).",
     "upcoming": "It is planned, scheduled or expected: future or intention (we fly in November; I'm planning a "
                 "party on December 22nd; my exam is on March 3rd; I'd like to attend on March 7th).",
     "unclear": "The wording does not say which, or `phrase` is not a calendar date or month."})
DIRECTION_AT = 0.85     # on the 600 sentences the rule was written on, its two wrong answers were at 0.77 and 0.79
_YEAR_WRITTEN = re.compile(r"\b(?:19|20)\d\d\b|\d{1,2}/\d{1,2}/\d{2,4}")
_YEAR_FOLLOWS = re.compile(r"^\s*(?:\d{1,2}(?:st|nd|rd|th)?)?\s*(?:,|of)?\s*'?(?:19|20)\d\d\b")


def year_written(phrase: str, sentence: str | None = None) -> bool:
    """A year is written in the phrase, or right after it in its sentence ("in February of 2008", "by September
    30, 2024")."""
    if _YEAR_WRITTEN.search(phrase):
        return True
    i = (sentence or "").lower().find(phrase.lower())
    return i >= 0 and bool(_YEAR_FOLLOWS.match(sentence[i + len(phrase):]))


# exp-date-counted-v1: "two weeks ago" has its number written, yet it came back with no date: Jev names the mode
# with certainty and picks the right offset, but among 25 offsets its probability for the pick stays near 0.5,
# under REVIEW_BELOW (twelve probe sentences on 2026-11-13: every "N weeks ago" with N of two or more was left
# without a date; so was "a couple of weeks ago" in the production live check). With the flag on, when the
# phrase is a written count of days, weeks or months before or after the writing date ("3 days ago", "a couple
# of weeks ago", "a week from now") and Jev's mode agrees on the unit, code reads the number from the phrase
# and the mode's own probability is the confidence. "a few" and "several" are not numbers and stay as built; so
# do "earlier", "before" and "later", which count from another event than the writing date. So does "in N
# weeks" (v1 read it and failed its gate on it: the words are also a duration, "a plan in 90 days", "haven't
# heard in four days", and the phrase finder cuts "in two weeks" out of "broke them in two weeks ago").
DATE_COUNTED = os.environ.get("BRAIN_DATE_COUNTED", "0") == "1"
DATE_COUNTED_OVERRIDE: contextvars.ContextVar = contextvars.ContextVar("date_counted", default=None)
_COUNT_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2}
_COUNTED = re.compile(rf"^({_NUM})\s+(day|week|month)s?\s+(ago|from\s+now)$", re.I)


def counted_offset(phrase: str) -> tuple[str, int] | None:
    """("week", -2) for "two weeks ago" or "a couple of weeks ago", ("month", 2) for "two months from now"; None
    for anything else: a count that is not a number ("a few"), a count from another event ("two days later"),
    or "in two weeks"."""
    m = _COUNTED.match(" ".join(phrase.split()))
    if not m:
        return None
    num = m.group(1).lower()
    n = int(num) if num.isdigit() else _COUNT_WORDS.get(num)
    if not n:
        return None
    return m.group(2).lower(), -n if m.group(3).lower() == "ago" else n


def _counted() -> bool:
    override = DATE_COUNTED_OVERRIDE.get()
    return DATE_COUNTED if override is None else bool(override)


def _direction() -> bool:
    override = DATE_DIRECTION_OVERRIDE.get()
    return DATE_DIRECTION if override is None else bool(override)


def date_questions(direction: bool | None = None) -> dict:
    on = _direction() if direction is None else direction
    return {**DATE_QUESTIONS, "direction": DIRECTION_QUESTION} if on else DATE_QUESTIONS


_USED = {"relative_day": ["offset_days"], "relative_week": ["week_offset"],
         "relative_weekday": ["weekday", "weekday_direction"], "relative_month": ["month_offset"],
         "absolute": ["month", "day", "year"], "none": []}


@dataclass(frozen=True)
class Mention:
    span: str
    date_from: date | None
    date_to: date | None
    granularity: str | None      # day, week, month
    confidence: float


def find_phrases(text: str) -> list[str]:
    out: list[str] = []
    for m in TIME_RE.finditer(text):
        p = m.group(0).strip()
        if p.lower() not in (x.lower() for x in out):
            out.append(p)
    return out[:MAX_PHRASES_PER_TURN]


def sentence_of(text: str, phrase: str, width: int = 300) -> str:
    i = text.lower().find(phrase.lower())
    lo = max(0, text.rfind(".", 0, max(0, i)) + 1) if i >= 0 else 0
    return text[lo:lo + width].strip()


def _month_shift(d: date, n: int) -> tuple[date, date]:
    y, m = divmod(d.month - 1 + n, 12)
    first = date(d.year + y, m + 1, 1)
    return first, first.replace(day=calendar.monthrange(first.year, first.month)[1])


def _year_by_direction(direction: tuple[str, float], anchor: date, first, last) -> int | None:
    """The year of a date with no year written, where the direction settles it; None leaves it as built.
    `first(year)` and `last(year)` give the date's first and last day in that year."""
    way, p = direction
    if p < DIRECTION_AT:
        return None
    if last(anchor.year) < anchor - timedelta(days=31):              # more than a month back in the writing year
        return anchor.year + 1 if way == "upcoming" else None
    if first(anchor.year) > anchor + timedelta(days=31):             # more than a month ahead
        return anchor.year - 1 if way == "past" else anchor.year if way == "upcoming" else None
    return None


def resolve(parts: dict[str, tuple[str, float]], anchor: date, phrase: str | None = None,
            sentence: str | None = None) -> Mention | None:
    """parts: question name -> (choice, probability). Returns the resolved dates (span filled by the caller).
    With `phrase` and a "direction" part (DATE_DIRECTION), a year that is not written is taken from the
    direction, where the direction settles it, and not from the year Choice."""
    mode, p_mode = parts["mode"]
    count = counted_offset(phrase) if phrase is not None and _counted() else None
    if count and mode == f"relative_{count[0]}" and p_mode >= REVIEW_BELOW:
        unit, n = count
        if unit == "day":
            d = anchor + timedelta(days=n)
            return Mention("", d, d, "day", p_mode)
        if unit == "week":
            monday = anchor - timedelta(days=anchor.weekday()) + timedelta(weeks=n)
            return Mention("", monday, monday + timedelta(days=6), "week", p_mode)
        lo, hi = _month_shift(anchor, n)
        return Mention("", lo, hi, "month", p_mode)
    used = _USED.get(mode, [])
    conf = min([p_mode] + [parts[q][1] for q in used])
    pick = {q: parts[q][0] for q in used}
    if mode == "none" or any(v == "none" for k, v in pick.items() if k not in ("year", "day")):
        return Mention("", None, None, None, conf)
    try:
        if mode == "relative_day":
            d = anchor + timedelta(days=int(pick["offset_days"]))
            return Mention("", d, d, "day", conf)
        if mode == "relative_week":
            monday = anchor - timedelta(days=anchor.weekday()) + timedelta(weeks=int(pick["week_offset"]))
            return Mention("", monday, monday + timedelta(days=6), "week", conf)
        if mode == "relative_month":
            lo, hi = _month_shift(anchor, int(pick["month_offset"]))
            return Mention("", lo, hi, "month", conf)
        if mode == "relative_weekday":
            w = WEEKDAYS.index(pick["weekday"])
            if pick["weekday_direction"] == "past":
                d = anchor - timedelta(days=((anchor.weekday() - w) % 7) or 7)
            elif pick["weekday_direction"] == "upcoming":
                d = anchor + timedelta(days=((w - anchor.weekday()) % 7) or 7)
            else:
                d = anchor - timedelta(days=anchor.weekday()) + timedelta(days=w)
            return Mention("", d, d, "day", conf)
        if mode == "absolute":
            m = MONTHS.index(pick["month"]) + 1
            if "direction" in parts and phrase is not None and not year_written(phrase, sentence):
                way = parts["direction"]
                if pick["day"] == "none":
                    y = _year_by_direction(way, anchor, lambda yr: date(yr, m, 1), lambda yr: _month_shift(date(yr, m, 1), 0)[1])
                else:
                    day = int(pick["day"])
                    y = _year_by_direction(way, anchor, lambda yr: date(yr, m, day), lambda yr: date(yr, m, day))
                if y is not None:
                    if pick["day"] == "none":
                        lo, hi = _month_shift(date(y, m, 1), 0)
                        return Mention("", lo, hi, "month", conf)
                    d = date(y, m, int(pick["day"]))
                    return Mention("", d, d, "day", conf)
            year = int(pick["year"]) if pick["year"] != "none" else anchor.year
            if pick["day"] == "none":
                lo, hi = _month_shift(date(year, m, 1), 0)
                if pick["year"] == "none" and lo > anchor + timedelta(days=31):   # memory: past-biased
                    lo, hi = _month_shift(date(year - 1, m, 1), 0)
                return Mention("", lo, hi, "month", conf)
            d = date(year, m, int(pick["day"]))
            if pick["year"] == "none" and d > anchor + timedelta(days=31):
                d = d.replace(year=year - 1)
            return Mention("", d, d, "day", conf)
    except (ValueError, KeyError):      # impossible dates (Feb 30), unknown options
        return Mention("", None, None, None, 0.0)
    return Mention("", None, None, None, conf)


def weekend_of(phrase: str, m: Mention) -> Mention:
    """exp-when-locomo-v1: "last weekend" resolved to its whole week (Monday to Sunday); the weekend is that
    week's Saturday and Sunday. Stored with granularity "range"."""
    if m.granularity == "week" and m.date_from and re.search(r"\bweekend\b", phrase, re.I):
        return Mention(m.span, m.date_from + timedelta(days=5), m.date_from + timedelta(days=6), "range", m.confidence)
    return m


_CACHE: dict[tuple, dict] = {}     # (phrase, sentence, writing date) -> parts; identical input,
_CACHE_MAX = 50_000                                # identical answer (shared sessions, re-sent history)


def mentions_for(providers, turns: list[tuple[int, str, date]]) -> list[tuple[int, Mention]]:
    """turns: (episode_id, text, anchor date in the client's own offset). One Jev request per new phrase, in
    parallel. Raises the provider's error on failure (the caller decides how to degrade)."""
    reqs, keys, todo = [], [], []
    asked = date_questions()
    for eid, text, anchor in turns:
        for phrase in find_phrases(text):
            state = {"phrase": phrase, "sentence": sentence_of(text, phrase),
                     "writing_date": f"{anchor.isoformat()} ({WEEKDAYS[anchor.weekday()]})"}
            key = (phrase, state["sentence"], state["writing_date"], "direction" in asked)
            keys.append((eid, phrase, anchor, key))
            if key not in _CACHE and key not in todo:
                todo.append(key)
                reqs.append((state, asked))
    if not keys:
        return []
    for key, r in zip(todo, providers.jev.ask_many(reqs) if reqs else []):
        parts = {}
        for q in asked:
            choice, probs = r.choice(q)
            parts[q] = (choice, float(probs.get(choice, 0.0)))
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[key] = parts
    out = []
    for eid, phrase, anchor, key in keys:
        parts = _CACHE.get(key)
        if parts is None:
            continue
        m = resolve(parts, anchor, phrase, key[1])
        if m is None:
            continue
        m = weekend_of(phrase, m)
        if m.confidence < REVIEW_BELOW:
            m = Mention(phrase, None, None, None, m.confidence)
        out.append((eid, Mention(phrase, m.date_from, m.date_to, m.granularity, round(m.confidence, 4))))
    return out


_QDATE = re.compile(r"(\d{4})[/-](\d{2})[/-](\d{2})")


def parse_day(s: str | None) -> date | None:
    m = _QDATE.search(s or "")
    try:
        return date(*map(int, m.groups())) if m else None
    except ValueError:
        return None


def _span(d: date, q: date) -> str:
    n = (q - d).days
    if n == 0:
        return "the question date"
    w, r = divmod(abs(n), 7)
    rel = f"{abs(n)} days ({w} weeks {r} days)" if w else f"{abs(n)} days"
    return f"{rel} {'before' if n > 0 else 'after'} the question date"


def block(conn, items: list[dict], question_date: str | None, max_dates: int = 6) -> str:
    """The when route's computed dates (DESIGN.md read path: code does the date arithmetic): each excerpt's
    session date and resolved time phrases, each date's distance from the question date, and the days between
    every pair of the first `max_dates` distinct dates. items: excerpts in rerank order (episode_id, ts)."""
    q = parse_day(question_date)
    ids = [it["episode_id"] for it in items]
    rows = conn.execute("""select episode_id, span, date_from, date_to, granularity from time_mention
                           where episode_id = any(%s) and date_from is not null order by id""", (ids,)).fetchall()
    by_ep: dict[int, list] = {}
    for eid, span, lo, hi, gran in rows:
        by_ep.setdefault(eid, []).append((span, lo, hi, gran))
    lines, points = [], []
    for it in items:
        sd = parse_day(it["ts"])
        ments = by_ep.get(it["episode_id"], [])
        desc = "; ".join(f"'{s}' = {lo.isoformat()}" if g == "day" else f"'{s}' = {lo.isoformat()} to {hi.isoformat()}"
                         for s, lo, hi, g in ments)
        lines.append(f"- [{sd.isoformat() if sd else '?'}] {it['text'][13:73].strip()}..." + (f" -> {desc}" if desc else ""))
        for d in [lo for _, lo, _, g in ments if g == "day"] + ([sd] if sd else []):
            if d not in points:
                points.append(d)
    points = points[:max_dates]
    out = ["Dates computed by code from these excerpts (use them instead of counting days yourself):", *lines]
    if q:
        out.append("Distance from the question date:")
        out += [f"- {d.isoformat()} ({WEEKDAYS[d.weekday()]}): {_span(d, q)}" for d in sorted(points)]
    pts = sorted(points)
    if len(pts) > 1:
        out.append("Days between dates:")
        out += [f"- {a.isoformat()} to {b.isoformat()}: {(b - a).days} days" for i, a in enumerate(pts) for b in pts[i + 1:]]
    return "\n".join(out)


def annotate(conn, items: list[tuple[int, str, str]]) -> list[str]:
    """exp-when-locomo-v1 (core.DATE_NOTES): each text with the date code resolved for a time phrase written
    beside that phrase, "Last weekend [2023-07-15 to 2023-07-16]". items: (episode_id, text, ts). The reader keeps
    the turn's words and gets the one thing code computes; no arithmetic between dates is added. A phrase that
    resolves to the turn's own day (today, tonight) or already states its date is left alone."""
    ids = list({eid for eid, _, _ in items})
    if not ids:
        return [t for _, t, _ in items]
    rows = conn.execute("""select episode_id, span, date_from, date_to from time_mention
                           where episode_id = any(%s) and date_from is not null order by id""", (ids,)).fetchall()
    by_ep: dict[int, list] = {}
    for eid, span, lo, hi in rows:
        by_ep.setdefault(eid, []).append((span, lo, hi))
    out = []
    for eid, text, ts in items:
        own = parse_day(ts)
        done: set[str] = set()
        for span, lo, hi in by_ep.get(eid, []):
            note = lo.isoformat() if lo == hi else f"{lo.isoformat()} to {hi.isoformat()}"
            if span.lower() in done or (lo == hi == own) or lo.isoformat() in span:
                continue
            m = re.search(rf"(?<!\w){re.escape(span)}(?!\w)", text, re.I)
            if m:
                text = f"{text[:m.end()]} [{note}]{text[m.end():]}"
                done.add(span.lower())
        out.append(text)
    return out


def store(conn, mentions: list[tuple[int, Mention]]) -> int:
    if not mentions:
        return 0
    with conn.cursor() as cur:
        cur.executemany("""insert into time_mention (episode_id, span, date_from, date_to, granularity, confidence)
                           values (%s, %s, %s, %s, %s, %s)""",
                        [(eid, m.span, m.date_from, m.date_to, m.granularity, m.confidence) for eid, m in mentions])
    return len(mentions)
