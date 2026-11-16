"""exp-ledger-set-v1: a question set about facts that change, built to ask what the ledger is for.

    python scripts/ledger_set.py plan                  # the scripts only (no model, no cost): prints the counts
    python scripts/ledger_set.py write [--only N] [--cap 5]    # a model writes the sessions; saves the set

LoCoMo and LongMemEval rarely ask about a fact that changed more than once, was corrected, ended, or was said
two ways. Here code scripts a timeline of such facts over 16 dated sessions in five domains (a friendship, a
software project, a contract negotiation, a clinic's follow-up calls, a shop), so every gold answer is computed,
not judged into being. A model (scripts/readers.py, Sonnet) then writes each session as a natural chat that
states each scripted point once; code checks every point is there with its literal value and that no tracked
value is said anywhere else, and asks again if not. Nothing of the brain runs here.

Cases (the question's `category`):
  chain          a value changes three or four times; asked what it is now
  as-of          the same chain; asked what it was on a date inside an earlier value's time
  correction     a value is stated, then the speaker says that was a mistake and gives the right one
  contradiction  the two speakers each state a different value and nobody settles it
  ended          something starts and is later ended; asked whether it holds now
  ongoing        the control for `ended`: something starts and is never ended (so "no" to every such question is wrong)
  buried         a value is stated plainly, then changed once in passing, in a turn about something else
  count          items are added and removed over the sessions; asked how many there are now
  stable         a control: said once, never changed

The set is written in LoCoMo's layout (evalsets/ledger_set_v1_<split>.json), so `locomo_eval.py --data` reads
it. Two conversations per domain: one in `dev`, one in `held` (not to be read until a design is fixed).
"""
from __future__ import annotations

import json
import random
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
OUT = SERVICE / "evalsets"
SESSIONS = 16
CASES = ("chain", "as-of", "correction", "contradiction", "ended", "ongoing", "buried", "count", "stable")

# A value track: (key, the fact as a noun phrase, question now, question as of {d}, values, both speakers know it)
# An ended track: (key, what holds, how it ends, question). A count: (key, add, remove, question, items).
# A stable fact: (key, statement, question, answer). {A} is the first speaker, {B} the second.
DOMAINS = {
    "friends": {
        "about": "{A} and {B} are old friends catching up by chat every couple of weeks.",
        "values": [
            ("employer", "the company {A} works for", "Which company does {A} work for now?", "Which company did {A} work for on {d}?",
             ["Halvorsen Freight", "Brightwell Labs", "Corvid Analytics", "Mossgate Studio", "Tallow & Pine"], False),
            ("city", "the city {A} lives in", "Which city does {A} live in now?", "Which city did {A} live in on {d}?",
             ["Leeds", "Tacoma", "Graz", "Dunedin", "Aarhus"], False),
            ("car", "the car {A} drives", "What car does {A} drive now?", "What car did {A} drive on {d}?",
             ["Skoda Octavia", "Mazda 3", "Volvo V60", "Kia Niro", "Honda Jazz"], False),
            ("phone_plan", "what {A} pays per month for their phone plan", "How much does {A} pay per month for their phone plan now?",
             "How much did {A} pay per month for their phone plan on {d}?", ["$38", "$52", "$27", "$44", "$61"], False),
            ("book_club", "the bookshop where the book club {A} and {B} both belong to meets", "Where does the book club meet now?",
             "Where did the book club meet on {d}?", ["Pilcrow Books", "The Dog-Eared Page", "Quire & Folio", "Marginalia", "Saltmarsh Books"], True),
            ("dentist", "the dentist {A} goes to", "Which dentist does {A} go to now?", "Which dentist did {A} go to on {d}?",
             ["Dr. Okafor", "Dr. Lindqvist", "Dr. Marchetti", "Dr. Yoon", "Dr. Abernathy"], False),
        ],
        "ended": [("gym", "{A} has a membership at the Ironbridge gym", "{A} cancelled the Ironbridge gym membership",
                   "Does {A} currently have a membership at the Ironbridge gym?"),
                  ("streaming", "{A} subscribes to the Lumen streaming service", "{A} cancelled the Lumen subscription",
                   "Is {A} currently subscribed to the Lumen streaming service?")],
        "count": ("houseplants", "{A} got a new houseplant: {v}", "{A} no longer has {v} (it died or was given away)",
                  "How many houseplants does {A} have now?",
                  ["the monstera", "the snake plant", "the fiddle-leaf fig", "the pothos", "the jade plant", "the peace lily", "the rubber plant"]),
        "stable": [("sister", "{A}'s sister is called Ingrid", "What is {A}'s sister called?", "Ingrid"),
                   ("allergy", "{A} is allergic to shellfish", "What food is {A} allergic to?", "shellfish")],
        "filler": ["a film one of them watched", "the weather at the weekend", "a recipe that went wrong", "a mutual friend's wedding",
                   "a hiking trail", "a noisy neighbour", "a podcast", "holiday ideas in general terms"],
    },
    "project": {
        "about": "{A} is an engineer and {B} the project manager on a software project called Kestrel; they chat about it every couple of weeks.",
        "values": [
            ("database", "the database the Kestrel project uses", "Which database does the Kestrel project use now?",
             "Which database did the Kestrel project use on {d}?", ["PostgreSQL", "CockroachDB", "MySQL", "SQLite", "MariaDB"], True),
            ("billing_owner", "who owns the billing module", "Who owns the billing module now?", "Who owned the billing module on {d}?",
             ["Priya", "Tomasz", "Leilani", "Henrik", "Odalys"], True),
            ("cloud_budget", "the project's quarterly cloud budget", "What is the project's quarterly cloud budget now?",
             "What was the project's quarterly cloud budget on {d}?", ["$18,000", "$24,500", "$31,000", "$12,750", "$27,200"], True),
            ("ci_tool", "the CI system the project builds on", "Which CI system does the project build on now?",
             "Which CI system did the project build on on {d}?", ["Buildkite", "CircleCI", "Jenkins", "TeamCity", "Drone"], True),
            ("region", "the cloud region the service is hosted in", "Which cloud region is the service hosted in now?",
             "Which cloud region was the service hosted in on {d}?", ["Frankfurt", "Oregon", "Singapore", "Mumbai", "Dublin"], True),
            ("design_lead", "who leads design on the project", "Who leads design on the project now?", "Who led design on the project on {d}?",
             ["Marisol", "Anselm", "Thandiwe", "Joaquin", "Greta"], True),
        ],
        "ended": [("oncall", "{A} is on the on-call rotation", "{A} was taken off the on-call rotation", "Is {A} currently on the on-call rotation?"),
                  ("vendor", "the team has a contract with the vendor Quillon for load testing", "the team ended the contract with Quillon",
                   "Does the team currently have a contract with Quillon?")],
        "count": ("blockers", "a new blocker bug was opened on the release: {v}", "the blocker {v} was fixed and closed",
                  "How many blocker bugs are open on the release now?",
                  ["the login timeout", "the export crash", "the duplicate invoice bug", "the timezone drift", "the sync memory leak",
                   "the broken password reset", "the stale cache bug"]),
        "stable": [("repo", "the code lives in a repository named kestrel-core", "What is the project's repository called?", "kestrel-core"),
                   ("client", "the project's main client is Northwind Dairy", "Who is the project's main client?", "Northwind Dairy")],
        "filler": ["code review manners", "a flaky test in general terms", "the office coffee machine", "a conference talk", "documentation debt",
                   "a team lunch", "a laptop upgrade", "a newcomer settling in"],
    },
    "contract": {
        "about": "{A} is a client and {B} their lawyer, negotiating a supply agreement with a company called Marlow Ltd; they chat every couple of weeks.",
        "values": [
            ("payment_term", "the payment term in the agreement", "What is the payment term in the agreement now?",
             "What was the payment term in the agreement on {d}?", ["30 days", "45 days", "60 days", "14 days", "90 days"], True),
            ("liability_cap", "the liability cap", "What is the liability cap now?", "What was the liability cap on {d}?",
             ["$2 million", "$5 million", "$750,000", "$3.5 million", "$1.2 million"], True),
            ("governing_law", "the governing law of the agreement", "What is the governing law of the agreement now?",
             "What was the governing law of the agreement on {d}?", ["English law", "New York law", "Swiss law", "Singapore law", "Ontario law"], True),
            ("min_volume", "the minimum annual order volume", "What is the minimum annual order volume now?",
             "What was the minimum annual order volume on {d}?", ["40,000 units", "55,000 units", "25,000 units", "70,000 units", "32,000 units"], True),
            ("signatory", "who signs for Marlow Ltd", "Who signs for Marlow Ltd now?", "Who was to sign for Marlow Ltd on {d}?",
             ["Ms. Devereux", "Mr. Nakagawa", "Ms. Oyelaran", "Mr. Castellanos", "Ms. Winterbourne"], True),
            ("delivery_port", "the port the goods are delivered to", "Which port are the goods delivered to now?",
             "Which port were the goods to be delivered to on {d}?", ["Rotterdam", "Felixstowe", "Gdansk", "Antwerp", "Bilbao"], True),
        ],
        "ended": [("exclusivity", "the agreement includes an exclusivity clause", "the exclusivity clause was removed from the agreement",
                   "Does the agreement currently include an exclusivity clause?"),
                  ("escrow", "there is an escrow arrangement for the tooling costs", "the escrow arrangement was dropped",
                   "Is there currently an escrow arrangement for the tooling costs?")],
        "count": ("open_points", "a new open point was raised in the negotiation: {v}", "the point about {v} was settled",
                  "How many open points remain to be negotiated now?",
                  ["the indemnity wording", "the delivery penalties", "the audit rights", "the tooling ownership", "the force majeure list",
                   "the price review formula", "the insurance certificates"]),
        "stable": [("their_lawyers", "Marlow's lawyers are the firm Hartwell Breck", "Which law firm represents Marlow?", "Hartwell Breck"),
                   ("product", "the agreement covers the supply of ceramic bearings", "What product does the agreement cover?", "ceramic bearings")],
        "filler": ["scheduling the next call", "a typo found in a draft", "how slow the other side is to reply", "an old deal of theirs",
                   "printing and signing logistics", "a holiday closure", "the lawyer's workload", "a visit to the client's plant"],
    },
    "clinic": {
        "about": "{A} is a patient with diabetes and {B} the practice nurse who phones or messages for follow-up every couple of weeks.",
        "values": [
            ("metformin", "{A}'s daily dose of metformin", "What is {A}'s daily dose of metformin now?", "What was {A}'s daily dose of metformin on {d}?",
             ["500 mg", "850 mg", "1000 mg", "750 mg", "1500 mg"], True),
            ("bp_drug", "the blood pressure medication {A} takes", "Which blood pressure medication does {A} take now?",
             "Which blood pressure medication did {A} take on {d}?", ["lisinopril", "amlodipine", "losartan", "ramipril", "valsartan"], True),
            ("hba1c", "{A}'s latest HbA1c result", "What is {A}'s latest HbA1c result?", "What was {A}'s latest HbA1c result as of {d}?",
             ["7.9%", "7.2%", "6.8%", "7.5%", "6.4%"], True),
            ("weight", "{A}'s latest recorded weight", "What is {A}'s latest recorded weight?", "What was {A}'s latest recorded weight as of {d}?",
             ["94 kg", "91 kg", "88 kg", "96 kg", "86 kg"], True),
            ("doctor", "the doctor {A} is under", "Which doctor is {A} under now?", "Which doctor was {A} under on {d}?",
             ["Dr. Hallberg", "Dr. Mbeki", "Dr. Fontaine", "Dr. Sato", "Dr. Grieve"], True),
            ("pharmacy", "the pharmacy {A} collects prescriptions from", "Which pharmacy does {A} collect prescriptions from now?",
             "Which pharmacy did {A} collect prescriptions from on {d}?",
             ["Calder Pharmacy", "Wren Street Chemist", "Oakfield Pharmacy", "Harbour Dispensary", "Linden Chemist"], False),
        ],
        "ended": [("statin", "{A} takes atorvastatin", "{A} was taken off atorvastatin", "Is {A} currently taking atorvastatin?"),
                  ("physio", "{A} is attending weekly physiotherapy", "{A}'s physiotherapy was finished and stopped",
                   "Is {A} currently attending physiotherapy?")],
        "count": ("supplements", "{A} started taking a supplement: {v}", "{A} stopped taking {v}", "How many supplements does {A} take now?",
                  ["vitamin D", "magnesium", "omega-3", "vitamin B12", "zinc", "iron", "folic acid"]),
        "stable": [("drug_allergy", "{A} is allergic to penicillin", "What medication is {A} allergic to?", "penicillin"),
                   ("blood_type", "{A}'s blood type is B negative", "What is {A}'s blood type?", "B negative")],
        "filler": ["parking at the clinic", "how {A} has been sleeping", "walking more, in general terms", "the flu season", "a grandchild",
                   "appointment reminders by text", "eating habits in general terms", "a broken lift at the clinic"],
    },
    "shop": {
        "about": "{A} owns a small bike shop and {B} is the shop's assistant; they message about the shop every couple of weeks.",
        "values": [
            ("tube_supplier", "the shop's inner-tube supplier", "Who is the shop's inner-tube supplier now?", "Who was the shop's inner-tube supplier on {d}?",
             ["Velox Parts", "Garrow Wholesale", "Tandem Supply Co", "Northgate Components", "Spindle & Co"], True),
            ("service_price", "the price of a basic service", "What is the price of a basic service now?", "What was the price of a basic service on {d}?",
             ["$45", "$55", "$60", "$49", "$65"], True),
            ("rent", "the shop's monthly rent", "What is the shop's monthly rent now?", "What was the shop's monthly rent on {d}?",
             ["$2,100", "$2,350", "$2,600", "$1,950", "$2,800"], True),
            ("lead_mechanic", "the shop's lead mechanic", "Who is the shop's lead mechanic now?", "Who was the shop's lead mechanic on {d}?",
             ["Dario", "Wilhelmina", "Kofi", "Saoirse", "Bartek"], True),
            ("till", "the till software the shop uses", "Which till software does the shop use now?", "Which till software did the shop use on {d}?",
             ["TillPoint", "Ledgerly", "ShopKeep One", "Cashwise", "Registra"], True),
            ("courier", "the courier the shop ships online orders with", "Which courier does the shop ship online orders with now?",
             "Which courier did the shop ship online orders with on {d}?", ["Parcelwing", "Fleetfoot", "Redline Couriers", "Dropswift", "Harrier Post"], True),
        ],
        "ended": [("rentals", "the shop offers bike rentals", "the shop stopped offering bike rentals", "Does the shop currently offer bike rentals?"),
                  ("sundays", "the shop opens on Sundays", "the shop stopped opening on Sundays", "Is the shop currently open on Sundays?")],
        "count": ("ebikes", "a new e-bike arrived in stock: {v}", "{v} was sold", "How many e-bikes are in stock now?",
                  ["the red Cube", "the Trek Verve", "the Giant Explore", "the Specialized Como", "the Orbea Vibe", "the Ribble Hybrid",
                   "the Cannondale Tesoro"]),
        "stable": [("landlord", "the shop's landlord is Mrs. Pemberton", "Who is the shop's landlord?", "Mrs. Pemberton"),
                   ("founded", "the shop first opened in 2011", "What year did the shop first open?", "2011")],
        "filler": ["a difficult customer", "punctures after the roadworks", "the window display", "a local race", "the kettle",
                   "a delivery van blocking the door", "cleaning the workshop", "a charity ride"],
    },
}
NAMES = {"dev": [("Maren", "Isak"), ("Dalia", "Corwin"), ("Yusuf", "Petra"), ("Edmund", "Noor"), ("Ffion", "Leandro")],
         "held": [("Sunniva", "Teodor"), ("Kwame", "Liesel"), ("Ottilie", "Rafiq"), ("Bronagh", "Emeka"), ("Casimir", "Anouk")]}
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven"]


def fmt_day(d: date) -> str:
    return f"{d.day} {d.strftime('%B')} {d.year}"


def build(domain: str, split: str, seed: int) -> dict:
    """One conversation's script: dated sessions, the points each must state, and the questions with their gold
    answers and evidence. All of it is code; the same arguments give the same script."""
    rng = random.Random(f"{domain}:{split}:{seed}")
    spec = DOMAINS[domain]
    a, b = NAMES[split][list(DOMAINS).index(domain)]
    f = lambda s: s.replace("{A}", a).replace("{B}", b)  # noqa: E731
    day = date(2025, 1, 6) + timedelta(days=rng.randrange(60))
    dates = []
    for _ in range(SESSIONS):
        dates.append(day)
        day += timedelta(days=rng.randrange(9, 21))
    points: list[dict] = []       # {id, session, speaker, say, need: [values that must be written], may: [values allowed]}
    qa: list[dict] = []

    def point(session: int, speaker: str, say: str, need: list[str], may: list[str] | None = None, after: str | None = None,
              quiet: bool = False) -> str:
        pid = f"p{len(points) + 1}"
        points.append({"id": pid, "session": session, "speaker": speaker, "say": say, "need": list(need), "may": list(may or []),
                       "after": after, "quiet": quiet})
        return pid

    def ask(case: str, question: str, gold: str, evidence: list[str], must: list[str], track: str,
            now: list[str] | None = None, old: list[str] | None = None) -> None:
        """`now` and `old`: the values a ledger that followed the changes holds as current, and no longer as current
        (scripts/ledger_report.py reads the ledger's own state against them; empty where the case has none)."""
        qa.append({"question": question, "answer": gold, "category": case, "points": evidence, "must": must, "track": track,
                   "now": list(now or []), "old": list(old or [])})

    shared = [t for t in spec["values"] if t[5]]
    contra = rng.choice(shared)
    rest = [t for t in spec["values"] if t is not contra]
    rng.shuffle(rest)
    chains, corrections, buried = rest[:2], rest[2:4], rest[4]

    for key, fact, q_now, q_asof, pool, both in chains:
        vals = rng.sample(pool, rng.choice((3, 4)))
        while True:
            at = sorted(rng.sample(range(0, SESSIONS - 2), len(vals)))
            gaps = [at[i + 1] - at[i] for i in range(len(at) - 1)]
            if max(gaps) >= 2:
                break
        ids = []
        for i, (s, v) in enumerate(zip(at, vals)):
            who = b if both and rng.random() < 0.4 else a      # a fact about {A} alone is {A}'s to tell
            say = f"{f(fact)} is {v}" if i == 0 else f"{f(fact)} has changed and is now {v}"
            ids.append(point(s, who, say, [v], vals[:i]))
        ask("chain", f(q_now), vals[-1], [ids[-1]], [vals[-1]], key, [vals[-1]], vals[:-1])
        i = gaps.index(max(gaps))                                    # the earlier value that held longest
        mid = dates[at[i]] + (dates[at[i + 1]] - dates[at[i]]) / 2
        ask("as-of", f(q_asof).replace("{d}", fmt_day(mid)), vals[i], [ids[i]], [vals[i]], key)

    for key, fact, q_now, _, pool, _ in corrections:
        wrong, right = rng.sample(pool, 2)
        s = rng.randrange(1, SESSIONS - 2)
        later = s + rng.choice((0, 1))
        p1 = point(s, a, f"{f(fact)} is {wrong}", [wrong])
        p2 = point(later, a, f"{a} corrects what they said before: {f(fact)} is not {wrong}, that was a mistake; it is {right}",
                   [right], [wrong], after=p1)
        ask("correction", f(q_now), right, [p1, p2], [right], key, [right], [wrong])

    key, fact, q_now, _, pool, _ = contra
    v1, v2 = rng.sample(pool, 2)
    s1 = rng.randrange(0, SESSIONS - 5)
    s2 = rng.randrange(s1 + 3, SESSIONS - 1)
    p1 = point(s1, a, f"{a} says, as a settled matter, that {f(fact)} is {v1}", [v1])
    p2 = point(s2, b, f"{b} says, as their own firm understanding and without referring to anything said before, that {f(fact)} is {v2} and has been all along (not a "
                      f"change); {a} does not react to it and the chat moves on, so it is never settled", [v2], quiet=True)
    ask("contradiction", f(q_now), f"It is disputed and was never settled: {a} said {v1}, {b} said {v2}.", [p1, p2], [v1, v2], key, [v1, v2])

    key, fact, q_now, _, pool, _ = buried
    v1, v2 = rng.sample(pool, 2)
    s1 = rng.randrange(0, SESSIONS - 5)
    s2 = rng.randrange(s1 + 3, SESSIONS - 1)
    p1 = point(s1, a, f"{f(fact)} is {v1}", [v1])
    p2 = point(s2, a, f"only in passing, as a short aside inside a turn that is mainly about another topic, and with nobody remarking on it: "
                      f"{f(fact)} is now {v2}", [v2], quiet=True)
    ask("buried", f(q_now), v2, [p2], [v2], key, [v2], [v1])

    stops = rng.randrange(2)                          # one of the two is ended, the other goes on
    for i, (key, holds, ends, q) in enumerate(spec["ended"]):
        s1 = rng.randrange(0, SESSIONS - 5)
        s2 = rng.randrange(s1 + 3, SESSIONS - 1)
        p1 = point(s1, a, f(holds), [])
        if i == stops:
            p2 = point(s2, a, f(ends), [])
            ask("ended", f(q), f"No. It ended: {f(ends)} (said on {fmt_day(dates[s2])}).", [p1, p2], [], key)
        else:
            ask("ongoing", f(q), f"Yes: {f(holds)}.", [p1], [], key)

    key, add, remove, q, items = spec["count"]
    have = rng.sample(items, 5)
    adds = sorted(rng.sample(range(0, SESSIONS - 4), 5))
    ids, gone = [], []
    for s, v in zip(adds, have):
        ids.append(point(s, a, f(add).replace("{v}", v), [v]))
    for v in rng.sample(have[:4], 2):
        s = rng.randrange(adds[have.index(v)] + 1, SESSIONS - 1)
        ids.append(point(s, a, f(remove).replace("{v}", v), [v]))
        gone.append(v)
    left = [v for v in have if v not in gone]
    ask("count", f(q), f"{len(left)}: {', '.join(left)}.", ids, [str(len(left))], key, left, gone)

    for key, statement, q, answer in spec["stable"]:
        p = point(rng.randrange(0, SESSIONS - 1), a, f(statement), [answer])
        ask("stable", f(q), answer, [p], [answer], key)

    tracked = sorted({v for t in spec["values"] for v in t[4]} | set(spec["count"][4]) | {t[3] for t in spec["stable"]})
    filler = [rng.sample([f(x) for x in spec["filler"]], 3) for _ in range(SESSIONS)]
    return {"sample_id": f"ledger-{domain}-{split}", "domain": domain, "split": split, "speakers": [a, b], "about": f(spec["about"]),
            "dates": [d.isoformat() for d in dates], "points": points, "qa": qa, "tracked": tracked, "filler": filler,
            "topics": [f(t[1]) for t in spec["values"]] + [f(e[1]) for e in spec["ended"]] + [spec["count"][0].replace("_", " ")]
                      + [f(s[1]) for s in spec["stable"]]}


def said(value: str, text: str) -> bool:
    """Is the literal value in the text, as a whole token ("$45" is not in "$450", "2.4" not in "12.45")."""
    return re.search(rf"(?<![\w$.]){re.escape(value)}(?![\w%]|[.,]\d)", text, re.I) is not None


WRITER_SYSTEM = ("You write one session of a realistic, casual text chat between two people. You follow the brief exactly and reply "
                 "with JSON only, no prose and no code fence.")


def brief(script: dict, session: int, note: str = "") -> str:
    a, b = script["speakers"]
    pts = [p for p in script["points"] if p["session"] == session]
    lines = "\n".join(f"- {p['id']} (said by {p['speaker']}): {p['say']}" + (f". Write the value exactly as: {' and '.join(p['need'])}" if p["need"] else "")
                      for p in pts) or "- (none this session)"
    return (f"{script['about']}\nThis is session {session + 1} of {len(script['dates'])}, on {fmt_day(date.fromisoformat(script['dates'][session]))}.\n\n"
            f"Points this session must state, each in exactly one turn by the speaker named, in this order, worked naturally into the chat "
            f"in the speaker's own everyday words (the lines below are instructions, not wording to copy: never write phrases like "
            f"'has changed and is now' or 'the car I drive is'; say it as a person would, keeping only the value's exact wording):\n{lines}\n\n"
            f"Small talk to fill the rest: {'; '.join(script['filler'][session])}.\n\n"
            f"Rules:\n- 14 to 18 turns, alternating loosely between {a} and {b}, one to three sentences each.\n"
            f"- Apart from the points above, say nothing at all about these subjects, not even a hint or a recap of earlier sessions: "
            f"{'; '.join(script['topics'])}.\n"
            f"- Do not state dates for the points; the session's own date is enough. No greetings that name the date.\n"
            f"- The turn right after a point may acknowledge it; after that, the point's value is not repeated.\n"
            f"- Do not invent other names, prices, amounts or counts that could be confused with the points.\n\n"
            f'Reply as {{"turns": [{{"speaker": "{a}", "text": "...", "point": null}}, ...]}}, with "point" set to the point\'s id on the '
            f"one turn that states it and null on every other turn." + (f"\n\n{note}" if note else ""))


def check(script: dict, session: int, turns: list[dict]) -> list[str]:
    """What is wrong with a written session, in words the writer can act on. Empty when it can be kept."""
    pts = {p["id"]: p for p in script["points"] if p["session"] == session}
    bad = []
    if not 10 <= len(turns) <= 24:
        bad.append(f"{len(turns)} turns; write 14 to 18")
    seen = [t.get("point") for t in turns if t.get("point")]
    for pid, p in pts.items():
        if seen.count(pid) != 1:
            bad.append(f"point {pid} is marked on {seen.count(pid)} turns; it must be on exactly one")
            continue
        t = next(t for t in turns if t.get("point") == pid)
        if t.get("speaker") != p["speaker"]:
            bad.append(f"point {pid} must be said by {p['speaker']}")
        for v in p["need"]:
            if not said(v, t.get("text", "")):
                bad.append(f"the turn for {pid} must contain exactly this wording: {v}")
    if [x for x in seen if x not in pts]:
        bad.append("a turn is marked with a point that is not in the brief")
    for pid, p in pts.items():
        if p["after"] in pts and seen.count(pid) == 1 and seen.count(p["after"]) == 1 and seen.index(pid) < seen.index(p["after"]):
            bad.append(f"point {pid} must come after point {p['after']}")
    for k, t in enumerate(turns):
        if t.get("speaker") not in script["speakers"] or not str(t.get("text", "")).strip():
            bad.append("every turn needs one of the two speakers and some text")
            break
        p = pts.get(t.get("point"))
        allowed = set(p["need"]) | set(p["may"]) if p else set()
        before = pts.get(turns[k - 1].get("point")) if k else None
        if before and not before["quiet"]:          # the other person may repeat what they were just told
            allowed |= set(before["need"])
        for v in script["tracked"]:
            if v not in allowed and said(v, t["text"]):
                bad.append(f'the turn "{t["text"][:50]}..." mentions {v}, which belongs to a tracked subject; leave it out')
    return bad


def to_locomo(script: dict, sessions: list[list[dict]]) -> dict:
    """The conversation in LoCoMo's layout, each question's evidence as the turn ids of its points."""
    a, b = script["speakers"]
    conv = {"speaker_a": a, "speaker_b": b}
    where = {}
    rng = random.Random(script["sample_id"])
    for n, turns in enumerate(sessions, 1):
        d = date.fromisoformat(script["dates"][n - 1])
        hour, minute = rng.randrange(8, 21), rng.choice((0, 5, 10, 15, 20, 30, 40, 45, 50))
        conv[f"session_{n}_date_time"] = f"{(hour - 1) % 12 + 1}:{minute:02d} {'am' if hour < 12 else 'pm'} on {d.day} {d.strftime('%B')}, {d.year}"
        conv[f"session_{n}"] = []
        for k, t in enumerate(turns, 1):
            conv[f"session_{n}"].append({"speaker": t["speaker"], "dia_id": f"D{n}:{k}", "text": t["text"].strip()})
            if t.get("point"):
                where[t["point"]] = f"D{n}:{k}"
    qa = [{"question": q["question"], "answer": q["answer"], "category": q["category"], "evidence": [where[p] for p in q["points"]],
           "must": q["must"], "track": q["track"], "now": q["now"], "old": q["old"]} for q in script["qa"]]
    return {"sample_id": script["sample_id"], "domain": script["domain"], "split": script["split"], "conversation": conv, "qa": qa,
            "script": {"points": script["points"], "dates": script["dates"], "turn_of_point": where}}


def scripts(seed: int = 1) -> list[dict]:
    return [build(d, split, seed) for split in ("dev", "held") for d in DOMAINS]


def main():
    args = sys.argv[1:]
    if not args or args[0] not in ("plan", "write"):
        sys.exit(__doc__)
    all_scripts = scripts()
    if args[0] == "plan":
        from collections import Counter
        for s in all_scripts:
            print(s["sample_id"], len(s["points"]), "points,", len(s["qa"]), "questions", dict(Counter(q["category"] for q in s["qa"])))
        print("questions:", sum(len(s["qa"]) for s in all_scripts))
        return
    import concurrent.futures as cf
    sys.path.insert(0, str(SERVICE))
    sys.path.insert(0, str(SERVICE / "scripts"))
    sys.path.insert(0, str(SERVICE.parent / "live-eval"))
    import env
    from brain.settings import load
    from brain.spend import MemorySpendStore, SpendMeter
    from readers import Reader

    def flag(name, default):
        return int(args[args.index(name) + 1]) if name in args else default

    only, cap = flag("--only", 0), flag("--cap", 5)
    if only:
        all_scripts = all_scripts[:only]
    env.load_env()
    meter = SpendMeter(MemorySpendStore(), float(cap))
    writer = Reader("sonnet", meter, load().anthropic_key)
    OUT.mkdir(exist_ok=True)
    saved_path = OUT / "ledger_set_v1_sessions.jsonl"       # every kept session, so a stopped start loses nothing
    saved = {}
    if saved_path.exists():
        for line in saved_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            saved[(r["sample_id"], r["session"])] = r

    def session(job):
        script, n = job
        if (script["sample_id"], n) in saved:
            return saved[(script["sample_id"], n)]
        note, tries = "", []
        for attempt in range(1, 5):
            raw = writer.complete(WRITER_SYSTEM, brief(script, n, note), 3000).text
            try:
                turns = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])["turns"]
                bad = check(script, n, turns)
            except (ValueError, KeyError, TypeError) as e:
                turns, bad = [], [f"the reply was not the JSON asked for ({type(e).__name__})"]
            tries.append(bad)
            if not bad:
                return {"sample_id": script["sample_id"], "session": n, "turns": turns, "attempts": attempt, "rejected": tries[:-1]}
            note = f"Attempt {attempt} was rejected. Fix exactly this and keep everything else: " + "; ".join(bad[:6])
        raise RuntimeError(f"{script['sample_id']} session {n + 1}: not written after 4 attempts: {tries[-1]}")

    jobs = [(s, n) for s in all_scripts for n in range(SESSIONS)]
    done = {}
    with cf.ThreadPoolExecutor(4) as ex:
        for k, r in enumerate(ex.map(session, jobs), 1):
            if (r["sample_id"], r["session"]) not in saved:
                with saved_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            done[(r["sample_id"], r["session"])] = r
            if k % 16 == 0:
                print(f"{k}/{len(jobs)} sessions, spent ${meter.spent_today():.3f}", flush=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    for split in ("dev", "held"):
        convs = [to_locomo(s, [done[(s["sample_id"], n)]["turns"] for n in range(SESSIONS)]) for s in all_scripts if s["split"] == split]
        if convs and not only:
            (OUT / f"ledger_set_v1_{split}.json").write_text(json.dumps(convs, ensure_ascii=False, indent=1), encoding="utf-8")
    retries = sum(r["attempts"] - 1 for r in done.values())
    print(json.dumps({"written": stamp, "conversations": len(all_scripts), "sessions": len(done), "rejected_attempts": retries,
                      "turns": sum(len(r["turns"]) for r in done.values()), "spend_usd_this_start": round(meter.spent_today(), 4)}, indent=2))


if __name__ == "__main__":
    main()
