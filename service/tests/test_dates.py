"""Dates (build step 3): phrase finding and calendar resolution in code. Offline."""
from datetime import date

from brain import dates
from brain.dates import Mention, find_phrases, resolve

SAT = date(2023, 5, 20)   # a Saturday


def parts(mode, p=0.95, **kw):
    out = {q: ("none", 0.99) for q in dates.DATE_QUESTIONS}
    out["mode"] = (mode, p)
    out.update({k: (v, 0.9) for k, v in kw.items()})
    return out


def test_find_phrases():
    t = ("Yesterday I finally finished the assay. Three weeks ago we ordered reagents, and last Friday the PI "
         "signed off. The review is on May 3rd, 2023; I moved in March. Every day I run 5 km for two weeks.")
    got = [p.lower() for p in find_phrases(t)]
    for want in ("yesterday", "three weeks ago", "last friday", "may 3rd, 2023", "in march"):
        assert want in got, (want, got)
    assert not any("every day" in p or "two weeks" == p for p in got)
    assert find_phrases("No dates at all here.") == []


def test_resolve_relative_forms():
    assert resolve(parts("relative_day", offset_days="-1"), SAT) == Mention("", date(2023, 5, 19), date(2023, 5, 19), "day", 0.9)
    m = resolve(parts("relative_week", week_offset="-3"), SAT)
    assert (m.date_from, m.date_to, m.granularity) == (date(2023, 4, 24), date(2023, 4, 30), "week")
    m = resolve(parts("relative_month", month_offset="-1"), SAT)
    assert (m.date_from, m.date_to) == (date(2023, 4, 1), date(2023, 4, 30))
    m = resolve(parts("relative_month", month_offset="-5"), date(2023, 3, 15))
    assert (m.date_from, m.date_to) == (date(2022, 10, 1), date(2022, 10, 31))


def test_resolve_weekdays_by_direction():
    past = resolve(parts("relative_weekday", weekday="Friday", weekday_direction="past"), SAT)
    assert past.date_from == date(2023, 5, 19)
    same = resolve(parts("relative_weekday", weekday="Saturday", weekday_direction="past"), SAT)
    assert same.date_from == date(2023, 5, 13)          # "last Saturday" written on a Saturday: a week back
    nxt = resolve(parts("relative_weekday", weekday="Tuesday", weekday_direction="upcoming"), SAT)
    assert nxt.date_from == date(2023, 5, 23)
    this = resolve(parts("relative_weekday", weekday="Monday", weekday_direction="this_week"), SAT)
    assert this.date_from == date(2023, 5, 15)


def test_resolve_absolute_and_past_bias():
    m = resolve(parts("absolute", month="May", day="3", year="2023"), SAT)
    assert m.date_from == date(2023, 5, 3) and m.granularity == "day"
    m = resolve(parts("absolute", month="December", day="24"), SAT)     # no year, far ahead: last December
    assert m.date_from == date(2022, 12, 24)
    m = resolve(parts("absolute", month="March"), SAT)
    assert (m.date_from, m.date_to, m.granularity) == (date(2023, 3, 1), date(2023, 3, 31), "month")
    assert resolve(parts("absolute", month="February", day="30", year="2023"), SAT).date_from is None


def test_confidence_is_the_weakest_used_part_and_none_mode():
    m = resolve(parts("relative_day", p=0.7, offset_days="-2"), SAT)
    assert m.confidence == 0.7
    n = resolve(parts("none"), SAT)
    assert n.date_from is None


def test_mentions_below_review_keep_the_phrase_without_a_date():
    class Jev:
        def ask_many(self, reqs):
            from brain.providers import JevResult
            out = []
            for state, qs in reqs:
                ans = {q: {"choice": "none", "probabilities": {"none": 0.99}} for q in qs}
                ans["mode"] = {"choice": "relative_day", "probabilities": {"relative_day": 0.55}}
                ans["offset_days"] = {"choice": "-1", "probabilities": {"-1": 0.9}}
                out.append(JevResult(ans, "fake"))
            return out

    class P:
        jev = Jev()
    [(eid, m)] = dates.mentions_for(P(), [(7, "Yesterday we closed.", SAT)])
    assert eid == 7 and m.span == "Yesterday" and m.date_from is None and m.confidence == 0.55


def test_capture_stores_time_mentions_and_answer_reports_route(make_client, pool):
    """Plumbing through the API with the offline stubs (the stub's date parts are not meaningful)."""
    on = make_client(dates_enabled=True, when_route_enabled=True)
    from conftest import ADMIN
    r = on.post("/v1/capture", headers=ADMIN, json={"namespace": "jeff", "turns": [
        {"role": "user", "text": "Yesterday I finished the assay, and last Friday the PI signed off.",
         "ts": "2023-05-20T10:00:00Z"}]})
    assert r.status_code == 200
    with pool.connection() as c:
        spans = sorted(s for (s,) in c.execute("select span from brain.time_mention").fetchall())
    assert spans == ["Yesterday", "last Friday"]
    a = on.post("/v1/answer", headers=ADMIN, json={"namespace": "jeff", "question": "When did I finish the assay?",
                                                   "question_date": "2023/05/30 (Tue) 10:00"})
    assert a.status_code == 200 and a.json()["route"] in ("when", "open", "current", "history", "count",
                                                         "preference", "assistant")
    off = make_client()
    off.post("/v1/capture", headers=ADMIN, json={"namespace": "jeff", "turns": [{"role": "user", "text": "Today we met."}]})
    with pool.connection() as c:
        assert c.execute("select count(*) from brain.time_mention").fetchone()[0] == 2     # unchanged when off
    assert off.post("/v1/answer", headers=ADMIN, json={"namespace": "jeff", "question": "When?"}).json()["route"] is None


def test_block_computes_distances_and_gaps(client, pool):
    from conftest import ADMIN
    from brain.db import tenant
    eid = client.post("/v1/capture", headers=ADMIN, json={"namespace": "jeff", "turns": [
        {"role": "user", "text": "I started yoga yesterday.", "ts": "2023-05-20T10:00:00Z"}]}).json()["episodes"][0]["id"]
    owner = None
    with pool.connection() as c:
        owner = c.execute("select id from brain.account where name = 'owner'").fetchone()[0]
    with tenant(pool, owner) as c:
        dates.store(c, [(eid, Mention("yesterday", date(2023, 5, 19), date(2023, 5, 19), "day", 0.95))])
        text = dates.block(c, [{"episode_id": eid, "ts": "2023-05-20T10:00:00+00:00",
                                "text": "[2023-05-20] user: I started yoga yesterday."}], "2023/06/02 (Fri) 09:00")
    assert "'yesterday' = 2023-05-19" in text
    assert "2023-05-19 (Friday): 14 days (2 weeks 0 days) before the question date" in text
    assert "2023-05-19 to 2023-05-20: 1 days" in text


def test_last_weekend_is_saturday_and_sunday_not_the_whole_week():
    """exp-when-locomo-v1: "Last weekend" said on Monday 2023-07-17 resolved to 10 to 16 July; the weekend is 15 and 16."""
    week = resolve(parts("relative_week", week_offset="-1"), date(2023, 7, 17))
    m = dates.weekend_of("Last weekend", week)
    assert (m.date_from, m.date_to, m.granularity) == (date(2023, 7, 15), date(2023, 7, 16), "range")
    assert dates.weekend_of("last week", week) == week
    day = resolve(parts("relative_day", offset_days="-1"), SAT)
    assert dates.weekend_of("last weekend", day) == day


def test_date_notes_write_the_resolved_date_beside_the_phrase(client, pool, monkeypatch):
    """exp-when-locomo-v1: off by default; on, the pack carries the turn's words with the computed date beside
    the phrase, and leaves alone a phrase for the turn's own day or a phrase that is not in the text."""
    from conftest import ADMIN
    from brain import core
    from brain.db import tenant
    assert core.DATE_NOTES is False                          # off until its gate is met
    eid = client.post("/v1/capture", headers=ADMIN, json={"namespace": "jeff", "turns": [
        {"role": "user", "text": "Last weekend I joined a mentorship program, and today I signed the form.",
         "ts": "2023-07-17T10:00:00Z"}]}).json()["episodes"][0]["id"]
    with pool.connection() as c:
        owner = c.execute("select id from brain.account where name = 'owner'").fetchone()[0]
    with tenant(pool, owner) as c:
        dates.store(c, [(eid, Mention("Last weekend", date(2023, 7, 15), date(2023, 7, 16), "range", 0.95)),
                        (eid, Mention("today", date(2023, 7, 17), date(2023, 7, 17), "day", 0.99)),
                        (eid, Mention("next Friday", date(2023, 7, 21), date(2023, 7, 21), "day", 0.9))])
        text = "[2023-07-17] user: Last weekend I joined a mentorship program, and today I signed the form."
        out = dates.annotate(c, [(eid, text, "2023-07-17T10:00:00+00:00"), (eid + 1000, "No dates here.", "2023-07-17")])
    assert out == ["[2023-07-17] user: Last weekend [2023-07-15 to 2023-07-16] I joined a mentorship program, "
                   "and today I signed the form.", "No dates here."]
    q = {"namespace": "jeff", "question": "When did I join the mentorship program?"}
    assert "[2023-07-15 to 2023-07-16]" not in client.post("/v1/context", headers=ADMIN, json=q).json()["prompt"]
    monkeypatch.setattr(core, "DATE_NOTES", True)
    assert "Last weekend [2023-07-15 to 2023-07-16] I joined" in client.post("/v1/context", headers=ADMIN, json=q).json()["prompt"]


def test_a_year_that_is_not_written_comes_from_the_direction_not_from_the_year_choice():
    """exp-date-direction-v1: "attended in November", written in February, with the year Choice saying 2023."""
    feb = date(2023, 2, 26)
    said = parts("absolute", month="November", year="2023")
    assert resolve(said, feb).date_from == date(2023, 11, 1)                      # as built: nine months ahead
    assert resolve(said, feb, "in November").date_from == date(2023, 11, 1)       # no direction asked: unchanged
    past = {**said, "direction": ("past", 0.9)}
    m = resolve(past, feb, "in November")
    assert (m.date_from, m.date_to, m.granularity) == (date(2022, 11, 1), date(2022, 11, 30), "month")
    assert resolve(past, date(2023, 11, 15), "in November").date_from == date(2023, 11, 1)    # this month counts
    plan = {**parts("absolute", month="October"), "direction": ("upcoming", 0.9)}
    assert resolve(plan, date(2023, 5, 23), "in October").date_from == date(2023, 10, 1)
    assert resolve({**parts("absolute", month="January"), "direction": ("upcoming", 0.9)},
                   date(2023, 11, 20), "in January").date_from == date(2024, 1, 1)             # next January
    # a date up to a month ahead is left as built, whatever the direction (a clock a little off, or last year)
    near = {**parts("absolute", month="March", day="3"), "direction": ("past", 0.9)}
    assert resolve(near, date(2023, 2, 1), "March 3rd").date_from == date(2023, 3, 3)
    # more than a month ahead: happened is last year; planned is this year (as built the past bias took last year)
    day = parts("absolute", month="April", day="10")
    assert resolve(day, date(2023, 1, 12), "April 10th").date_from == date(2022, 4, 10)
    assert resolve({**day, "direction": ("past", 0.9)}, date(2023, 1, 12), "April 10th").date_from == date(2022, 4, 10)
    assert resolve({**day, "direction": ("upcoming", 0.9)}, date(2023, 1, 12), "April 10th").date_from == date(2023, 4, 10)
    # earlier in the writing year: only a plan moves it, to next year
    jan = parts("absolute", month="January", day="15")
    assert resolve({**jan, "direction": ("upcoming", 0.9)}, date(2023, 5, 12), "January 15th").date_from == date(2024, 1, 15)
    assert resolve({**jan, "direction": ("past", 0.9)}, date(2023, 5, 12), "January 15th").date_from == date(2023, 1, 15)
    # a plan dated a few days before the writing date stays in the writing year: a clock a day off, not next year
    may = {**parts("absolute", month="May", day="22"), "direction": ("upcoming", 0.99)}
    assert resolve(may, date(2023, 5, 27), "May 22nd").date_from == date(2023, 5, 22)
    # an unsure answer, "unclear", and a written year (in the phrase or right after it) are left as built
    assert resolve({**said, "direction": ("past", 0.8)}, feb, "in November").date_from == date(2023, 11, 1)
    assert resolve({**said, "direction": ("unclear", 0.9)}, feb, "in November").date_from == date(2023, 11, 1)
    written = {**parts("absolute", month="November", year="2021"), "direction": ("upcoming", 0.9)}
    assert resolve(written, feb, "in November 2021").date_from == date(2021, 11, 1)
    y2008 = {**parts("absolute", month="February", year="2008"), "direction": ("past", 0.9)}
    assert resolve(y2008, date(2023, 3, 15), "In February", "In February of 2008, Kimbo fought.").date_from == date(2008, 2, 1)
    assert dates.year_written("September 30", "must be issued visas by September 30, 2024.")
    assert dates.year_written("by September", "must be issued visas by September 30, 2024.")
    assert not dates.year_written("in November", "a workshop I attended in November at a festival in 2 days")


def test_the_direction_question_is_asked_only_with_the_setting_and_kept_apart_in_the_cache():
    from brain.providers import JevResult
    assert dates.DATE_DIRECTION is False and "direction" not in dates.date_questions()
    seen = []

    class Jev:
        def ask_many(self, reqs):
            out = []
            for state, qs in reqs:
                seen.append(sorted(qs))
                ans = {q: {"choice": "none", "probabilities": {"none": 0.99}} for q in qs}
                ans.update(mode={"choice": "absolute", "probabilities": {"absolute": 0.95}},
                           month={"choice": "November", "probabilities": {"November": 0.9}},
                           year={"choice": "2023", "probabilities": {"2023": 0.9}})
                if "direction" in qs:
                    ans["direction"] = {"choice": "past", "probabilities": {"past": 0.9}}
                out.append(JevResult(ans, "t"))
            return out

    class P:
        jev = Jev()
    turn = [(1, "I have been writing since a workshop I attended in November at a festival.", date(2023, 2, 26))]
    dates._CACHE.clear()
    assert dates.mentions_for(P, turn)[0][1].date_from == date(2023, 11, 1)
    token = dates.DATE_DIRECTION_OVERRIDE.set(True)
    try:
        got = dates.mentions_for(P, turn)[0][1]            # asked again: the answer kept without it is not reused
    finally:
        dates.DATE_DIRECTION_OVERRIDE.reset(token)
    assert (got.date_from, got.date_to) == (date(2022, 11, 1), date(2022, 11, 30))
    assert len(seen) == 2 and "direction" not in seen[0] and "direction" in seen[1]
    dates._CACHE.clear()


def test_a_written_count_is_read_by_code_when_the_flag_is_on():
    """exp-date-counted-v1: "two weeks ago" names its own offset. Jev is sure of the mode and unsure among 25
    offsets, so the date was dropped; with BRAIN_DATE_COUNTED code reads the number and keeps the mode's
    probability."""
    co = dates.counted_offset
    assert co("two weeks ago") == co("a couple of  weeks ago") == co("2 Weeks Ago") == ("week", -2)
    assert co("two months from now") == ("month", 2) and co("a week from now") == ("week", 1) and co("3 days ago") == ("day", -3)
    for not_counted in ("in two months", "in 90 days", "a few weeks ago", "several months ago", "two days later", "two weeks before", "two weeks",
                        "in 2 weeks ago", "last week", "two years ago", "yesterday"):
        assert co(not_counted) is None, not_counted
    unsure = {**parts("relative_week", p=1.0), "week_offset": ("-2", 0.5)}
    fri = date(2026, 11, 13)
    assert resolve(unsure, fri, "two weeks ago").confidence == 0.5                   # as built: under REVIEW_BELOW
    token = dates.DATE_COUNTED_OVERRIDE.set(True)
    try:
        assert resolve(unsure, fri, "two weeks ago") == Mention("", date(2026, 10, 26), date(2026, 11, 1), "week", 1.0)
        assert resolve(unsure, fri, "a couple of weeks ago").date_from == date(2026, 10, 26)
        assert resolve(parts("relative_day", p=0.9), fri, "3 days ago") == Mention("", date(2026, 11, 10), date(2026, 11, 10), "day", 0.9)
        m = resolve(parts("relative_month", p=0.9), fri, "two months from now")
        assert (m.date_from, m.date_to, m.granularity) == (date(2027, 1, 1), date(2027, 1, 31), "month")
        # Jev's mode must agree on the unit and be sure of it; "in two weeks" is left to Jev (v1 failed on it)
        assert resolve(parts("none", p=0.9), fri, "two weeks ago").date_from is None
        assert resolve({**parts("relative_week", p=1.0), "week_offset": ("2", 0.5)}, fri, "in two weeks").confidence == 0.5
        assert resolve(parts("relative_month", p=0.9, month_offset="-1"), fri, "two weeks ago").date_from == date(2026, 10, 1)
        assert resolve({**parts("relative_week", p=0.5), "week_offset": ("-2", 0.5)}, fri, "two weeks ago").confidence == 0.5
        assert resolve(unsure, fri, "a few weeks ago").confidence == 0.5             # not a number: as built
        assert resolve(unsure, fri).confidence == 0.5                                # no phrase given: as built
    finally:
        dates.DATE_COUNTED_OVERRIDE.reset(token)
    assert dates.DATE_COUNTED is False
