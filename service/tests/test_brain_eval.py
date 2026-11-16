"""scripts/brain_eval.py: the sessions behind an answer's excerpts are recorded with each row. No keys."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from conftest import ADMIN  # noqa: E402


def test_sessions_of_names_the_sessions_behind_the_cited_chunks(client, pool):
    import brain_eval
    for sid, text in (("s-denver", "I moved to Denver last spring."), ("s-boat", "My boat is called Juniper.")):
        r = client.post("/v1/capture", headers=ADMIN, json={
            "namespace": "jeff", "session_id": sid, "turns": [{"role": "user", "text": text}]})
        assert r.status_code == 200, r.text
    with pool.connection() as c:
        rows = c.execute("""select k.id, e.session_id from brain.chunk k join brain.episode e on e.id = k.ref_id
                            where k.kind = 'episode_line' order by k.id""").fetchall()
        by_session = {s: i for i, s in rows}
        assert set(by_session) == {"s-denver", "s-boat"}
        assert brain_eval.sessions_of(c, [by_session["s-boat"]]) == ["s-boat"]
        assert brain_eval.sessions_of(c, [i for i, _ in rows]) == ["s-boat", "s-denver"]
        assert brain_eval.sessions_of(c, []) == [] == brain_eval.sessions_of(c, [10 ** 9])


def test_majority_read_takes_the_majority_verdict_of_an_odd_number_of_reads():
    import brain_eval
    import pytest
    answers = iter(["Denver", "Boulder", "Denver", "x", "y", "z", "only"])
    verdict = {"Denver": True, "Boulder": False, "x": True, "y": None, "z": False, "only": None}
    judge = lambda a: (verdict[a], [verdict[a]] * 3)  # noqa: E731
    r = brain_eval.majority_read(lambda: next(answers), judge, 3)
    assert r["correct"] is True and r["answer"] == "Denver" and r["votes"] == [True] * 3
    assert [x["answer"] for x in r["reads"]] == ["Denver", "Boulder", "Denver"]
    split = brain_eval.majority_read(lambda: next(answers), judge, 3)      # one yes, one unjudged, one no
    assert split["correct"] is None
    one = brain_eval.majority_read(lambda: next(answers), judge)           # one read: as before, no "reads" key
    assert one == {"answer": "only", "correct": None, "votes": [None] * 3}
    with pytest.raises(ValueError):
        brain_eval.majority_read(lambda: "a", judge, 2)
    rows = [{"brain": r, "flat": one}, {"brain": split, "flat": one}]
    assert brain_eval.read_splits(rows, ("flat", "brain")) == {"flat": 0, "brain": 2}


def test_flag_int_takes_the_flag_and_its_number_out_of_the_arguments():
    import brain_eval
    args = ["brain", "test", "--reads", "3", "20"]
    assert brain_eval.flag_int(args, "--reads", 1) == 3 and args == ["brain", "test", "20"]
    assert brain_eval.flag_int(args, "--skip", 0) == 0 and args == ["brain", "test", "20"]


def test_memo_jev_answers_a_repeated_request_as_it_did_the_first_time():
    import brain_eval

    class Wobbly:                                # a Jev whose answer moves on every call
        name, calls = "jev", 0

        def ask_many(self, requests):
            out = []
            for _ in requests:
                Wobbly.calls += 1
                out.append(Wobbly.calls)
            return out
    jev = brain_eval.MemoJev(Wobbly())
    a, q, q2 = {"text": "We moved."}, {"q": {"type": "noul", "instructions": "A move?"}}, {"q": {"type": "noul", "instructions": "A dog?"}}
    first = jev.ask(a, q)
    assert jev.ask_many([(a, q2), (a, q), (a, q2)]) == [2, first, 2]      # a batch: one call per new request
    assert jev.ask({"text": "We moved."}, dict(q)) == first and Wobbly.calls == 2
    assert (jev.hits, jev.misses) == (3, 2) and jev.name == "jev"


def test_memo_haiku_gives_the_same_prompt_the_same_answer():
    import brain_eval

    class Changeable:                            # a model that never writes the same answer twice
        model, calls = "m", 0

        def complete(self, system, user, max_tokens=1024):
            Changeable.calls += 1
            return f"answer {Changeable.calls}"
    haiku = brain_eval.MemoHaiku(Changeable())
    first = haiku.complete("sys", "prompt A", 400)
    assert haiku.complete("sys", "prompt A", 400) == first == "answer 1"
    assert haiku.complete("sys", "prompt B", 400) == "answer 2" and haiku.complete("other", "prompt A", 400) == "answer 3"
    assert (haiku.hits, haiku.misses) == (1, 3) and haiku.model == "m"


def test_a_stopped_locomo_run_keeps_its_finished_conversations(tmp_path):
    import locomo_eval
    path = tmp_path / "cache" / "partial.jsonl"
    assert locomo_eval.partial_load(path) == ({}, 0.0)
    locomo_eval.partial_add(path, "start-1", 3, [{"qid": "conv-3:1"}], 0.2)
    locomo_eval.partial_add(path, "start-1", 0, [{"qid": "conv-0:5"}, {"qid": "conv-0:9"}], 0.5)
    locomo_eval.partial_add(path, "start-2", 7, [], 0.25)            # a second start has its own meter
    done, spent = locomo_eval.partial_load(path)
    assert done == {3: [{"qid": "conv-3:1"}], 0: [{"qid": "conv-0:5"}, {"qid": "conv-0:9"}], 7: []}
    assert spent == 0.75
    # the packs of a finished conversation are saved with it, so a resumed run can still be read by reader_swap
    assert locomo_eval.partial_packs(path) == {}
    rows = [{"qid": "conv-9:1", "flat": {"context_sha": "aa"}, "near1seats": {"context_sha": "bb"}, "ledger_totals": {}}]
    held = {"aa": "pack a", "bb": "pack b", "cc": "another conversation's pack"}
    assert locomo_eval.packs_of(rows, held) == {"aa": "pack a", "bb": "pack b"}
    locomo_eval.partial_add(path, "start-2", 9, rows, 0.3, locomo_eval.packs_of(rows, held))
    assert locomo_eval.partial_packs(path) == {"aa": "pack a", "bb": "pack b"}
    assert locomo_eval.partial_load(path)[0][9] == rows


def test_locomo_rows_say_which_evidence_turns_reached_the_reader():
    from datetime import datetime, timedelta, timezone

    import locomo_eval
    base = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
    turn_at = {base + timedelta(seconds=k): f"D1:{k + 1}" for k in range(5)}
    excerpts = [{"chunk_id": 1, "ts": "2023-05-08T13:56:02+00:00"}, {"chunk_id": 2, "ts": "2023-05-08T15:56:04+02:00"},
                {"chunk_id": 3, "ts": "2024-01-01T00:00:00+00:00"}]          # the last is no captured turn
    gold = locomo_eval.gold_turns(["D1:3", "D1:5; D2:9"])
    assert gold == ["D1:3", "D1:5", "D2:9"] and locomo_eval.gold_turns(None) == []
    assert locomo_eval.evidence_reached(excerpts, turn_at, gold) == ["D1:3", "D1:5"]
    assert locomo_eval.evidence_reached([], turn_at, gold) == []


def test_a_row_whose_routed_arm_has_no_route_marks_the_run_void():
    import brain_eval
    rows = [{"flat": {"route": None}, "brain": {"route": "open"}}, {"flat": {"route": None}, "brain": {"route": None}}]
    assert brain_eval.unrouted(rows[:1], ["brain"]) == 0          # the flat arm never asks the router
    assert brain_eval.unrouted(rows, ["brain", "twin"]) == 1


def test_locomo_rows_can_say_where_each_evidence_turn_stood_among_the_candidates():
    from datetime import datetime, timedelta, timezone

    import locomo_eval
    base = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
    turn_at = {base + timedelta(seconds=k): f"D1:{k + 1}" for k in range(5)}
    items = [{"ts": (base + timedelta(seconds=k)).isoformat()} for k in (4, 0, 2)]      # the reranked order
    assert locomo_eval.evidence_ranks(items, turn_at, ["D1:1", "D1:5", "D1:2"]) == {"D1:1": 2, "D1:5": 1, "D1:2": None}


def test_wide_ranks_place_each_evidence_turn_in_the_wide_candidate_list():
    from datetime import datetime, timedelta, timezone

    import locomo_eval
    base = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
    turn_at = {base + timedelta(seconds=k): f"D1:{k + 1}" for k in range(5)}
    wide = [(base + timedelta(seconds=3), 1, None), (base + timedelta(seconds=0), 7, 2),
            (base + timedelta(seconds=0), 9, None)]                  # a turn's second piece does not move its place
    assert locomo_eval.wide_ranks(wide, turn_at, ["D1:1", "D1:3"]) == {
        "D1:1": {"rrf": 2, "dense": 7, "keyword": 2}, "D1:3": None}


def test_probe_jev_samples_real_requests_and_reports_which_cross_a_line():
    import brain_eval

    class R:
        def __init__(self, answers):
            self.answers = answers

    class Wobbly:                                 # "near" moves across 0.5 between calls, "clear" does not
        name, calls = "jev", 0

        def ask_many(self, requests):
            Wobbly.calls += 1
            near = 0.45 if Wobbly.calls % 2 else 0.55
            return [R({n: ({"noul": near if "near" in s["text"] else 0.97} if q["type"] == "noul"
                           else {"choice": "a" if Wobbly.calls % 2 else "b"}) for n, q in qs.items()})
                    for s, qs in requests]
    jev = brain_eval.ProbeJev(Wobbly(), per_kind=2)
    noul, choice = {"fits": {"type": "noul", "instructions": "?"}}, {"pick": {"type": "choice", "instructions": "?"}}
    for text in ("near one", "clear two", "near three", "clear four"):
        jev.ask({"text": text}, noul)
    jev.ask({"text": "x"}, choice)
    assert jev.seen == {"fits": 4, "pick": 1} and len(jev.samples["fits"]) == 2      # a sample, not everything
    summary, rows = jev.replay(times=4)
    assert summary["pick"]["pick"] == {"pick_changed": 1.0} and summary["fits"]["seen_in_run"] == 4
    for (state, _), stat in zip(jev.samples["fits"], rows["fits"]):
        assert stat["fits"]["crosses"] == ([0.5] if "near" in state["text"] else [])
    stats = brain_eval.probe_stats(noul, [{"fits": {"noul": 0.28}}, {"fits": {"noul": 0.31}}, {"fits": {"noul": 0.71}}])
    assert stats["fits"]["crosses"] == [0.3, 0.5, 0.7] and (stats["fits"]["min"], stats["fits"]["max"]) == (0.28, 0.71)


def test_a_question_the_judge_left_open_is_not_a_flip():
    from brain_eval import arm_diffs
    rows = [{"a": {"correct": None}, "b": {"correct": True}},
            {"a": {"correct": False}, "b": {"correct": True}},
            {"a": {"correct": True}, "b": {"correct": False}},
            {"a": {"correct": True}, "b": {"correct": True}}]
    assert arm_diffs(rows, "a", "b") == [1, -1, 0]


def test_the_probe_reads_a_score_as_a_number_not_a_pick():
    from brain_eval import probe_stats, probe_summary
    qs = {"a0": {"type": "score"}, "a1": {"type": "score"}}
    repeats = [{"a0": {"score": 1.62}, "a1": {"score": 1.45}}, {"a0": {"score": 1.64}, "a1": {"score": 1.55}}]
    st = probe_stats(qs, repeats)
    assert st["a0"]["crosses"] == [] and st["a1"]["crosses"] == [1.5]
    s = probe_summary({"a0,a1": [st]})["a0,a1"]
    assert s["a0"]["crosses_1.5"] == 0.0 and s["a1"]["crosses_1.5"] == 1.0 and s["a1"]["max_spread"] == 0.1


def test_the_probe_sees_a_pick_that_stays_while_its_probability_crosses_a_gate():
    from brain_eval import probe_stats, probe_summary
    qs = {"which": {"type": "choice"}}
    repeats = [{"which": {"choice": "home", "probabilities": {"home": 0.72, "work": 0.28}}},
               {"which": {"choice": "home", "probabilities": {"home": 0.66, "work": 0.34}}}]
    st = probe_stats(qs, repeats)
    assert st["which"]["changed"] is False and st["which"]["crosses"] == [0.7]
    s = probe_summary({"which": [st]})["which"]["which"]
    assert s["pick_changed"] == 0.0 and s["pick_crosses_0.7"] == 1.0 and s["pick_crosses_0.5"] == 0.0


def test_the_probe_keeps_each_sampled_request_with_its_raw_repeats_and_more_of_a_focus_question():
    from types import SimpleNamespace

    from brain_eval import ProbeJev

    class Jev:
        def ask_many(self, requests):
            return [SimpleNamespace(answers={n: {"type": "noul", "noul": 0.4} for n in q}) for _, q in requests]

    probe = ProbeJev(Jev(), per_kind=2, focus=("fits",), focus_n=5)
    for i in range(8):
        probe.ask({"i": i}, {"fits": {"type": "noul"}})
        probe.ask({"i": i}, {"other": {"type": "noul"}})
    assert len(probe.samples["fits"]) == 5 and len(probe.samples["other"]) == 2
    probe.replay(times=3)
    kept = probe.payloads["fits"][0]
    assert set(kept) == {"state", "questions", "answers"} and len(kept["answers"]) == 3
    assert kept["answers"][0]["fits"]["noul"] == 0.4


def test_note_counts_tell_the_kinds_of_ledger_note_in_a_pack():
    import locomo_eval
    pack = ("[2023-05-01] user: I live in Denver. [memory: later replaced: 'Denver' -> 'Boston' on 2023-06-01]\n"
            "[2023-05-02] user: My dog is Rex. [memory: 'Rex' is disputed by another statement; 'cat' no longer holds]\n"
            "Facts: pet name: Rex (disputed)")
    assert locomo_eval.note_counts(pack) == {"disputed": 1, "replaced": 1, "ended": 1, "disputed_in_blocks": 1}


def test_resolve_replay_counts_what_the_name_rule_would_change():
    import resolve_replay

    def w(mention, cand, score, outcome):
        return {"stage": "write.resolve", "outcome": outcome, "output": {"scores": [score, 0.0]},
                "input": {"mention": {"name": mention, "said_in": "q"},
                          "candidates": [{"name": cand, "also_called": [cand.lower()]}, {"name": "user", "also_called": []}]}}

    def n(mention, ent, outcome):
        return {"stage": "nightly.settle", "outcome": outcome, "output": {"score": 1.0},
                "input": {"mention": {"name": mention, "said_in": "q"}, "entity": {"name": ent, "also_called": []}}}
    logged = [w("Sophia", "Emma", 0.78, "tentative"), w("Sophia", "Emma", 0.78, "tentative"),     # one, logged twice
              w("Rachel", "my best friend", 1.4, "tentative"), w("Max", "Nate", 1.52, "link"),
              n("Sophia", "Emma", "still_tentative"), n("The Nightingale", "user", "still_tentative"),
              n("Maya", "My sister", "settled"),
              {"stage": "write.resolve", "outcome": "link", "input": {"pronoun": "she"}, "output": {"who": "Emma"}}]
    counts, changed = resolve_replay.replay(logged)
    assert counts == {("write.resolve", "tentative", "new (different name)"): 1,
                      ("write.resolve", "tentative", "unchanged"): 1,
                      ("write.resolve", "link", "unchanged"): 1,           # a firm link is left alone
                      ("nightly.settle", "still_tentative", "split (different name)"): 1,
                      ("nightly.settle", "still_tentative", "split (BRAIN_SETTLE_SPLIT only)"): 1,
                      ("nightly.settle", "settled", "unchanged"): 1}
    assert changed[("write.resolve", "new (different name)")] == [(0.78, "Sophia", "Emma", "q")]


def _reader_mock(stop="end_turn", text=" Seven. "):
    import json

    import httpx
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"model": seen[-1]["model"], "stop_reason": stop,
                                         "content": [{"type": "text", "text": text}] if text else [],
                                         "usage": {"input_tokens": 1000, "output_tokens": 100}})
    return httpx.Client(transport=httpx.MockTransport(handle)), seen


def test_readers_send_each_model_its_own_thinking_off_setting_and_price():
    import readers
    from brain.spend import MemorySpendStore, SpendMeter
    http, seen = _reader_mock()
    meter = SpendMeter(MemorySpendStore(), 5.0)
    s = readers.Reader("sonnet", meter, "k", http=http)
    c = s.complete("You answer.", "Q?", 400)
    assert (c.text, c.model, c.stop_reason) == ("Seven.", "claude-sonnet-5-5", "end_turn")
    assert seen[0] == {"model": "claude-sonnet-5-5", "max_tokens": 400, "thinking": {"type": "between_tools"},
                       "messages": [{"role": "user", "content": "Q?"}], "system": "You answer."}
    assert abs(meter.spent_today() - (1000 * 2.0 + 100 * 10.0) / 1e6) < 1e-12
    s.complete("You answer.", "Q?", 400)                   # the same prompt is read once
    assert len(seen) == 1
    readers.Reader("haiku", meter, "k", http=http).complete("", "Q?", 400)
    assert seen[1] == {"model": "claude-haiku-5-5", "max_tokens": 400, "thinking": {"type": "disabled"},
                       "messages": [{"role": "user", "content": "Q?"}]}
    readers.Reader("opus", meter, "k", http=http).complete("", "Q?", 400)
    assert seen[2]["max_tokens"] == 400 + readers.OPUS_EXTRA and seen[2]["output_config"] == {"effort": "low"}
    assert "thinking" not in seen[2] and all("temperature" not in b for b in seen)


def test_a_refusal_is_an_empty_answer_that_is_counted():
    import readers
    from brain.spend import MemorySpendStore, SpendMeter
    http, _ = _reader_mock(stop="refusal", text="")
    r = readers.Reader("sonnet", SpendMeter(MemorySpendStore(), 5.0), "k", http=http)
    c = r.complete("", "Q?", 400)
    assert (c.text, c.stop_reason, r.refusals) == ("", "refusal", 1)


def test_reader_swap_builds_its_prompts_from_the_stored_packs_and_counts_the_flips():
    import reader_swap
    pack = "Excerpts:\n[2023-05-01] user: I have 7 fish.\nMemory ledger: x\n\nToday's date: 2023-06-01\nQuestion: How many fish?\nAnswer:"
    assert reader_swap.no_memory_prompt(pack) == ("Excerpts:\n(no excerpts)\n\nToday's date: 2023-06-01\n"
                                                  "Question: How many fish?\nAnswer:")
    rows = [{"qid": "a", "cat": 1, "brain": {"context_sha": "s1", "correct": False}},
            {"qid": "b", "cat": 1, "brain": {"context_sha": "s1", "correct": True}}]
    todo = reader_swap.jobs(rows, {"s1": pack}, ["brain", "none"])
    assert [(i, a) for i, a, _, _ in todo] == [(0, "brain"), (0, "none"), (1, "brain"), (1, "none")]
    assert todo[0][2] == pack and "(no excerpts)" in todo[1][2]
    rows[0]["brain@sonnet"] = {"answer": "7", "correct": True}
    rows[1]["brain@sonnet"] = {"answer": "", "correct": False, "stop_reason": "refusal"}
    rows[0]["none@sonnet"] = rows[1]["none@sonnet"] = {"answer": "UNKNOWN - nothing given", "correct": False}
    got = reader_swap.summarise(rows, ["brain", "none"], "sonnet")
    assert (got["brain"]["haiku"], got["brain"]["sonnet"], got["brain"]["gained"], got["brain"]["lost"]) == (1, 1, 1, 1)
    assert got["brain"]["refused"] == 1 and got["none"]["said_unknown"] == 2 and got["none"]["sonnet"] == 0
    assert got["brain"]["by_category"] == {"1": {"n": 2, "haiku": 1, "sonnet": 1}}
    lme = [{"qid": "x", "qtype": "multi-session", "near1seats": {"context_sha": "s1", "correct": True, "style": "preference"},
            "near1seats@sonnet": {"answer": "7", "correct": True}}]
    assert reader_swap.jobs(lme, {"s1": pack}, ["near1seats", "none"])[1][3] == "preference"     # the style goes with it
    assert reader_swap.summarise(lme, ["near1seats"], "sonnet")["near1seats"]["by_category"] == {
        "multi-session": {"n": 1, "haiku": 1, "sonnet": 1}}


def test_date_direction_replay_picks_month_phrases_without_a_year_and_counts_the_changes():
    import date_direction_replay as ddr
    got = [p.lower() for p in ddr.yearless("I went in November, and again on March 3rd, 2023. Back in May 2021 "
                                           "and on 5/3/22 it rained; yesterday too. My exam is on June 24th.")]
    assert got == ["in november", "june 24th"]
    c = ddr.compare([["2023-11-01", "2023-11-30"], ["2023-06-24", "2023-06-24"], None, ["2023-05-01", "2023-05-31"]],
                    [["2022-11-01", "2022-11-30"], ["2023-06-24", "2023-06-24"], ["2023-01-01", "2023-01-31"], None])
    assert c == {"changed": 1, "same": 1, "resolved only with": 1, "resolved only without": 1}


def test_lenient_judge_reads_one_label_and_counts_where_the_two_judges_differ():
    import lenient_judge as rejudge
    assert rejudge.parse_label('It names the same necklace. {"label": "CORRECT"}') is True
    assert rejudge.parse_label('{"label": "WRONG"}') is False
    assert rejudge.parse_label("correct") is True
    assert rejudge.parse_label("Not CORRECT, so WRONG") is None      # both words: never counted correct
    assert rejudge.parse_label("INCORRECT") is None and rejudge.parse_label("") is None and rejudge.parse_label(None) is None
    prompt = rejudge.lenient_user("Where to?", "Lisbon", "To {Lisbon}")
    assert prompt.endswith('with the key as "label".') and "Gold answer: Lisbon\nGenerated answer: To {Lisbon}" in prompt
    rows = [{"cat": "temporal", "a": {"correct": True, "lenient": True}},
            {"cat": "temporal", "a": {"correct": False, "lenient": True}},
            {"cat": "single-hop", "a": {"correct": True, "lenient": None}},
            {"cat": "single-hop", "a": {"correct": None, "lenient": False}},
            {"cat": "single-hop", "a": {"correct": True}}]                 # not marked a second time: left out
    assert rejudge.summarise(rows, ["a"]) == {"a": {
        "n": 4, "ours": 2, "lenient": 2, "no_label": 1, "lenient_only": 1, "ours_only": 1,
        "by_category": {"single-hop": {"n": 2, "ours": 1, "lenient": 0}, "temporal": {"n": 2, "ours": 1, "lenient": 2}}}}


def test_a_stopped_reader_swap_keeps_the_answers_it_judged(tmp_path):
    import reader_swap
    path = tmp_path / "cache" / "partial.jsonl"
    assert reader_swap.saved_load(path) == ({}, 0.0)
    reader_swap.saved_add(path, "start-1", "a", "brain", "pack one", {"answer": "7", "correct": True}, 0.1)
    reader_swap.saved_add(path, "start-1", "a", "none", "no pack", {"answer": "?", "correct": False}, 0.3)
    reader_swap.saved_add(path, "start-2", "b", "brain", "pack two", {"answer": "9", "correct": None}, 0.2)
    done, spent = reader_swap.saved_load(path)
    assert spent == 0.5                                              # each start has its own meter
    assert done[("a", "brain", reader_swap._sha("pack one"))] == {"answer": "7", "correct": True}
    assert ("a", "brain", reader_swap._sha("another pack")) not in done    # an answer goes with the prompt it read
    assert len(done) == 3
    # the key is the run, the reader and its request, not the arms: a later start may ask for more arms
    k = reader_swap.swap_key("locomo_eval_x", "sonnet", {"thinking": {"type": "adaptive"}})
    assert k == reader_swap.swap_key("locomo_eval_x", "sonnet", {"thinking": {"type": "adaptive"}})
    assert k != reader_swap.swap_key("locomo_eval_y", "sonnet", {"thinking": {"type": "adaptive"}})
    assert k != reader_swap.swap_key("locomo_eval_x", "sonnet", {})
