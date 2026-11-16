"""scripts/locomo_eval.py: a relative gold answer as the dates it means, for the judge. No keys."""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "live-eval"))


def test_gold_dates_turns_a_phrase_before_a_date_into_calendar_dates():
    """exp-judge-dates-v1: the judge marked "2022-06-17" wrong against "The Friday before 24June, 2022"."""
    from locomo_eval import gold_dates
    assert gold_dates("The Friday before 24June, 2022.") == (date(2022, 6, 17), date(2022, 6, 17))
    assert gold_dates("Friday before 13 March, 2023") == (date(2023, 3, 10), date(2023, 3, 10))
    assert gold_dates("The weekend before 17 July 2023") == (date(2023, 7, 15), date(2023, 7, 16))     # said on a Monday
    assert gold_dates("on the weekend before March 26, 2023") == (date(2023, 3, 18), date(2023, 3, 19))  # on a Sunday
    # "the week before": the calendar week before and the seven days before are both accepted
    assert gold_dates("week before August 7, 2023.") == (date(2023, 7, 31), date(2023, 8, 6))
    assert gold_dates("The week before 9 June 2023") == (date(2023, 5, 29), date(2023, 6, 8))
    assert gold_dates("Two weeks before 11 August 2023") == (date(2023, 7, 24), date(2023, 7, 30))
    assert gold_dates("few days before 17 December, 2023.") == (date(2023, 12, 10), date(2023, 12, 16))
    for other in ("September 2023", "first week of June 2023", "19 days", "Boston", "21 October 2023",
                  "The Friday before 31 Smarch 2023", "The Friday before 31 February 2023"):
        assert gold_dates(other) is None


def test_dated_verdict_lets_code_compare_the_date_the_model_picked():
    """exp-judge-dates-v2: handed the gold dates, the judge still marked dates inside them wrong (v1); now the
    model only picks the answer's date and code compares."""
    from locomo_eval import dated_verdict, parse_span
    assert parse_span("2022-06-17") == (date(2022, 6, 17), date(2022, 6, 17))
    assert parse_span("2023-05-29 to 2023-06-04") == (date(2023, 5, 29), date(2023, 6, 4))
    for bad in ("NONE", None, "", "2023-06-04 to 2023-05-29", "2023-02-30", "2023-01-01, 2023-02-01 or 2023-03-01"):
        assert parse_span(bad) is None
    friday = "The Friday before 24June, 2022."
    assert dated_verdict(friday, ["2022-06-17"] * 3) is True
    assert dated_verdict(friday, ["2022-06-24", "2022-06-24", "2022-06-17"]) is False
    week = "The week before 9 June 2023"
    assert dated_verdict(week, ["2023-05-29 to 2023-06-04", "2023-06-02", "NONE"]) is True
    assert dated_verdict(week, ["2023-06-09"] * 3) is False                      # the session date itself
    assert dated_verdict("The weekend before 17 July 2023", ["2023-07-08 to 2023-07-16"] * 3) is False
    assert dated_verdict(week, ["NONE", "NONE", "2023-06-02"]) is None           # no date: the usual judge decides
    assert dated_verdict("September 2023", ["2023-09-01"] * 3) is None           # not a phrase before a date


def test_an_ablated_arm_takes_a_provider_down_the_way_an_outage_does():
    """exp-ablate-providers-v1: the arms that measure what Jev's rerank and the vectors add. Jev down raises
    ProviderError on any call; the embedder gives no vector for a question but still embeds what is stored; the
    providers handed to the other arms are untouched."""
    import dataclasses
    import pytest
    from brain.providers import ProviderError
    from locomo_eval import ABLATE, ARM_LABELS, ablated

    @dataclasses.dataclass
    class P:
        embedder: object
        jev: object
        haiku: object

    class Embedder:
        def embed_query(self, q):
            return [1.0]

        def embed_documents(self, docs):
            return [[2.0] for _ in docs]

    class Jev:
        def ask_many(self, items):
            return ["answer"]

    prov = P(Embedder(), Jev(), "haiku")
    assert ablated(prov, ()) is prov
    no_jev = ablated(prov, ABLATE["flat_norerank"])
    with pytest.raises(ProviderError):
        no_jev.jev.ask_many([])
    with pytest.raises(ProviderError):
        no_jev.jev.ask({}, {})
    assert no_jev.embedder.embed_query("q") == [1.0] and no_jev.haiku == "haiku"
    no_vec = ablated(prov, ABLATE["flat_kw"])
    with pytest.raises(ProviderError):
        no_vec.embedder.embed_query("q")
    assert no_vec.embedder.embed_documents(["a"]) == [[2.0]] and no_vec.jev.ask_many([]) == ["answer"]
    both = ablated(prov, ABLATE["flat_kw_norerank"])
    with pytest.raises(ProviderError):
        both.jev.ask_many([])
    with pytest.raises(ProviderError):
        both.embedder.embed_query("q")
    assert prov.jev.ask_many([]) == ["answer"] and prov.embedder.embed_query("q") == [1.0]     # untouched
    assert set(ABLATE) <= set(ARM_LABELS)
