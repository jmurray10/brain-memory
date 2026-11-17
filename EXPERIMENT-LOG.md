# Experiment log

This is a reader's record of every experiment run on the brain, including the ones that failed. Each bar was
written down before its run. It is a summary of a longer working log, in roughly the order the work was done.
"Met" and "not met" refer to the bar as written. "No gain shown" means the run finished and the difference was
inside what two identical runs produce. Differences of a few questions are inside run-to-run noise: two arms with
the same settings disagreed on about 7% of questions until a memo was added. Unless it says otherwise, Haiku
(thinking off) plays the user's model and the judge is Haiku, majority of three. [RESULTS.md](RESULTS.md) gives the
headline numbers, [DESIGN.md](DESIGN.md) the design, and [REFERENCE.md](REFERENCE.md) the settings named here.

Arm names: "retrieval only" is the service's own search with the ledger and routes off. "Full brain" adds the
ledger's facts, counts, dates, preferences and notes. "Shipped pack" is the full brain with the settings on at
that date.

## The evaluation harness and its noise

**Perfect retrieval against flat retrieval** (exp-lme100-v1, 2026-11-06)
- Question: On LongMemEval S, is finding the evidence the gap, or reading it?
- Bar set before the run: the run invariants (perfect retrieval scores at least as well as retrieval, and so on).
- Result: 100 questions. Perfect retrieval 0.85 (95% interval 0.78 to 0.92), flat retrieval with Jev rerank 0.84
  (0.77 to 0.91); paired difference +0.01 (-0.06 to +0.08). A first single-pass grading made the run invalid.
- Outcome: valid baseline. Grading became a majority of three verdicts for every later run.

**The noise floor of the answer path** (exp-noise-v1, 2026-11-09)
- Question: How often do two identical arms disagree?
- Bar set before the run: none; a measurement. Later gates had to clear half the flip rate per type.
- Result: 100 LongMemEval questions, 25 per type. The arms scored 0.66 and 0.69 and disagreed on 12% of
  questions (knowledge-update 0.08, multi-session 0.08, preference 0.16, temporal 0.16).
- Outcome: used as the noise threshold for the slice gates below.

**Three reads per answer** (exp-reads-v1, 2026-11-10)
- Question: Does a majority of three reads remove the noise between identical arms?
- Bar set before the run: the majority flip rate is at most half the single-read rate, with at least 15 flips.
- Result: 449 LoCoMo questions, full brain against its twin. Single-read flips 33 (0.073), majority-of-three
  flips 30 (0.067). Not met. The two identical arms were handed a different context on 276 of 449 questions.
- Outcome: dropped. Three reads are not the standard.

**One answer per distinct request within a run** (exp-memo-v1, 2026-11-10)
- Question: Does a memo of Jev and reader answers make identical arms agree?
- Bar set before the run: at most 4 flips of 361 between twins; the same context on at least 95% of questions.
- Result: 0 flips; the same context on 361 of 361. Met.
- Outcome: the memo is the standard for runs that compare arms inside one run. It does not make one arm's
  accuracy more exact and does not carry between runs.

**Which Jev decisions change between identical calls** (exp-jev-probe-v1 and exp-jev-probe-v2, 2026-11-11)
- Question: Which of the brain's typed decisions flip when the same request is asked again?
- Bar set before the run: a decision is unstable if more than 5% of sampled requests cross its line or change pick.
- Result: confirmed unstable in both runs: slot fits (11.7% then 16.7% of 60), slot pick (7.8% then 12.8% of 180),
  slot cardinality (6.7% then 10.0% of 120), relation to an existing claim (6.9% of 1,103 then 5.7% of 1,527).
  Four findings that rested on 60 requests did not repeat. Rerank, both injection screens and the claim checks
  were stable in both runs.
- Outcome: led to exp-ledger-stable-v1. The router's hedge gate was found to be unread code.

**Do wrong answers lack the evidence or have it** (exp-evidence-v1, 2026-11-10 and 2026-11-11)
- Question: When an answer is wrong, were the evidence turns among the excerpts?
- Bar set before the run: none; a measurement. A first run was void because Jev stopped answering partway.
- Result: 360 LoCoMo questions. All evidence in the excerpts on 274 (76%): retrieval only 0.854, full brain 0.850.
  Some on 50: 0.520 and 0.580. None on 36: 0.250 and 0.417. Of the full brain's 83 wrong answers, 41 had all
  their evidence and 42 did not.
- Outcome: the loss is split between retrieval and reading. The ledger helped where retrieval missed.

**A date-aware judge for LoCoMo** (exp-judge-dates-v1 and exp-judge-dates-v2, 2026-11-12)
- Question: Can the judge be given the dates a relative gold answer means, so a correct date is not marked wrong?
- Bar set before the run: v1, the judge agrees with a code check on at least 90% of dated answers; v2 (the model
  picks the date, code compares), at most 4 of 84 verdicts wrong on a hand reading.
- Result: v1 agreed on 34 of 84. Not met. v2 had 6 wrong and 1 doubtful of 84. Not met. By the same reading the
  stored judge was wrong on at least 13 of the 84.
- Outcome: dropped. Neither judge was adopted and the stored judge stays the one of record.

**What a 45-question guard can tell** (guard-noise-reading-v1, 2026-11-12, no paid run)
- Question: How much does one unchanged configuration move between three captures of the same 45 questions?
- Bar set before the run: none; a reading of stored rows.
- Result: 37 right each time, but 7 of the 45 questions did not get the same mark in all three runs. An unchanged
  arm moved 4 to 6 answers between two runs.
- Outcome: the gate "at most 1 net lost of 45" stands for the two experiments that used it and is not reused.

## Retrieval

**The service's recall against the test harness** (exp-service-m5-parity, 2026-11-08)
- Question: Does the service's own recall find the same evidence as the harness it was designed in?
- Bar set before the run: recall of all evidence sessions and turn recall each within 0.03 of the harness.
- Result: 70 scored LongMemEval questions. All evidence sessions kept on 0.986 (69 of 70) against 0.96; turn
  recall 0.904 against 0.91. Met.
- Outcome: build step 1 met.

**Keyword side on short conversational turns** (the LoCoMo keyword parity check, 2026-11-10)
- Question: Is Postgres keyword ranking worse than BM25 on LoCoMo?
- Bar set before the run: both recall measures within 0.03 of the harness.
- Result: 300 questions. All evidence kept 0.807 against 0.780 (+0.027); turn recall 0.797 against 0.765 (+0.032).
  Not met as written: the gate is two-sided and the service is over it by 0.002, in the service's favour.
- Outcome: recorded as not met. The service's keyword side is not worse.

**The best passage of each session first** (exp-session-slots-v1, 2026-11-09)
- Question: Does reserving a place per session keep more evidence sessions for multi-session questions?
- Bar set before the run: multi-session accuracy not lower and evidence recall higher; LoCoMo no worse than -0.02.
- Result: 20 LongMemEval questions. Retrieval only 12 of 20 and full brain 15 of 20, both with the flag off and
  on. Evidence recall could not be measured by that run. Not met. The LoCoMo half was not run.
- Outcome: built and off (`BRAIN_SESSION_SLOTS`).

**Where the missing evidence stands** (exp-more-excerpts-v1, two spikes, 2026-11-11)
- Question: Is evidence that misses the pack just below the cut, or never a candidate?
- Bar set before the run: none; two spikes.
- Result: 360 LoCoMo questions, 578 evidence turns: 426 in the top ten, 21 at 11 to 20, 4 at 21 to 30, 127 never
  among the 30 candidates. Questions with all evidence at k = 10 / 15 / 20 / 30: 276 / 280 / 283 / 284. Of the 127,
  71 were already in the dense top 60 and 39 sat within two turns of an excerpt that was handed over.
- Outcome: a larger k is not the fix. Led to neighbouring turns and to the shortlist seats.

**The same ten excerpts when asked twice** (exp-stable-pack-v1, v2 and v3, 2026-11-12)
- Question: Does cutting rerank scores into steps give the same ten excerpts on a repeat ask, at no cost?
- Bar set before the run: v1, steps of 0.1: same ten at least 20 points above shipped on an unused conversation.
  v2, steps of 0.2: at least 15 points, and evidence turns not more than 1% below shipped. Answer check: at most 4
  net lost of 361 on LoCoMo. Guard: at most 1 net lost of 45 on LongMemEval.
- Result: shipped gives the same ten on 42% to 63% of repeat asks over three conversations. v1: 75% against 63%,
  12 points. Not met. v2: 79% against 50%, but 480 evidence turns against 492, 2.4% below. Not met. The answer
  check for 0.1 was then run on the owner's word: 280 against 280 of 361. Met. Guard: 35 against 37 of 45, 2 net
  lost. Not met.
- Outcome: built and off (`BRAIN_RERANK_BAND`). The instability is recorded as a known property.

**Seats for the dense and keyword lists** (exp-shortlist-v1, 2026-11-12)
- Question: Do more evidence turns reach the reader if the 30 candidates are the dense top 25 and keyword top 5?
- Bar set before the run: G1, at least 10 more evidence turns handed over. G2, at least +3 net answers of 361, no
  category losing more than 3. G3, at most 1 net lost of 45 on LongMemEval. G4 (added before its run, on 131
  held-out questions): at most 3 net lost and no fewer evidence sessions reached.
- Result: G1 +15 (453 to 468 of 578). Met. G2 277 to 285 (23 gained, 15 lost). Met. G3 37 to 34 of 45. Not met.
  G4 99 to 102 of 131 (6 gained, 3 lost), evidence sessions 278 and 281 of 299. Met. No gain shown on LongMemEval;
  the LoCoMo gain is in-sample for the seat counts.
- Outcome: shipped on at "25,5" (`BRAIN_SHORTLIST_SEATS`), with G3 recorded as not met.

**A wider shortlist and twenty excerpts** (exp-wide-pack-v1, 2026-11-13)
- Question: Does letting Jev read 45 candidates and handing over 20 excerpts give more right answers?
- Bar set before the run: G1, all evidence on at least 15 more questions. G2, Haiku's and Sonnet's correct answers
  together at least 12 higher, neither reader more than 5 lower.
- Result: 361 LoCoMo questions. G1: 322 against 306, 16 more. Met. G2: Haiku 284 against 290, 6 lower. Not met.
  Sonnet's read stopped at its cost cap and was not rerun.
- Outcome: built and off (`BRAIN_PACK_WIDE`). More evidence reached the reader and the weaker reader did not use it.

## The router

**Router v1, scored by question type** (exp-route-v1, 2026-11-08)
- Question: Can one Jev request send a question to the right store?
- Bar set before the run: accuracy at least 0.85; confident errors at most 8%; every type at least 0.70; above the
  keyword baseline.
- Result: 420 LongMemEval test questions. Accuracy 0.833, confident errors 14.3%, lowest type 0.64, keyword
  baseline 0.721. Not met.
- Outcome: superseded by exp-route-v2. The question type proved a weak proxy for the right route.

**Router v2, scored by per-question labels** (exp-route-v2, 2026-11-08)
- Question: Does a router built on the function-calling pattern pass against labels written blind?
- Bar set before the run: the same four checks against the labels; v2 replaces v1 only if it is at least as good
  as v1 rescored (0.898, confident errors 4.5%).
- Result: 420 questions. Accuracy 0.943, confident errors 0.7% (3), lowest type 0.86, keyword baseline 0.786. Met.
- Outcome: shipped on. Build step 2 met.

**The router outside LongMemEval** (exp-route-domains, 2026-11-08)
- Question: Does router v2, unchanged, hold on LoCoMo and on questions from 14 professional fields?
- Bar set before the run: per set, the same four checks (every route with at least 15 questions at least 0.70).
- Result: LoCoMo sample, 301 questions: 0.884, confident errors 1.7%, keyword baseline 0.880. Complex domains, 280
  questions: 0.954, confident errors 1.1%, keyword baseline 0.539. Both met. The domain labels agreed with the
  question writer on every question, so 0.954 is likely optimistic.
- Outcome: router v2 kept unchanged.

**A small trained model in front of Jev** (exp-booster-route-shadow, 2026-11-08)
- A small trained model in front of Jev was tried on synthetic data, did not hold up on real questions, and is off.

## The write path: claims, entities, slots, relations

**Claim checks with a weak client** (exp-claims-v1, 2026-11-09)
- Question: Of the claims the brain keeps from a weak client, how many does the turn really support?
- Bar set before the run: at least 0.95 of kept claims judged supported by a stronger model, blind to the checks.
- Result: 180 turns. 181 of 192 audited claims supported, 0.943 (Wilson 95% 0.900 to 0.968). Not met. The client
  sent 325 claims on 90 turns; code dropped 34 and Jev 25, of which 9 were judged to have been good.
- Outcome: recorded as not met, and the setting stayed off at the time. The write path runs in the reference
  deployment today (`WRITE_PATH_ENABLED`); this bar was not measured again. A rerun with a stronger client
  (exp-claims-v2) was withdrawn before any result, to keep a low-end client as the yardstick.

**Are page lines supported by their turns** (exp-page-audit-v1, 2026-11-10)
- Question: Does a rendered page line state what its source turn says?
- Bar set before the run: at least 0.95 of sampled lines supported, with a hand check of 20 lines.
- Result: 3,791 of 3,983 lines supported, 0.952 (Wilson 0.945 to 0.958). Met, narrowly. Hand check agreed with Jev
  on 19 of 20. The ten failures read by hand were all claims filed under a slot that only resembled their own.
- Outcome: build step 8 met with the place check below; without it the same audit later gave 0.9465.

**A support check before a claim joins another slot** (exp-place-check-v1, 2026-11-10)
- Question: Does asking "does the turn support this slot" stop claims being filed under a look-alike slot?
- Bar set before the run: page support at least 0.97 and above the flag-off arm; accuracy on 20 questions not
  lower by more than one; values ended fall by no more than 15%.
- Result: support 0.9892 (3,935 of 3,978) against 0.9465 (3,768 of 3,981); full brain 20 against 19 of 20; values
  ended 54 against 52. All three met. The check refused 172 of 1,172 placements.
- Outcome: shipped on (`BRAIN_PLACE_CHECK`).

**An "adds" no longer replaces a value** (exp-supersede-v1, 2026-11-09)
- Question: If only "updates" and "retracts" end a value, do the notes stop costing and do updates still work?
- Bar set before the run: full brain minus the no-notes arm at least 0 on LoCoMo; knowledge-update at least 0.95
  on the seen 20; on 60 unseen LongMemEval questions no type's interval wholly below 0.
- Result: LoCoMo, 291 questions: -0.017. Not met. Seen 20: 0.95. Met. Unseen 60: retrieval only 0.750, full brain
  0.780, +0.017 (-0.067 to +0.100). Met. No gain shown.
- Outcome: the rule is kept: an "adds" never ends a value.

**One question per slot** (exp-ledger-stable-v1, 2026-11-11)
- Question: Can slot placement be made repeatable without harming answers?
- Bar set before the run: G1, at least halve the share of requests whose action differs between repeats. G2, no
  harm: LongMemEval at least the flag-off arm minus 0.02 with knowledge-update losing at most one; LoCoMo at least
  minus 0.01.
- Result: G1: placement differs on 12.0% of repeats as built and 3.7% with one question per slot. Met. G2:
  LongMemEval 47 against 45 of 60 (+0.033), knowledge-update 18 of 20 both ways; LoCoMo 274 against 271 of 361
  (+0.008). Met. No gain shown. No rule was found for cardinality; the relation decision needed no change.
- Outcome: shipped on (`BRAIN_PLACE_PER_SLOT`).

**End a value only when it stopped being true** (exp-relation-ends-v1, 2026-11-11)
- Question: Does a second question ("has the older value stopped being true?") stop wrong endings?
- Bar set before the run: G1, values ended fall by at least half on both benchmarks. G2, no harm: LoCoMo at least
  -0.01; LongMemEval at least -0.02 with knowledge-update losing at most one.
- Result: values ended 62 to 6 on LoCoMo and 53 to 14 on LongMemEval. Met. Answers 277 against 277 of 361 and 31
  against 31 of 45, knowledge-update 14 of 15 both ways. Met. No gain shown, as stated before the runs.
- Outcome: shipped on (`BRAIN_RELATE_ENDS`). The check misses some real changes, which then stay beside the new value.

**Different proper names are not joined on a "maybe"** (exp-resolve-names-v1, 2026-11-12)
- Question: Does refusing tentative links between two different proper names stop one person's facts being filed
  under another?
- Bar set before the run: the rule reaches the write path, and the lookup no longer hands the wrong person's row
  on the question the fault was read from. Answers reported, not gated.
- Result: 15 LongMemEval questions, two ingests. Four decisions logged under the rule; the lookup fault was gone.
  Met. Answers 12 against 11 of 15, inside chance. On all logged runs the rule would change 96 of 1,654 tentative
  links, about 92 of them joining two different people or things on a hand reading.
- Outcome: shipped on (`BRAIN_RESOLVE_NAMES`). A wider rule that splits every unconfirmed link was read for free
  (settle-split-reading-v1, 596 links), judged not worth a paid run, and is built and off (`BRAIN_SETTLE_SPLIT`).

**A contradiction is a dispute only when the older value ended** (exp-relate-disputes-v1, 2026-11-12 and 2026-11-13)
- Question: Does gating "contradicts" on the same second question remove wrong "disputed" marks and help answers?
- Bar set before the run: on LoCoMo, marks and contested claims fall and released pairs are mostly things that can
  both be true; at most 8 net lost of 361 for each reader; answers on marked questions reported.
- Result: marks 35 to 0, contested claims 8 to 2; 3 of 5 released pairs can both be true. Met narrowly. Haiku 285
  and 285, Sonnet 305 and 303. Met. On the 19 marked questions Haiku 14 and 14, Sonnet 15 and 15. No gain shown.
- Outcome: built and off (`BRAIN_RELATE_DISPUTES`). Two of every five pairs it released were real conflicts.

**The step 5 gate on a larger held-out set** (exp-step5-gate-v2, 2026-11-10)
- Question: On knowledge-update questions, does the full brain beat retrieval only?
- Bar set before the run: the 95% interval of full brain minus retrieval only lies above 0.
- Result: 46 questions. 0.870 against 0.870, difference 0.000 (-0.109 to +0.109), 3 flips each way. Not met.
- Outcome: step 5 stays on with no gain shown on its gate.

## The upper layers against retrieval alone

**The ledger, counts and preferences by question type** (exp-slices-v1 and exp-slices-v2, 2026-11-09)
- Question: Does the full brain beat retrieval only on the LongMemEval types each build step targets?
- Bar set before the run: per type, the difference at or above the noise threshold with an interval excluding 0.
- Result: 20 questions per type on the built brain. Knowledge-update 1.00 against 0.90 (0.00 to +0.25);
  multi-session 0.60 against 0.60; temporal 0.80 against 0.75; assistant 1.00 against 1.00. No gain shown on any.
  Preference 0.37 against 0.60, -0.25 (-0.45 to -0.10): not met, a regression.
- Outcome: the preference block was changed to follow the excerpts as support. A post-hoc rerun gave 0.50 against
  0.50, which is not a clean test. Steps 5, 6 and 7 are on with no gain shown.

**The full brain on LoCoMo** (exp-locomo-brain-v1 and exp-locomo-brain-v2, 2026-11-09)
- Question: Does the full brain regress against retrieval only on LoCoMo?
- Bar set before the run: full brain minus retrieval only no worse than -0.06 overall; in v2 also no significant
  single-hop regression.
- Result: v1, 400 questions: 0.734 against 0.762, -0.028 (-0.06 to +0.01). Met, with single-hop a significant
  regression (-0.058). v2, the next 400 unseen: 0.717 against 0.727, -0.010; single-hop -0.024 (-0.07 to +0.03).
  Met. No gain shown.
- Outcome: side lookups made stricter. The result reads as parity with retrieval, not as the fixes working.

## Dates

**A date block and the when route** (exp-when-v1, 2026-11-08 and 2026-11-09)
- Question: Does a block of resolved dates and day counts help date questions?
- Bar set before the run: when arm minus open arm at least +0.03 on temporal questions; not more than 0.02 below
  on 40 guard questions.
- Result: 111 temporal questions: 0.631 against 0.667, -0.036 (-0.117 to +0.045). Guard: 0.900 against 0.950.
  Not met on either.
- Outcome: built and off (`BRAIN_WHEN_ROUTE`).

**The resolved date written beside its phrase** (exp-when-locomo-v1, 2026-11-11 and 2026-11-12)
- Question: Does writing the computed date range beside a time phrase in the excerpt help date answers?
- Bar set before the run: G1, at least +4 on 78 temporal LoCoMo questions and +4 over all 361. G2, at most 2 net
  lost outside temporal. G3, LongMemEval guard: at least -0.02 over 45, temporal losing at most one net.
- Result: G1: +1 and +1 (59 against 58; 280 against 279). Not met. G2: 6 gained, 6 lost. Met. G3, run on the
  owner's word although G1 was not met: 37 against 35 of 45. Met. On a hand reading of the 21 questions with a
  relative gold answer the notes were right on 20 against 16.
- Outcome: not adopted by the rule as written. Turned on by the owner's decision on the hand-read evidence
  (`BRAIN_DATES`, `BRAIN_DATE_NOTES`).

**The year of a date said without one** (exp-date-direction-v1, 2026-11-12 and 2026-11-13)
- Question: Can the wording (happened or planned) decide the year when none is written?
- Bar set before the run: T1, past events dated into the future are fixed and no plan is moved into the past, by
  hand reading. T2, at most 4 net lost for each reader on 131 held-out questions.
- Result: a first capture was void (Jev stopped answering). The first version of the rule failed T1 on 600
  sentences. The second version, on 460 sentences held back: 43 changed, all in line with the wording (a margin added afterwards
  takes 4 of them back, leaving 39). Met. T2:
  Haiku 105 against 107, Sonnet 118 against 120. Met. No answer turned on a changed date.
- Outcome: shipped on (`BRAIN_DATE_DIRECTION`).

**A written count of days, weeks or months** (exp-date-counted-v1 and exp-date-counted-v2, 2026-11-13)
- Question: Can code read "two weeks ago" when Jev agrees on the unit but is unsure of the offset?
- Bar set before the run: T1, at least 95% of resolved sentences right by hand and at least 20 newly resolved.
  T2, no date that was right becomes wrong.
- Result: v1, 300 sentences: 75 of 79 right (94.9%) and two right dates made wrong. Not met. v2 leaves "in N
  units" to Jev; on 226 unseen sentences it dated 54 that had no date, 52 right and 2 doubtful (96.3% counting both
  as wrong), and changed none. Met.
- Outcome: v2 shipped on (`BRAIN_DATE_COUNTED`). Answers on a benchmark were not measured.

## The context pack

**Neighbouring turns** (exp-more-excerpts-v1, 2026-11-11)
- Question: Does adding the turn before and after each excerpt complete the evidence and help answers?
- Bar set before the run: for two turns each side: evidence complete on at least 20 more questions; answers at
  least +0.01; at least -0.01 where nothing was missing. LongMemEval guard: at least -0.02 on 60 questions.
- Result: 361 LoCoMo questions. Full brain 0.756, one turn each side 0.803, two turns 0.797. Two turns: 28 more
  complete, +0.042, +0.036. Met. One turn met its own bars (13 more against 10). Guard: +0.017 and +0.033. Met,
  with no gain shown on LongMemEval. Most of the LoCoMo gain came on questions that already had their evidence.
- Outcome: shipped on at one turn each side (`BRAIN_NEIGHBOURS`).

**The open route with less of the ledger** (exp-open-route-v1 and exp-open-route-v2, 2026-11-10)
- Question: On single-detail questions, do answers improve if the ledger's blocks and notes are left out?
- Bar set before the run: the plain arm at least +0.01 over the full brain, pooled over two LoCoMo runs.
- Result: v1, 447 questions: +0.004. Not met. v2, with the memo, 449 questions: 0.744 against 0.744, 13 flips each
  way. Not met.
- Outcome: dropped as a default. The switch stays in the code (`BRAIN_OPEN_ROUTE`).

**The memory check** (exp-memory-check-v1 and exp-memory-check-v1b, 2026-11-09)
- Question: Does an "is this enough?" line in the pack help abstention without costing answers?
- Bar set before the run: the check stays on only if it does not cost LoCoMo. For the setting without it: full
  brain no worse than -0.06 against retrieval only; abstention at least 0.90.
- Result: LoCoMo, 361 questions: 0.723 with the check, 0.765 without, -0.042 (-0.072 to -0.014). Abstention, 30
  questions: 0.933 with, 0.967 without. The check cost answers and did not help abstention. Without it, on 291
  LoCoMo questions: full brain 0.770, retrieval only 0.729; abstention 0.967. Met.
- Outcome: built and off (`BRAIN_MEMORY_CHECK`). The first run's switch also removed the notes; v1b separated them.

**The wording of the ledger's notes** (exp-ledger-wording-v1, 2026-11-11)
- Question: Do softer notes on excerpts, or none, beat the notes as built?
- Bar set before the run: at least +0.01 (4 questions net of 361) and more gained than lost on packs with a note.
- Result: a spike without neighbouring turns found notes on 102 of 361 packs and 1 answer gained against 8 lost.
  In the shipped pack: full notes 282 of 361, soft 279 of 360 judged, none 283 of 361. Not met.
- Outcome: the default stays as built (`BRAIN_NOTE_STYLE`). Neighbouring turns had removed the harm.

**The count route's gathered list** (count-route-reading-v1 and exp-count-block-v1, 2026-11-12)
- Question: Does the code-counted list help on questions sent to the count route?
- Bar set before the run: shipped ahead by 3 or more net on count-route questions means it helps; level or
  behind means it does not earn its place; ahead by 1 or 2 is undecided.
- Result: a free reading of 61 count-route questions found the code-counted number equal to the gold on 2. The
  run, 27 count-route questions: 21 with the list, 19 without. Undecided by the rule.
- Outcome: the list stays as built (`BRAIN_COUNT_BLOCK`).

**The count list worded as evidence** (exp-count-style-v1, 2026-11-12)
- Question: Does presenting the list as further evidence, with no count in the header, help?
- Bar set before the run: 3 or more net either way decides; within 2 is no difference shown.
- Result: 36 unread multi-session questions; the pack differed on 25: 19 against 18. No gain shown.
- Outcome: the wording stays as built (`BRAIN_COUNT_STYLE`).

**The count route reads more of the ledger** (exp-gather-top-v1, 2026-11-13)
- Question: Does a wider ledger search (200 claims a side, not 40) give more right counts?
- Bar set before the run: 5 or more net ahead with Haiku is a finding; within 4 with Haiku and under 5 ahead with
  Sonnet is nothing shown.
- Result: 141 LoCoMo count-route questions; the pack differed on 116. Haiku 66 against 70; Sonnet 79 against 78.
  Over all 141: Haiku 78 against 82, Sonnet 97 against 97. No gain shown. Packs carrying the "search hit its
  limit" note fell from 44 to 8.
- Outcome: the limit stays at 40 (`BRAIN_GATHER_TOP`). Two free readings before it (count-coverage-reading-v1,
  rollup-reading-v1) had found a wider pack unsupported and a night-built list unable to hold the missing items.
  Counting across many sessions is a stated limit.

## Safety: the injection screen

**Screening retrieved passages for injected instructions** (stress-c, 2026-11-08)
- Question: Does the screen remove attacks without removing legitimate passages or the user's own preferences?
- Bar set before the run: at least 95% of attacks removed on a blind set written after the rule was fixed, with
  further bars on legitimate passages and on the user's preferences.
- Result: second blind set, 160 passages. The one-question screen: 61 of 70 attacks, 18 of 90 legitimate passages
  wrongly removed, 18 of 33 user preferences removed. The two-question provenance screen: 64 of 70 (91%), 0 of 90,
  0 of 33. Not met on attacks; met on the other two. On an earlier blind set of 140 the one-question screen had
  wrongly removed 31 of 80 legitimate passages.
- Outcome: the two-question screen ships, recorded as failing its own attack bar. A further version with two
  more questions was not shipped: a third blind set to score it could not be produced.

## Stress, scale and load

**Hostile input, races, isolation and recovery** (stress-c, 2026-11-08)
- Question: Does the service hold under bad input, concurrent writes and failures?
- Bar set before the run: 40 checks, each pass or fail.
- Result: all 40 pass after fixes. Found and fixed: a lock-up at 10 or more concurrent requests, NUL characters,
  and deep nesting. 30 accounts running 1,500 random concurrent operations leaked nothing. A backup loaded into
  an empty database matched the live one, including 1,321 stored turns.
- Outcome: fixes shipped with regression tests.

**Scale and load** (stress-c, 2026-11-08)
- Question: Does recall stay fast as one namespace grows, and how much load does one replica take?
- Bar set before the run: recall at the 95th percentile under 1 s; planted facts still found.
- Result: all 20 planted facts found at every size up to 200,000 lines. Recall p95 was 1,536 ms at 200,000 lines.
  Not met. A fix for very common words was re-measured at 10,000 lines only (mixed-query p95 197 ms, was 417 ms).
  On a 1 GB database writes stalled at about 33,000 to 35,000 lines. One replica: 9.5 requests a second with 4
  clients, 12.8 with 16 (recall p95 3,128 ms), no errors.
- Outcome: keyword fix shipped. The vector index must fit in memory: about 780 MB per 100,000 lines.

**Six complex-domain worlds, first pass** (stress-b, 2026-11-08)
- Question: Do retrieval and answers hold on dense professional material?
- Bar set before the run: none; a measurement under the run invariants.
- Result: 150 questions over 884 turns. Answer accuracy 0.913; never-said questions answered "not available" on 18
  of 18; preference questions 0.42; count 0.83.
- Outcome: named the two weak spots: preferences asked as tasks, and counts spread over many sessions.

**The shipped brain on the six worlds** (exp-worlds-shipped-v1, 2026-11-13)
- Question: Does the shipped pack beat retrieval only on the same worlds?
- Bar set before the run: a difference under 6 of 150 is not a finding; retrieval only with Sonnet at least 100.
- Result: Haiku 141 against 134 (+7, a finding by the rule, one sample). Sonnet 142 against 138 (no gain shown).
  Sound-run check met (138). No world fell away. The eight misses with Sonnet were the known kinds.
- Outcome: no change. Not comparable with the first pass (another path, prompt and judge).

**A live check of the deployed service** (2026-11-13)
- Question: Do the shipped settings work on the deployed service?
- Bar set before the run: none; a check with a labelled synthetic session of seven turns, archived afterwards.
- Result: capture, the date direction rule, the router, the ledger's current value after a move, and two named
  people kept apart all worked. Not as wanted: two time phrases got no date note, and one claim with a pronoun
  as its subject was not filed.
- Outcome: the first fault led to exp-date-counted-v1.

## Cost and latency

**Write cost and recall latency with the shipped settings** (measure-shipped-v1, 2026-11-12)
- Question: What does a turn cost to store and a question cost to answer?
- Bar set before the run: none; a report. One conversation of 369 turns and 40 questions, one machine.
- Result: capture 0.16 s median, 0.26 s at the 95th percentile. Write cost $0.14 per 1,000 turns. Context pack
  1.22 s median, 1.36 s at the 95th percentile; $1.35 per 1,000 questions, about 52 Jev calls each.
- Outcome: led to the profile below.

**Running the router and ledger beside recall** (profile-context-v1, 2026-11-12)
- Question: Where does the time go, and can the reads run side by side with the same pack?
- Bar set before the run: none; a report and a setting.
- Result: 40 questions. One after another: 1.15 s and 1.12 s median over two runs. Side by side: 0.79 s median,
  0.89 s at the 95th percentile, the same Jev requests. Also found: two identical requests gave a different
  prompt on 24 of 40 questions.
- Outcome: shipped on (`BRAIN_CONTEXT_PARALLEL`). The second finding led to exp-stable-pack-v1.

**One Jev request for the whole rerank** (a spike, 2026-11-13)
- Question: Would one request holding all 30 passages be cheaper and faster than 30 requests?
- Bar set before the run: none; a spike on 12 questions with easier shortlists than the service's own.
- Result: 0.08 s against 0.28 s, and $0.00018 against $0.00062 per question, but a different ranking: 6.6 of the
  top 10 shared on average.
- Outcome: not built. It would be a different reranker and would need its own gates.

**Retrieval with each provider taken away** (exp-ablate-providers-v1, 2026-11-15)
- Question: How much do Jev's rerank and Voyage's vectors each add to retrieval?
- Bar set before the run: a measurement of the frozen brain; a difference of 10 answers or more of 361 is a
  finding, under 10 is not.
- Result: 361 LoCoMo questions, retrieval alone. As shipped 265; no rerank 221; keyword search only 217; neither
  196. All evidence among the excerpts on 275, 222, 210 and 183 of 360. Both are findings: 44 and 48 answers.
- Outcome: no change. Recorded as what each provider is worth to retrieval on this benchmark.

**Marking weak passages by their rerank score** (exp-recall-floor-v1, 2026-11-16)
- Question: Is there a rerank score below which a returned passage is almost never evidence, so recall could
  mark it as a weak match without cutting anything?
- Bar set before the run: dev picks the largest threshold marking at most 2% of evidence passages; held half, at
  that threshold, at most 3% of evidence and at least 40% of the other passages marked.
- Result: 150 LoCoMo questions a half, top ten each. Dev picked 0.05. Held half: 1.2% of evidence marked, 25.3%
  of the others. Not met. The two overlap: at 0.15, half the others and 5% to 8% of the evidence fall below.
  A hard cut at 0.10 would have left 4 of 125 held-half questions with none of their evidence. The injection
  screen scored 9000 ordinary turns and dropped none.
- Outcome: no change. Recall keeps its top ten with the scores shown, and the tool description says a low
  rerank score is a weak match.

## Readers and benchmarks

**The answering model with no memory** (exp-no-memory-v1, 2026-11-09)
- Question: What does the same model score with no excerpts at all?
- Bar set before the run: an expectation, not a gate: near 0 outside abstention questions; above about 0.15 would
  be treated as a bug.
- Result: LongMemEval, 60 questions: 0.033 (2 of 60), against 0.750 for retrieval only and 0.780 for the full
  brain. LoCoMo, 291 questions: 0.034 (10 of 291), against 0.769 and 0.763. The threshold was not reached overall.
- Outcome: results are reported as a three-step ladder from here on.

**A stronger reader on the same packs** (exp-reader-swap-v1, 2026-11-12)
- Question: Are the remaining errors the reader's or the pack's?
- Bar set before the run: none; a measurement. A difference between readers under about 10 answers of 361 is not
  a finding.
- Result: LoCoMo, 361 questions: Haiku 272 (retrieval only) and 285 (shipped pack); Sonnet 289 and 306; Sonnet with
  no memory 43 (11.9%). LongMemEval, 131 held-out questions: Haiku 92 and 108; Sonnet 109 and 119; Sonnet with no
  memory 9, all abstention questions.
- Outcome: the reader was a limit, and the pack adds to retrieval for both readers. Hand readings of the questions
  neither reader got (both-wrong-reading-v1, 9 questions; locomo-both-wrong-reading-v1, 43 questions) found faulty
  or strict gold answers, lists spread over many turns, and wrongly marked disputes.

**Every LoCoMo question** (locomo-full-1540, 2026-11-13)
- Question: What does the shipped pack score on all 1,540 questions?
- Bar set before the run: none; a measurement. Not held out: earlier experiments used part of the set.
- Result: Haiku 1,132 (73.5%) retrieval only and 1,190 (77.3%) shipped pack. Sonnet 1,244 (80.8%) and 1,292
  (83.9%); Sonnet with no memory 225 (14.6%). Marked a second time with a lenient judge prompt, one vote: 1,229
  (79.8%), 1,296 (84.2%), 1,317 (85.5%), 1,363 (88.5%).
- Outcome: the stricter judge stays the one of record.

**A stronger model extracts the claims** (exp-full-client-v1, 2026-11-13)
- Question: Do answers improve when Sonnet, not Haiku, sends the claims at capture?
- Bar set before the run: a difference under about 60 answers of 1,540 is not a finding.
- Result: Haiku reading the shipped pack: 1,190 with Haiku's claims, 1,187 with Sonnet's. Claims placed 3,324
  against 5,279. No gain shown on any arm. The run took three starts.
- Outcome: no change. On LoCoMo the answers come from the read path.

## Consolidation and the nightly pass

**A question set about facts that change** (exp-ledger-set-v1, 2026-11-13)
- Question: On scripted conversations where values change, does the ledger add to answers?
- Bar set before the run: the control questions answered on at least 8 of 10; a difference under 6 over the
  change questions is not a finding.
- Result: dev half, 65 questions. Control 10 of 10. Met. Change questions: 41 against 40 of 50. No gain shown;
  retrieval only is at the ceiling. The ledger recorded 16 replacements where the scripts hold about 50.
- Outcome: the set is read by the ledger's own state from here on, not by answers.

**When consolidation runs** (exp-ledger-dream-v1, 2026-11-13)
- Question: Does running the nightly pass never, once, three times or after every session change what the ledger
  follows?
- Bar set before the run: a difference under 4 of 35 between two settings is not a finding.
- Result: scripted changes followed: 8, 10, 12 and 11 of 35. Never against three times is 4; every other pair is
  under it. All the movement was in corrections. Answers did not move.
- Outcome: no change. Most misses are upstream: the claim was never sent, or the fact was filed under two names.

**Compare a new claim with same-named facts under another entity** (exp-relate-across-v1, 2026-11-13)
- Question: Does comparing across entities end old values the write path filed under another name?
- Bar set before the run: dev, at least 14 of 35 followed and at most 2 wrong endings; held half, at least 4 above
  the flag-off run and at most 2 wrong endings.
- Result: dev 15 against 10. Met. Held half 13 against 12. Not met. On the held half it ended nine claims, eight
  rightly and one wrongly (an unsettled dispute between two speakers turned into an update).
- Outcome: built and off (`BRAIN_RELATE_ACROSS`).

## Growth

**The same early questions as one memory grows** (exp-over-time-v1, 2026-11-13)
- Question: Do answers about early material hold as the memory grows, and do the nights help?
- Bar set before the run: drift under 4 of 40 either way means it holds; nightly against none under 4 means the
  nights are not shown to change answers.
- Result: 40 questions at four sizes, about 95 to 788 turns. Shipped pack right on 35, 34, 33 and 34 with the
  nights and 33, 35, 33 and 34 without. Holds. Complete evidence in the pack fell from 38 to 34.
- Outcome: no change. The brain does not get worse as it fills and does not get better with use.

## Where it ended

The brain was frozen at the shipped settings on 2026-11-13. From there the benchmarks are regression checks only.
Known limits carried forward: counting across many sessions; a client that sends no claims gets retrieval only;
the ledger follows about a third of changed facts and answers do not depend on it; nothing improves with use.
