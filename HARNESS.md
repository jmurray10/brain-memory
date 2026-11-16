# The evaluation harness: what was wrong with it, and what that changed

Before any number in [RESULTS.md](RESULTS.md) was trusted, the harness that produces the numbers was reviewed.
The first version, written for LongMemEval, had faults that made its results worthless, in both directions. They
are listed here because they are easy to make and hard to see: every one of them produced a plausible score.

## Faults found in the first harness

1. **The oracle arm was not an oracle.** The oracle arm is handed the sessions that hold the answer, and is the
   ceiling every other arm is read against. It was packing only the first 10 turns of those sessions: 90 of 100
   oracle answers had exactly 10 turns. A question about a value that changed needs the latest session, so the
   oracle mostly saw the old value, and retrieval appeared to beat it. Fix: the caller passes the turn cap and
   token budget; the oracle gets all the evidence turns, and the run logs when a cap cuts any off.
2. **The question's date was never in the prompt.** A question such as "how many days ago" cannot be answered
   without today's date. Oracle accuracy on temporal questions was 0.27, mostly "unknown". Fix: the question's
   date goes in the prompt of every arm.
3. **Preference questions could not pass.** The answer prompt asked for a few words; the benchmark's expected
   answer for a preference question is a rubric ("the user would prefer responses that..."). Retrieval scored 0
   of 10. Fix: answer and judge prompts per question type, following the benchmark's own evaluation. The answer
   prompt is chosen from the question's text alone, never from the benchmark's type label, which a real client
   would not have.
4. **The judge's parser counted `CORRECT: no` as correct.** It matched the first word. The test for it covered
   only a bare yes or no. Fix: parse the verdict exactly; anything else is a parse failure, logged and counted as
   an error, never as correct. The parser is tested on the judge's real output format.
5. **The recall measure was too loose.** It counted a question as retrieved if any one of its evidence sessions
   was in the pack. Questions that need several sessions need all of them, which is why recall read 1.00. Fix:
   report recall of all evidence sessions, and of the evidence turns themselves.
6. **Smaller ones.** Each turn was cut to 1,200 characters in the prompt, and long assistant turns often hold the
   answer. Packing stopped at the first turn that overflowed the budget, where it should skip it and continue.
   The rerank read only 12 candidates and kept 10, so it barely reordered anything; it now reads 30.

The first scores from this harness (oracle 0.51, retrieval 0.71) were thrown away.

## What was learned after the fixes

- **One judge pass is not a measurement.** The first run on the fixed harness was also invalid: retrieval beat
  the oracle on preference questions by 30 points. Nothing in the pipeline was wrong. Judging the same 20
  answers three times changed the verdict on 4 of the 10 oracle answers, and one verdict came back as prose.
  Since then every answer is judged three times and the majority counts, with a retry when the judge does not
  answer yes or no.
- **Two identical runs disagree.** With nothing changed, two arms differed on 12% of 100 LongMemEval questions,
  and on about 7% of LoCoMo questions. The reader and Jev each move a little between calls. A difference smaller
  than that is not a result.
- **So arms share a memo.** Within one run, each distinct Jev request and each distinct reader prompt is
  answered once and reused. With the memo, two identical arms agreed on all 361 questions, and a run cost about
  half as much.
- **Even a strong reader with thinking off is not steady.** Reading the same pack twice, it gave a differently
  marked answer about once in twenty reads. One read per answer cannot resolve a difference under about 18
  questions in 361.
- **The benchmark's judge is sometimes wrong.** On LoCoMo answers that give a date in relative words, a hand
  check found the judge wrong on at least 13 of 84.
- **A reader may already know the benchmark.** With no memory at all, the stronger reader answered 14.6% of
  LoCoMo correctly, and none of LongMemEval beyond the questions whose answer is "memory does not hold this".
  The no-memory arm is run so this can be seen.

## The rules every run has to pass

A run is thrown away if any of these fails:

- the oracle arm scores at least as well as retrieval, overall and per question type;
- no arm scores above 95% without a hand check of a sample;
- the judge's parser is tested on the judge's exact output format;
- spend is computed from the provider's real price table, and the run stops at its cap;
- the sample, the settings and the hashes of the code are logged with the result.

And one habit: the bar a change has to clear is written down before the run, and the result is recorded whether
it clears it or not. [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md) is that record.
