# Results

What was measured on the brain as released, and how to read it. [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md) lists
every experiment behind these numbers, and [HARNESS.md](HARNESS.md) says why the harness can be trusted. Every number is one run unless it says
otherwise; small differences are within run-to-run noise and are said to be.

## Setup

- **The client is deliberately weak.** Claude Haiku (thinking off) plays the user's model: it extracts the claims
  at capture and writes the answers. A stronger reader (Claude Sonnet) answers from the same context packs, to
  separate what the brain retrieved from what the reader made of it.
- **Arms.** "Full pack" is what the service hands a model: dated excerpts, the turn before and after each, the
  ledger's facts, resolved dates. "Retrieval alone" is the service's own search with the ledger and routes off.
  It is the brain's first layer, not a model without memory.
- **Judge.** The benchmark's own judging prompt, Haiku, majority of three.
- **Settings.** The shipped ones: per-slot placement with its support check, the "has it stopped being true"
  check before a value is ended, different proper names never joined tentatively, one neighbouring turn each
  side, dates stored at capture and written beside the phrase, 25 dense and 5 keyword seats among the 30
  candidates the rerank reads.

## Answers

| Material | Questions | Reader | Full pack | Retrieval alone |
| --- | --- | --- | --- | --- |
| LoCoMo (ten long two-person conversations) | 1,540 | Haiku | 1,190 (77.3%) | 1,132 (73.5%) |
| LoCoMo | 1,540 | Sonnet | 1,292 (83.9%) | 1,244 (80.8%); no memory at all: 225 (14.6%) |
| LongMemEval, 131 held-out questions | 131 | Haiku | 108 (82.4%) | 92 (70.2%) |
| LongMemEval, 131 held-out questions | 131 | Sonnet | 119 (90.8%) | 109 (83.2%); no memory at all: 9 (6.9%) |
| Six generated professional worlds | 150 | Haiku | 141 (94.0%) | 134 (89.3%) |
| Six generated professional worlds | 150 | Sonnet | 142 (94.7%) | 138 (92.0%) |

The pack is about 1,400 tokens. It helps the weaker reader more than the stronger one.

The LongMemEval questions are the 131 that were never used while building: 31 knowledge-update, 50
multi-session and 50 temporal-reasoning, so three of the benchmark's question types and not the whole of it. By
type the full pack scored 28, 42 and 38 with Haiku and 31, 45 and 43 with Sonnet. With no memory Sonnet answered
"unknown" to all 131 and was right only on the 9 questions whose answer is that memory does not hold it. This is
the cleanest reading of what memory adds, and the widest gap between the full pack and retrieval alone. A later
repeat of the same 131 gave 107 and 105 with Haiku and 120 and 118 with Sonnet.

**How much two identical runs differ.** On LoCoMo, two runs with the same settings disagreed on about 7% of
questions, because Jev's answer to the same request moves a little between calls and so does a reader's. Arms of
one run are therefore compared with a memo (one Jev answer and one reader answer per distinct request). Read
every difference above against that.

## Does it hold as a memory grows?

The same 40 questions about the first quarter of two conversations, asked as the memory grew from about 95
turns to 788 (the last step adds an unrelated conversation to the same namespace):

| Memory holds | Full pack right, of 40 |
| --- | --- |
| first quarter | 35 |
| first half | 34 |
| whole conversation | 33 |
| plus an unrelated conversation | 34 |

It holds. Complete evidence reached the pack on 38 questions at the start and 34 at the end: later talk on the
same subjects competes for the pack. The nightly pass made no difference to these answers.

## The parts, measured on their own

Each of these had a bar fixed before it was run. Two did not meet theirs and are reported as such.

| Part | What was measured | Result |
| --- | --- | --- |
| Claim checks | Of the claims the brain kept from a weak client, how many does the turn really support? 180 turns, judged by a stronger model | 181 of 192 (94.3%). The bar was 95%: narrowly not met |
| Claim checks | What the checks removed, of 325 claims the client sent | Code dropped 34 (the quote was not in the turn, or the value was not in the quote); Jev dropped 25, of which 9 were judged to have been good |
| Injection screen | Attacks removed from retrieved passages, on a blind set of 160 written after the rule was fixed | 64 of 70 (91%). The bar was 95%: not met |
| Injection screen | Legitimate passages wrongly removed, same set | 0 of 90, including 0 of 33 of the user's own standing preferences ("always answer me in metric units") |
| Slot placement | How often the same claim is filed differently when asked again | 12.0% with one "which slot" question; 3.7% with one question per slot, which is what ships |
| Retrieval | Asked twice, how often the same question gets the same ten excerpts | 42% to 63% of the time; nine of the ten are the same on average |

The six attacks the screen missed were four messages from someone else that the user had pasted into their own
chat, which it trusts as the user's words, and two faked tool calls in an agent's log. The earlier screen, with
one question instead of two, caught about as many attacks but removed 18 of the 90 legitimate passages.

## What Jev and Voyage each do, and what was measured

The brain uses two outside models: Voyage for embeddings and TypeSafe's Jev for every judgment. This table says
where each is used and what that use was measured to do.

**Retrieval with each one taken away.** The same 361 LoCoMo questions, answered from retrieval alone, with a
provider made unavailable the way an outage would. The bar, fixed before the run: 10 answers or more is a finding.

| Retrieval | Right, of 361 | All the evidence among the excerpts, of 360 |
| --- | --- | --- |
| As shipped: keyword and vector search, Jev reranks | 265 (73.4%) | 275 |
| No Jev: the fused search order, no rerank | 221 (61.2%) | 222 |
| No Voyage: keyword search only, Jev reranks | 217 (60.1%) | 210 |
| Neither: keyword search in keyword order | 196 (54.3%) | 183 |

Jev's rerank is worth 44 answers and Voyage's vectors 48; both together are worth 69. The two overlap: each
recovers part of what the other finds. With neither, keyword search alone still answers 54% of the questions,
which is also what a deployment with no keys set falls back to. One benchmark, one run, one reader.

| Where | Who | What was measured |
| --- | --- | --- |
| Embed every stored line and every question | Voyage | 48 more right answers of 361 than keyword search alone (table above). Of 127 evidence turns the candidate list missed, 71 were in the dense top 60, which is why 25 of the 30 candidates now come from the dense list |
| Rerank the 30 candidates | Jev | 44 more right answers of 361 than the same search with no rerank (table above). With it, retrieval scored 0.84 on 100 LongMemEval questions where perfect retrieval scored 0.85 |
| Screen passages for injected instructions | Jev | 64 of 70 attacks removed, 0 of 90 legitimate passages removed |
| Route the question | Jev | Right route on 94.3% of 420 questions, 88.4% of 301 on another benchmark, 95.4% of 280 from professional fields |
| Check the claims a client sends | Jev, after code | Of 325 claims sent, code dropped 34 and Jev 25 (9 of those were good). 94.3% of what was kept is supported by its turn |
| File a claim under the right attribute | Jev | Asked twice, the same claim is filed differently 3.7% of the time (12.0% before one question per slot) |
| Decide whether a newer value ends an older one | Jev | Values ended fell from 62 to 6 on one benchmark and 53 to 14 on another; answers did not change |
| Read time phrases | Jev names the parts, code computes | Of 54 phrases with a written count that had no date before, 52 were dated correctly |

**What the two cost.** A question makes about 52 small Jev requests and costs $1.35 per 1,000 questions. Checking
and filing claims costs $0.125 per 1,000 turns, all Jev. Embedding and the date questions at capture cost $0.013
per 1,000 turns. Jev is nearly all of the bill.

**What is not known.** Vector search with no keyword side was not measured: the service has no switch for it.
Nothing here compares Jev or Voyage with another reranker or embedding model. When either provider is down the
service falls back to what is left and says so in the response.

## Speed, cost and load

| | Result |
| --- | --- |
| Storing a turn | 0.16 s median, 0.26 s at the 95th percentile |
| Cost of writing | $0.14 per 1,000 turns, checks and filing included |
| Building a context pack | 0.79 s median, 0.89 s at the 95th percentile, with the reads run side by side |
| Cost of a question | $1.35 per 1,000 questions, about a tenth of a cent each; all of it is Jev (about 52 small calls) |
| One API replica under load | 9.5 requests a second with 4 clients (recall median 551 ms); 12.8 with 16 clients (recall median 1,122 ms, 95th percentile 3,128 ms). No errors: it slows down instead |
| One namespace grown to 200,000 lines | All 20 planted facts still found at every size; recall at the 95th percentile rose to 1,536 ms, over the 1 s bar. A fix for very common words followed and was re-measured at 10,000 lines only |

Timings are from one machine and one conversation, and the load test predates the current pack. The vector
index has to fit in the database's memory: about 780 MB per 100,000 lines. On a 1 GB database writes slowed
sharply at about 33,000 lines.

## Isolation and recovery

- **Accounts stay apart.** 30 accounts ran 1,500 random operations at once (capture, recall, export,
  directives, forgetting another account's ids): nothing leaked and nothing failed. Run with stub providers.
- **Races hold.** 50 captures at once with one idempotency key store one turn; a forget racing a capture leaves
  no searchable piece of the forgotten turn; 40 interleaved forgets and restores end consistent.
- **A backup restores exactly.** A backup of a live deployment was loaded into an empty database and matched it:
  accounts, key hashes, directives, 1,321 stored turns and what had been forgotten.
- This testing found a real fault, since fixed: under ten or more requests at once the service could lock up
  waiting for database connections.

## Where it is weak

- **Counting across many sessions.** On the 141 LoCoMo questions routed to counting: 58% with Haiku and 69%
  with Sonnet, against about 79% elsewhere with Haiku. Four changes were tried (the gathered list, its wording,
  a wider pack, a wider search); none showed a gain. It is a stated limit.
- **The ledger follows about a third of changed facts.** On generated conversations with scripted changes it
  ended the old value on 10 to 13 of 35. Most misses are upstream of the ledger: the client never sent the
  claim, or the same fact was filed under two names. Answers did not depend on it: the excerpts carried them.
- **Nothing improves with use.** Consolidation, run never, once or every night, moved no answers.
- **Preferences asked as tasks** ("draft my email in my usual format") are the client's job at session start,
  through standing directives, not something an answer prompt does well.

## What is not measured

Months of real use by one person; several clients writing to one brain; material that is not conversation at
all; the other LongMemEval question types; latency of a hosted deployment as its users see it. The generated sets are tidy and small. The benchmark numbers above are tests the brain was checked
against, not targets it was built for.
