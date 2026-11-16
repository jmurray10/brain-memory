# Telling the brain how to work for you

A memory is only useful if your models use it the way you want: check it at the right moments, save the right
things, and follow the rules you set. This guide shows how to set that up, and how to change it as your use
changes. [SETUP.md](SETUP.md) gets the brain running first.

There are two places an instruction can live. Use both.

| | In the app's custom instructions | In the brain, as a standing instruction |
| --- | --- | --- |
| Who reads it | That one app, on every message | Every client that starts a session with the brain |
| Good for | Making the model use the brain at all | Rules about you that every client should follow |
| Changed by | You, in the app's settings | You, with the commands below |
| Shared between apps | No | Yes |

The first makes a model reach for the brain. The second is what it finds there. A rule kept only in the brain is
never seen by a model that does not look, which is why the first one matters.

## 1. One line in each app

Add this to the app's custom instructions (in Claude: Settings, Profile; most apps have an equivalent):

> Use the brain tools for anything about me: check memory before answering, and save what I tell you.

Without it, a model often answers from what it already knows, and "remember that..." may go to the app's own
built-in memory and never reach the brain. Pick the wording for how you work:

| You want | Put this in the app |
| --- | --- |
| Memory used all the time | Use the brain tools for anything about me: check memory before answering, and save what I tell you. |
| Memory only when you ask | Use the brain tools only when I say "check memory" or "save this". |
| Read, but never write | Check the brain before answering questions about me. Never save to it. |
| One project | For anything about the Atlas project, check the brain first (namespace `atlas`) and save decisions there. |

## 2. Standing instructions in the brain

A standing instruction is a rule the brain hands to a client at the start of each session: "Use metric units",
"Reply in British English", "Never suggest restaurants with shellfish". Every client that connects gets the
same ones, so you say it once.

Commands are shown for Docker, from the `service` folder. `KEY` is a client key, `ADMIN` is the operator key
(`BRAIN_ADMIN_KEY` in `.env`). Without Docker, drop everything before `python`.

**Add one.** It is stored as pending:

    docker compose exec -e BRAIN_KEY=$KEY api python -m brain.cli directives add "Use metric units." -n me

A model can propose one too, with the `memory_add_directive` tool: say "propose a standing instruction: ...".

**Confirm it.** Nothing is followed until you confirm it yourself, with the operator key:

    docker compose exec -e BRAIN_KEY=$ADMIN api python -m brain.cli directives list -n me
    docker compose exec -e BRAIN_KEY=$ADMIN api python -m brain.cli admin confirm-directive 1 -n me --all-clients

`--all-clients` serves it to every client. Leave it off and only the client that proposed it gets it. This step
is deliberate: text in a memory is untrusted, and a model, or something a model read, must not be able to give
itself standing orders. There is no tool that confirms.

**Change or remove one.** Retire the old one and add the new one:

    docker compose exec -e BRAIN_KEY=$KEY api python -m brain.cli directives retire 1 -n me

**For one client only.** Add `--scope` with the client's name, for example a rule only your coding tool needs:

    docker compose exec -e BRAIN_KEY=$KEY api python -m brain.cli directives add "Use tabs, not spaces." -n me --scope claude-code

The brain also suggests them: when you state a lasting preference in conversation, the nightly pass turns it into
a pending instruction. It appears in `directives list` and waits for you like any other.

## 3. Recipes

**A personal assistant across several apps.** One namespace (`me`), one key per app, the "all the time" line in
each app. Standing instructions for how you like answers: units, language, tone, things to avoid. Confirm them
for all clients.

**Coding across tools.** A namespace per project (`docker compose run --rm setup --client claude-code
--namespace atlas`). In each tool: "For this project, check the brain before changing anything, and save every
decision and its reason." Standing instructions for the project's rules: the test command, what must never be
committed, naming. Scope a rule to one tool when only that tool needs it.

**A bot with many users.** One account per user, so memories never mix (see accounts in
[service/README.md](service/README.md)). The bot's own prompt carries the "use the brain" line; each user's
standing instructions are theirs alone. Give the bot's key `speaks-as-user` only if what it saves really is the
user's own words.

**A shared notebook for a team or a study.** A namespace for the subject, a key per person or tool, and a
standing instruction about sources: "Say where each fact came from." Use `memory_page` to read what the brain
holds about one person or thing, with the sentence behind every line.

**Keeping it quiet.** The "only when you ask" line in the app, no standing instructions, and forget what you do
not want kept: `python -m brain.cli forget --episode N`. Forgetting can be undone; nothing expires on its own.

## 4. When it is not doing what you want

| What you see | What to change |
| --- | --- |
| The model answers without checking memory | The line in the app (section 1). Until then, add "check memory first" to the question |
| "Remember that..." is not saved | Say "save this to the brain", or add the line in the app. Look for a `memory_capture` step in the reply |
| It saves too much | Switch the app to "only when I ask" |
| A rule is not followed | Is it confirmed? `directives list` shows pending ones. Was it confirmed for all clients? |
| One app follows a rule and another does not | It was confirmed for one client. Confirm it again with `--all-clients`, or add it with the other client's `--scope` |
| An old rule is still applied | Retire it. Clients pick up the change at their next session |
| Two apps disagree about a fact | Ask either one for the fact's history (`memory_history`): the brain keeps both values and what replaced what |
