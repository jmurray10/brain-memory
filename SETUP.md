# Set it up

From nothing to a brain your models can use. The only thing to install is Docker. Allow fifteen minutes, most of
it the first build. [README.md](README.md) says what the brain is; [REFERENCE.md](REFERENCE.md) lists every setting.

## What you need

- **Docker** (Docker Desktop on Windows or macOS; Docker Engine with the Compose plugin on Linux) and **git**.
- **Two API keys, when you want real answers.** You can skip them at first: with none, setup turns on trial mode,
  where both services are stand-ins. Trial mode proves the wiring; its search results are not meaningful.

| Key | Where to get it | What it does |
| --- | --- | --- |
| `TYPESAFE_API_KEY` | [TypeSafe](https://docs.typesafe.ai) | Jev: every judgment the brain makes |
| `VOYAGE_API_KEY` | [Voyage AI](https://docs.voyageai.com) | Embeddings: finding things by meaning |

## 1. Get the code

    git clone https://github.com/jmurray10/brain-memory
    cd brain-memory/service

Every command below is run from this `service` folder.

## 2. Run setup

    docker compose run --rm setup

The first time, this builds the image (a few minutes) and then writes a file named `.env` in the folder, holding:

- an operator key it generates for you (it is never shown; it stays in that file);
- trial mode, because you have not given it provider keys yet;
- the settings the published results were measured with, so a new install behaves like the brain the results
  describe.

It then says the service is not running yet. That is expected.

## 3. Start the brain

    docker compose up -d --build

This starts the database, the API on `http://localhost:8000`, and two workers that file what you save every ten
minutes and tidy up once a day. Nothing needs scheduling, and it comes back by itself after a restart.

Check it is up: open `http://localhost:8000/healthz` in a browser. It should say `"ok": true`.

## 4. Connect your first client

    docker compose run --rm setup

The same command again. Now that the brain is up, it creates a key for a client named `claude`, shows the key
once, and prints two things to copy:

- **For Claude Code:** one `claude mcp add ...` line. Paste it into a terminal.
- **For Claude Desktop:** a short block. Open Claude Desktop, Settings, Developer, Edit Config, paste the block
  inside `"mcpServers": { }`, save, and restart Claude Desktop. The memory tools then appear in every chat.
  Claude Desktop runs the connection through Docker, so Docker has to be running when you use it.

## 5. Check that it remembers

In your client, say something worth keeping, then ask for it back in a new chat:

> Remember that my sister lives in Austin.

> Where does my sister live?

Or from the terminal, with the key from step 4:

    docker compose exec -e BRAIN_KEY=bk_... api python -m brain.cli capture "My sister lives in Austin." -n me
    docker compose exec -e BRAIN_KEY=bk_... api python -m brain.cli recall "Where does my sister live?" -n me

## 6. Go live

Open `.env` in a text editor, add your two provider keys, and change the trial line:

    TYPESAFE_API_KEY=...
    VOYAGE_API_KEY=...
    BRAIN_OFFLINE=0

What you saved in trial mode was indexed by the stand-ins, so begin real use with a fresh database:

    docker compose down -v
    docker compose up -d --build
    docker compose run --rm setup

`down -v` erases the trial memory and its keys; the last command makes a new client key, so connect your client
again with it (in Claude Desktop, replace the old block). From here the brain costs about $0.14 per 1,000 turns
saved and about a tenth of a cent per question. `BRAIN_DAILY_CAP_USD` (default 5) stops spending for the day when
it is reached.

## One memory for several clients

This is what the brain is for: every model, session and bot you use reads and writes the same memory. Give each
one its own key, all on the same namespace, by running setup once per client:

    docker compose run --rm setup --client claude-code
    docker compose run --rm setup --client claude-desktop
    docker compose run --rm setup --client my-bot

| Client | How it connects |
| --- | --- |
| Claude Code, or any client that speaks MCP over HTTP | The `claude mcp add ...` line setup prints |
| Claude Desktop, or any client that speaks MCP over stdio | The config block setup prints |
| A bot or script, on any model | REST with its key: `POST /v1/capture` to save, `POST /v1/context` to ask. The reply's `prompt` is ready to hand to the model |
| A hosted model that calls tools over the internet | It must be able to reach your brain, so the brain needs a public HTTPS address, not `localhost`. `service/scripts/grok_bot.py` is an example: Grok, through xAI's remote tools |

A separate key per client matters: each turn records which client wrote it, a key can be revoked alone, and a
standing instruction can be for one client or for all. What one client saves, the others can find at once.

Checked by the author with two clients on one hosted brain: a question asked with one client's key returned what
the other had saved. Many clients writing at the same moment has not been measured.

## Desktop apps on a subscription: no model API key

A desktop app you already pay for (Claude Desktop, the Grok app) can use the brain through a connector. The app
talks to the model on your subscription; the brain only needs its own two keys. Which kind of connector depends
on where your brain runs.

**Your brain runs in Docker on the same machine (`localhost`).** Use a *command* connector: the app starts a
small container that speaks to the brain for it. Docker has to be running.

| Field | Value |
| --- | --- |
| Name | `brain` |
| Type | Command (also called local, or stdio) |
| Command | `docker` |
| Arguments | `run -i --rm --add-host host.docker.internal:host-gateway -e BRAIN_URL=http://host.docker.internal:8000 -e BRAIN_KEY=bk_... brain-memory python -m brain.mcp_stdio` |

Put the client's own key in place of `bk_...`. In Claude Desktop this is the block that
`docker compose run --rm setup` prints (step 4). In an app that asks for the arguments one per line, each word
above is one argument.

**Your brain is on a public HTTPS address** (a server or a hosting service). Use a *remote* connector:

| Field | Value |
| --- | --- |
| Name | `brain` |
| Type | Remote HTTPS |
| Server URL | `https://your-brain.example.com/mcp` |
| HTTP header | Key `Authorization`, value `Bearer bk_...` |
| Client ID | Leave empty: it is for servers that use sign-in, and the brain uses the header |

A remote connector does not work with `localhost`. In Claude (Settings, Connectors, Add custom connector) the
connection is made from Anthropic's servers, not from your machine, so the brain must be reachable from the
internet; the same custom connector then also works in Claude on the web and on a phone.

What is tested: the command form runs in this repository's CI on every push (it lists the memory tools), and
Grok, through xAI's API with the remote form on a hosted brain, answered from what another client had saved. The
connector screens of the apps themselves change often; the fields above are what they asked for when this was
written.

## Day to day

Put the operator key from `.env` (`BRAIN_ADMIN_KEY`) or a client key after `BRAIN_KEY=`:

    docker compose exec -e BRAIN_KEY=... api python -m brain.cli status
    docker compose exec -e BRAIN_KEY=... api python -m brain.cli forget --episode 12 -n me          # archive
    docker compose exec -e BRAIN_KEY=... api python -m brain.cli admin restore --episode 12 -n me   # bring it back
    docker compose logs -f api writer                # what it is doing
    docker compose down                              # stop; your memory is kept
    docker compose up -d                             # start again

[service/README.md](service/README.md) covers accounts, directives, backups, and running without Docker.

## If something is wrong

| What you see | What to do |
| --- | --- |
| Setup says the service is not running | Do step 3, then run setup again |
| Setup says the service did not accept the operator key | The brain started before `.env` had this key. `docker compose up -d` again |
| Setup says a client with that name exists | Names are never reused. Pick another with `--client` |
| The tools do not appear in Claude Desktop | Is Docker running? Restart Claude Desktop fully, then check Settings, Developer: the `brain` server and its log |
| `401` or `403` from the API | The key is wrong, revoked, or has no access to that namespace |
| Answers ignore what you saved | You are in trial mode (step 6), or the workers are not running (`docker compose ps`) |
| `"degraded": "no_rerank"` or `"no_embed"` in a reply | A provider key is missing or the daily cap was reached. Nothing is lost |
| Port 8000 or 5432 is already in use | Add `BRAIN_API_PORT=8001` or `BRAIN_DB_PORT=5433` to `.env` and start again |
