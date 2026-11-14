<!-- mcp-name: io.github.swapnanil/vectr -->

<p align="center">
  <img src="https://raw.githubusercontent.com/swapnanil/vectr/main/assets/banner.svg" alt="vectr: semantic codebase search and persistent working memory for AI code editors" width="560">
</p>

# Vectr

> **Delivery, not storage.** A local daemon that gives an AI code editor semantic search over your codebase, plus working memory that shows up on its own when it is relevant.

[![CI](https://github.com/swapnanil/vectr/actions/workflows/ci.yml/badge.svg)](https://github.com/swapnanil/vectr/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/downloads/)
[![Version 1.12.0](https://img.shields.io/badge/version-1.12.0-blue.svg)](CHANGELOG.md)
[![MCP: 23 tools](https://img.shields.io/badge/MCP-23%20tools-blue.svg)](#23-mcp-tools)

Version 1.12.0 · Last updated 2026-09-03 · [CHANGELOG](CHANGELOG.md)

## In 30 seconds

Your AI code editor pays for the same knowledge twice. It re-reads files it read yesterday, re-greps symbols it already found, and loses the exact signature it discovered at turn 5 the moment the conversation is compacted. Vectr is one local process that removes both halves of that bill:

- **Retrieval that understands code.** AST-aware chunks, a symbol and call graph, and hybrid semantic plus BM25 ranking. Describe a behaviour and get the whole function back, in one call instead of a grep loop and three blind file reads.
- **Working memory that gets delivered.** Save a finding once. It comes back verbatim in under 50ms, and on editors that expose session hooks it arrives automatically at the moment it applies, without the agent choosing to ask for it.
- **Local, keyless, zero config.** One `pip install`, one `vectr start`. The embedding model runs on your machine. Nothing is transmitted anywhere, and there is no API key.

```bash
pip install vectr && cd /path/to/your/project && vectr start
```

That is the whole setup. Vectr writes the MCP config for your editor, then indexes in the background.

---

## How vectr differs from other AI agent memory tools

There is a healthy category of general-purpose memory layers for AI agents, and the good ones solve a real problem well. Vectr is most often compared to them, so here is the difference stated plainly. The left column describes the design these tools tend to share, not any one product; individual tools vary, so check the one you are considering.

| | A typical memory layer | Vectr |
|---|---|---|
| **How memory reaches the model** | The agent or developer calls an explicit store-and-retrieve API, `add()` and `search()` or the equivalent. Memory arrives only when something decides to ask for it. | Every note carries trigger conditions over `{path, symbol, semantic, event, temporal}`. On editors with session hooks the harness evaluates them deterministically and injects the match. The agent never has to remember to ask. |
| **Domain** | Generic conversational memory, aimed broadly at assistants and applications rather than at source code. Notes are strings about a user or a session. | Code native. A symbol graph, AST chunking, semantic code search, and working memory fused in one daemon, so a note can be anchored to a real symbol or path rather than a string. |
| **Inference** | Commonly runs note extraction through an LLM and embeds through a provider API, so an API key and a provider account are part of the setup. | Zero internal LLM calls, by design. A local embedding model, no provider account, no API key. |
| **Deployment** | Usually a library or a self-hosted server, frequently with a managed cloud tier alongside it. | One local daemon bound to `127.0.0.1`, one per workspace. Team mode is available and opt in. |

**Why the first row is the one that matters.** Storage is not the bottleneck. Retrieval that an agent must volunteer to call is, because it largely does not call it. In our controlled evaluation the agent performed **0 memory operations across 114 turns** even when the store was pre-seeded with knowledge directly relevant to its task. Deterministic injection delivered in every injection-equipped run, with zero false-alarm fires across the audit-logged trigger evaluations. Under repeated compaction, ten facts held only in the conversation were absent from 106 of 108 forced compactions, while the same ten facts injected from a harness-owned store arrived intact across 138 of 138 compact-resumes.

Full method and results: [Delivery, Not Storage: Cue-Anchored Working Memory as a Harness Property for Coding Agents](https://arxiv.org/abs/2607.20972) (arXiv:2607.20972).

---

## What it costs and what it saves

**Measured, not hypothetical.** Recalling 3 stored notes with `vectr_recall` costs 360 tokens in one tool call. Re-deriving the same three facts with grep plus Read costs about 2,060 tokens across six tool calls on the same 182-file Python repo. That is roughly **5.7x fewer tokens and 6x fewer tool calls**, in under 50ms (chars/4 tokenization, full breakdown in [Measured costs, honestly](#measured-costs-honestly)).

Notes are persisted to disk, not held in the conversation, so they survive `/compact` and a fresh session equally. The session boundary does not matter: saving at turn 5 and recalling at turn 15 is the same mechanism as recalling three days later.

**Where it pays off:** unfamiliar or large codebases, work you resume, and long sessions with many turns.
**Where it does not:** a one-off grep on code you already know cold. Reach for grep instead, and see [When vectr can hurt](#when-vectr-can-hurt).

---

## Benchmarks

Public results live in [`benchmarks/`](https://github.com/swapnanil/vectr/tree/main/benchmarks). The corpora below are witnesses chosen because they are large and unfamiliar to the model, nothing more.

A six-task sprint on a large unfamiliar C codebase simulates a week of feature work. One research session stores findings with `vectr_remember`. Six isolated implementation sessions each open cold and call `vectr_recall`.

**Implementation sessions only, 6 tasks combined:**

| Metric | Vanilla | Vectr | Delta |
|---|---|---|---|
| Cost | $2.50 | $1.97 | **-21%** |
| Wall time | 17.6 min | 13.5 min | **-24%** |
| Turns | 123 | 94 | **-24%** |
| Read + Bash calls | 102 | 62 | **-39%** |

**Per-task re-discovery (Read and Bash calls before the first write):**

| Task | Vanilla | Vectr | Delta |
|---|---|---|---|
| `debug_gc_finalizer` | 16 | 6 | -62% |
| `feature_dict_pop_last` | 13 | 3 | -77% |
| `cross_session_set_cartesian` | 23 | 9 | -61% |
| `debug_descriptor_priority` | 6 | 6 | 0% |
| `cross_session_bytes_find_all` | 13 | 2 | -85% |
| `cross_session_list_rotate` | 21 | 16 | -24% |

The 0% row is real and kept on purpose: that task was one the model could already navigate from training knowledge alone.

**Research versus implementation, stated honestly.** The research phase is paid once and costs *more* with vectr (+94%), because storing rich code stubs and signatures produces output tokens. The implementation phases repeat every task and cost less, because recall replaces re-discovery. The overhead breaks even after roughly 8 tasks of note reuse.

| Phase | Vanilla | Vectr | Why |
|---|---|---|---|
| Research (1 session, paid once) | $1.36 | $2.63 | Storing notes costs output tokens |
| Impl (6 sessions, repeating) | $2.50 | $1.97 | Notes replace re-discovery |
| Total sprint | $3.86 | $4.60 | Inverts to a net gain after about 8 tasks |

An earlier run on a 5,856-file Java corpus measured **-58% implementation cost, -72% implementation tool calls, and -39% wall time.**

---

## Measured costs, honestly

Per-call token cost (median, 182-file Python repo, chars/4 tokenization):

| Tool | Median tokens | Range |
|---|---:|---|
| `vectr_search` | ~2,320 | 1,437 to 3,091 (n=8) |
| `vectr_locate` | ~192 | |
| `vectr_trace` | ~720 | |
| `vectr_recall` (index tier) | ~180 | |

The trade-off, stated plainly: for a single pointed lookup on a small, already-familiar repo, grep is cheaper. Vectr's median cost across 5 single-fact tasks was **60% more tokens**, and it is slower too, since a `vectr_search` round trip takes 1.7 to 3.6 seconds against about 28ms for grep. Vectr does not win on per-call cost. It wins on tool-call count (one round trip instead of several), on answer completeness (a whole symbol back, not a partial file read), and on everything in working memory, where the 5.7x recall refund compounds with every task you resume.

Fine print: the automatic eviction and reminder banners riding along on tool responses cost tokens too. An always-on re-fetch footer runs about 27 tokens, a light nudge about 89 tokens, and the escalated action-required banner (which fires only after both the chunk and token thresholds are crossed without a save) scales from about 480 to 535 tokens before it plateaus.

---

## Quick start

**Local (recommended)**

```bash
python3.14 -m venv ~/.vectr-env
source ~/.vectr-env/bin/activate   # Windows: ~/.vectr-env/Scripts/activate
pip install vectr
cd /path/to/your/project
vectr start
```

**Requires Python 3.14+.** To install:
- macOS: `brew install python@3.14`
- Ubuntu/Debian: `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.14 python3.14-venv`
- Windows: [python.org/downloads](https://www.python.org/downloads/)

`vectr start` returns immediately. Indexing runs in the background, so run `vectr status` to check progress. On first run the embedding model downloads once (about 290 MB). Restart your AI code editor once to pick up the new MCP config.

**Docker (CI and servers)**

```bash
git clone https://github.com/swapnanil/vectr
cd vectr
docker-compose up api
```

Exposes port 8765. Docker does not auto-write IDE config files, so use the local install for IDE integration.

---

## Connect to your AI code editor

`vectr start` writes the MCP config for your editor automatically. Restart your editor once.

| Editor | Config | Status |
|---|---|---|
| Claude Code | Auto: `.claude/settings.json`, guidance file, and session hooks (memory auto-injected at session start, per prompt, and before file read or edit) | **Verified** |
| Cursor | Auto: `.cursor/mcp.json` | Experimental |
| VS Code / GitHub Copilot | Auto: `.vscode/mcp.json` | Experimental |
| Windsurf | Manual, see below | Experimental |
| Cline | Manual, see below | Experimental |
| Continue | Manual, see below | Experimental |
| Codex CLI | Auto: `.codex/config.toml`, `AGENTS.md` guidance, and `.codex/hooks.json` (`vectr init --hooks`) | Experimental |

"Verified" means the full integration (config, guidance, and hooks) has been exercised end to end. "Experimental" means the MCP config is written and works, but the integration has not been run through the same verification pass.

Automatic delivery of memory depends on the host exposing session hooks. Where it does not, every tool below still works on demand over MCP.

<details>
<summary>Manual setup</summary>

**Claude Code**, `.claude/settings.json`:
```json
{ "mcpServers": { "vectr": { "type": "http", "url": "http://localhost:8765/mcp" } } }
```

**Cursor**, `.cursor/mcp.json`:
```json
{ "mcpServers": { "vectr": { "url": "http://localhost:8765/mcp" } } }
```

**VS Code / GitHub Copilot** (1.99+), `.vscode/mcp.json`:
```json
{ "servers": { "vectr": { "type": "http", "url": "http://localhost:8765/mcp" } } }
```

**Windsurf**, `~/.codeium/windsurf/mcp_settings.json`:
```json
{ "mcpServers": { "vectr": { "serverUrl": "http://localhost:8765/mcp" } } }
```

**Continue.dev**, `.continue/config.json`:
```json
{ "mcpServers": [{ "name": "vectr", "transport": { "type": "http", "url": "http://localhost:8765/mcp" } }] }
```

**Codex CLI**, project-scoped `.codex/config.toml` (written automatically by `vectr start`):
```toml
[mcp_servers.vectr]
url = "http://localhost:8765/mcp"
```
On its first run in the workspace, Codex shows a one-time interactive prompt to trust the project before it loads a project-scoped config. If you also ran `vectr init --hooks`, Codex shows a second one-time prompt to trust the hook commands. Both persist after you accept them once, and a config writer cannot clear them for you without weakening your security posture.

</details>

### Stdio transport

The editors above connect over HTTP (`vectr start` plus `POST /mcp`). For MCP clients and hosting platforms that spawn the server as a subprocess and speak MCP over its stdin and stdout instead of opening an HTTP connection, run:

```bash
vectr mcp-stdio [WORKSPACE]
```

No port, no daemon: a single foreground process framed as newline-delimited JSON-RPC 2.0 (one JSON object per line on each of stdin and stdout, no `Content-Length` headers). `initialize` and `tools/list` answer immediately, while the embedding model load and initial indexing happen on a background thread. A `tools/call` made before that finishes gets a graceful "still starting up" response instead of hanging. Stdout carries protocol JSON only, and all logging goes to stderr. The process exits cleanly on stdin EOF. `--memory-only` and `--search-only` behave the same as on `vectr start`.

---

## How it works

1. **AST-aware chunking.** tree-sitter parses each file and splits at function, class, and method boundaries. No chunk breaks mid-logic.
2. **Code embeddings.** `ibm-granite/granite-embedding-english-r2` (local, CPU-fast, overridable) maps natural-language queries to code symbols, so "JWT validation" finds `verify_jwt_token`. BM25 handles exact symbol names.
3. **Hybrid search.** Vector similarity and BM25 combined, weighted by codebase characteristics. Large and unfamiliar leans semantic, small and well-documented leans BM25.
4. **Symbol graph.** Call edges, import chains, and HTTP routes (Flask, FastAPI, Express, Spring) are extracted and stored. `vectr_locate` resolves a name through a 6-strategy cascade: exact match, suffix, same-module, import-chain, substring, then fuzzy. A fuzzy answer is always rendered as explicitly inexact rather than presented as a confident match, because a confident wrong symbol is worse than a not-found.
5. **Working memory.** `vectr_remember` stores structured notes to SQLite and ChromaDB. `vectr_recall` runs semantic search over notes rather than a SQL LIKE, so multi-word queries still find the relevant context.
6. **Trigger evaluation.** Each note's conditions are checked deterministically by the daemon at lifecycle moments, with a per-session fire ledger and injection budgets, so a note resurfaces exactly when it applies and never twice in the same window.
7. **MCP protocol.** 23 tools served over HTTP. Any MCP-compatible AI code editor connects without plugins.

---

## 23 MCP tools

`vectr start` writes a guidance file into your workspace with this table, so your AI code editor knows which tool to reach for without being prompted.

**Search tools**, which retrieve code from the index:

| Situation | Tool |
|---|---|
| You know a concept or behaviour, not a name | `vectr_search("description")` |
| You know a symbol name, not its file | `vectr_locate("SymbolName")`, with a multi-strategy fallback cascade and optional `caller_file` |
| You need callers or callees of a symbol | `vectr_trace("symbol_name")` |
| You need an architectural overview | `vectr_map()` |
| You want to save a synthesised map summary | `vectr_map_save(summary)` |
| You have runtime call data to inject | `vectr_ingest_traces([{caller, callee}])` |
| You need index health or note count | `vectr_status()` |

**Memory tools**, which store and recall across sessions:

| Situation | Tool |
|---|---|
| Notes exist from a prior session | `vectr_recall(query)`, semantic vector search rather than substring match, two-tier (crisp index by default, expand one note with `note_id=N` or all bodies with `detail='full'`) |
| One note must survive every recall regardless of the query | `vectr_pin(note_id=N)`, which places it in Tier 0 so it is injected on every `vectr_recall(query=...)` call. Bounded, so pin sparingly. `vectr_remember(..., pin=true)` does the same at write time |
| You found something worth preserving | `vectr_remember(content, tags, priority, kind, title, agent)`. `kind` controls delivery: `directive` fires unconditionally every session, `task` carries current-work state, `gotcha` resurfaces when its file is touched, `operational` covers build and environment facts, `finding` (the default) is relevance-ranked, `reference` is a pointer, and `decision` is an architectural decision recallable as a chronological ADR-style timeline via `vectr_recall(kind="decision", sort_by="chronological")`. `title` labels the note in index output, and `agent` attributes it to a subagent or orchestrator |
| A stored note turns out to be about a file you did not declare | `vectr_anchor(note_id=N, anchors=[...])`, attaching file anchors to an existing note without re-storing it. Idempotent, hashed at write time, and it emits no lifecycle event, so the note's history stays clean (also `vectr anchor --id N PATH...` on the CLI) |
| A stored note was anchored to the wrong file, or the anchored file's relevance is gone | `vectr_unanchor(note_id=N, anchors=[...])`, the inverse of `vectr_anchor`. Removes file paths from a note's anchor set; a removed anchor simply stops being a staleness candidate, never a claim that the note is wrong. Idempotent, paths the note was never anchored to are reported back rather than treated as failures |
| Starting a session, want to pick up where you left off | `vectr_resume()`, returning the most recent task note, the latest snapshot, and open gotchas with their file anchors in one call (also `vectr resume` on the CLI) |
| Context is filling up | `vectr_evict_hint()`, which identifies chunks vectr can re-retrieve, with the exact re-fetch ids |
| A chunk shown earlier has left your context | `vectr_fetch(ids=[...])`, a deterministic byte-verbatim re-fetch by id, with no re-search and no file re-read. Flags a truncation warning if the index itself stored a capped chunk |
| End of a long session, want a checkpoint | `vectr_snapshot("label")` |
| Looking for a prior checkpoint | `vectr_snapshot_list()` |
| Notes are stale after a large refactor | `vectr_forget(note_id=N)` per note, or `vectr_forget(all=true)` to clear |
| An auto-captured note has been reviewed and still holds | `vectr_promote(note_id=N)`, raising its trust class one step from `auto` to `agent`. Promotion to `human` is reserved for user-side surfaces, never the agent's call |
| Automatically captured failure-to-success moments are waiting | `vectr_distill()`, which renders pending arcs (a command failed, then passed after an edit) for review. Persist a lesson with `vectr_remember(..., distilled_from=[arc_id])` or dismiss with `vectr_distill(dismiss=[...], reason)` |
| A stored note turned out to be wrong | `vectr_revoke(note_id=N, reason)`, which keeps the note visible as a deterrent ("previously believed ..., revoked ...") instead of deleting it, so the mistake is not silently re-derived. `vectr_remember(contradicts=N)` corrects and revokes in one step |
| A note is simply out of date, not wrong | `vectr_supersede(note_id=N, successor_note_id=M)`, which retires it in favour of a newer note without revoke's "proven wrong" framing. The audit trail records post-hoc retirement distinctly from write-time supersession, and `vectr_reinstate` reverses it |
| A revoked note was right after all | `vectr_reinstate(note_id=N)`, restoring the original content |

### Triggers: the delivery layer

Those `kind` values carry sensible defaults. A note can also declare explicit per-note `triggers` on `vectr_remember`:

| Axis | Fires when |
|---|---|
| `path` | A glob matches the file the current lifecycle moment targets |
| `symbol` | The targeted file defines or references that symbol, resolved exactly against the same symbol graph `vectr_locate` uses, never fuzzy |
| `semantic` | Prompt similarity clears a fixed per-kind threshold |
| `event` | A lifecycle moment occurs: session-start, prompt-submit, pre-edit, pre-run, pre-commit, or post-compaction |
| `command` | The normalized verb of an about-to-run shell command matches a glob |
| temporal | `not_before`, `expires_visibility`, and `cooldown` guards, which modify a fire but never cause one alone |

Conditions AND within one entry and OR across entries. Every fire carries a one-line explanation of which trigger matched. Injection is budgeted (a hard per-turn cap spent jointly across surfaces), deduplicated by note id across every surface within a turn, and framed by provenance, so an unreviewed auto-captured note never renders with the authority of a standing rule from you. A note is never partially truncated: it arrives whole, as an index-tier line, or not at all.

Workspace-scoped notes double as a shared bus for multi-agent workflows. An orchestrator and its subagents all read and write the same note store, so a subagent can call `vectr_remember(..., agent="coder-2")` with its findings before finishing, and the orchestrator recalls them instead of re-reading the subagent's full transcript. The `agent` parameter is never inferred. It is explicit attribution, and it shows up as a tag in `vectr_recall` index output.

---

## CLI reference

```bash
vectr start                           # index + start daemon for current dir
vectr start /project/api              # positional workspace: a directory or .code-workspace file
vectr start --path /project/api       # specific workspace (repeatable, multi-root)
vectr start --memory-only             # working memory + hooks only, no code index, no watcher
vectr status                          # index health, chunk count, notes count
vectr status --all                    # all running instances
vectr stop /project/api               # stop one instance (same positional as start)
vectr stop --path /project/api        # stop one instance (equivalent --path form)
vectr stop --all                      # stop all instances
vectr index --path .                  # re-index without restarting daemon
vectr fetch src/auth.py:10-42         # re-fetch a chunk by exact id, verbatim
vectr resume                          # most recent task note, latest snapshot, open gotchas
vectr init --path .                   # write guidance file + MCP config without starting
vectr init --exclude vendor           # exclude directories from indexing
vectr forget --path .                 # delete all working-memory notes
vectr memory export                   # render the working-memory store as a greppable, git-diffable markdown file (default MEMORY.md)
vectr memory export --path notes.md   # write the export to a specific file
vectr memory export --disable         # stop the auto-refresh that keeps the export file in sync
vectr memory edit                     # open the working-memory store in your $EDITOR, one block per note
vectr memory import                   # bring existing on-disk memory files into the working-memory store (see below)
vectr memory import --path NOTES.md   # import a specific file instead of the default discovery list
vectr memory import --dry-run         # preview what would be created (kind, title, source file, line range), change nothing
vectr cache prune                     # remove empty per-workspace cache dirs (live instances skipped)
vectr cache prune --dry-run           # preview what would be removed, delete nothing
vectr proxy                           # experimental: localhost API proxy (see below)
vectr mcp-stdio                       # foreground stdio MCP transport, no port or daemon (see above)
```

### Importing existing memory

`vectr memory export` and `vectr memory edit` are the one-way-out and check-out/check-in paths. `vectr memory import` is the one-way-in: it brings existing on-disk memory into the working-memory store without re-entering it by hand. This is the on-ramp for a new user who already has memory written down.

**Discovery.** With no `--path`, the importer walks a closed list of default candidate files at the workspace root: `MEMORY.md`, `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, and `.github/copilot-instructions.md`. Pass `--path` (repeatable) to point at any other file or directory; a directory is walked recursively for `*.md` and `*.mdc` files. Files that don't exist are reported but never raised on, since most workspaces will not have every default file.

**Granularity.** A source file is not one note and not one note per line. It is split at Markdown heading boundaries (`^#{1,6}\s+`), the same way a human skims a `MEMORY.md`. A heading-less file becomes one note. Headings inside fenced code blocks and HTML comment blocks are skipped, so a code example or the export-file header comment does not become a section boundary. A file previously exported by `vectr memory export` is detected by its `<!-- Generated by ... -->` header and parsed through the same block grammar `vectr memory edit` uses, recovering each original block as a note with its declared `kind`, `priority`, `tags`, and `provenance` carried forward.

**Dry run.** `--dry-run` prints exactly what would be created (kind, title, source file, source-line range, provenance) and changes nothing. Memory is the one resource the user cannot easily reconstruct, so dry run is the path of first resort:

```bash
vectr memory import --dry-run
# + [finding/auto] 'Project conventions'  (MEMORY.md:12-34)
# + [operational/agent] 'Test runner'  (CLAUDE.md:5-9)
# DRY RUN: would create 2 note(s), skip 0 already-imported, ...
```

**Idempotency.** Every imported note carries a `src:`, `src-sha:`, and `src-lines:` tag triple. A re-run of `vectr memory import` against an unchanged source file recognizes those tags and creates zero new notes. Editing a section of the source file produces a new body whose hash does not match the stored one, so it becomes a new note, and the old one is left untouched, since silently deleting a prior import behind the user's back is exactly the silent loss the design defends against on the export side.

**Provenance.** Imported notes carry a source-file tag (`imported` and `src:FILENAME`, where FILENAME is the basename of the source file), so a recall that lists a note carrying `imported` is telling the user "this came from a file, not from this session's observations." The note's `provenance` defaults to `auto`, the class machine-captured facts with no reviewing judgment already carry, since a hand-written `MEMORY.md` is structurally indistinguishable from a generator's output to a parser. The exception is narrow and structural, not semantic: a note whose body contains imperative-rule markers (the closed keyword set `MUST`, `MUST NOT`, `NEVER`, `ALWAYS`, `DO NOT`, `DON'T`, `REQUIRED`, `FORBIDDEN`) is classified as `kind="directive"` and upgraded to `provenance="agent"`. The reason is the `auto + directive` write-time guard in `remember()`: an unreviewed standing rule is a contradiction in terms, and an imported rule is exactly that pair. Promoting to `agent` is the smallest possible deviation from `auto`, exactly enough to satisfy the guard, and the user promotes further with `vectr_promote` after review.

---

## Excluding paths

Create `.vectrignore` in your project root (same syntax as `.gitignore`):

```
vendor/
node_modules/
*.pb.go
dist/
```

Or pass `--exclude` at init time:

```bash
vectr init --exclude vendor --exclude dist
```

Exclusions apply to both the initial index walk **and** the live file watcher, so adding a directory to `.vectrignore` stops a running instance from re-indexing it. The next index also **prunes** any chunks already stored for now-excluded or deleted files, so you do not have to rebuild from scratch. If you ever need a clean rebuild, for example after changing the embedding model, force one:

```bash
vectr index --path . --force      # ignore the incremental cache, re-embed everything
```

---

## Supported languages

| Language | Chunking | Symbol graph |
|---|---|---|
| Python | AST (functions, classes) | Yes |
| JavaScript | AST (functions, classes, arrow fns) | Yes |
| TypeScript | AST | Yes |
| Go | AST | Yes |
| Rust | AST | Yes |
| Java | AST | Yes |
| C | AST | Yes |
| C++ | AST | Yes |
| Zig | AST | Yes |
| All others | 200-line windows, 50-line overlap | No |

HTTP routes (Flask and FastAPI decorators, Express `app.get()`, Spring `@GetMapping`) are extracted as symbols and searchable via `vectr_locate("GET /api/users")`.

---

## Cost

| | Cost |
|---|---|
| Embedding model | $0.00, a one-time download of about 290 MB, cached at `~/.cache/vectr/` |
| Re-index (10k files, first run) | About 10 min on CPU, under 5 sec on subsequent runs (mtime cache) |
| Incremental re-index per changed file | About 0.5 sec |
| vectr_search / vectr_recall | $0.00, local inference only |

---

## Security

The default is unchanged and stays the headline: **local, no API key, zero config**, for a solo developer on a personal machine. Out of the box, the daemon binds to `127.0.0.1` only, CORS is restricted to localhost origins, each workspace gets its own isolated DB directory, port, and process (owner-only `0700` on POSIX), and the index and notes persist locally in `~/.cache/vectr/`.

Everything below is **opt in**. Enabling nothing changes nothing.

**Authentication.** Set `VECTR_API_KEY` and every request to `/v1/*` and `/mcp` must carry it (`X-Api-Key: <key>` or `Authorization: Bearer <key>`, constant-time comparison, with `/v1/health` left open for liveness probes). Generate a key with `vectr key`. When the key is set at start time, the editor MCP configs vectr writes include the header automatically. Those files (`.mcp.json`, `.cursor/mcp.json`, `.vscode/mcp.json`) then hold the key in plaintext, so treat them as secrets and keep them out of shared or public version control.

**Encryption at rest.** Set `VECTR_ENCRYPT_KEY`, or store a passphrase in the OS keychain (service `vectr`, username `encrypt-key`, requires `pip install vectr[encryption]`), and note content, note titles, and snapshot payloads are encrypted with Fernet and a PBKDF2-derived key. Honest boundary: the **code index is not encrypted**, because the search engine needs readable chunk text and vectors. Protect it with OS full-disk encryption. Note tags and metadata stay plaintext, and note embedding vectors (a lossy projection of note text) are kept for semantic recall unless you set `VECTR_ENCRYPT_DISABLE_NOTE_VECTORS=1`.

**Retention and audit.** Notes are kept until you delete them. Set `VECTR_NOTES_TTL_DAYS` to auto-purge older notes at startup. `vectr_forget(all=true)` and `vectr forget --all` delete notes, snapshots, and note vectors, so everything means everything. Set `VECTR_AUDIT_LOG=<path>` for a rotating local log of index, search, remember, and recall events (off by default, and it records query text, which is its purpose, and is never transmitted). Full policy: [docs/data-handling.md](docs/data-handling.md).

**Team mode (shared instance).** One central daemon can serve a team on one repo. Run `VECTR_API_KEY=<key> vectr start --host 0.0.0.0` on the server (a non-loopback bind **refuses to start without a key**), then `vectr connect --url http://<host>:<port> --api-key <key> --label <you>` on each client to point the editor at it. Working memory is shared: a note one agent stores, every connected agent can recall, and `--label` attributes notes and audit lines. Note IDs are allocated by the central store in the order writes arrive, so with concurrent clients your notes are **not** a contiguous block. They interleave with other clients'. The `Stored note #N` line the write returns is the canonical reference to that specific note, so do not assume the next ID is also yours. Plain limits: one shared key means every holder is an equal, trusted peer (no roles, no per-user permissions), the server operator can read everything, search results reference the **server's** checkout, which may differ from your local tree, and vectr speaks plain HTTP, so put TLS at a reverse proxy or tunnel if the network is not trusted.

---

## Proactive context (experimental, localhost only)

Pull-based recall means the agent has to ask. **Proactive context** extends deterministic delivery to hosts without session hooks, by sitting on the wire instead. It is **experimental** and **localhost only**. Since 1.6.0 proactive delivery is **on by default** on local instances (`proactive.enabled`; turn off with `VECTR_PROACTIVE=0`). The wire channel below is governed by launch consent instead of that switch: starting the proxy and pointing your agent at it is the opt-in, and stopping it or unsetting the base URL is the opt-out.

The `vectr proxy` command runs a small local proxy between your agent and the model API:

```bash
vectr proxy                            # starts a localhost proxy (default :8785)
export ANTHROPIC_BASE_URL=http://127.0.0.1:8785   # point your agent at it
```

What it does and does not do, plainly:

- **Transparent by default.** It forwards every request to the real API, and streaming responses and tool calls pass through byte for byte. Your **API key is forwarded untouched and never stored or logged.**
- **Deterministic injection.** With the workspace daemon running, it appends matched working-memory notes to the request *after* the last prompt-cache breakpoint, so your prompt cache is never invalidated. Triggering is a similarity threshold plus exact structural matches, never keyword guessing, with a strict per-request budget so a hint only lands when it is worth the tokens.
- **Fail open.** If the intelligence layer is slow or errors, your request goes through unchanged. **To bypass the proxy entirely, unset the base URL:** `unset ANTHROPIC_BASE_URL`. The proxy is on the request path, so if it is down, unset the variable to talk to the API directly.
- **Solo and localhost only.** It reads your conversation to compute context, so it refuses any non-loopback bind and is mutually exclusive with team mode.
- **Caveats on a non-first-party base URL,** per the host's own documentation: MCP tool search is disabled unless `ENABLE_TOOL_SEARCH=true` and the proxy forwards tool-reference blocks, and Remote Control is disabled on a non-`api.anthropic.com` base URL.

**Org-wide caching (team mode).** With a central shared instance, vectr can cache its own expensive artifacts, namely semantic search and recall results, keyed by exact identity and the current index state, so a re-index or note change invalidates them automatically and every connected developer benefits. It is off by default (`proactive.cache`), and `vectr status` reports its hit rate and estimated tokens saved so the value is measured rather than asserted. Vectr does **not** cache LLM responses across similar requests, only byte-identical ones, locally, and opt in, because a wrong cache hit would silently corrupt a conversation.

---

## When vectr can hurt

**Stale notes after codebase churn.** Notes store file paths at write time. After a large refactor, `vectr_recall` flags changed referenced files with `[STALE]`. Re-verify before acting, delete the stale note with `vectr_forget(note_id=N)`, or clear everything with `vectr_forget(all=true)`.

**Over-retrieval on a well-known API.** If the model already knows a framework deeply from training, vectr's research overhead may exceed the savings. The benchmark above shows exactly that on `debug_descriptor_priority`, a task where training knowledge alone was enough to navigate.

**Stale architectural overview.** A passport saved by `vectr_map_save` describes the architecture at index time. After renamed modules or restructured entry points it can be confidently wrong, so `vectr_map` surfaces a staleness warning and you should re-run `vectr_map_save`.

**Broad recall costs tokens.** `vectr_recall()` with no query returns everything. Pass a targeted query, and check `vectr_status()` first if you are unsure whether recall is warranted at all.

---

## Built with

Python 3.14 · FastAPI · sentence-transformers · tree-sitter · ChromaDB · BM25 · Docker

## Author

Swapnanil Saha · [swapnanilsaha.com](https://swapnanilsaha.com)
