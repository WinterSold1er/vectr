# Changelog

## Unreleased

### Integrations
- DeepSeek Harness (dsh) is now a supported host. `vectr start` / `vectr init` register vectr in the active dsh profile's `cordis.patch.yml` as an `@deepseek-ai/dsh-mcp-client` insert row over Streamable HTTP (`mcp-client.vectr`, `http://localhost:<port>/mcp`), merge-safe by a comment-delimited region that preserves any other patch entries the user has. The write is skipped when no dsh home (`~/.dsh` / `$DSH_HOME`) exists; the profile comes from `$DSH_PROFILE` / `$VECTR_DSH_PROFILE`, defaulting to `web`. dsh guidance rides the existing `AGENTS.md` vectr block, which dsh auto-loads from the workspace. Honoring `--no-ide-config` and the durable `.vectr/ide_config` opt-out as every other editor config does. Authenticated daemons get the `headers` block in the row. A malformed existing patch is left untouched with a pointer to the README.

## 1.9.0 - 2026-08-07

A trace answer stops hiding real callers behind a same-named one and becomes
chainable into `vectr_fetch`; a mistyped qualified name gets the same
near-miss help a mistyped bare name already got; a failed fetch says which of
its two very different causes it hit; `--path` resolves from anywhere inside a
workspace; and shell episodes stop being classified `unknown` on hosts that
report success without an exit code.

### Retrieval
- `vectr_trace` collapses callers by `(name, file)` instead of by name alone.
  Two unrelated functions sharing a leaf name in different files (two classes
  each defining `__call__`) used to merge under one elected representative
  edge, so a real caller in the second file never appeared in the rendered
  trace even though its calls were counted in the `xN` suffix. Same-file call
  sites still fold into one entry with a count, which is a genuine ambiguity a
  receiver-blind static graph cannot resolve further. Callees keep the
  name-only collapse.
- Trace paths render workspace-relative, and every call site carries a
  `vectr_fetch`-ready chunk id, so reading the code around a caller is one
  chained call instead of a path fix-up plus a file read.
- Qualified-name near-miss now covers a typo in the **leaf** segment, not only
  in the qualifier (`QuerySt.get_or_creat`, both halves misspelled). The
  fallback resolves the bare leaf through the same per-segment machinery an
  unqualified token already gets, then ranks by qualifier similarity, so a
  doubly-mistyped token still surfaces its real symbol as a labeled-inexact
  suggestion. The widen step is also guarded so it can no longer displace an
  exact resolution that already succeeded.
- `vectr_fetch` distinguishes its two miss shapes. A requested id whose line
  range does not align to a stored chunk span, in a file that still has chunks
  indexed, returns `reason="misaligned"` plus `nearest_ids` naming the closest
  real chunk ids in that file, so the caller's next fetch succeeds. Only a file
  with no indexed chunks at all returns `reason="file_changed"`. New config
  `fetch.misalign_nearest_max` (default 5) caps the suggestion list. Callers
  reading the old shape are unaffected: the field defaults to `file_changed`.
- The caller in-degree tiebreak query now has index statistics to plan with. A
  full graph build gathers them once via a bounded `ANALYZE`, which flips a
  selective `to_symbol IN (...)` lookup from a whole-workspace covering-index
  scan to a direct index seek (measured ~18.6ms to ~1.0ms for a 50-name lookup
  on a synthetic 300k-edge graph). Enclosing-container lookups during
  qualifier scoring are batched per file rather than issued per candidate.

### CLI
- `vectr start --path` / query subcommands resolve a path through the
  workspace instance that **owns** it: pointing at a subdirectory of an
  indexed workspace now finds that workspace's daemon instead of reporting
  nothing running.
- `vectr stop --port N` stops the instance bound to a port, bypassing
  workspace-path resolution entirely and exiting non-zero if nothing is
  registered there. Deterministic for scripts and harnesses that manage
  scratch daemons by port.
- `vectr start --strict-port` fails instead of walking to the next free port
  when the requested one is taken, and `vectr start --json` prints a
  machine-readable result (port, workspace, mode, whether an existing instance
  was reused). Exit codes are reliable on the `--json` surface.

### Working-memory delivery
- The trigger-engine per-injection token cap now **trims** an oversized note
  body to the cap instead of demoting the note to title-only. A capped note
  previously delivered its label and none of its guidance. Truncation backs off
  to a sentence or word boundary, controlled by the new
  `memory_triggers.injection.body_truncation_min_boundary_fraction` (default
  0.5), rather than cutting mid-word.
- The proactive request window is built from user and assistant text only.
  Host-injected system reminders could be an order of magnitude longer than the
  user's actual message, so the tail-window slice was filled with boilerplate
  and the real task text was evicted before scoring ever saw it.

### Episode capture
- Bash episodes are classified from the host's success signal when no exit code
  is available. Claude Code's post-tool event carries no `rc`, which left the
  outcome cascade with nothing to key on for most everyday commands: measured
  across the live episode store, `outcome=unknown` fell from 91.6% to 0.0% and
  detected failure-to-success arcs rose from 0.675 to 1.605 per 100 episodes.
  The new signal sits below markers and `rc` in the cascade, and below
  `is_error`, since some hosts report one event name for both outcomes. No
  previously non-`unknown` classification changes.

### Internals
- Cache path resolution is centralized and reads `VECTR_CACHE_DIR` at call
  time rather than import time. Precedence is `VECTR_DB_DIR` >
  `VECTR_CACHE_DIR` > `~/.cache/vectr`. Test sessions now get an isolated cache
  root instead of writing into the real one.

## 1.8.0 - 2026-08-03

A restart now means "the same daemon again", version staleness is decided by
the code revision rather than the version string, and a note that transcribes
a direct user statement can now prove it.

### CLI
- `vectr restart <workspace>` inherits the recorded mode (full, memory-only,
  or search-only) and bind host of the instance it replaces. Explicit
  `--memory-only`/`--search-only` still win, and a new `--full` flag exits an
  inherited reduced mode on purpose. Previously a memory-only daemon came
  back in full mode (indexing and file watcher on) unless the flag was
  re-passed, and the staleness banner recommended exactly the bare command
  that caused the flip; the banner now prints the mode-complete command.
- The version-staleness warning keys on the git revision, not the version
  string. Same revision with differing packaged versions reports a "version
  metadata mismatch ... no restart needed" (the usual cause is a stale
  editable install). "Older" is never claimed from a lexical version compare,
  and abbreviated revision ids of the same commit are not a mismatch.

### Working memory
- New derived provenance class `user-stated` for notes that transcribe a
  direct user statement. A write may attach a `user_quote`; the store binds
  it only when the excerpt appears verbatim in the note content
  (whitespace-insensitive, case-sensitive, minimum 12 characters: a
  deterministic string check, never a semantic one). A bound note renders
  with its own recall frame naming the verbatim excerpt; an excerpt that
  fails the check is discarded and the note stays in the ordinary
  agent-recorded class, with the reason surfaced in the write response. The
  class cannot be claimed directly, only earned by the check. Available over
  MCP and REST.

## 1.7.0 — 2026-07-29

Proactive delivery becomes worth reading: an injected item now carries the
note's actual guidance instead of just its label, a file-anchored note
retires when its file is actually edited rather than on a timer, and the
framing envelope scales with note provenance. Also: `vectr stop` followed by
`start`/`restart` keeps the workspace's previous port so already-written
editor MCP configs stay valid, and `vectr init --hooks` survives an
unwritable git hooks directory.

### Proactive context
- Injected items now carry the note's **body**, not just its title. A titled
  note previously delivered only the title, leaving the actual guidance —
  the caveat, the fix, the "do not do X" reason — in `content` and never on
  the wire. Rendering is now `title: body` (the title is skipped when empty
  or when it is already a prefix of the body), and character-budget
  truncation backs off to a sentence or word boundary before resorting to a
  hard cut.
- **Event-anchored retirement** for file-anchored matches. A note delivered
  because the request touched its declared anchor file used to enter a fixed
  cooldown whether or not the agent had acted on it yet — measured as "too
  early, then evicted" in six of seven benchmark cells. Such a note now
  stays eligible for re-delivery while its file is merely being read, and
  retires the first time a request's window shows the file actually edited.
- The injection envelope's opening clause now scales with **note
  provenance**: the weakest provenance among the notes selected for one
  delivery (auto < agent < human) picks the wording, so an unreviewed
  auto-capture is introduced more skeptically than a rule a human wrote. The
  shared no-authority / verify-before-relying clause is identical across all
  tiers, since provenance is a caller-declared field, not a verified
  identity claim. Wording is configurable via `proactive.envelope` in
  `config.yaml`.

### Reliability
- `vectr stop` followed by `start` or `restart` now reuses the workspace's
  previous port instead of walking to a new one that already-written editor
  MCP config files don't point at. The port-free probe now treats a socket
  in TCP `TIME_WAIT` — the state every `vectr stop` leaves behind — as
  reusable, with a short bounded retry
  (`instance_registry.port_reuse_retry_attempts` /
  `port_reuse_retry_delay_s`), and `restart` no longer erases the registry
  entry the reuse step depends on before consulting it.
- Editor config writes now always sync vectr's **own** MCP server entry to
  the current port, while continuing to never overwrite user-added keys.
  Previously the merge-only-add rule kept the file pointing at the old port
  after a port change. After `start`/`restart`, vectr also prints an
  explicit warning naming any known editor config file under the workspace
  that still points at a different port, instead of leaving the editor's
  MCP connection to fail silently.
- `vectr init --hooks` no longer aborts when the git hooks directory is
  unwritable (for example a stale or misconfigured `core.hooksPath`): the
  git post-commit writer skips with a disclosed message like every other
  precondition failure in that path, and the editor hook installers that
  already succeeded stay in place.

### Research artifacts
- Injection-utility benchmark: six trap scenarios whose correct answer
  exists only in working memory rather than the workspace, an A/B analysis
  pipeline (arm hit rates, inject-vs-control efficiency, note-awareness
  scoring), and a hook-channel arm so proxy-versus-hook delivery is measured
  rather than assumed. The first seed-0 measured run is recorded, including
  a delivery-side finding: in the agent CLI's non-interactive print mode,
  prompt-time hook injections do not render into the machine-readable
  transcript (session-start injections do), so hook-channel utility must be
  verified against an interactive session.

## 1.6.0 — 2026-07-28

Proactive context — surfacing the relevant working-memory note at a
sanctioned delivery point instead of waiting to be asked — becomes the
default on local instances, together with the precision, provenance, consent,
and audit work that makes that default defensible. Also: write-time
related-note and process-anchor offers on every note write, a
compaction-boundary preservation surface, and a gitignore-matching fix that
could silently exclude an entire workspace from the index.

### Proactive context
- `proactive.enabled` now defaults to **true**. It gates ambient delivery
  channels — a hook surface that reads the transcript with no per-session
  user action. The `vectr proxy` channel is exempt from the switch by launch
  consent: starting the proxy and pointing an editor at it is itself the
  opt-in for that channel. Proactive context is still refused outright on a
  non-loopback (team / shared-instance) bind, independent of this key — the
  bind check runs before any config or channel branch. Turn off with
  `VECTR_PROACTIVE=0` or `proactive.enabled: false`.
- Structural (exact file-path) matches no longer bypass the similarity floor.
  They are scored in tiers instead: the note declares the file in its own
  `anchors` (1.0), a `gotcha` mentions it (0.9), any other eligible kind
  merely name-drops it (0.6). At most one weak-mention item is selected per
  delivery, so low-confidence file mentions can no longer fill the item
  budget ahead of a genuine semantic match. New config:
  `proactive.structural_scores`, `proactive.max_weak_structural_items`.
- The structural channel is restricted to kinds that state a durable fact
  about a file (`proactive.structural_kinds`: gotcha, finding, decision,
  operational, reference). `task` notes — a moment in time that happens to
  name a file — are excluded, and the channel now over-fetches before
  filtering so that exclusion cannot starve it
  (`proactive.structural_overfetch_multiplier` / `_ceiling`).
- Injected context carries a role-provenance envelope and a per-item
  provenance marker, so a recalled note reads as recalled memory rather than
  as an instruction from the current turn. `directive` notes are excluded
  from the proxy channel by default
  (`proactive.proxy.exclude_directive_notes`); session-start injection
  delivers them through its own path and is unaffected.
- A revoked note reaching the proactive matcher renders with the same
  deterrent framing recall uses, and delivery fails closed when a note's
  lifecycle state cannot be read rather than emitting the original content.
- Cooldown identity is the proxy process, not a hash of the conversation's
  first user message. One proxy serves one editor session, so every request
  through it — subagent turns included — shares one cooldown ledger.
  Previously each new first message minted a fresh ledger, and a note could
  be re-injected seconds after its last emission.
- Cooldown slots are charged on confirmed delivery, not on retrieval. The
  proxy checks up front whether a request can carry an appended block at all
  and skips retrieval entirely when it cannot, then confirms delivery with an
  opaque token. A caller that passes no token keeps retrieval-time charging
  exactly as before.
- Audit split to match: `PROACTIVE_RETRIEVE` is written at selection,
  `PROACTIVE_INJECT` only once the block is confirmed delivered — so the
  injection count is an honest lower bound on what reached the model instead
  of counting blocks that were never appended. `PROACTIVE_INJECT` lines now
  also carry `chars=` and per-item `states=`, recording delivered size and
  each item's lifecycle state.
- `vectr status` / `GET /v1/status` report the **effective** proactive state
  (the config flag ANDed with the bind gate) instead of the bare config
  value, which on a non-loopback bind claimed injection was on where it was
  in fact refused.
- The proxy's counters distinguish `inject_skipped_not_appendable` — the
  request could carry no block, so nothing was ever asked for — from the
  general skipped count.

### Working memory
- `vectr_remember` returns write-time offers alongside the stored note id:
  the nearest existing active notes by similarity, so a superseded one can be
  corrected in the same turn, and — for an `operational` note written without
  anchors — the process files actually present in the workspace it could be
  anchored to. Both are additive; neither gates, rewrites, or reorders the
  write, and similarity is closeness, never a contradiction verdict. Config:
  `memory_write.related_notes`, `memory_write.proxy_anchor_suggestions`.
  `POST /v1/remember` gains `related[]` (with each candidate's priority) and
  `proxy_anchor_suggestions[]`.
- New versioned proxy-anchor manifest (`agent/proxy_anchors.yaml`) mapping
  process domains — how dependencies are pinned, how CI runs, how the app is
  containerized, built, and deployed — to the ecosystem-standard files that
  encode them. Presence is glob-detected only; no file content is read.
- Drift on an anchored `operational` note now renders honestly: "changed
  since — verify; <file> is a proxy for this process, last confirmed <date>",
  rather than asserting the note is wrong. A changed process file means the
  process *may* have changed; the judgment stays caller-side.
- Stale-anchor flagging is keyed on a signature of the drifted anchors rather
  than on the most recent lifecycle event of any kind. An unrelated event in
  between no longer admits a duplicate audit row, and a genuinely new drift
  of the same anchor is no longer permanently suppressed.
- `recall_for_path()` matches on path boundaries instead of raw substrings —
  a note about `gate.py` no longer matches when `regate.py` is touched — and
  over-fetches candidates before that filter, so a true match can no longer
  be dropped from a too-small candidate pool. New config block:
  `memory_recall_for_path`.

### Compaction boundary
- New `GET /v1/boundary/precompact` and a PreCompact hook branch: on editors
  with session hooks, vectr emits a short boundary-preservation nudge (plus
  the count of arcs still awaiting distillation) as plain text before the
  conversation is compacted, which the harness appends to its own compact
  instructions. Emitted as plain text, never the JSON hook envelope other
  events use. Config: `episodes.boundary_precompact_enabled` (default true),
  `episodes.boundary_precompact_token_cap` (default 200).

### Indexing
- gitignore patterns are matched against the workspace-relative path, not the
  absolute one. A directory-style entry such as `tmp/` previously matched
  ancestor components of the workspace root itself, so a workspace checked
  out under a matching prefix had every one of its files silently excluded
  from the index. Path-scoped entries such as `docs/*` consequently now match
  as gitignore intends. Fixed in both the initial scan and the file watcher.

### Reliability
- Working-memory note operations — remember, recall, forget, commit notes,
  and proactive selection — dispatch off the request event loop, extending
  the 1.5.0 vector-store fix to the notes collection. A slow store call no
  longer stalls every other HTTP and MCP endpoint.
- The proxy-anchor workspace walk is bounded (depth, directory budget, and
  pruned dependency/build trees) and runs once per call instead of once per
  recursive glob: ~600ms → ~22ms on a 3,600-directory tree. It also no longer
  offers files from inside `node_modules/`, `.venv/`, or `target/`, which
  encode a dependency's process rather than the workspace's own.

### Editor extension
- The extension now finds a CLI installed by `pip install --user`. After the
  PATH probe fails it checks the per-user script directories a GUI
  application's inherited PATH commonly omits — `~/Library/Python/<X.Y>/bin`
  on macOS, `%APPDATA%\Python\Python<XY>\Scripts` on Windows, `~/.local/bin`
  elsewhere — before falling back to `python -m vectr`. Candidates are
  stat-checked before they are executed, so a machine with none of them costs
  no extra process spawns.
- The not-found message lists every location searched instead of claiming
  only PATH was checked, and the output channel names which CLI was launched.
- The `python -m vectr` fallback now spawns with its `-m vectr` arguments
  intact; it previously reported success and then spawned the bare
  interpreter with vectr's arguments.

### Hygiene
- Generated editor-guidance files list `vectr_distill`, `vectr_revoke`, and
  `vectr_reinstate`, which shipped in 1.5.0 but were missing from the tool
  tables.
- The proxy-anchor manifest version is derived from the manifest file instead
  of being duplicated in code, so the two cannot silently drift.
- New injection-utility A/B harness under `benchmarks/` — scenario set,
  mechanical scorer, and runner — with its non-vacuity check gated on
  confirmed delivery rather than on retrieval alone.

## 1.5.0 — 2026-07-23

Automatic episode capture with arc distillation, a note lifecycle with
revoke/reinstate, session resume, and an off-event-loop vector-store fix that
keeps the daemon responsive while its index compacts. MCP surface grows from
16 tools to 19.

### Working memory
- Episode capture: on editors with session hooks, each tool invocation is
  recorded as a deterministic local episode (command, outcome, failure
  markers) — no model calls involved. A streaming detector groups episodes
  into **arcs**: failure→success moments, e.g. a command that failed twice
  and then passed after an edit. `vectr status` / `GET /v1/status` report
  `episodes_count` and `arcs_pending_distill`.
- New `vectr_distill` MCP tool + `GET /v1/arcs`: review pending arcs
  (oldest and most-confident first, token-bounded), persist the ones worth
  keeping as notes via `vectr_remember(..., distilled_from=[arc_id])`, and
  dismiss the rest with a reason.
- Note lifecycle: `vectr_revoke` flags a note as wrong **without deleting
  it** — future recall shows a deterrent framing ("previously believed …,
  revoked …") instead of the original content, so the mistake is not
  silently re-derived. `vectr_reinstate` reverses it.
  `vectr_remember(contradicts=N)` records a correction and revokes the old
  note in one step. All transitions land in a note lifecycle event log.
- Serving policy: new `operational` note kind with `command` triggers,
  delivered on the command lane's hook fast path; a per-turn injection
  ledger keeps any note from firing twice in the same turn, and fires are
  recorded only when actually delivered.
- New `vectr_resume` MCP tool + `vectr resume` CLI + `GET /v1/resume`:
  the most recent task note, the latest snapshot, and open gotchas with file
  anchors in one deterministic call, selected by the same shared helper
  session-start injection uses. Config: `behavior.resume.max_gotchas`
  (default 5).
- New note kind `decision` and recall sort `chronological` (oldest-first,
  dated index lines): architectural decisions accrue as notes and recall as
  an ADR-style timeline. Decisions are not auto-injected at session start.
- Commit provenance: a git post-commit hook records each commit's message
  and touched files as a working-memory note, so "when did we change X"
  recalls instead of re-running `git log`.
- Note ages under an hour render as seconds/minutes in recall output
  instead of rounding to "0h".

### Search correctness
- Java `enum` methods now chunk correctly (the chunker descends through the
  enum member-group wrapper node), and the chunker recurses into any AST
  container node rather than class bodies only.
- Rust `impl`-block members resolve for qualified `Type.method` locate and
  trace.
- `vectr_trace` with a qualified `Class.method` validates the class
  qualifier before answering the callers path — no more caller lists for a
  same-named method on a different class.
- Near-miss suggestions for a misremembered qualified name are ranked by
  similarity, and caller ties are broken beyond name identity.
- Pointer-mode search keeps a result's code body when that result's own
  score clears the confidence floor, instead of demoting it with the rest.

### Reliability
- Vector-store calls no longer run on the request event loop: search/fetch
  dispatch through a dedicated executor, chunk counts are served from
  caches refreshed on every mutation and embed batch, and any store call
  slower than 5s logs a warning. Previously a single store call issued
  during the store's internal compaction could freeze every HTTP and MCP
  endpoint (observed: 41 minutes on a 2.6 GB index). New config:
  `indexing.vector_store_bridge.dispatch_max_workers` (must stay 1) and
  `indexing.vector_store_bridge.slow_call_warn_seconds`.
- Version stamps no longer absorb a foreign git SHA when vectr is installed
  editable under an unrelated repository checkout.
- The MCP server reports its version from package metadata instead of a
  hardcoded constant.

### Hygiene
- Two staleness tests pinned file mtimes explicitly, fixing a CI-only
  coarse-clock flake.
- GitHub release creation is skipped when the tag's release already exists,
  making the release workflow re-runnable.

## 1.4.0 — 2026-07-20

Search-correctness fixes, leaner MCP responses, and acceptance-corpus hardening.

### Search correctness
- Zig: struct- and enum-scoped `const`/`var` declarations are now extracted as
  symbol-graph members (previously silently dropped; function-locals stay
  excluded). Symbol schema v12.
- Qualified `Class.method` locate now resolves modifier-prefixed class
  declarations (`public class`, `export default class`, `export abstract
  class`, …) across Java/TypeScript/JavaScript.
- The low-confidence "may be unrelated" banner no longer false-fires on
  high-confidence paraphrase matches: new config key
  `ranking.notfound_floor.ce_override_min_relevance` (default 0.70) suppresses
  the zero-vocabulary trigger when the top result's cross-encoder relevance is
  confidently high. Genuine misses still banner; set above 1.0 to disable.
- `class_importance` no longer counts barrel re-export lines (`export { X }
  from`, `export * from`) as usage, removing a display-order inversion in
  re-export-heavy JS/TS codebases.
- `ranking.class_importance.lambda` default raised 0.25 → 0.35 after a
  full-corpus regression audit (recovers two known ranking regressions with
  zero regressions elsewhere).

### Leaner MCP responses
- All MCP text output renders workspace-relative paths, with the absolute
  root printed once per response header. `vectr_fetch` accepts both relative
  (new canonical) and absolute (back-compat) chunk ids. The absolute-path
  prefix previously accounted for ~9% of a default search response, ~26% of
  pointer mode, and ~42% of `vectr_evict_hint`.
- `vectr_evict_hint` renders each chunk once, in id-ready form
  `relpath:start-end  (symbol)`, with a single fetch template — ~70% of the
  old payload was duplicate serialization.
- Low-confidence pointer mode is now actually slim: the duplicated
  symbol-graph section is deduped against the pointer list and the re-fetch
  footer is dropped.
- `vectr_remember`'s tool schema trimmed (−317 tokens on tools/list) with no
  parameter, enum, default, or required-field changes.

### Hygiene
- Acceptance corpus cases now carry an `embed_model_stamp`; the harness
  reports stamp/embedder mismatches (informational) so a future embedder swap
  cannot silently stale-ify verified labels.
- README version/tool counts refreshed (15 MCP tools); generated-guidance
  template now lists `vectr_promote` and documents `vectr_remember`'s
  `triggers` argument.
- `scripts/release.sh`: one-command tag + push + GitHub release.

## 1.3.0 — 2026-07-18

Per-memory trigger engine, stdio MCP transport, instant memory readiness.

### Per-memory trigger engine
- `vectr_remember` accepts `triggers`: explicit per-note conditions for when
  a note resurfaces — `path` globs, lifecycle `event`s (session-start,
  prompt-submit, pre-edit, pre-run, pre-commit, post-compaction), exact
  `symbol` references resolved against the code symbol graph, `semantic`
  prompt-similarity with a fixed per-kind threshold, and temporal guards
  (`not_before`, `expires_visibility`, `cooldown`). Conditions AND within an
  entry and OR across entries; omitting `triggers` keeps the kind defaults.
- Trigger evaluation is wired into the live hook pipeline (session-start /
  prompt-submit / pre-tool-use / pre-compact) with a per-session fire
  ledger, cumulative injection budgets, scope enforcement, and
  double-injection prevention against the legacy relevance path.
- Provenance classes on notes with framing gates: auto-captured content is
  injected as epistemic memory, never as imperative instruction; forged
  human-provenance writes are rejected at both MCP and store boundaries.
- New `vectr_promote` MCP tool (the 15th): raise a reviewed auto-captured
  note's trust class one step (`auto` → `agent`). Promotion to `human` is
  reserved for user-side surfaces; the full chain remains available on REST.
- Kind-default scopes are resolved at write time; path triggers match both
  absolute and workspace-relative forms.

### Transport and readiness
- New foreground stdio MCP transport: `vectr mcp-stdio`.
- Two-phase service init: memory tools are live from process start on every
  transport, and warm-up notes are vector-backfilled when the embedder
  attaches — remember/recall no longer wait on model load.
- `vectr hook <event>` runs a stdlib-only fast path, cutting the per-hook
  subprocess import tax.

### Availability and correctness
- Embedding provider's torch thread pool is capped and MCP tool dispatch
  runs off the event loop; a full-workspace index or embed burst no longer
  starves concurrent requests.
- Fixed a shutdown-vs-init race in two-phase startup.
- `VECTR_WORKSPACE` pointing at a nonexistent path now fails loudly instead
  of silently indexing nothing.
- `vectr_map`'s raw-metadata path walks only indexed files (no more venv
  walks); the workspace fingerprint scan honors indexer exclusions.
- Injection packing stops at the first eviction and never backfills
  lower-ranked items; the per-turn recall relevance floor is config-driven
  (default 0.72).

### Tooling and display
- One displayed score scale per result set; resolved note scope is surfaced
  back to the caller; `vectr_status` nudges about stale task notes; the
  instruction-style label is renamed `memory-first`; a failing acceptance
  case can no longer crash the harness run.

### Research artifacts
- New `research/` directory: published evaluation artifacts for the
  brain-memory work — controlled-matrix and forced-compaction decay-probe
  protocols, graders, and complete run archives with per-directory READMEs.
  Not part of the PyPI package.

## 1.2.0 — 2026-07-13

Proactive context injection and daemon availability under load. All new
behavior is opt-in; with no new flags or consent given, behavior is
unchanged.

### Proactive context (new, opt-in)
- New `vectr proxy` command: a localhost, Anthropic-API-shaped proxy that
  deterministically injects relevant working-memory notes into an AI
  agent's request context when their trigger conditions match — no reliance
  on the agent voluntarily calling recall. Injection is consent-gated (a
  proactive master switch plus per-launch consent for the proxy channel)
  and fully observable: injection counts and end-to-end state are exposed
  in `/v1/status` and rendered in the proxy banner.
- Scored recall behind a new `/v1/proactive` endpoint; injections are
  budgeted, deduplicated, and fail open (an injection-path error never
  blocks the underlying request — it is logged and bypassed).
- Org-wide artifact cache and exact-match response cache wired into the
  proactive path.

### Daemon availability
- `/v1/index` now runs off the event loop (threadpool), so a full-workspace
  index call no longer blocks every other request for its duration.
- New `reindex_in_progress` field in `/v1/status`: a lock-free, always-cheap
  signal that bulk index work (an explicit index or the watcher's coalesced
  batch) is running right now.
- Per-language index statistics no longer trigger a full metadata scan on
  every status call while the index is changing; chunk totals read the
  vector store's native count directly.

### Watcher
- Live file events now honor `.gitignore` exactly like the bulk indexer
  (previously only `.vectrignore` was consulted on create/modify/delete/move
  events, so a gitignored file could enter the index through a live edit).

### Fixes
- `vectr forget --all` sweeps the current cache layout, not only the legacy
  nested layout.
- The proxy banner's status probe retries with a longer timeout instead of
  reporting a transient failure.

## 1.1.1 — 2026-07-12

### Indexing
- Indexing now streams embeddings into the vector store batch by batch
  instead of accumulating every embedding for the whole workspace in memory
  before writing. Both embedding passes (content and purpose vectors)
  previously held full-workspace embedding lists concurrently — gigabytes of
  memory on large workspaces, pushing memory-constrained machines into swap
  and slowing the second pass several-fold while it paged. Peak memory for
  this step is now one embedding batch. Indexed content is unchanged: chunk
  ids, documents, metadata, and embedding values are identical, so existing
  indexes remain valid and no reindex is required.

## 1.1.0 — 2026-07-11

Security and shared-deployment features. All opt-in; with none of the new
environment variables or flags set, behavior is unchanged — local, keyless,
zero-config stays the default.

### Authentication
- API key comparison now uses a constant-time comparison (`hmac.compare_digest`)
  instead of `!=`, removing a response-timing side channel. `VECTR_API_KEY`
  protects both the REST `/v1` routes and the `/mcp` endpoint; `/v1/health`
  stays open for liveness probes; the key is never echoed in responses or logs.
- New `vectr key` command prints a fresh high-entropy key (stdout) with usage
  guidance (stderr); vectr never persists it. Generated keys never start with
  `-` (a leading dash made `--api-key <key>` parse as a flag and fail); the
  usage guidance shows the always-safe `--api-key=<key>` form.
- When `VECTR_API_KEY` is set at start time, the editor MCP configs vectr
  writes (`.mcp.json`, `.cursor/mcp.json`, `.vscode/mcp.json`) include the
  `X-Api-Key` header so the editor keeps reaching its own authenticated daemon.
  These files then hold the key in plaintext; the CLI and docs warn to treat
  them as secrets and keep them out of shared version control.

### Team mode (shared central instance)
- `vectr start --host <addr>` selects the daemon bind address (default
  `127.0.0.1`). A non-loopback bind refuses to start unless `VECTR_API_KEY` is
  set — a network-reachable index is never served unauthenticated.
- New `vectr connect --url <url> [--api-key <key>] [--label <name>]` configures
  the local editor to use a remote vectr instance: writes the MCP configs with
  the auth header (and optional `X-Vectr-Client` attribution label) and the
  guidance block, spawning no local daemon. Working memory on the central
  instance is shared: a note one connected agent stores, every other can
  recall. Search results reference the server's indexed checkout.
- The client label attributes notes (author) and audit-log lines.
- Notes DB connections set an SQLite busy timeout so concurrent writes from
  multiple clients wait for the lock instead of failing.

### Encryption at rest
- `VECTR_ENCRYPT_KEY` now also encrypts note titles (previously plaintext, and
  derived from the first content line — leaking note content) and snapshot
  payloads (previously plaintext JSON embedding full decrypted note text).
  Reads are tolerant: pre-existing plaintext rows keep working.
- The passphrase can be stored in the OS keychain (service `vectr`, username
  `encrypt-key`) instead of the environment; the env var wins when both exist.
- New `VECTR_ENCRYPT_DISABLE_NOTE_VECTORS=1` omits note embedding vectors (a
  lossy plaintext projection of note text) for the strictest posture; recall
  falls back to exact-text matching.
- Honest boundary, unchanged: the code index (chunk text + vectors) is NOT
  encrypted — documented in docs/data-handling.md and the README.

### Data handling and retention
- The audit log is now strictly opt-in: `VECTR_AUDIT_LOG` unset means nothing
  is recorded (previously it silently defaulted to `~/.vectr/audit.log`,
  recording every recall query undisclosed). When enabled it also records
  INDEX and SEARCH events, and rotating-handler initialization is now
  race-free under concurrent writes.
- Purge is now complete: `vectr_forget(all=true)` / `POST /v1/forget` /
  `/v1/memory/clear` also delete the workspace's snapshots (whose payloads
  embed note contents), and `vectr forget --all` additionally clears snapshots
  and note embedding vectors across all workspaces. Previously all of these
  deleted only the notes table.
- `VECTR_NOTES_TTL_DAYS` (existing startup TTL purge) is now documented and
  covered by tests; unset = notes are kept until deleted.
- New data-handling policy: docs/data-handling.md — what vectr stores, where,
  plaintext vs encrypted, retention, deletion, team-mode caveats.

### Filesystem
- `~/.cache/vectr/` and `~/.vectr/` (and per-workspace subdirectories) are
  created owner-only (0700) on POSIX systems; existing directories are
  tightened at startup.

### Proactive context (experimental, off by default)
- New `vectr proxy` command runs a localhost Anthropic-shaped API proxy the
  agent harness targets with `ANTHROPIC_BASE_URL`. It forwards `/v1/messages`
  (and everything else) to the real API transparently — streaming SSE and
  tool_use pass through byte-for-byte, and the upstream API key is forwarded
  untouched and never stored or logged. Localhost-only: a non-loopback bind is
  refused. Bypass it at any time by unsetting `ANTHROPIC_BASE_URL`.
- When injection is enabled and the workspace daemon is running, the proxy
  appends deterministic proactive context (matched working-memory notes /
  structural matches) **after the last prompt-cache breakpoint**, so existing
  cache prefixes are never invalidated. Injection is fail-open: if the
  intelligence layer errors or exceeds a tight time budget, the request is
  forwarded unmodified; a proxy that cannot reach upstream returns an honest
  upstream-shaped error.
- Injection triggering is deterministic — structural exact file-path matches +
  a numeric cosine floor + additive packing — with a per-request item/char
  budget and per-session dedup/cooldown. No keyword/regex classification of
  conversation content anywhere.
- New `POST /v1/proactive` daemon route returns packed context for an assembled
  window (used by the proxy); scored recall (`recall_scored`) surfaces the
  per-note cosine similarity the semantic path already computes.
- Org-wide artifact cache (`proactive.cache`, off by default): caches
  `/v1/search` and scored-recall results keyed by exact identity + the current
  index epoch, so a re-index or note change invalidates automatically. On a
  team instance the cache is shared by every connected client. Exact-match
  local LLM-response caching in the proxy (`proactive.cache.response_cache`,
  off by default) serves a cached response only for a byte-identical request
  within a TTL; semantic-similarity response caching is deliberately not
  offered (see the design doc's cache-safety analysis).
- `vectr status` gains proactive-injection counts and artifact-cache metrics
  (hits/misses/hit-rate/entries/estimated tokens saved) when either is active.
  A `PROACTIVE_INJECT` audit event records metadata only (channel, item count,
  anchor ids) — never conversation text or note bodies.
- Config: new `proactive:` block in `config.yaml`; env overrides under the
  `VECTR_PROACTIVE*` prefix. All off by default; with nothing set, behavior is
  unchanged.

## 1.0.3 — 2026-07-11

- Added the MCP Registry ownership marker (`mcp-name: io.github.swapnanil/vectr`) to the README as an HTML comment — required by registry.modelcontextprotocol.io to validate that the PyPI package and the registry entry belong to the same publisher.
- `server.json` added for the Official MCP Registry listing (`pypi` package, `streamable-http` transport at `http://localhost:{port}/mcp`).
- README version badge and version line corrected (still said 1.0.0).

## 1.0.2 — 2026-07-11

- PyPI project page now renders the README (`readme` was missing from package metadata), plus license, author, and project URL metadata.
- Removed the vestigial `LLM_MODEL` env var, `agent/llm_client.py`, and the dead `model` field it echoed into `/v1/health`, `/v1/index`, `/v1/search`, and `/v1/status` responses. Vectr makes no LLM calls; the field had no consumers.
- `.env.example` updated to the shipped defaults: `ibm-granite/granite-embedding-english-r2` embedding model; removed the unused `ANTHROPIC_API_KEY` entry.
- `.vectrignore` untracked (repository-local indexing config, not product content).

## 1.0.1 — 2026-07-11

- Package description now covers both halves of the product: semantic codebase search and persistent working memory. The 1.0.0 description mentioned only the indexer.
- Install instructions updated to `pip install vectr` (PyPI release) in README, extension README, and docs page.
- Personal development configuration untracked from the repository (`CLAUDE.md`, `.mcp.json`, `.cursor/mcp.json`); local absolute paths in benchmark data and harness scripts genericized.

## 1.0.0 — 2026-07-08

**Semantic search**
- Hybrid dense (`ibm-granite/granite-embedding-english-r2`, local and overridable) + BM25 retrieval, weighted by codebase size/documentation characteristics.
- Dual-vector indexing: a body-stripped "purpose" vector (signature + docstring) alongside the full-body vector, closing the embedding-dilution gap where a symbol's own doc paraphrase of a query fails to surface it in the dense pool.
- Symbol graph with call edges, import chains, and HTTP route extraction (Flask/FastAPI/Express/Spring); `vectr_locate` (5 fallback strategies) and `vectr_trace` for callers/callees.
- AST-aware chunking across 7 languages (Python, JavaScript, TypeScript, Go, Rust, Java, Zig) plus C and C++; all other file types fall back to overlapping window chunking.

**Working memory**
- Five note kinds (`directive`, `task`, `gotcha`, `finding`, `reference`), each with distinct injection semantics — directives fire unconditionally, gotchas resurface when their anchored file is touched.
- Two-tier `vectr_recall`: a token-bounded index by default, full note bodies on request (`note_id=` or `detail='full'`).
- Session-start boot injection of directives and high-priority tasks, recency-ordered.
- `vectr_snapshot` / `vectr_snapshot_list` to checkpoint and browse working-memory state.
- Multi-agent shared memory: workspace-scoped notes act as a shared bus for orchestrator/subagent handoff, with explicit (never inferred) `agent` attribution on `vectr_remember`.

**Context relief**
- `vectr_fetch`: deterministic, byte-verbatim re-fetch of a previously-seen chunk by id, with a truncation warning when the index's own storage cap capped the original chunk.
- `vectr_evict_hint`, with recency-ordered re-fetch ids, so the calling model knows what it can safely drop from context.
- Remember nudges with dual-gate escalation (chunk count and token count) so a light reminder scales into a stronger banner only once both thresholds are crossed.

**Editor integration**
- Zero-config MCP setup: auto-written config for Claude Code, Cursor, and VS Code / GitHub Copilot; manual config documented for Windsurf, Cline, and Continue.
- Auto-generated `CLAUDE.md` guidance template describing the tool surface and when to use each tool.
- Claude Code session hooks (`SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PreCompact`) for automatic recall injection, with an injection-observability status line.

**Ops**
- `--memory-only` daemon mode: working memory + hooks without code indexing or the file watcher, for actively-edited or performance-sensitive projects.
- File watcher burst governor and RSS self-limit to bound resource use during large or rapid file-change bursts.
- Multi-workspace instance registry (`vectr status --all`, `vectr stop --all`) and per-workspace status surfaces.
