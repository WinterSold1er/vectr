"""Deterministic, streaming failure->success arc detection.

Pure functions plus one small stateful detector class — no HTTP-layer
dependency, no imports from `app.routes` / `app.service`. Interface
contract: `ArcDetector.observe(episode: dict) -> list[Arc]`.

An "arc" is a discovery moment: one or more failed attempts at a command
followed by a success that resolves them, chained together. This module
never writes notes, never calls an LLM, and never touches query/prompt
content — every classification here is over TOOL OUTPUT (episode outcome,
markers) or TOOL-CALL argv structure (via app.cmdnorm), both sanctioned
under R5.

`episode` dict — the fields this module reads (the episode schema, using
the in-memory/API-level names, not the `_json`-suffixed DB column names):
`session_id`, `ts`, `cwd`, `tool` ("bash" | "edit"), `cmd_raw`, `outcome`
("success" | "failure" | "soft_failure" | "interrupted" | "unknown"),
`termination`, `markers` (list[str] of matched marker ids), `env_delta_names`
(list[str]), `file_path` (edit rows). `verb`/`flags`/`args`, if already
present, are used only as a fallback when `cmd_raw` is absent (e.g. a
caller that pre-normalized) — whenever `cmd_raw` is present it is always
the source of truth, so both this detector and the persisted episode row
derive from the identical string via app.cmdnorm.
"""
from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agent.config import (
    ARC_FLAKE_NEAR_THRESHOLD_MIN,
    ARC_FLAKE_SUPPRESS_MIN_COUNT,
    ARC_MUTATION_BAND_MAX,
    ARC_MUTATION_BAND_MIN,
    ARC_SIMILARITY_ARG_WEIGHT,
    ARC_SIMILARITY_FLAG_WEIGHT,
    ARC_SIMILARITY_VERB_SOFT_MATCH_MIN_RATIO,
    ARC_SIMILARITY_VERB_SOFT_MATCH_SCORE,
    ARC_SIMILARITY_VERB_WEIGHT,
    ARC_TRANSIENT_MARKER_IDS,
    ARC_TS_MONOTONIC_FALLBACK_SECONDS,
    ARC_WINDOW_MAX_COMMANDS,
    ARC_WINDOW_MAX_PENDING_PER_VERB_FAMILY,
    ARC_WINDOW_TTL_SECONDS,
)
from app.cmdnorm import (
    CD_ABSOLUTE,
    CD_NONE,
    CD_RELATIVE,
    CD_SYMBOLIC_HOME,
    NormalizedCommand,
    classify_arg,
    leading_cd_resolution,
    normalize_command,
)

_PENDING_OUTCOMES = frozenset({"failure", "soft_failure"})
_IGNORED_OUTCOMES = frozenset({"interrupted", "unknown", None})
# agent/outcome.py's TERMINATION_VALUES that mean "the run was cut short by
# the user or a signal, not by its own completion" — trap (d): none of these
# may ever enter or resolve a pending arc, independent of the `outcome`
# check above (adversarial review 2026-07-22).
_INTERRUPTED_TERMINATIONS = frozenset({"interrupted", "cancelled", "signal"})
_MUTATION_DIFF_AXES = ("verb", "flag", "arg", "env", "files")


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class Arc:
    """One emitted failure(s)->success discovery moment.

    `failures_chain` is oldest-first; `mutation_diff` maps a subset of
    {verb, flag, arg, env, files} to an (old, new) pair — only the axes
    that actually changed are present. `cwd` is never a mutation_diff axis:
    it is a pending-bucket key (review 2026-07-22; refined to the command's
    EFFECTIVE directory, not just the raw `cwd` field, by
    UPG-ARC-CWD-VS-EFFECTIVE-DIR — see `_effective_cwd`), so it cannot
    differ between a chain's failures and its resolving success by
    construction. `confidence` is "normal" or "low" (a chained failure's
    stderr matched a transient-error marker).
    """

    session_id: str
    failures_chain: tuple[dict[str, Any], ...]
    success: dict[str, Any]
    mutation_diff: dict[str, tuple[Any, Any]]
    confidence: str = "normal"


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def _verb_component(a: str, b: str) -> float:
    if a == b:
        return 1.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    if ratio > ARC_SIMILARITY_VERB_SOFT_MATCH_MIN_RATIO:
        return ARC_SIMILARITY_VERB_SOFT_MATCH_SCORE
    return 0.0


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _args_component(a: NormalizedCommand, b: NormalizedCommand) -> float:
    if len(a.args) == len(b.args) and a.args:
        # Same arity: per-position Levenshtein on the RAW tokens is more
        # sensitive than the abstracted class (a path swap within the same
        # <PATH> class must still show up as a real difference).
        ratios = [
            difflib.SequenceMatcher(None, x, y).ratio()
            for x, y in zip(a.args, b.args)
        ]
        return sum(ratios) / len(ratios)
    if not a.args and not b.args:
        return 1.0
    # Different arity: jaccard over the abstracted classes, penalized by
    # the relative arity gap. Without this penalty, adding/removing a
    # positional argument (e.g. `cp src/a.py` -> `cp src/a.py src/b.py`,
    # both all-<PATH>) would jaccard to a perfect 1.0 purely because the
    # SET of classes present is unchanged — but 1.0 is reserved for
    # is_identical_command's exact-equality check (an identical normalized
    # command is never a mutation), so an unpenalized 1.0 here would make
    # a genuine "added a missing argument" fix silently invisible to the
    # mutation-band check.
    class_jaccard = _jaccard(frozenset(a.arg_classes), frozenset(b.arg_classes))
    arity_ratio = min(len(a.args), len(b.args)) / max(len(a.args), len(b.args))
    return class_jaccard * arity_ratio


def similarity(a: NormalizedCommand, b: NormalizedCommand) -> float:
    """Composite mutation-similarity score: 0.5*verb + 0.3*jaccard(flags) +
    0.2*jaccard(args), verb Levenshtein-softened above the configured
    ratio. Identical normalized commands score exactly 1.0."""
    verb_score = _verb_component(a.verb, b.verb)
    flag_score = _jaccard(frozenset(a.flags), frozenset(b.flags))
    arg_score = _args_component(a, b)
    return (
        ARC_SIMILARITY_VERB_WEIGHT * verb_score
        + ARC_SIMILARITY_FLAG_WEIGHT * flag_score
        + ARC_SIMILARITY_ARG_WEIGHT * arg_score
    )


def is_mutation_band(score: float) -> bool:
    return ARC_MUTATION_BAND_MIN <= score < ARC_MUTATION_BAND_MAX


def is_identical_command(a: NormalizedCommand, b: NormalizedCommand) -> bool:
    """True iff two normalized commands are textually the same invocation
    (same verb, same flag set, same positional args in order) — the
    identical-command branch, decided by direct equality rather than a
    similarity-score threshold (never subject to floating-point slack)."""
    return a.verb == b.verb and frozenset(a.flags) == frozenset(b.flags) and a.args == b.args


def _command_signature(n: NormalizedCommand) -> tuple[Any, ...]:
    return (n.verb, frozenset(n.flags), n.args)


def _verb_family(n: NormalizedCommand) -> str:
    """Coarse bucketing key for the pending-failure window, keyed by
    verb-family — the first token of the normalized verb (the invoked
    binary: `pip`, `npm`, `git`, `./mvnw`), NOT the full absorbed verb
    string.

    The verb-absorption cap (app.cmdnorm) folds a trailing bareword target
    into the verb itself for shapes like `pip install requests` or `npm run
    build` — so the token that actually varies between two retries of the
    "same" command
    (a package name, a script name, a branch) can end up INSIDE the verb
    string rather than in args. Bucketing on the full verb would then put
    every distinct retry in its own singleton bucket and the fine-grained
    similarity() comparison would never run. Bucketing on the family (first
    token) instead guarantees same-tool invocations land together; the
    actual match/no-match decision is still made entirely by similarity()
    over the complete verb/flags/args, so unrelated same-family commands
    (`git commit` vs `git checkout`) are filtered there, not here."""
    return n.verb.split(" ", 1)[0] if n.verb else ""


def _effective_cwd(n: NormalizedCommand, episode: dict[str, Any]) -> Any:
    """The directory a command actually ran in, for bucket-key purposes
    (UPG-ARC-CWD-VS-EFFECTIVE-DIR): the episode's `cwd` FIELD is only
    where the harness started the command, not necessarily where it ran —
    an agent that writes `cd /other/repo && cmd` inside `cmd_raw` changes
    the effective directory without changing the episode's own `cwd`
    field, so two episodes sharing a harness cwd can still execute in
    unrelated repos.

    Resolution rules (see `app.cmdnorm.leading_cd_resolution` for the full
    classification — structural command-shape parsing, same category as
    the rest of app.cmdnorm's verb/flag/arg normalization, never
    query-content classification):
      - no leading `cd`          -> the episode `cwd` field (pre-fix behavior);
      - absolute target          -> the target itself, lexically normalized
        (`os.path.normpath` only — never `abspath`, which would consult
        THIS process's cwd, and never `realpath`, which would touch the
        filesystem and break the detector's replay determinism);
      - relative target          -> composed with the `cwd` field, again
        normpath'd — so `cwd=/work/A` + `cd sub && x` keys as `/work/A/sub`
        and can never pair with `/work/B`'s `sub`;
      - tilde/$HOME forms        -> the symbolic tilde-form itself: constant
        per session/machine even though its absolute value is unknowable;
      - unresolvable (`cd -`, `$var` targets, declined shapes) or a
        relative target with NO usable `cwd` field to compose against ->
        None, meaning "refuse": the caller must not bucket this episode at
        all (see `_bucket_key`). Refusing costs a missed lesson; falling
        back to a KNOWN-wrong directory key risks a false lesson the
        distiller may persist into memory permanently."""
    status, target = leading_cd_resolution(n.cmd_raw)
    if status == CD_NONE:
        return episode.get("cwd")
    if status == CD_ABSOLUTE:
        return os.path.normpath(target)
    if status == CD_SYMBOLIC_HOME:
        return target
    if status == CD_RELATIVE:
        base = episode.get("cwd")
        if not isinstance(base, str) or not base.strip():
            return None
        return os.path.normpath(os.path.join(base, target))
    # CD_UNRESOLVABLE (imported vocabulary lives in app.cmdnorm): refuse.
    return None


def _bucket_key(n: NormalizedCommand, episode: dict[str, Any]) -> tuple[str, Any] | None:
    """Pending-failure bucket key (review 2026-07-22, refined
    UPG-ARC-CWD-VS-EFFECTIVE-DIR): `(verb-family, effective_cwd)`, or None
    when the effective directory is unresolvable — a refused episode
    neither enters pending nor resolves one (an arc that cannot be placed
    truthfully is never offered at all).
    effective_cwd is a BUCKET KEY, not a mutation axis — the original draft
    listed cwd on both sides, which is contradictory: cwd-as-mutation-axis
    binds unrelated repos' builds into false arcs (the same command failing
    in `/work/serviceA` and succeeding in `/work/serviceB` must never
    match), while the genuine "cd to the right dir fixed it" case normally
    has the `cd` inside the same invocation, where normalization already
    strips it (`app.cmdnorm._strip_leading_cd`) — so cross-cwd matching is
    nearly all downside. See `_effective_cwd` for why the bucket key is
    derived from `cmd_raw`'s leading `cd`, not the raw `cwd` field alone."""
    key_dir = _effective_cwd(n, episode)
    if key_dir is None:
        return None
    return (_verb_family(n), key_dir)


# ---------------------------------------------------------------------------
# Timestamp parsing — always derived from episode data, never wall clock, so
# the detector is exactly reproducible against a historical transcript.
# ---------------------------------------------------------------------------


def _parse_ts(ts: Any) -> float:
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        s = ts.strip()
        try:
            return float(s)
        except ValueError:
            pass
        try:
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s).timestamp()
        except ValueError:
            pass
    raise ValueError(f"Unparseable episode timestamp: {ts!r}")


# ---------------------------------------------------------------------------
# mutation_diff: which of {verb, flag, arg, env, files} changed, old->new —
# cwd is the bucket key, never a diff axis (review 2026-07-22).
# ---------------------------------------------------------------------------


def diff_episodes(
    old_norm: NormalizedCommand,
    old_episode: dict[str, Any],
    new_norm: NormalizedCommand,
    new_episode: dict[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Structural diff over {verb, flag, arg, env} — only the axes that
    actually changed are present. `cwd` is deliberately never computed
    here: it is the pending-bucket key (review 2026-07-22), so
    `old_episode`/`new_episode` always share the same cwd by construction
    whenever this is called."""
    d: dict[str, tuple[Any, Any]] = {}
    if old_norm.verb != new_norm.verb:
        d["verb"] = (old_norm.verb, new_norm.verb)
    old_flags, new_flags = frozenset(old_norm.flags), frozenset(new_norm.flags)
    if old_flags != new_flags:
        d["flag"] = (tuple(sorted(old_flags)), tuple(sorted(new_flags)))
    if old_norm.args != new_norm.args:
        d["arg"] = (old_norm.args, new_norm.args)
    old_env = frozenset(old_episode.get("env_delta_names") or ())
    new_env = frozenset(new_episode.get("env_delta_names") or ())
    if old_env != new_env:
        d["env"] = (tuple(sorted(old_env)), tuple(sorted(new_env)))
    return d


def _normalize_episode(episode: dict[str, Any]) -> NormalizedCommand:
    cmd_raw = episode.get("cmd_raw")
    if cmd_raw:
        return normalize_command(cmd_raw)
    # Fallback: caller already supplied a normalized triple directly (no
    # cmd_raw available) — build a NormalizedCommand from it rather than
    # requiring every caller to fabricate a command string.
    args = tuple(episode.get("args") or ())
    return NormalizedCommand(
        verb=str(episode.get("verb") or ""),
        flags=tuple(sorted(episode.get("flags") or ())),
        args=args,
        arg_classes=tuple(classify_arg(a) for a in args),
        env_prefix_names=tuple(episode.get("env_delta_names") or ()),
        cmd_raw="",
    )


# ---------------------------------------------------------------------------
# Detector state machine
# ---------------------------------------------------------------------------


@dataclass
class _PendingFailure:
    episode: dict[str, Any]
    normalized: NormalizedCommand
    ts: float
    command_index: int


@dataclass
class _EditRecord:
    episode: dict[str, Any]
    ts: float
    command_index: int
    file_path: str | None


@dataclass
class _SessionState:
    command_count: int = 0
    last_ts: float = 0.0
    # Keyed by (verb-family, cwd) — review 2026-07-22.
    pending: dict[tuple[str, Any], list[_PendingFailure]] = field(default_factory=dict)
    edits: list[_EditRecord] = field(default_factory=list)
    # Keyed by the identical-command signature that has proven flaky
    # (flipped outcome with no intervening edit/env delta — cwd cannot
    # differ within a bucket).
    flake_counts: dict[tuple[Any, ...], int] = field(default_factory=dict)


class ArcDetector:
    """Streaming, per-session failure->success arc detector.

    One instance is expected to live for the daemon's process lifetime (or
    per-workspace); sessions are tracked independently and never compared
    across each other.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionState] = {}

    # -- public API ----------------------------------------------------

    def observe(self, episode: dict[str, Any]) -> list[Arc]:
        session_id = episode.get("session_id")
        if not session_id:
            return []
        tool = episode.get("tool")
        if tool not in ("bash", "edit"):
            return []

        state = self._sessions.setdefault(session_id, _SessionState())
        ts = self._resolve_ts(state, episode.get("ts"))
        state.command_count += 1
        state.last_ts = ts
        self._age_out(state)

        if tool == "edit":
            state.edits.append(
                _EditRecord(
                    episode=episode,
                    ts=ts,
                    command_index=state.command_count,
                    file_path=episode.get("file_path"),
                )
            )
            return []

        # tool == "bash"
        outcome = episode.get("outcome")
        if outcome in _IGNORED_OUTCOMES or episode.get("termination") in _INTERRUPTED_TERMINATIONS:
            # Trap (d): interrupted/unknown episodes never enter pending and
            # never resolve one either — an interleaved command that
            # "simply matches nothing". The termination check is a second,
            # independent gate on the outcome one above (adversarial review
            # 2026-07-22): outcome.py already maps a signal/cancelled
            # termination to outcome="interrupted" before this ever runs,
            # but this guard holds even if a future outcome-derivation path
            # regresses that ordering.
            return []

        normalized = _normalize_episode(episode)
        if not normalized.verb:
            # An episode that normalizes to an empty verb (`2>&1` alone, an
            # env-assignment-only command) carries no comparable command
            # structure — treated like outcome `unknown` (review
            # 2026-07-22): never enters pending, never resolves one.
            return []

        if outcome in _PENDING_OUTCOMES:
            self._add_pending(state, episode, normalized, ts)
            return []

        if outcome == "success":
            return self._match_success(state, session_id, episode, normalized)

        return []

    def _resolve_ts(self, state: _SessionState, ts_raw: Any) -> float:
        """Episode timestamps are drift-tolerant — a missing/None/
        unparseable `ts` must never raise (review 2026-07-22: the caller
        may not always supply one). Falls back to a small monotonic
        step past the session's own last-seen ts so window/TTL math stays
        well-defined and strictly ordered; deterministic for a fixed replay
        sequence, never wall-clock-derived."""
        if ts_raw is None:
            return state.last_ts + ARC_TS_MONOTONIC_FALLBACK_SECONDS
        try:
            return _parse_ts(ts_raw)
        except ValueError:
            return state.last_ts + ARC_TS_MONOTONIC_FALLBACK_SECONDS

    # -- window maintenance ---------------------------------------------

    def _in_window(self, state: _SessionState, ts: float, command_index: int) -> bool:
        return (
            (state.last_ts - ts) <= ARC_WINDOW_TTL_SECONDS
            and (state.command_count - command_index) <= ARC_WINDOW_MAX_COMMANDS
        )

    def _age_out(self, state: _SessionState) -> None:
        for bucket_key in list(state.pending.keys()):
            kept = [p for p in state.pending[bucket_key] if self._in_window(state, p.ts, p.command_index)]
            if kept:
                state.pending[bucket_key] = kept
            else:
                del state.pending[bucket_key]
        state.edits = [e for e in state.edits if self._in_window(state, e.ts, e.command_index)]

    def _add_pending(
        self,
        state: _SessionState,
        episode: dict[str, Any],
        normalized: NormalizedCommand,
        ts: float,
    ) -> None:
        # UPG-ARC-CWD-VS-EFFECTIVE-DIR refusal bias: an episode whose
        # effective directory cannot be determined truthfully never enters
        # pending at all.
        bucket_key = _bucket_key(normalized, episode)
        if bucket_key is None:
            return
        bucket = state.pending.setdefault(bucket_key, [])
        bucket.append(
            _PendingFailure(episode=episode, normalized=normalized, ts=ts, command_index=state.command_count)
        )
        if len(bucket) > ARC_WINDOW_MAX_PENDING_PER_VERB_FAMILY:
            bucket.pop(0)

    def _intervening_edits(self, state: _SessionState, from_index: int, to_index: int) -> list[_EditRecord]:
        return [e for e in state.edits if from_index < e.command_index <= to_index]

    def _confidence_for(self, chain: list[_PendingFailure]) -> str:
        for p in chain:
            markers = p.episode.get("markers") or ()
            if any(m in ARC_TRANSIENT_MARKER_IDS for m in markers):
                return "low"
        return "normal"

    def _build_chain(
        self, bucket: list[_PendingFailure], anchor_idx: int
    ) -> tuple[list[_PendingFailure], set[int]]:
        """Walk backward through older pending failures, chaining each one
        that is similar to the CURRENT frontier (the success first, then
        each newly-added chain member — a failure is chained if it is
        similar to either the success or the previous chain member).
        Non-matching interleaved failures are simply skipped, not treated
        as a stop condition."""
        anchor = bucket[anchor_idx]
        chain = [anchor]
        used = {anchor_idx}
        frontier = anchor.normalized
        frontier_index = anchor.command_index

        older = sorted(
            ((i, p) for i, p in enumerate(bucket) if i not in used and p.command_index < frontier_index),
            key=lambda t: t[1].command_index,
            reverse=True,
        )
        for i, p in older:
            if is_identical_command(p.normalized, frontier) or is_mutation_band(similarity(p.normalized, frontier)):
                chain.insert(0, p)
                used.add(i)
                frontier = p.normalized
                frontier_index = p.command_index
        return chain, used

    def _remove_indices(self, state: _SessionState, bucket_key: tuple[str, Any], indices: set[int]) -> None:
        bucket = state.pending.get(bucket_key, [])
        remaining = [p for i, p in enumerate(bucket) if i not in indices]
        if remaining:
            state.pending[bucket_key] = remaining
        else:
            state.pending.pop(bucket_key, None)

    def _match_success(
        self,
        state: _SessionState,
        session_id: str,
        success_episode: dict[str, Any],
        success_norm: NormalizedCommand,
    ) -> list[Arc]:
        bucket_key = _bucket_key(success_norm, success_episode)
        if bucket_key is None:
            # Same refusal bias as _add_pending: a success whose effective
            # directory is unknowable resolves nothing.
            return []
        bucket = state.pending.get(bucket_key)
        if not bucket:
            return []

        identical_idxs = [i for i, p in enumerate(bucket) if is_identical_command(p.normalized, success_norm)]
        if identical_idxs:
            anchor_idx = identical_idxs[-1]
            anchor = bucket[anchor_idx]
            edited = self._intervening_edits(state, anchor.command_index, state.command_count)
            # cwd is never checked here: it is the bucket key above, so
            # `anchor` and `success_episode` share it by construction
            # (review 2026-07-22).
            env_delta = frozenset(anchor.episode.get("env_delta_names") or ()) != frozenset(
                success_episode.get("env_delta_names") or ()
            )

            if edited or env_delta:
                chain, used = self._build_chain(bucket, anchor_idx)
                diff = diff_episodes(chain[0].normalized, chain[0].episode, success_norm, success_episode)
                if edited:
                    files = tuple(sorted({e.file_path for e in edited if e.file_path}))
                    diff["files"] = ((), files)
                arc = Arc(
                    session_id=session_id,
                    failures_chain=tuple(p.episode for p in chain),
                    success=success_episode,
                    mutation_diff=diff,
                    confidence=self._confidence_for(chain),
                )
                self._remove_indices(state, bucket_key, used)
                return [arc]

            # True flaky retry: identical command, no edit, no env delta —
            # suppressed, never emitted as an arc.
            signature = _command_signature(success_norm)
            state.flake_counts[signature] = state.flake_counts.get(signature, 0) + 1
            self._remove_indices(state, bucket_key, {anchor_idx})
            return []

        candidates: list[tuple[int, _PendingFailure, float]] = []
        for i, p in enumerate(bucket):
            score = similarity(p.normalized, success_norm)
            if is_mutation_band(score):
                candidates.append((i, p, score))
        if not candidates:
            return []

        filtered = []
        near_threshold_suppressed: set[int] = set()
        for i, p, score in candidates:
            sig = _command_signature(p.normalized)
            if score >= ARC_FLAKE_NEAR_THRESHOLD_MIN and state.flake_counts.get(sig, 0) >= ARC_FLAKE_SUPPRESS_MIN_COUNT:
                near_threshold_suppressed.add(i)
                continue
            filtered.append((i, p, score))

        if not filtered:
            # Every mutation-band candidate was near-threshold flaky noise —
            # resolve them out of pending without emitting an arc.
            self._remove_indices(state, bucket_key, near_threshold_suppressed)
            return []

        # Highest score wins; ties broken by recency (the most recent
        # pending failure is the more direct antecedent of this success) —
        # otherwise a stable sort on score alone would silently prefer the
        # oldest tied candidate and strand a more-recent equally-similar
        # failure in pending, unconsumed.
        filtered.sort(key=lambda t: (t[2], t[1].command_index), reverse=True)
        anchor_idx = filtered[0][0]
        chain, used = self._build_chain(bucket, anchor_idx)
        diff = diff_episodes(chain[0].normalized, chain[0].episode, success_norm, success_episode)
        arc = Arc(
            session_id=session_id,
            failures_chain=tuple(p.episode for p in chain),
            success=success_episode,
            mutation_diff=diff,
            confidence=self._confidence_for(chain),
        )
        used |= near_threshold_suppressed
        self._remove_indices(state, bucket_key, used)
        return [arc]
