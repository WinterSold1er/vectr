#!/usr/bin/env python3
"""EVAL-LONGITUDINAL-REDISCOVERY -- mechanical, arm-blind, per-leg scorer.

Grades one completed LEG against the ground truth its `LegSpec` declared before the
leg ran (see `scenarios.py`). Every verdict comes from exactly one of: the final bytes
of a workspace file (regex / occurrence count / sha256), the ordered `tool_use` blocks
of the leg's `stream-json` transcript, the exit code of a scenario-owned verify script,
or the transcript's own `result` / `assistant.message.usage` fields (DESIGN.md 6.1).
No LLM judges a run. Nothing here reads the agent's prose: an agent that merely SAYS it
will use `./tools/t` scores exactly like one that says nothing.

`score_run()` is deliberately arm-blind -- it takes a workspace, leg-start baselines
and a transcript, and has no `arm` parameter at all. Arm-specific reasoning lives only
in `leg_non_vacuity(arm=...)`, which answers a different question (did this leg's
memory channel actually fire?) and gates validity, never the outcome verdict.

The check primitives (`FileMatches`, `FileMatchCountAtMost`, `FileUnchanged`,
`CommandRan`, `VerifyCommand`, `AllOf`, `sha256_file`, plus this eval's own
`FileMatchCountAtLeast`) come from `scenarios.py` in this directory, which itself
re-exports the trap harness's primitives by import (see that file's docstring) --
never redefined here either.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

# `benchmarks/injection_utility/` has its own `scorer.py` AND its own `scenarios.py`.
# A plain `sys.path.insert` + `from scenarios import ...` keys off the bare module
# name `scenarios`, which both harnesses claim; whichever imports first in a shared
# interpreter (pytest collecting both harnesses' test files in one process) wins
# `sys.modules["scenarios"]` for the rest of the process, so the loser silently binds
# the wrong module's attributes (observed: `AttributeError` / stale `SCENARIOS`
# resolving to the trap harness's slugs). Loading THIS directory's `scenarios.py` by
# explicit file path under a fixed, private `sys.modules` key sidesteps the collision
# regardless of import order -- the same technique `scenarios.py`'s own
# `_load_trap_harness_scenarios()` already uses for the reverse direction. The key is
# checked first (not re-executed) so every caller in the process -- this module, and
# `tests/test_longitudinal_scorer.py`, which loads the identical key/path pair --
# converges on ONE module object, and therefore the same check-primitive classes
# `isinstance()` dispatch in `evaluate_check()` below depends on.
_LONGITUDINAL_SCENARIOS_KEY = "_vectr_eval_longitudinal_scenarios"


def _load_sibling_scenarios():
    cached = sys.modules.get(_LONGITUDINAL_SCENARIOS_KEY)
    if cached is not None:
        return cached
    path = Path(__file__).resolve().parent / "scenarios.py"
    spec = importlib.util.spec_from_file_location(_LONGITUDINAL_SCENARIOS_KEY, path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses (3.14) resolves a class's module via sys.modules[cls.__module__]
    # while building it, so the module must be registered before exec_module runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_scenarios = _load_sibling_scenarios()
AllOf = _scenarios.AllOf
CommandRan = _scenarios.CommandRan
FileMatchCountAtLeast = _scenarios.FileMatchCountAtLeast
FileMatchCountAtMost = _scenarios.FileMatchCountAtMost
FileMatches = _scenarios.FileMatches
FileUnchanged = _scenarios.FileUnchanged
LegSpec = _scenarios.LegSpec
NoteVariant = _scenarios.NoteVariant
VerifyCommand = _scenarios.VerifyCommand
sha256_file = _scenarios.sha256_file

ActionPattern = Any  # BashAction | PathAction | ContentAction | ToolAction (scenarios.py)


# ---------------------------------------------------------------------------
# Transcript reading
# ---------------------------------------------------------------------------


def load_transcript(path: Path) -> list[dict]:
    """Parse a stream-json transcript file (one JSON object per line).

    Unparseable lines are skipped: the CLI occasionally interleaves non-JSON
    diagnostics on stdout and a single bad line must not void a whole leg.
    """
    events: list[dict] = []
    if not path.exists():
        return events
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            events.append(ev)
    return events


# ---------------------------------------------------------------------------
# The action stream (DESIGN.md 6.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    idx: int  # position within the action stream (0-based)
    event_index: int  # position within the FULL event list this action's event sits at
    name: str
    input: Mapping[str, Any]


def build_action_stream(events: Sequence[dict]) -> list[Action]:
    """Every `tool_use` block, in transcript order. Text blocks are NOT actions --
    prose is not evidence (DESIGN.md 6.1, 6.2).
    """
    actions: list[Action] = []
    for ev_i, ev in enumerate(events):
        if ev.get("type") != "assistant":
            continue
        message = ev.get("message")
        if not isinstance(message, dict):
            continue
        for blk in message.get("content") or []:
            if isinstance(blk, dict) and blk.get("type") == "tool_use":
                actions.append(
                    Action(
                        idx=len(actions),
                        event_index=ev_i,
                        name=str(blk.get("name") or ""),
                        input=blk.get("input") or {},
                    )
                )
    return actions


def bash_commands(events: Iterable[dict]) -> list[str]:
    """Every `command` string from Bash tool_use blocks, in order."""
    return [
        str(a.input.get("command"))
        for a in build_action_stream(list(events))
        if a.name == "Bash"
        and isinstance(a.input.get("command"), str)
        and a.input.get("command").strip()
    ]


_PATH_INPUT_KEYS = ("file_path", "path", "notebook_path")


def _path_values(input_: Mapping[str, Any]) -> list[str]:
    return [str(input_[k]) for k in _PATH_INPUT_KEYS if input_.get(k) is not None]


# ---------------------------------------------------------------------------
# UPG-EVAL-TOUCHES-ANCHOR-SUBSTRING: anchor matching by path component / shell
# token, never a raw substring search.
#
# `a in v` (bare substring) matches whenever the anchor's characters appear ANYWHERE
# inside a longer string -- a `.bak` sibling (`config.py.bak` contains `config.py`),
# an unrelated file whose name happens to end the same way, or a Bash command that
# merely PRINTS the anchor's name (`echo config.py`) rather than touching the file.
# None of those are the agent inspecting the anchor. The fix scopes matching to the
# unit that actually denotes a path or a shell argument: a trailing PATH COMPONENT
# run for Read/Grep, and a whitespace/quoting-delimited TOKEN for Bash.
# ---------------------------------------------------------------------------


def _anchor_matches_path(value: str, anchor: str) -> bool:
    """True when `value`'s path components end with `anchor`'s path components (in
    order), or vice versa -- covers a deeper real path naming the same file
    (`src/pkg/config.py` for anchor `config.py`) without requiring either side to know
    the other's workspace root. Rejects a `.bak` sibling (`config.py.bak`'s trailing
    component is `config.py.bak`, not `config.py`) and an unrelated file that merely
    contains the anchor's characters (`old_config.py`), because neither shares a
    trailing path COMPONENT with the anchor -- a bare substring test would match both.
    """
    value_parts = PurePosixPath(value).parts
    anchor_parts = PurePosixPath(anchor).parts
    if not value_parts or not anchor_parts:
        return False
    shorter, longer = (
        (anchor_parts, value_parts)
        if len(anchor_parts) <= len(value_parts)
        else (value_parts, anchor_parts)
    )
    return longer[-len(shorter):] == shorter


# Commands whose own arguments are literal OUTPUT TEXT, never a file the shell or an
# invoked program opens -- `echo config.py` never touches `config.py`. Every genuine
# inspection command a scenario's own verify_hint uses (`grep`/`head`/`cat`/`python
# <path>`, DESIGN.md 7.2) is deliberately absent from this set and must keep counting.
_BASH_LITERAL_ARG_COMMANDS = frozenset({"echo", "printf"})

# Shell command separators a real shell allows between distinct invocations, reusing
# DEFECT 9's own separator set (`_EXEC_BOUNDARY`) so a multi-command Bash string is
# tokenized per invoked command, not as one run-on argument list.
_SHELL_SEGMENT_SEPARATOR = re.compile(r"[;&|\n]+")


def _bash_command_touches_anchor(command: str, anchors: Sequence[str]) -> bool:
    """True when `command` invokes an anchor path as a genuine shell TOKEN -- either
    as the program being run (`./tools/t status`, `scripts/envctl get FOO`) or as one
    of that program's own arguments (`head -5 config.py`) -- never as a substring of
    some unrelated longer token. Splits on shell command separators and tokenizes each
    segment with `shlex.split` (POSIX quoting rules) after stripping any leading
    interpreter/exec-wrapper prefix (`_strip_exec_prefixes`, DEFECT 9), then compares
    each token against every anchor by path component (`_anchor_matches_path`).
    """
    for segment in _SHELL_SEGMENT_SEPARATOR.split(command):
        segment = _strip_exec_prefixes(segment)
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            # Unbalanced quoting (e.g. a heredoc body) -- fall back to whitespace
            # splitting rather than dropping the segment; still token-boundary
            # matching, never a substring search.
            tokens = segment.split()
        if not tokens or tokens[0] in _BASH_LITERAL_ARG_COMMANDS:
            continue
        if any(_anchor_matches_path(token, anchor) for token in tokens for anchor in anchors):
            return True
    return False


# ---------------------------------------------------------------------------
# Exec-position anchoring (DEFECT 9)
#
# A plain `re.search` finds `pattern` ANYWHERE in a Bash `command` string, so
# `cat deploy.sh` or `grep -n foo deploy.sh` -- both read-only -- match a
# mistake_signature written to detect `./deploy.sh` just as readily as an actual
# `./deploy.sh staging` invocation. That collapses "the agent ran X" and "the agent's
# command MENTIONED X" into the same verdict, which is wrong whenever a signature's
# declared semantic is the former (DESIGN.md 6.2's BashAction/CommandRan primitives,
# `exec_anchor=True`). Anchoring instead requires the pattern to match starting at a
# genuine command-EXECUTION position: the start of the string, or immediately after
# a shell command separator (`;`, `&&`, `||`, `|`, `&`, or a newline), optionally
# through one or more interpreter/exec-wrapper tokens a real shell allows to precede
# the invoked program (`env`, one or more `VAR=value` assignments, `sh`/`bash`/`zsh`/
# `exec`, `timeout N`, `caffeinate -flags`).
# ---------------------------------------------------------------------------

_EXEC_PREFIX_TOKEN = re.compile(
    r"""
    \s*
    (?:
        [A-Za-z_][A-Za-z0-9_]*=(?:'[^']*'|"[^"]*"|\S*)\s+   # one VAR=value assignment
      | env\s+                                              # env (itself often followed
                                                              # by more VAR= tokens, looped)
      | (?:sh|bash|zsh|exec)\s+                              # interpreter / exec keyword
      | timeout\s+\S+\s+                                     # timeout N
      | caffeinate(?:\s+-\S+)*\s+                            # caffeinate [-flags]
    )?
    """,
    re.VERBOSE,
)

_EXEC_BOUNDARY = re.compile(r"\A|[;&|\n]")


def _strip_exec_prefixes(command: str) -> str:
    """Repeatedly strip leading whitespace and interpreter/exec-wrapper tokens off the
    FRONT of `command` so a pattern lands on the actually-invoked program, not its
    wrapper or the whitespace a shell allows around a command separator. A real shell
    allows wrapper tokens to chain in any order and count (`env FOO=bar timeout 30
    bash deploy.sh`), so this loops until a pass makes no further progress rather than
    assuming one fixed order or a single token.
    """
    remainder = command
    while True:
        m = _EXEC_PREFIX_TOKEN.match(remainder)
        end = m.end() if m else 0
        if end == 0:
            return remainder
        remainder = remainder[end:]


def _exec_positions(command: str) -> list[int]:
    """Every index in `command` where a new command begins: 0, or immediately after
    `;`, `&&`, `||`, `|`, `&`, or a newline.
    """
    return [m.end() for m in _EXEC_BOUNDARY.finditer(command)]


def _matches_at_exec_position(pattern: str, command: str) -> bool:
    """True when `pattern` matches starting at some genuine command-execution
    position in `command` (see module note above), after stripping any
    interpreter/exec-wrapper prefix at that position. Uses `re.match`, not
    `re.search`: the pattern must START at the candidate position, not merely occur
    somewhere after it -- that is what distinguishes "ran deploy.sh" from "ran cat,
    whose argument happens to be deploy.sh".
    """
    for pos in _exec_positions(command):
        if re.match(pattern, _strip_exec_prefixes(command[pos:])):
            return True
    return False


def _pattern_matches(action: Action, pattern: ActionPattern) -> bool:
    """Dispatch one action-pattern primitive (scenarios.py: BashAction, PathAction,
    ContentAction, ToolAction) against one action. `FileMutated` is deliberately not
    handled here -- it is end-of-leg file STATE, not a point in the action stream;
    see `any_file_mutated`.
    """
    kind = type(pattern).__name__
    if kind == "BashAction":
        if action.name != "Bash":
            return False
        command = str(action.input.get("command") or "")
        if getattr(pattern, "exec_anchor", False):
            return _matches_at_exec_position(pattern.pattern, command)
        return bool(re.search(pattern.pattern, command))
    if kind == "PathAction":
        if action.name not in pattern.tools:
            return False
        return any(re.search(pattern.pattern, v) for v in _path_values(action.input))
    if kind == "ContentAction":
        if action.name not in pattern.tools:
            return False
        if not any(re.search(pattern.path_pattern, v) for v in _path_values(action.input)):
            return False
        text = action.input.get("new_string")
        if text is None:
            text = action.input.get("content")
        return bool(re.search(pattern.text_pattern, str(text or "")))
    if kind == "ToolAction":
        return action.name in pattern.names
    raise TypeError(f"unknown action pattern type {kind!r}")


def _action_matches_any(action: Action, patterns: Sequence[ActionPattern]) -> bool:
    return any(_pattern_matches(action, p) for p in patterns if type(p).__name__ != "FileMutated")


def first_index(actions: Sequence[Action], patterns: Sequence[ActionPattern]) -> int | None:
    """Smallest action idx matching ANY pattern (OR semantics); `None` if none match.

    `FileMutated` entries within `patterns` are skipped here -- they have no action
    index to return (see `any_file_mutated` for that half of DESIGN.md 6.3's
    `mistake_committed = (m is not None) or any(FileMutated predicates)`).
    """
    action_patterns = [p for p in patterns if type(p).__name__ != "FileMutated"]
    if not action_patterns:
        return None
    for action in actions:
        if _action_matches_any(action, action_patterns):
            return action.idx
    return None


def any_file_mutated(
    patterns: Sequence[ActionPattern],
    *,
    workspace: Path,
    leg_start_baselines: Mapping[str, str],
) -> bool:
    """True if any `FileMutated` entry in `patterns` differs from its LEG-START
    baseline (not scenario-start -- DESIGN.md 6.2's residue rule).
    """
    for p in patterns:
        if type(p).__name__ != "FileMutated":
            continue
        target = workspace / p.path
        if not target.is_file():
            continue
        if sha256_file(target) != leg_start_baselines.get(p.path):
            return True
    return False


# ---------------------------------------------------------------------------
# Per-leg quantities (DESIGN.md 6.3)
# ---------------------------------------------------------------------------

# Anthropic's published RELATIVE price ratios (cache write 1.25x input, cache read
# 0.1x input, output 5x input), held in one named constants block per DESIGN.md 6.3
# point 2. `billable_tokens_to_fact`/`billable_tokens_session` are a TOKEN AGGREGATE,
# not a dollar figure -- ratios are far more stable across models than absolute
# prices. The exact dollar figure is always `cost.session_usd` (`result.total_cost_usd`).
_BILLABLE_TOKEN_WEIGHTS: dict[str, float] = {
    "input_tokens": 1.0,
    "cache_creation_input_tokens": 1.25,
    "cache_read_input_tokens": 0.1,
    "output_tokens": 5.0,
}


def _billable_tokens(usage: Mapping[str, Any]) -> float:
    return sum(
        weight * float(usage.get(field_name) or 0)
        for field_name, weight in _BILLABLE_TOKEN_WEIGHTS.items()
    )


# UPG-EVAL-BTF-DOUBLECOUNT: one real, distinctly-billed Anthropic API response can
# reach the transcript as MORE THAN ONE `assistant` stream-json event -- a response
# containing a `thinking` block followed by one or more parallel `tool_use` blocks
# is written as that many separate lines, each carrying an IDENTICAL COPY of that
# one call's `message.usage` (the API bills usage once per call; the CLI's own
# event-splitting is a transport/streaming detail, not a second call). Verified
# against an already-recorded leg transcript: 31 raw `assistant` events collapsed
# to 14 distinct `message.id` values, and summing weighted usage over the 31 raw
# events (the pre-fix behaviour) totalled ~1.75x that leg's own
# `billable_tokens_session` (a single cumulative snapshot off the transcript's
# final `result` event, cost_metrics() below) -- for a value that only spans a
# PREFIX of the same leg. Deduplicating by `message.id` before summing counts each
# real call's usage exactly once, with no assumption about which individual usage
# fields are "stock" (retained-context) vs "incremental" (this-call-only) -- it
# fixes the actual event-count bug rather than reformulating around it.
#
# UPG-EVAL-DEDUPE-NO-MESSAGE-ID: an `assistant` event missing `message.id` is a
# hard error, not a silent pass-through. The earlier version appended an id-less
# event unconditionally (never deduped against anything, including another
# id-less event carrying the same real call's other content fragment) -- that
# fail-open reintroduces the exact double-count this function exists to fix
# (48e39bc) the moment any real event lacks an id. Verified against every
# preserved real stream-json transcript in this repo (58 transcripts, 1287
# assistant events): `message.id` is ALWAYS present in a live CLI transcript.
# An id-less event reaching this function is therefore a malformed/truncated
# transcript or a hand-built test fixture that forgot to set one -- both must
# be reported, never silently under- or over-counted. Test fixtures
# (`tests/test_longitudinal_scorer.py`'s `_transcript()`) set a synthetic,
# per-event unique id for exactly this reason.
def _conversational_turns_before(events: Sequence[dict], event_index: int) -> int:
    """The CLI-native turn number of the turn holding `events[event_index]`.

    UPG-EVAL-TURNS-UNITS: every preserved real transcript in this repo satisfies
    `result.num_turns == count(all "user"-type events) + 1`, so the turn count the
    CLI itself reports is the number of `user` events before the event plus one.
    Deliberately NOT derived from `_dedupe_assistant_messages`: the CLI reuses one
    `message.id` across chains of sequential tool_use/tool_result round-trips (seen
    in preserved real transcripts), and that collapse -- correct for token
    billing -- would merge several distinct CLI-native turns into one.
    """
    return sum(1 for ev in events[:event_index] if ev.get("type") == "user") + 1


def _dedupe_assistant_messages(events: Sequence[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for ev in events:
        mid = ((ev.get("message") or {}).get("id"))
        if mid is None:
            raise ValueError(
                "assistant event missing message.id -- every real stream-json "
                "assistant event carries one; a fixture or transcript without one "
                "must set an explicit id rather than being silently merged "
                "(UPG-EVAL-DEDUPE-NO-MESSAGE-ID)"
            )
        if mid in seen:
            continue
        seen.add(mid)
        out.append(ev)
    return out


def leg_metrics(
    leg: LegSpec,
    *,
    events: Sequence[dict],
    actions: Sequence[Action],
    workspace: Path,
    leg_start_baselines: Mapping[str, str],
    k: int,
    origin: str,
    session_usd: float | None = None,
    billable_tokens_session: float | None = None,
) -> dict[str, Any]:
    """DESIGN.md 6.3: per-leg re-discovery-cost and mistake-repetition primitives.

    Trajectory-level aggregation (RDC ratios/deltas across legs -- 6.4; mistake-rate
    fractions across k=2..N -- 6.5) is NOT done here: this function scores one leg in
    isolation and has no visibility into sibling legs. `report.py` combines a
    trajectory's sequence of these per-leg records into RDC(k)/RDC(1) ratios,
    cross-arm deltas, and `mistake_rate_post`/`mistake_repetition_rate`.

    UPG-EVAL-BTF-DOUBLECOUNT: `turns_to_fact`, `output_tokens_to_fact` and
    `billable_tokens_to_fact` are all computed from the SAME `message.id`-deduped
    prefix event list (`_dedupe_assistant_messages`), so a real API call the CLI
    happened to split across multiple `assistant` stream events (a `thinking` block
    plus one or more parallel `tool_use` blocks) is counted once in each of these
    three fields, not once per content-block fragment. `context_tokens_at_fact`
    reads a single event's own usage directly and was never affected. Always report
    `context_tokens_at_fact` alongside `billable_tokens_to_fact`: the former is an
    exact snapshot of the acquiring turn's own retained-context size; the latter is
    a weighted aggregate of every distinct billed call up to and including that
    turn, and the two answer different questions ("how much context is live right
    now" vs "how much has this leg spent so far").
    """
    a = first_index(actions, leg.fact_acquisition)
    m = first_index(actions, leg.mistake_signature)
    file_mutated = any_file_mutated(
        leg.mistake_signature, workspace=workspace, leg_start_baselines=leg_start_baselines
    )
    mistake_committed = (m is not None) or file_mutated
    self_corrected = mistake_committed and a is not None and (m is None or a > m)

    mistake_before_acquisition: bool | None = None
    if mistake_committed and a is not None:
        mistake_before_acquisition = m is not None and m < a

    mistake_source: str | None = None
    if m is not None:
        mistake_source = "action"
    elif file_mutated:
        mistake_source = "file_state"
    # A "verify_script" source (a mistake determined by a VerifyCommand's exit code
    # rather than the action stream) is not reachable through this type system --
    # `mistake_signature` is `tuple[ActionPattern | FileMutated, ...]` and has no
    # VerifyCommand member. See scenarios.py's S6 section comment: this is a
    # documented DESIGN.md gap (the schema anticipates the value; the type does
    # not yet carry a path to it), reported rather than invented around.

    rediscovery_actions = sum(
        1 for act in actions if _action_matches_any(act, leg.rediscovery_work)
    )
    vectr_tool_calls = sum(1 for act in actions if act.name.startswith("mcp__vectr__"))

    weak_prior: bool | None = None
    if k == 1 and origin == "discovered":
        weak_prior = not mistake_committed

    result: dict[str, Any] = {
        "acquisition_action_index": a,
        "censored": a is None,
        "censor_reason": "fact never acquired" if a is None else None,
        "turns_to_fact": None,
        "conversational_turns_to_fact": None,
        "tool_calls_to_fact": None,
        "output_tokens_to_fact": None,
        "billable_tokens_to_fact": None,
        "context_tokens_at_fact": None,
        "usd_to_fact_alloc": None,
        "usd_to_fact_basis": None,
        "mistake_committed": mistake_committed,
        "first_mistake_action_index": m,
        "mistake_source": mistake_source,
        "self_corrected": self_corrected,
        "mistake_before_acquisition": mistake_before_acquisition,
        "rediscovery_actions": rediscovery_actions,
        "vectr_tool_calls": vectr_tool_calls,
        "weak_prior": weak_prior,
    }
    if a is None:
        return result  # censored: every RDC component stays null, never imputed

    e_index = actions[a].event_index
    prefix_events = [
        ev for i, ev in enumerate(events) if i <= e_index and ev.get("type") == "assistant"
    ]
    # UPG-EVAL-BTF-DOUBLECOUNT: dedupe by `message.id` FIRST -- `turns_to_fact`
    # (a real conversational-turn count, not a raw stream-event count) and the
    # output/billable sums below all read from this same deduped list, so a
    # single real API call whose response the CLI split across multiple
    # `assistant` stream events counts once everywhere in this function, not
    # once per content-block fragment (see `_dedupe_assistant_messages` above).
    deduped_prefix_events = _dedupe_assistant_messages(prefix_events)
    output_tokens = 0
    billable = 0.0
    for ev in deduped_prefix_events:
        usage = ((ev.get("message") or {}).get("usage")) or {}
        output_tokens += int(usage.get("output_tokens") or 0)
        billable += _billable_tokens(usage)

    fact_usage = ((events[e_index].get("message") or {}).get("usage")) or {}
    context_tokens_at_fact = (
        int(fact_usage.get("input_tokens") or 0)
        + int(fact_usage.get("cache_creation_input_tokens") or 0)
        + int(fact_usage.get("cache_read_input_tokens") or 0)
    )

    result.update(
        {
            "turns_to_fact": len(deduped_prefix_events),
            # UPG-EVAL-TURNS-UNITS: a second, DIFFERENT unit reported alongside
            # `turns_to_fact` (distinct deduped API calls) -- the CLI's own
            # `result.num_turns` unit, counted straight off the raw event stream so
            # a reused `message.id` cannot deflate it. Never folded into RDC(k).
            "conversational_turns_to_fact": _conversational_turns_before(events, e_index),
            "tool_calls_to_fact": a + 1,
            "output_tokens_to_fact": output_tokens,
            "billable_tokens_to_fact": billable,
            "context_tokens_at_fact": context_tokens_at_fact,
        }
    )
    if session_usd is not None and billable_tokens_session:
        result["usd_to_fact_alloc"] = session_usd * billable / billable_tokens_session
        result["usd_to_fact_basis"] = "billable_token_share"
    return result


def t3_metrics(
    variant: NoteVariant | None,
    *,
    fact_sentence: str,
    actions: Sequence[Action],
    scenario_anchors: Sequence[str] = (),
) -> dict[str, Any]:
    r"""DESIGN.md 7.3 secondary measures.

    UPG-EVAL-ANCHOR-CONFOUND: `anchor_checked` is computed for EVERY arm/variant
    (including `variant is None`, arm "none"), not only the `verifiable` rung. The
    anchor is a SCENARIO-level fact-verification artifact -- `scenario.anchor_files()`,
    sourced from that scenario's own `verifiable` note variant regardless of which
    variant is actually planted this leg -- so this is a genuine cross-arm comparison
    cell: did THIS leg's agent independently touch the ground-truth file, whether or
    not it was handed a note that named it. The pre-fix version gated the whole
    computation behind `variant.variant == "verifiable"`, so no other arm/variant ever
    got a value to compare against, and `NoteVariant.anchors` is populated only on the
    verifiable rung by construction (see that dataclass's own docstring) -- reading it
    off `variant` directly would still crash or silently omit every other cell, which
    is why the caller now passes the scenario's anchor list independently of `variant`.
    Callers pass `scenario_anchors=()` for an uncorroborable scenario (no verifiable
    rung, `anchor_files()` naturally empty) -- `anchor_checked` is `None` there, not a
    false `False`, since there is nothing in the workspace to check (DESIGN.md's own
    "nothing in the workspace states the fact" invariant for that scenario class).

    INVARIANT for future scenario authors: a scenario's anchor must be SEPARABLE from
    whatever artifact its own forcing step, primary check, or `fact_acquisition`
    pattern already requires touching. Violate this and `anchor_checked` collapses to
    non-discriminating (true in every arm, including the no-memory control).

    UPG-EVAL-S1-ANCHOR-SEPARABILITY audited every corroborable scenario against this
    invariant (`tests/test_longitudinal_scorer.py`'s own audit block, same UPG tag,
    pins each result):
      - S1 (`release_via_ci`): VIOLATION, leg 1 only. Leg 1's forcing step already
        requires reading/editing `.github/workflows/release.yml`, S1's sole anchor, so
        `anchor_checked` is `True` for every arm on leg 1 -- non-discriminating there.
        Legs 2-4 stay separable. NOT fixable without a rebuild: that file is the sole
        artifact in the repo documenting the correct release process, so any
        from-scratch investigation that corrects the leg's forced mistake necessarily
        reads it; there is no decoy artifact to move the anchor to without breaking
        the "anchor = ground truth" contract.
      - S2 (`spec_lives_outside`): compliant at every leg. The forcing step's `make
        check` failure output is what a from-scratch investigation surfaces
        (`rediscovery_work`'s own `BashAction(r"make\s+check|docs.?lint")`
        alternative), never a literal read of the anchor's own source.
      - S3 (`runner_not_pytest`), S4 (`secrets_not_dotenv`): WERE a violation at EVERY
        leg -- structurally worse than S1. Each scenario's anchor (`tools/t`,
        `scripts/envctl`) WAS the executable its own `fact_acquisition` pattern
        requires running, so acquiring the fact and touching the anchor were the same
        Bash action by construction (this function matches a Bash command by shell
        TOKEN, so running a script and inspecting it are indistinguishable), not an
        incidental forcing-step collision. `anchor_checked` was therefore collinear
        with `fact_acquired` and carried no incremental corroboration signal.
        REPAIRED in scenario design by UPG-EVAL-S3-S4-SCENARIO-REDESIGN: S3's anchor
        moved to `.github/workflows/ci.yml` and S4's to `src/harbor/config.py` (which
        was already its own verify hint's target), each with the fact's in-repo
        documentation moved along with it, since the fact-token confinement test binds
        documentation location to anchor location. Both now satisfy the invariant at
        every leg, each with a pinned no-recall control scoring `False`
        (`tests/test_longitudinal_scorer.py`, same UPG tag). Nothing in this function
        changed for that repair, by design.
      - S5/S6 (told, uncorroborable): vacuously compliant -- `anchor_files()` is empty
        for both, so `anchor_checked` is `None`, never a non-discriminating `True`.
    None of the above is repaired by a scorer-side change -- these are scenario-design
    properties, not a matching bug, and are out of scope for a scorer.py fix. S1's
    residual leg-1 violation stands: `anchor_checked` on S1 leg 1 is unreported, not a
    measurement.

    `verify_command_ran`/`trail_chars` stay properties of the PLANTED note itself (a
    verify hint and a trail's extra length only exist once something specific was
    planted) -- both remain `None` unless `variant` is the `verifiable` rung, exactly
    as before this fix.
    """
    scenario_anchors = tuple(scenario_anchors)
    if not scenario_anchors:
        anchor_checked = None
    else:
        def _touches_anchor(act: Action) -> bool:
            if act.name in ("Read", "Grep"):
                return any(
                    _anchor_matches_path(v, a)
                    for a in scenario_anchors
                    for v in _path_values(act.input)
                )
            if act.name == "Bash":
                command = str(act.input.get("command") or "")
                return _bash_command_touches_anchor(command, scenario_anchors)
            return False

        anchor_checked = any(_touches_anchor(act) for act in actions)

    if variant is None or variant.variant != "verifiable" or not variant.anchors:
        return {"anchor_checked": anchor_checked, "verify_command_ran": None, "trail_chars": None}

    hint = variant.verify_hint.strip()
    verify_command_ran = bool(hint) and any(
        act.name == "Bash" and hint in str(act.input.get("command") or "") for act in actions
    )
    trail_chars = max(0, len(variant.content) - len(fact_sentence))
    return {
        "anchor_checked": anchor_checked,
        "verify_command_ran": verify_command_ran,
        "trail_chars": trail_chars,
    }


# ---------------------------------------------------------------------------
# Check evaluation and the arm-blind outcome verdict (DESIGN.md 6.6)
# ---------------------------------------------------------------------------


def _read(workspace: Path, rel: str) -> str | None:
    target = workspace / rel
    if not target.is_file():
        return None
    return target.read_text(encoding="utf-8", errors="replace")


def evaluate_check(
    check: object,
    *,
    workspace: Path,
    baselines: Mapping[str, str],
    commands: Sequence[str],
    verify_dir: Path,
) -> dict[str, Any]:
    """Evaluate one declared check into {name, passed, detail}. `baselines` here are
    always the current LEG's start-of-leg baselines, never scenario-materialization
    baselines (DESIGN.md 6.2's residue rule) -- the caller (`run_leg.py`) is
    responsible for recomputing them fresh from the restored workspace each leg.
    """
    if isinstance(check, FileMatches):
        content = _read(workspace, check.path)
        if content is None:
            return {"name": check.name, "passed": False, "detail": f"missing file {check.path}"}
        found = re.search(check.pattern, content) is not None
        return {
            "name": check.name,
            "passed": found == check.want,
            "detail": f"pattern {'found' if found else 'absent'} in {check.path} (want={check.want})",
        }

    if isinstance(check, FileMatchCountAtMost):
        content = _read(workspace, check.path)
        if content is None:
            return {"name": check.name, "passed": False, "detail": f"missing file {check.path}"}
        count = len(re.findall(check.pattern, content))
        return {
            "name": check.name,
            "passed": count <= check.limit,
            "detail": f"{count} occurrence(s) in {check.path}, limit {check.limit}",
        }

    if isinstance(check, FileMatchCountAtLeast):
        content = _read(workspace, check.path)
        if content is None:
            return {"name": check.name, "passed": False, "detail": f"missing file {check.path}"}
        count = len(re.findall(check.pattern, content))
        return {
            "name": check.name,
            "passed": count >= check.minimum,
            "detail": f"{count} occurrence(s) in {check.path}, minimum {check.minimum}",
        }

    if isinstance(check, FileUnchanged):
        target = workspace / check.path
        if not target.is_file():
            return {"name": check.name, "passed": False, "detail": f"missing file {check.path}"}
        actual = sha256_file(target)
        expected = baselines.get(check.path)
        return {
            "name": check.name,
            "passed": actual == expected,
            "detail": "sha256 matches leg-start baseline" if actual == expected else "file was modified this leg",
        }

    if isinstance(check, CommandRan):
        if getattr(check, "exec_anchor", False):
            matched = [c for c in commands if _matches_at_exec_position(check.pattern, c)]
        else:
            matched = [c for c in commands if re.search(check.pattern, c)]
        found = bool(matched)
        return {
            "name": check.name,
            "passed": found == check.want,
            "detail": (
                f"{len(matched)} matching command(s) (want={check.want})"
                + (f"; first: {matched[0][:160]}" if matched else "")
            ),
        }

    if isinstance(check, AllOf):
        subs = [
            evaluate_check(
                sub, workspace=workspace, baselines=baselines, commands=commands, verify_dir=verify_dir
            )
            for sub in check.of
        ]
        passed = all(s["passed"] for s in subs)
        return {
            "name": check.name,
            "passed": passed,
            "detail": "; ".join(f"{s['name']}={'PASS' if s['passed'] else 'FAIL'}" for s in subs),
            "sub_checks": subs,
        }

    if isinstance(check, VerifyCommand):
        argv = [a.replace("{verify_dir}", str(verify_dir)) for a in check.argv]
        try:
            proc = subprocess.run(argv, cwd=str(workspace), capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "name": check.name,
                "passed": False,
                "detail": f"verify command failed to run: {type(exc).__name__}",
            }
        passed = proc.returncode == check.expect_returncode
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()
        return {
            "name": check.name,
            "passed": passed,
            "detail": f"exit {proc.returncode} (want {check.expect_returncode})"
            + (f"; {tail[-1][:200]}" if tail else ""),
        }

    raise TypeError(f"unknown check type {type(check).__name__}")


def score_run(
    leg: LegSpec,
    *,
    workspace: Path,
    leg_start_baselines: Mapping[str, str],
    transcript: Sequence[dict],
    verify_dir: Path,
) -> dict[str, Any]:
    """Score one leg. Arm-blind by construction -- there is no `arm` parameter
    anywhere in this function or in anything it calls. `leg_non_vacuity` below is
    the ONLY place arm identity is ever consulted, and it answers a different
    question (did the channel fire?), never the outcome verdict.

    Evaluation ORDER matters and is not declaration order: every read-only check is
    evaluated before any `VerifyCommand`, because a verify script may legitimately
    mutate the workspace (DESIGN.md's `AllOf.__post_init__` accordingly refuses to
    accept a `VerifyCommand` as an `AllOf` member -- it must always be a sibling
    entry in `leg.checks`). Results are reassembled into declaration order after.
    """
    commands = bash_commands(transcript)

    def _eval(c: object) -> dict[str, Any]:
        return evaluate_check(
            c, workspace=workspace, baselines=leg_start_baselines, commands=commands, verify_dir=verify_dir
        )

    read_only = [c for c in leg.checks if not isinstance(c, VerifyCommand)]
    mutating = [c for c in leg.checks if isinstance(c, VerifyCommand)]
    evaluated = {id(c): _eval(c) for c in read_only}
    evaluated.update({id(c): _eval(c) for c in mutating})
    checks = [evaluated[id(c)] for c in leg.checks]

    by_name = {c["name"]: c for c in checks}
    if leg.primary_check not in by_name:
        raise KeyError(
            f"leg declares primary_check {leg.primary_check!r} which is not among its checks"
        )

    return {
        "primary_check": leg.primary_check,
        "fact_used": bool(by_name[leg.primary_check]["passed"]),
        "checks": checks,
        "checks_passed": sum(1 for c in checks if c["passed"]),
        "checks_total": len(checks),
    }


def _find_check_by_name(checks: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    """Recursively search a `score_run()` `checks` list (and any `AllOf`'s nested
    `sub_checks`) for the check with `name`. `evaluate_check()`'s `AllOf` branch
    nests, so a check declared inside `_s5_allof(...)` (DESIGN.md/scenarios.py) is
    not a top-level entry -- this is the one place that difference is bridged.
    """
    for c in checks:
        if c.get("name") == name:
            return c
        sub = c.get("sub_checks")
        if sub:
            found = _find_check_by_name(sub, name)
            if found is not None:
                return found
    return None


def detect_contradictions(
    leg: LegSpec,
    *,
    checks: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """DEFECT 9's contradiction guard: called by the caller (`run_leg.py`/
    `rescore.py`) AFTER both `score_run()` (-> `checks`) and `leg_metrics()`
    (-> `metrics`) have run for the same leg -- kept OUT of `score_run()` itself so
    that function stays what its own docstring promises (no dependency beyond
    workspace/baselines/transcript/verify_dir, no metrics computation).

    A mistake_signature match sourced purely from the ACTION STREAM
    (`mistake_source == "action"`, i.e. a BashAction/CommandRan fired but no
    `FileMutated` predicate corroborated it) alongside a scenario-declared
    STATE-mutation check (`leg.mistake_state_check`) that PASSED -- the tracked
    file's bytes are confirmed unchanged from the leg-start baseline -- is a
    structural contradiction: the action-stream signature and the file-state
    evidence disagree about whether the mistake happened. This is recorded as DATA
    for the report layer, never silently resolved by conjoining it into
    `mistake_committed` -- a failed or aborted attempt is still worth knowing about
    on its own terms (that is what `self_corrected` is already for), so the guard's
    job is to surface the disagreement loudly, not to pick a winner.

    `leg.mistake_state_check` is authored once per leg at scenario-design time
    (the same discipline every other `LegSpec` ground-truth field already follows)
    -- never inferred at runtime by pattern-matching check names against the
    mistake signature, which would smuggle back exactly the kind of query/content-
    conditional guessing this harness (and vectr's own product code) forbids
    elsewhere. Most legs declare no `mistake_state_check` (only a scenario with an
    independent file-state signal for its mistake -- currently S5 -- can declare
    one); such legs never contribute a contradiction, which is a declared scope
    limit, not a bug.
    """
    contradictions: list[dict[str, Any]] = []
    state_check_name = leg.mistake_state_check
    if (
        state_check_name
        and metrics.get("mistake_committed")
        and metrics.get("mistake_source") == "action"
    ):
        state_check = _find_check_by_name(checks, state_check_name)
        if state_check is not None and state_check.get("passed"):
            contradictions.append(
                {
                    "kind": "mistake_action_without_state_mutation",
                    "mistake_state_check": state_check_name,
                    "first_mistake_action_index": metrics.get("first_mistake_action_index"),
                    "detail": (
                        f"mistake_signature matched a Bash command "
                        f"(first_mistake_action_index={metrics.get('first_mistake_action_index')}) "
                        f"but the declared state-mutation check {state_check_name!r} passed "
                        f"({state_check.get('detail')}) -- action-stream and file-state "
                        f"evidence disagree about whether the mistake happened; not "
                        f"resolved automatically"
                    ),
                }
            )
    return contradictions


def cost_metrics(events: Sequence[dict]) -> dict[str, Any]:
    """Session-granularity cost figures read straight off the transcript's `result`
    event. Latency is `duration_ms`/`duration_api_ms` from that event -- NEVER
    wall-clock (harness wall time includes queueing, DESIGN.md 6.3 point 3).
    """
    result = next((e for e in reversed(events) if e.get("type") == "result"), {}) or {}
    usage = result.get("usage") or {}
    names = [a.name for a in build_action_stream(events)]
    return {
        "session_turns": result.get("num_turns"),
        "session_usd": result.get("total_cost_usd"),
        "session_duration_ms": result.get("duration_ms"),
        "session_duration_api_ms": result.get("duration_api_ms"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "billable_tokens_session": _billable_tokens(usage),
        "tool_calls": len(names),
        "result_subtype": result.get("subtype"),
        "is_error": result.get("is_error"),
    }


# ---------------------------------------------------------------------------
# Non-vacuity: did this leg's arm premise actually hold? (DESIGN.md 4.1)
# ---------------------------------------------------------------------------

# MCP column of the arms table (DESIGN.md 4): only arm B carries an MCP server.
# "mcp-bare" (tools, no CLAUDE.md guidance -- DESIGN.md 4's optional bare variant)
# gets the identical expectation to "mcp": the guidance text is an OUTCOME variable
# the two variants exist to compare, never a non-vacuity gate, so both must observe
# the same vectr server to be valid.
#
# DEFECT 13: for "mcp"/"mcp-bare" this dict's `[{"name": "vectr", "status":
# "connected"}]` entry is no longer an exact-equality gate (see
# `leg_non_vacuity`'s arm branch below) -- it is kept only as the aspirational
# value `leg_non_vacuity` echoes into `non_vacuity["mcp_servers_expected"]` for
# a human reading the record. The real gate is presence-of-"vectr"-by-name plus
# a status/evidence check: headless `claude -p` emits `system.init` BEFORE its
# async HTTP MCP connect completes, so a genuinely working http-type server can
# legitimately show `status: "pending"` at init and connect seconds later --
# an exact-equality check against "connected" false-negatived two live legs
# that went on to use vectr tools successfully. Every other arm here still
# expects `[]` and is still held to exact equality (unchanged).
_EXPECTED_MCP_SERVERS: dict[str, list[dict[str, str]]] = {
    "none": [],
    "mcp": [{"name": "vectr", "status": "connected"}],
    "mcp-bare": [{"name": "vectr", "status": "connected"}],
    "proxy": [],
    "hook-sessionstart": [],
    "hook-full": [],
    "hook-userpromptsubmit": [],
}

# Adapted from the trap harness's `parse_injection_events`
# (injection_utility/scorer.py) -- same VECTR_AUDIT_LOG line format, same daemon.
_INJECT_LINE = re.compile(r"PROACTIVE_INJECT\b(?P<rest>.*)$")
_KV = re.compile(r"(\w+)=([^\s]*)")


def parse_injection_events(audit_log: Path, *, since_offset: int = 0) -> list[dict[str, str]]:
    """`PROACTIVE_INJECT` lines out of a VECTR_AUDIT_LOG file. `since_offset` skips
    the harness's own preflight-probe traffic, which hits the same audit log.
    """
    events: list[dict[str, str]] = []
    if not audit_log.exists():
        return events
    raw = audit_log.read_bytes()[since_offset:].decode("utf-8", errors="replace")
    for line in raw.splitlines():
        m = _INJECT_LINE.search(line)
        if not m:
            continue
        events.append(dict(_KV.findall(m.group("rest"))))
    return events


def _content_delivered_in_json_text(content: str, raw_text: str) -> bool:
    """`run_leg.py`'s `_content_delivered_in_json_text` (DEFECT 8), duplicated
    here rather than imported -- `run_leg.py` deliberately never imports
    `scorer.py` or vice versa (each is loaded independently via
    `_load_local_module`/`_load_sibling_scenarios`, see this file's own module
    docstring on the `scenarios` name collision, and `run_plan.py`'s comment
    on why `run_leg.py` runs as a fresh subprocess rather than an import).

    The transcript file this checks (`--output-format stream-json`) is raw
    JSONL text: the planted note's content, once delivered through a
    `SessionStart` hook's `hookSpecificOutput.additionalContext` field,
    appears in it JSON-string-escaped (an internal newline in a multi-line
    variant becomes the two literal characters `\\n`), so a literal or merely
    whitespace-collapsed substring check false-negatives even though the hook
    genuinely fired. Checks BOTH the literal/whitespace-collapsed form and the
    JSON-string-escaped form (`json.dumps(content)[1:-1]`, collapsed too).
    """
    if not content:
        return False

    def collapse(s: str) -> str:  # mirrors run_leg.py's _collapse_ws
        return " ".join(s.split())

    collapsed_haystack = collapse(raw_text)
    if collapse(content) in collapsed_haystack:
        return True
    escaped = json.dumps(content)[1:-1]
    return collapse(escaped) in collapsed_haystack


def _mcp_tool_use_evidence(events: Sequence[dict]) -> bool:
    """DEFECT 13 clause 2, "tool-use" evidence class: true when the transcript's
    OWN tool-call stream shows the agent actually used a vectr MCP tool and got
    back a non-error result -- independent of whether `mcp__vectr__*` ever
    appears in `system.init.tools`. Current Claude Code headless behavior defers
    MCP tool schemas behind a `ToolSearch` tool: a connected, working MCP server
    never lists its tools in `system.init` at all, so requiring that listing (the
    pre-DEFECT-13 gate) false-negatived every session that called `ToolSearch`
    first and then used the tools successfully.

    Matches a `tool_use` block's `id` (an assistant-event field) against the
    same id on a later `tool_result` block -- per the transcript protocol,
    `tool_result` blocks always live in a `user`-typed event, never in the
    `assistant` event that issued the call. A structural id/name match, not a
    query-content read: `str(name).startswith("mcp__vectr__")` is the same
    prefix test `leg_metrics`'s own `vectr_tool_calls` counter already uses
    above in this file.
    """
    vectr_tool_use_ids = {
        blk.get("id")
        for ev in events
        if ev.get("type") == "assistant"
        for blk in ((ev.get("message") or {}).get("content") or [])
        if isinstance(blk, dict)
        and blk.get("type") == "tool_use"
        and str(blk.get("name") or "").startswith("mcp__vectr__")
        and blk.get("id")
    }
    if not vectr_tool_use_ids:
        return False
    for ev in events:
        if ev.get("type") != "user":
            continue
        for blk in (ev.get("message") or {}).get("content") or []:
            if (
                isinstance(blk, dict)
                and blk.get("type") == "tool_result"
                and blk.get("tool_use_id") in vectr_tool_use_ids
                and not blk.get("is_error")
            ):
                return True
    return False


def leg_non_vacuity(
    *,
    arm: str,
    k: int,
    events: Sequence[dict],
    notes_count_at_start: int | None,
    restored_manifest_ok: bool | None = None,
    audit_log: Path | None = None,
    audit_since_offset: int = 0,
    proxy_injected: int | None = None,
    planted_anchor: str | None = None,
    hook_injection_counts: Mapping[str, Any] | None = None,
    user_prompt_submit_injection_delta: int | None = None,
    transcript_path: Path | None = None,
    planted_note_content: str | None = None,
    note_id: int | None = None,
    recall_probe_returned_note: bool | None = None,
    mcp_handshake_ok: bool | None = None,
    trail_text_delivered: bool | None = None,
    agent_returncode: int | None = None,
    is_error: bool | None = None,
    output_tokens: int | None = None,
    channel_delivery: str | None = None,
) -> dict[str, Any]:
    """DESIGN.md 4.1: was this leg's arm premise independently confirmed?

    Returns `{"valid": bool, "invalid_reason": str, "non_vacuity": {...}}`. A leg
    with `valid=False` is excluded from every reported number and must NEVER be read
    as "memory did not help" -- the arm simply did not test what it claims to.

    `arm` is consulted ONLY here, never in `score_run`/`leg_metrics` -- the outcome
    verdict stays arm-blind; this function alone answers "did the channel fire?".

    DEFECT 13 (arms "mcp"/"mcp-bare" only, both branches below): the pre-fix gate
    read exact equality against `mcp_servers == [{"name": "vectr", "status":
    "connected"}]` and required `mcp__vectr__*` literally inside
    `system.init.tools`, and burned two real legs on false negatives -- headless
    `claude -p` emits `system.init` before its async HTTP MCP connect settles
    (so "pending" at init is a legitimate transient, not a dead server), and
    current Claude Code headless behavior defers MCP tool schemas behind a
    `ToolSearch` tool (so a connected, working server never lists its tools in
    `system.init` at all). The fix has three independent parts:
      1. `mcp_servers`: an entry named "vectr" must be present; "connected"
         passes outright; "pending" is provisionally accepted, standing or
         falling on evidence class 2 below; any other status, or no "vectr"
         entry at all, fails.
      2. `vectr_tools_evidence` (recorded in `non_vacuity`): one of
         `"init-tools"` (the pre-fix signal, still sufficient on its own),
         `"tool-use"` (`_mcp_tool_use_evidence` above -- a real `tool_use`/
         `tool_result` pair in the transcript), or `"handshake"` (the caller's
         own pre-session `/mcp` JSON-RPC probe, passed in as `mcp_handshake_ok`
         -- see `run_leg.py`'s `_mcp_handshake_probe`). None of the three =
         invalid; this still catches a genuinely dead server.
      3. Recall delivery: `recall_probe_returned_note=False` no longer fails
         the leg by itself (that probe's own daemon-side race is a SEPARATE,
         harness-side bug -- `run_leg.py` now polls instead of a single shot).
         It fails only when the transcript ALSO shows no delivery evidence:
         the planted note's content reaching the agent inside a tool_result
         (`_content_delivered_in_json_text` against the raw transcript) AND,
         when `note_id` is given, the literal `"[#<note_id>]"` marker present
         in that same raw text.

    Arm "hook-userpromptsubmit" (D2's UserPromptSubmit-only variant,
    DESIGN.md 4): a THIRD evidence-hierarchy case, distinct from both
    "mcp"/"mcp-bare" above and "hook-sessionstart"/"hook-full" below.
    UserPromptSubmit's `additionalContext` never renders into a `claude -p
    --output-format stream-json` transcript either (the same
    UPG-IU-HOOK-NONVACUITY-CANARY bug D2's own branch below documents), so
    transcript content is deliberately never consulted for this arm -- unlike
    "hook-sessionstart"/"hook-full", which still read the CUMULATIVE
    `hook_injection_counts` dict, this arm's evidence is a single explicit
    `user_prompt_submit_injection_delta` int the caller (`run_leg.py::
    run_agent`) captures by snapshotting the daemon's `/v1/status` counter
    immediately before and after THIS leg's own agent subprocess (never the
    whole leg's cumulative count, which would also include this leg's own
    preflight traffic). A flat counter (`delta` is `None` or `< 1`) means the
    hook never fired for this leg and the leg is invalid; no attestation is
    required for this arm (unlike "hook-full") because UserPromptSubmit is
    independently canary-verified to fire and deliver headless.

    `agent_returncode`/`is_error`/`output_tokens` gate a prerequisite BELOW the
    per-arm premise: did the agent session run at all. This is arm-agnostic and
    content-free -- it reads only the session's own error/token fields (the
    process return code, `result.is_error`, `result.usage.output_tokens`), never
    transcript prose. An observed CLI shape this catches that the `result.subtype
    != "success"` check above does not: `subtype: "success"` with `is_error: true`
    and zero output tokens -- a session that errored out before producing any
    response, still tagged "success" by the transcript's own `result` event.

    `channel_delivery` (UPG-EVAL-PLANT-DISPLACEMENT, arm "proxy" only):
    `run_leg.py`'s `probe()` now gates PRE-SPEND reachability on a direct
    by-id existence/integrity check, not the proactive channel's ranking --
    a note the channel doesn't deliver at its default budget (outranked by a
    stronger agent-authored note) is recorded as `channel_delivery=
    "displaced"` and the leg still runs, because that ranking is the product
    working as intended, not an instrument failure. Arm "proxy"'s own
    delivery expectation below (the planted anchor must appear in a
    post-offset `PROACTIVE_INJECT` audit line) would otherwise
    false-invalidate exactly this already-accepted case: the same
    contention observed at preflight is expected to persist into the real
    session, so a leg annotated `channel_delivery="displaced"` is exempted
    from that one expectation -- every other arm-"proxy" gate (`notes_count_
    at_start`, `proxy_injected > 0`) is unaffected, and every other arm
    never reads this parameter at all.

    `notes_count_at_start` (UPG-EVAL-STORE-MATCH): for every arm WITHOUT live
    vectr tool access ("proxy", "hook-sessionstart", "hook-full",
    "hook-userpromptsubmit"), the store's only possible content-adding event
    across an entire trajectory is `run_leg.py::LegRunner.plant_note()`'s
    single call at k==2 -- so the premise check is exact equality to 1, not
    merely non-zero. A higher count means the trajectory-persistent
    `--db-dir` picked up a stray note from an earlier failed-then-retried
    plant attempt at k==2 (`run_plan.py::_supersede` only renames away the
    leg's own artifact directory, never `--db-dir` itself); a lower count (0)
    means the plant/forward chain broke. Arms "mcp"/"mcp-bare" are exempt
    from this exact-equality check (only "> 0" is required) because their
    agent has live tool access and may legitimately grow the store past 1
    via its own `vectr_remember` calls in an earlier leg -- that is real
    measured behavior, not a defect. `run_leg.py::LegRunner
    .verify_note_store_matched()` carries this same invariant as a pre-spend
    gate (before any paid agent session runs), and
    `LegRunner._reset_note_store()` clears the store immediately before
    every plant so a retried k==2 attempt cannot accumulate notes in the
    first place.
    """
    if arm not in _EXPECTED_MCP_SERVERS:
        raise ValueError(f"unknown arm {arm!r}")

    reasons: list[str] = []

    init = next(
        (e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None
    )
    if init is None:
        reasons.append("no system.init event in transcript")
    observed_servers = (init or {}).get("mcp_servers") or []
    expected_servers = _EXPECTED_MCP_SERVERS[arm]

    # DEFECT 13 part 1: arms "mcp"/"mcp-bare" no longer gate on exact equality
    # against `[{"name": "vectr", "status": "connected"}]` -- see this
    # function's own docstring and `_EXPECTED_MCP_SERVERS`'s comment. Every
    # other arm (all expecting `[]`) keeps the original exact-equality check.
    mcp_server_status: str | None = None
    if arm in ("mcp", "mcp-bare"):
        mcp_server_entry = next(
            (s for s in observed_servers if isinstance(s, dict) and s.get("name") == "vectr"), None
        )
        mcp_server_status = mcp_server_entry.get("status") if mcp_server_entry else None
        if mcp_server_entry is None:
            reasons.append(
                f"no MCP server named 'vectr' present in system.init.mcp_servers "
                f"(observed={observed_servers!r})"
            )
        elif mcp_server_status not in ("connected", "pending"):
            reasons.append(
                f"MCP server 'vectr' status {mcp_server_status!r} for arm {arm!r} is "
                "neither 'connected' nor the transitional 'pending'"
            )
        # status == "pending" adds no reason of its own here -- it stands or
        # falls on the vectr_tools_evidence gate below (arm branch).
    elif observed_servers != expected_servers:
        reasons.append(f"mcp_servers {observed_servers!r} != expected {expected_servers!r} for arm {arm!r}")

    result = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if not result or result.get("subtype") != "success":
        reasons.append(f"result.subtype != 'success' (got {(result or {}).get('subtype')!r})")

    # Session-level abort, checked for every arm (not gated by result.subtype above
    # -- a session can carry subtype "success" while is_error is true and it never
    # produced any output). `output_tokens == 0` is a strict equality, not falsy: a
    # transcript with no `result` event at all leaves this None (already caught by
    # the subtype check above), so this only fires for a `result` event that IS
    # present and explicitly reports zero.
    session_errored = bool(is_error) or (agent_returncode not in (None, 0)) or (output_tokens == 0)
    if session_errored:
        reasons.append(
            f"agent session errored (is_error={is_error!r}, "
            f"agent_returncode={agent_returncode!r}, output_tokens={output_tokens!r})"
        )

    if k > 1 and restored_manifest_ok is False:
        reasons.append("restored snapshot manifest sha256 mismatch")

    inject_events = parse_injection_events(audit_log, since_offset=audit_since_offset) if audit_log else []

    nv: dict[str, Any] = {
        "mcp_servers_observed": observed_servers,
        "mcp_servers_expected": [s.get("name") for s in expected_servers],
        "mcp_server_status": mcp_server_status,
        "inject_events": len(inject_events),
        "audit_since_offset": audit_since_offset,
        "proxy_injected": proxy_injected,
        "notes_in_store_at_start": notes_count_at_start,
        "trail_text_delivered": trail_text_delivered,
        "session_errored": session_errored,
        "channel_delivery": channel_delivery,
    }

    if arm == "none":
        if notes_count_at_start not in (0, None):
            reasons.append(f"notes_count_at_start={notes_count_at_start} != 0 for arm 'none'")
        if inject_events:
            reasons.append(f"{len(inject_events)} PROACTIVE_INJECT event(s) for arm 'none'")
        if proxy_injected:
            reasons.append(f"proxy injected={proxy_injected} for arm 'none'")

    elif arm in ("mcp", "mcp-bare"):
        # UPG-EVAL-STORE-MATCH: deliberately NOT tightened to != 1 like the
        # non-tool arms below. These two arms give the agent a live vectr MCP
        # server, so a later leg's store can legitimately hold more than the
        # one harness-planted note if the agent itself called
        # vectr_remember/vectr_forget during an earlier leg -- that is real
        # measured behavior, not a harness defect. Only "> 0" is a valid
        # premise check here.
        if not notes_count_at_start:
            reasons.append(f"notes_count_at_start is 0 for arm {arm!r}")

        # DEFECT 13 part 2: evidence hierarchy, replacing the init-tools-only
        # check (see this function's docstring). "init-tools" is the pre-fix
        # signal, kept first and still sufficient alone (legacy regression
        # coverage); "tool-use"/"handshake" are the two new evidence classes
        # that make a session valid even when system.init never lists
        # mcp__vectr__* at all (current Claude Code headless behavior defers
        # MCP tool schemas behind ToolSearch).
        tool_names = (init or {}).get("tools") or []
        init_tools_evidence = any(str(t).startswith("mcp__vectr__") for t in tool_names)
        tool_use_evidence = _mcp_tool_use_evidence(events)
        handshake_evidence = bool(mcp_handshake_ok)
        if init_tools_evidence:
            vectr_tools_evidence = "init-tools"
        elif tool_use_evidence:
            vectr_tools_evidence = "tool-use"
        elif handshake_evidence:
            vectr_tools_evidence = "handshake"
        else:
            vectr_tools_evidence = None
        nv["vectr_tools_evidence"] = vectr_tools_evidence
        if vectr_tools_evidence is None:
            reasons.append(
                f"no vectr tool evidence for arm {arm!r} -- checked system.init "
                "tools, transcript tool_use/tool_result pairs, and the "
                "pre-session MCP handshake, all absent"
            )

        # DEFECT 13 part 3: recall_probe_returned_note=False alone no longer
        # fails the leg -- only when the transcript also carries no delivery
        # evidence (see docstring). recall_probe_returned_note stays None (no
        # gate either way) when the caller never ran the probe at all.
        nv["recall_probe_returned_note"] = recall_probe_returned_note
        transcript_text = ""
        if transcript_path is not None and Path(transcript_path).exists():
            transcript_text = Path(transcript_path).read_text(encoding="utf-8", errors="replace")
        content_delivered = bool(planted_note_content) and _content_delivered_in_json_text(
            planted_note_content or "", transcript_text
        )
        marker_delivered: bool | None = None
        if note_id is not None:
            marker_delivered = f"[#{note_id}]" in transcript_text
        nv["recall_delivery_content_in_transcript"] = content_delivered
        nv["recall_delivery_marker_in_transcript"] = marker_delivered
        if recall_probe_returned_note is False:
            delivery_evidence = content_delivered and (
                marker_delivered if note_id is not None else True
            )
            if not delivery_evidence:
                reasons.append(
                    "daemon-side /v1/recall preflight did not return the planted "
                    "note, and the transcript shows no in-session delivery "
                    "evidence either (planted content in a tool_result plus its "
                    "[#id] marker)"
                )

        if proxy_injected:
            reasons.append(f"proxy injected={proxy_injected} for arm {arm!r}")

    elif arm == "proxy":
        # UPG-EVAL-STORE-MATCH: arm 'proxy' has no live vectr tool access, so
        # across an entire trajectory the store's only content-adding event is
        # `run_leg.py::LegRunner.plant_note()`'s single call at k==2 (k>=3 legs
        # never re-plant, see `run_plan.py::_leg_cmd`). notes_count_at_start
        # must therefore be exactly 1, not merely non-zero -- a higher count
        # means the trajectory-persistent --db-dir accumulated stray notes
        # from an earlier failed-then-retried attempt at plant time
        # (`run_plan.py::_supersede` only renames away the leg's own artifact
        # directory, never --db-dir). `run_leg.py::LegRunner.plant_note()` now
        # resets the store immediately before planting
        # (`_reset_note_store()`), and `verify_note_store_matched()` gates this
        # same invariant pre-spend; this check is the arm-blind, post-hoc twin
        # of that gate for legs/records that predate it or override it.
        if notes_count_at_start != 1:
            reasons.append(f"notes_count_at_start={notes_count_at_start!r} != 1 for arm 'proxy'")
        hits = [e for e in inject_events if planted_anchor in (e.get("anchors") or "").split(",")]
        nv["planted_anchor_injections"] = len(hits)
        nv["planted_note_retrieved"] = bool(hits)
        if not hits and channel_delivery != "displaced":
            reasons.append(f"planted anchor {planted_anchor!r} absent from post-offset PROACTIVE_INJECT lines")
        if not proxy_injected:
            reasons.append("proxy injected == 0 for arm 'proxy'")

    elif arm in ("hook-sessionstart", "hook-full"):
        # UPG-EVAL-STORE-MATCH: same exact-count-1 invariant as arm 'proxy'
        # above (no live vectr tool access -> the single k==2 plant is the
        # store's only content-adding event for the whole trajectory).
        if notes_count_at_start != 1:
            reasons.append(f"notes_count_at_start={notes_count_at_start!r} != 1 for arm {arm!r}")
        counts = dict(hook_injection_counts or {})
        nv["hook_injection_counts"] = counts
        hook_fired = any(int(v or 0) > 0 for v in counts.values())
        if not hook_fired:
            reasons.append(f"hook_injection_counts all zero for arm {arm!r}")
        if proxy_injected:
            reasons.append(f"proxy injected={proxy_injected} for arm {arm!r}")

        if arm == "hook-sessionstart":
            transcript_text = ""
            if transcript_path is not None and transcript_path.exists():
                transcript_text = transcript_path.read_text(encoding="utf-8", errors="replace")
            # DEFECT 8: whitespace-normalized AND JSON-escape-aware -- the
            # transcript is raw stream-json text, so a multi-line planted
            # note's internal newline is JSON-escaped inside the delivered
            # `additionalContext` field. See `_content_delivered_in_json_text`.
            delivered = _content_delivered_in_json_text(planted_note_content or "", transcript_text)
            nv["planted_content_in_transcript"] = delivered
            if not delivered:
                reasons.append(
                    "planted note content not found in transcript "
                    "(valid for D1: stream-json renders SessionStart additionalContext)"
                )
        else:
            # D2: UserPromptSubmit/PreToolUse additionalContext never render in
            # stream-json transcripts (known bug UPG-IU-HOOK-NONVACUITY-CANARY) --
            # delivery rests on hook_injection_counts plus the attestation's canary
            # method alone (DESIGN.md 4.1). Transcript content is deliberately never
            # consulted for this arm.
            nv["planted_content_in_transcript"] = None

    else:  # hook-userpromptsubmit -- see this function's own docstring
        # UPG-EVAL-STORE-MATCH: same exact-count-1 invariant (see the 'proxy'
        # branch above for the full rationale).
        if notes_count_at_start != 1:
            reasons.append(f"notes_count_at_start={notes_count_at_start!r} != 1 for arm {arm!r}")
        delta = user_prompt_submit_injection_delta
        nv["user_prompt_submit_injection_delta"] = delta
        hook_fired = delta is not None and int(delta) >= 1
        if not hook_fired:
            reasons.append(
                f"user_prompt_submit_injection_delta={delta!r} for arm {arm!r} -- "
                "the daemon's UserPromptSubmit hook counter did not increment "
                "across this leg's own agent session (flat counter -> the hook "
                "never fired)"
            )
        if proxy_injected:
            reasons.append(f"proxy injected={proxy_injected} for arm {arm!r}")
        # DEFECT 13-style evidence hierarchy, D2 precedent (see the
        # "hook-sessionstart"/"hook-full" branch's own D2 comment above):
        # UserPromptSubmit's additionalContext never renders in a stream-json
        # transcript either (same UPG-IU-HOOK-NONVACUITY-CANARY bug), so
        # transcript content is deliberately never consulted here -- delivery
        # rests solely on the before/after hook_injection_counts delta
        # run_leg.py::run_agent captures bracketing the agent session
        # (DESIGN.md 4.1).
        nv["planted_content_in_transcript"] = None

    if trail_text_delivered is False:
        reasons.append("T3: variant's provenance-trail text absent from probe's returned context")

    return {"valid": not reasons, "invalid_reason": "; ".join(reasons), "non_vacuity": nv}
