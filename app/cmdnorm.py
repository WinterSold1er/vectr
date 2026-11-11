"""Deterministic, structural command-line normalization for arc detection.

Pure functions only — no I/O, no state, no imports from `app.routes` /
`app.service`. Shared by `app/arcs.py` (in-memory similarity matching) and
the episode write path (persisted `verb`/`flags_json`/`args_json` columns)
so both derive the exact same normalized triple from the same raw command
string — nothing to reconcile between the two call sites.

R5 scope note: everything classified here is the **argv structure of an
already-issued tool call** (which token is a flag, a path, a version
string, semantics-neutral shell decoration) — tool-call structure, not
prompt content. Sanctioned. This module never reads user/task-prompt text.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

from agent.config import (
    ARC_NORM_ENV_ASSIGNMENT_REGEX,
    ARC_NORM_MAX_VERB_TOKENS,
    ARC_NORM_NICE_NICENESS_FLAG,
    ARC_NORM_NUM_REGEX,
    ARC_NORM_PATH_EXTENSION_REGEX,
    ARC_NORM_PIPELINE_DISPLAY_ONLY_VERBS,
    ARC_NORM_STDERR_MERGE_TOKEN,
    ARC_NORM_UUID_REGEX,
    ARC_NORM_VERSION_REGEX,
    ARC_NORM_WRAPPER_PREFIXES,
)

_UUID_RE = re.compile(ARC_NORM_UUID_REGEX)
_VERSION_RE = re.compile(ARC_NORM_VERSION_REGEX)
_NUM_RE = re.compile(ARC_NORM_NUM_REGEX)
_PATH_EXT_RE = re.compile(ARC_NORM_PATH_EXTENSION_REGEX)
_ENV_ASSIGNMENT_RE = re.compile(ARC_NORM_ENV_ASSIGNMENT_REGEX)

# Compound-command boundaries (top-level, unquoted, unescaped). Four guards
# keep quoted text and literal backslash-escapes out of this set's way:
#   1. `tokenize`'s pre-tokenize padding pass (UPG-CMDNORM-GLUED-SEPARATOR)
#      tracks quote state and never pads a `;`/`&&`/`||`/`&` inside quotes.
#   2. The same pass also pads a glued single `|` (UPG-CMDNORM-SINGLE-
#      PIPE-AMP) so `cat a|grep b` and `cat a | grep b` tokenize the same
#      way, letting the existing `|` split on the primary segment see it,
#      and a glued single `&` (UPG-CMDNORM-BACKGROUND-AMP) so `make&make`
#      and `make & make` tokenize the same way, letting the compound-
#      separator split on `&` see the boundary.
#   3. It also marks a backslash-escaped compound separator at a token
#      boundary (`<bs>;`, `<bs>&`, `<bs>&<bs>&`, `<bs>|`, `<bs>|<bs>|`)
#      with an internal marker (UPG-CMDNORM-ESCAPED-SEPARATOR-SPLIT and
#      UPG-CMDNORM-ESCAPED-SINGLE-PIPE-STAGE); `_split_on_any` strips
#      the marker and either skips the bare separator (for `;`/`&&`/`||`/
#      `&`, which are control operators and never valid argv) or adds it
#      to the current segment as a literal `|` arg (for `|`, which the
#      user may have escaped precisely to pass as a literal char rather
#      than a pipeline boundary).
# After all four, a literal token equal to one of these is never something
# the user quoted (`grep "a|b"` stays one shlex token) or escaped (`find
# . -exec rm {} \;` keeps the literal `;` from being misread).
_COMPOUND_SEPARATORS = frozenset({"&&", "||", ";", "&"})

# Internal marker used to carry a backslash-escape's "this separator is
# literal" signal through shlex (which strips the backslash) into
# `_split_on_any`. U+2063 INVISIBLE SEPARATOR is never typed by a user
# and never appears in a real shell command, so it is safe to use as a
# private sentinel — and so it can be safely stripped before the
# segment's tokens reach the stored `NormalizedCommand`. The marker is
# only ever inserted by the padding pass for a `<bs><sep>` at a token
# boundary; in every other position the escape rides through shlex as a
# literal char inside its adjacent token.
_ESCAPED_SEP_MARKER = "⁣"


@dataclass(frozen=True)
class NormalizedCommand:
    """The normalized (verb, flags, args) triple for one Bash invocation.

    `args` holds the RAW positional-argument tokens (concrete values, order
    preserved) alongside `arg_classes` — the same-length abstraction of each
    (`<PATH>`/`<VERSION>`/`<UUID>`/`<NUM>`, or the literal token when
    unclassified) used for comparison only.
    """

    verb: str
    flags: tuple[str, ...]
    args: tuple[str, ...]
    arg_classes: tuple[str, ...]
    env_prefix_names: tuple[str, ...] = field(default_factory=tuple)
    cmd_raw: str = ""


def _pad_glued_compound_separators(cmd_raw: str) -> str:
    """Pre-tokenize pass (UPG-CMDNORM-GLUED-SEPARATOR): pad bare `;`, `&&`
    and `||` occurrences with spaces so `shlex.split` yields them as their
    own tokens — `cd /path;cmd` must split exactly like `cd /path ; cmd`,
    because in the shell that RAN this command an unquoted `;`/`&&`/`||` IS
    a control operator regardless of surrounding whitespace. Padding an
    unquoted separator therefore aligns tokenization with what actually
    executed; leaving it glued mis-attributes both segments to one command.

    Quote state is tracked character-by-character so quoted text is never
    touched: `echo "a;b"` stays one argument, and `&&` inside a quoted
    argument is never split. Backslash escapes are honored OUTSIDE quotes
    (`\\;` in a `find -exec` tail is literal) and inside double quotes
    (`\\"` does not close the string for scanning purposes).

    Deliberately NOT handled (a stated boundary, not an oversight):
      - `$()` / backtick substitution bodies are scanned like plain text,
        so an unquoted separator inside them gets padded even though it
        belongs to the inner command (`echo $(a;b)`);
      - heredoc bodies (`<<EOF ... EOF`) are scanned like plain text too;
      - nested/odd quote interleavings across a boundary ('"a'"'"';b')
        follow the same single-pass rules shlex itself applies, but were
        not exhaustively verified.
    The pass only ever affects how vectr NORMALIZES an already-recorded
    command for comparison — never execution, which happened before vectr
    ever saw the string.
    """
    out: list[str] = []
    i = 0
    n = len(cmd_raw)
    in_single = False
    in_double = False
    while i < n:
        c = cmd_raw[i]
        if in_single:
            out.append(c)
            if c == "'":
                in_single = False
            i += 1
        elif in_double:
            if c == "\\" and i + 1 < n:
                # Inside double quotes a backslash escapes the next char
                # for scanning purposes (\\" must not close the string).
                out.append(cmd_raw[i:i + 2])
                i += 2
            else:
                out.append(c)
                if c == '"':
                    in_double = False
                i += 1
        elif c == "\\" and i + 1 < n:
            # Outside quotes: an escaped pair is literal (`\\;`, `\\&\\&`)
            # — never a separator, never a quote-state change. The one
            # exception is a `<bs><sep>` AT A TOKEN BOUNDARY: shlex will
            # strip the backslash and emit a bare `<sep>` token that
            # `_split_on_any` would then misread as a real compound
            # separator (UPG-CMDNORM-ESCAPED-SEPARATOR-SPLIT). For that
            # shape we mark the escape with `_ESCAPED_SEP_MARKER` so
            # `_split_on_any` can keep the bare sep out of the separator
            # set; glued escapes like `echo a\;b` are not at a boundary
            # and shlex already keeps the `;` inside one token, so they
            # need no marker.
            nxt = cmd_raw[i + 1]
            if (
                _out_ends_at_token_boundary(out)
                and _is_escaped_compound_sep(cmd_raw, i)
            ):
                # Order matters: the doubled `<bs><X><bs><X>` check must
                # fire BEFORE the single-char `&`/`|` branch, so a
                # `\|\|` at a boundary is marked as one escaped `||`
                # (advance 4), not as two single escaped `|` tokens
                # (advance 2 twice). Mirrors the ordering hazard in
                # `_pad_glued_compound_separators`'s `&&`/`||` vs single
                # `|` branch.
                if (
                    nxt in "&|"
                    and i + 3 < len(cmd_raw)
                    and cmd_raw[i + 2] == "\\"
                    and cmd_raw[i + 3] == nxt
                ):
                    # Doubled form `<bs><X><bs><X>` for `&&`/`||`.
                    out.append(_ESCAPED_SEP_MARKER + nxt + nxt)
                    i += 4
                elif nxt == ";":
                    out.append(_ESCAPED_SEP_MARKER + ";")
                    i += 2
                else:
                    # Single-char `&` (UPG-CMDNORM-BACKGROUND-AMP) or `|`
                    # (UPG-CMDNORM-ESCAPED-SINGLE-PIPE-STAGE) at a token
                    # boundary: mark with the marker so the downstream
                    # `_split_on_any` calls can keep the bare sep out of
                    # the compound-separator split and (for `|`) out of
                    # the pipeline-staging split.
                    out.append(_ESCAPED_SEP_MARKER + nxt)
                    i += 2
            else:
                out.append(cmd_raw[i:i + 2])
                i += 2
        elif c == "'":
            in_single = True
            out.append(c)
            i += 1
        elif c == '"':
            in_double = True
            out.append(c)
            i += 1
        elif c == ";":
            out.append(" ; ")
            i += 1
        elif c in "&|" and i + 1 < n and cmd_raw[i + 1] == c:
            # Doubled `&&` / `||` — must fire BEFORE the single-`|` / `&`
            # branches below so `||` is treated as the doubled compound
            # separator, not a single `|` followed by an unmatched `|`,
            # and `&&` is read as one doubled separator, not two single
            # `&` background markers.
            out.append(" " + c + c + " ")
            i += 2
        elif c == "&":
            # Single `&` (UPG-CMDNORM-BACKGROUND-AMP): a glued single `&`
            # like `make&make` would shlex into one token, so the
            # compound-separator split on `&` would never see it. Padding
            # the glued form makes spaced and glued `&` tokenize
            # identically, with no vocabulary change for downstream code.
            #
            # Two `&`-as-redirect shapes must NOT be padded, since they
            # are not background operators but part of a shell redirect:
            #   - `>&` (the `&` follows `>`, e.g. `2>&1`, `1>&2`):
            #     padding would split the stderr-merge token and break
            #     `normalize_command`'s trailing-stderr-merge stripping.
            #   - `&>` (the `&` is followed by `>`, e.g. `cmd &>file`,
            #     bash-specific redirect of both streams to `file`):
            #     padding would route the primary to `>` and mis-attribute
            #     the redirect. Not tested in the suite but handled for
            #     the same reason.
            prev_is_gt = bool(out) and out[-1][-1] == ">"
            next_is_gt = i + 1 < n and cmd_raw[i + 1] == ">"
            if prev_is_gt or next_is_gt:
                out.append(c)
                i += 1
            else:
                out.append(" & ")
                i += 1
        elif c == "|":
            # Single `|` (UPG-CMDNORM-SINGLE-PIPE-AMP): a glued single pipe
            # like `cat a|grep b` would shlex into one token per glued
            # group, so the existing `|` split on the primary segment
            # would never see it. Padding the glued form makes spaced and
            # glued pipes tokenize identically, with no vocabulary change
            # for downstream code.
            out.append(" | ")
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _is_escaped_compound_sep(cmd_raw: str, i: int) -> bool:
    """At position `i` (which is a backslash), does the escape form a
    top-level compound separator? Detected shapes:

      - `<bs>;`             (single-char `;`)
      - `<bs>&`             (single-char `&`, UPG-CMDNORM-BACKGROUND-AMP)
      - `<bs>|`             (single-char `|`, UPG-CMDNORM-ESCAPED-SINGLE-
                             PIPE-STAGE; see `_split_on_any` for the
                             special handling that keeps the bare `|`
                             as a literal arg instead of a stage boundary)
      - `<bs><X><bs><X>`    (doubled `&&` / `||` for X in `&|`)

    The single-`<bs>|` case is the one that differs from the generic
    marked-separator behavior: `normalize_command` runs a second
    `_split_on_any` over the primary segment with `seps={"|"}` for
    pipeline staging, and a bare `|` would be misread as a stage
    boundary there. The marker carries the "this was escaped" signal
    through both calls; the first call preserves it (does not strip
    the marker from `|`) and the second call strips it and adds the
    bare `|` to the current segment."""
    if i + 1 >= len(cmd_raw):
        return False
    nxt = cmd_raw[i + 1]
    if nxt in ";&|":
        # Single-char `;`, `&`, or `|` at a token boundary.
        return True
    if (
        nxt in "&|"
        and i + 3 < len(cmd_raw)
        and cmd_raw[i + 2] == "\\"
        and cmd_raw[i + 3] == nxt
    ):
        # Doubled `&&` / `||`.
        return True
    return False


def _out_ends_at_token_boundary(out: list[str]) -> bool:
    """Is the most-recently-emitted character a token boundary for shlex
    purposes (whitespace, or the very start of input)? Used to decide
    whether an escape would, after shlex drops the backslash, arrive at
    `_split_on_any` as its own bare token.

    `out` is the padding pass's running emit list — entries are single
    chars except for escape pairs (`<bs>X`), so the last character of
    the last entry is the right thing to look at."""
    if not out:
        return True
    return out[-1][-1].isspace()


def tokenize(cmd_raw: str) -> list[str]:
    """Quote-aware tokenize. Falls back to a naive whitespace split on
    unbalanced quotes rather than raising — a malformed command is still a
    real episode we must not crash on. The fallback splits the ORIGINAL
    string, not the padded one: with unbalanced quotes the padding pass's
    own quote tracking is unreliable, so its output is not trusted there."""
    if not cmd_raw:
        return []
    try:
        return shlex.split(_pad_glued_compound_separators(cmd_raw), posix=True)
    except ValueError:
        return cmd_raw.split()


def classify_arg(token: str) -> str:
    """Return the abstraction class placeholder for a positional-argument
    token, or the token itself verbatim when none applies. Order: uuid >
    version > unambiguous-path (slash / "." / ".." / "~" prefix) > num >
    extension-shaped-path > literal.

    The extension-shaped path check (`name.ext`) is deliberately tried
    AFTER num: it is a loose shape that a bare decimal number like "3.14"
    also satisfies, so num (a strictly digits-and-one-dot pattern) must
    get first refusal on anything that looks purely numeric. The
    unambiguous path indicators checked earlier never collide with a
    number, so they keep precedence over num."""
    if _UUID_RE.match(token):
        return "<UUID>"
    if _VERSION_RE.match(token):
        return "<VERSION>"
    if "/" in token or token in (".", "..") or token.startswith("~"):
        return "<PATH>"
    if _NUM_RE.match(token):
        return "<NUM>"
    if _PATH_EXT_RE.match(token):
        return "<PATH>"
    return token


def _split_on_any(tokens: list[str], seps: frozenset[str]) -> list[list[str]]:
    """Split a token list on any token present in `seps`, dropping the
    separator tokens. Also recognizes `_ESCAPED_SEP_MARKER`-tagged tokens
    (UPG-CMDNORM-ESCAPED-SEPARATOR-SPLIT) — a marked `<MARKER><sep>` is
    the padding pass's signal that the user wrote `<bs><sep>` and meant
    the separator literally. For `;`/`&&`/`||`/`&` (control operators,
    never valid argv), we keep the bare sep out of argv AND out of the
    separator set (mirroring the pre-fix behavior for the terminal
    `find -exec ... <bs>;` shape, where the phantom split produced a
    trailing empty segment that the final filter discarded). For `|`
    (UPG-CMDNORM-ESCAPED-SINGLE-PIPE-STAGE), the user may have escaped
    it precisely to pass it as a literal char rather than a pipeline
    boundary, so the second-call behavior strips the marker and adds
    the bare `|` to the current segment instead of skipping it. The
    first call (compound separators) preserves the marker on a `|` so
    the second call can see it — otherwise the bare `|` would survive
    into the primary segment and the staging split would misfire. The
    marker itself never reaches a stored `NormalizedCommand` — it is
    either consumed by the marked-sep branch or stripped from any
    non-separator token that happened to start with it."""
    segments: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if tok.startswith(_ESCAPED_SEP_MARKER):
            rest = tok[len(_ESCAPED_SEP_MARKER):]
            if rest in seps:
                if rest == "|":
                    # Marked escaped pipe (UPG-CMDNORM-ESCAPED-SINGLE-
                    # PIPE-STAGE): the user wrote `<bs>|` at a token
                    # boundary, meaning a literal `|` arg, NOT a
                    # pipeline boundary. Strip the marker and add the
                    # bare `|` to the current segment.
                    current.append("|")
                # else: marked escaped compound separator: do not add
                # to argv, do not split.
                continue
            # Marker as a prefix in a non-separator token: strip it so
            # it does not leak into stored argv. Exception: a marked
            # `|` must keep its marker here so the pipeline-staging
            # call (which uses a different `seps` set) can recognize
            # it and add the bare `|` to the current segment.
            if rest == "|":
                current.append(tok)
                continue
            tok = rest
        if tok in seps:
            segments.append(current)
            current = []
        else:
            current.append(tok)
    segments.append(current)
    return [s for s in segments if s]


def _consume_leading_cd(segments: list[list[str]]) -> tuple[list[list[str]], str | None]:
    """Strip leading `cd <path> &&`/`cd <path>;` segments and return the
    stripped segments together with the LAST leading `cd`'s target —
    repeatedly, so `cd a && cd b && real-cmd` reduces to `(real-cmd, "b")`:
    the last `cd` is the one whose target is the directory the primary
    command actually runs in. A bare `cd` (no argument, meaning $HOME)
    resets the running target to None — there is no concrete path in argv
    for $HOME — but a later `cd <path>` in the same chain still overrides
    it, matching real shell semantics."""
    target: str | None = None
    while segments and segments[0][0] == "cd" and len(segments[0]) <= 2:
        seg = segments[0]
        target = seg[1] if len(seg) == 2 else None
        segments = segments[1:]
    return segments, target


def _strip_leading_cd(segments: list[list[str]]) -> list[list[str]]:
    """Strip leading `cd <path> &&` segments (semantics-neutral decoration)
    — repeatedly, so `cd a && cd b && real-cmd` reduces to `real-cmd`."""
    stripped, _ = _consume_leading_cd(segments)
    return stripped


# `leading_cd_resolution`'s status vocabulary. Plain strings rather than an
# Enum: callers compare against them directly and the closed set lives in
# this module's docstrings, same reasoning as NOTE_EVENT_KINDS.
CD_NONE = "none"                # no leading `cd` chain — caller may trust the episode cwd field
CD_ABSOLUTE = "absolute"        # concrete target starting with `/`: REPLACES the cwd field
CD_RELATIVE = "relative"        # concrete relative target: COMPOSES with the cwd field
CD_SYMBOLIC_HOME = "symbolic_home"  # `$HOME` family (`cd`, `~`, `~/x`, `~user`): constant per machine, unknowable absolutely
CD_UNRESOLVABLE = "unresolvable"    # a leading cd exists but its effective dir is not statically derivable


def leading_cd_target(cmd_raw: str) -> str | None:
    """Return the directory a leading `cd <path> &&` / `cd <path>;` chain
    in `cmd_raw` sets before the real command runs, or None when there is
    no such leading chain — or the chain's final target is not a concrete,
    statically-usable path (bare `cd` to $HOME yields the symbolic "~";
    see `leading_cd_resolution` for the full classification).

    Thin wrapper over `leading_cd_resolution` (UPG-ARC-CWD-VS-EFFECTIVE-DIR),
    kept as a separate name because its original contract — "the raw target
    token, unresolved" — is what tests pin; arc bucket-key derivation uses
    `leading_cd_resolution` directly so it can also distinguish "no cd at
    all" from "a cd whose target cannot be resolved"."""
    status, target = leading_cd_resolution(cmd_raw or "")
    if status in (CD_ABSOLUTE, CD_RELATIVE, CD_SYMBOLIC_HOME):
        return target
    return None


def leading_cd_resolution(cmd_raw: str) -> tuple[str, str | None]:
    """Classify the LEADING `cd` chain of `cmd_raw` for effective-directory
    bucketing (UPG-ARC-CWD-VS-EFFECTIVE-DIR). Returns `(status, target)`:

      (CD_NONE, None)          — no leading cd chain at all; the episode's
                                 own cwd field IS where the command ran.
      (CD_ABSOLUTE, "/abs")    — `cd /abs ...`: replaces the cwd field.
      (CD_RELATIVE, "sub/dir") — `cd sub/dir ...`: composes with the cwd
                                 field (caller joins + normalizes lexically).
      (CD_SYMBOLIC_HOME, "~"-form) — bare `cd`, `~`, `~/x`, `~user/x` all
                                 land in $HOME-derived directories: absolute
                                 value unknowable from a static string, but
                                 CONSTANT for the life of a session/machine,
                                 so the tilde-form itself is a truthful
                                 bucket key (two episodes that both ran
                                 `cd ~/repo && x` ran in the SAME directory
                                 regardless of their harness cwd fields).
      (CD_UNRESOLVABLE, None)  — a leading cd chain EXISTS but its effective
                                 directory is not derivable: `cd -`
                                 ($OLDPWD varies per call site), a target
                                 containing `$` (value varies), or a shape
                                 this parser deliberately declines (flagged
                                 `cd -- <p>` / `cd -L <p>`). Callers must
                                 REFUSE TO PAIR on this status, never fall
                                 back to the cwd field — the cd proves the
                                 field wrong, so falling back reintroduces
                                 exactly the cross-directory false pairing
                                 this item exists to kill.

    Shape recognition (which leading segments get consumed) mirrors
    `_strip_leading_cd`/`_consume_leading_cd` exactly — same tokenize/
    split, same `len(segment) <= 2` gate — so verb-normalization stripping
    and bucket-key classification always agree on what counts as a leading
    `cd`. What resolution ADDS is the verdict for shapes both leave alone:
    a declined shape is UNRESOLVABLE (refuse), not NONE (trust the field).

    Still deliberately NOT recognized as a leading cd at all (status
    CD_NONE, i.e. today's trust-the-field behavior):
      - `pushd`/`popd` (directory-STACK semantics, not a linear prefix);
      - a `cd` appearing anywhere other than leading (`mkdir -p d && cd d
        && x`) — only a LEADING cd is unambiguous prefix decoration."""
    segments = _split_on_any(tokenize(cmd_raw or ""), _COMPOUND_SEPARATORS)
    target: str | None = None
    saw_cd = False
    idx = 0
    while (
        idx < len(segments)
        and segments[idx]
        and segments[idx][0] == "cd"
        and len(segments[idx]) <= 2
    ):
        saw_cd = True
        target = segments[idx][1] if len(segments[idx]) == 2 else None
        idx += 1
    if not saw_cd:
        if segments and segments[0] and segments[0][0] == "cd":
            # Leading `cd` in a declined shape (`cd -- <p>`, `cd -L <p>` —
            # len > 2, the same shape _strip_leading_cd leaves alone).
            return (CD_UNRESOLVABLE, None)
        return (CD_NONE, None)
    if target is None:
        # Bare `cd` (no argument): goes home. "~" is the canonical
        # symbolic key for $HOME itself.
        return (CD_SYMBOLIC_HOME, "~")
    if target.startswith("~"):
        # `~`, `~/x`, `~user/x`: HOME-relative, constant per machine.
        return (CD_SYMBOLIC_HOME, target)
    if target == "-" or "$" in target:
        # `cd -` ($OLDPWD) and unexpanded env-var targets vary per call
        # site — no truthful key exists.
        return (CD_UNRESOLVABLE, None)
    if target.startswith("/"):
        return (CD_ABSOLUTE, target)
    return (CD_RELATIVE, target)


def _is_display_only_stage(stage: list[str]) -> bool:
    return bool(stage) and stage[0] in ARC_NORM_PIPELINE_DISPLAY_ONLY_VERBS


def _strip_trailing_display_stages(pipeline_stages: list[list[str]]) -> list[list[str]]:
    """Drop a TRAILING run of display-only stages (`| tail -30`, `| cat`) —
    these only reshape output for a human, never change what actually ran.
    A stage is only ever eligible while it is the last remaining one, so a
    genuine multi-stage pipeline (`cat data.csv | python train.py`) keeps
    every non-trailing stage's tokens (unconditionally collapsing to stage
    0 made distinct pipelines normalize identical)."""
    while len(pipeline_stages) > 1 and _is_display_only_stage(pipeline_stages[-1]):
        pipeline_stages = pipeline_stages[:-1]
    return pipeline_stages


def _strip_env_and_wrapper_prefixes(stage: list[str]) -> tuple[list[str], list[str]]:
    """Strip, iteratively and in any interleaving, leading bare env-var
    assignments (`FOO=bar cmd`) and transparent wrapper-prefix tokens
    (`timeout N`, `env VAR=...`, `nice [-n N]`, `nohup`, `stdbuf -xX`) from
    the front of a pipeline stage so the WRAPPED command, not the wrapper,
    is what verb extraction sees.
    Returns (remaining_tokens, env_assignment_names)."""
    tokens = list(stage)
    env_names: list[str] = []
    while tokens:
        head = tokens[0]
        if _ENV_ASSIGNMENT_RE.match(head):
            env_names.append(head.split("=", 1)[0])
            tokens = tokens[1:]
            continue
        kind = ARC_NORM_WRAPPER_PREFIXES.get(head)
        if kind is None:
            break
        tokens = tokens[1:]
        if kind == "fixed_arg":
            tokens = tokens[1:]
        elif kind == "nice_niceness":
            if tokens and tokens[0] == ARC_NORM_NICE_NICENESS_FLAG:
                tokens = tokens[1:]
                if tokens:
                    tokens = tokens[1:]
        elif kind == "dash_flags":
            while tokens and tokens[0].startswith("-"):
                tokens = tokens[1:]
        # "bare" and "env_assignments" wrappers consume only their own
        # name token — nothing further to drop here.
    return tokens, env_names


def normalize_command(cmd_raw: str) -> NormalizedCommand:
    """Normalize one raw Bash command string into (verb, flags, args)."""
    tokens = tokenize(cmd_raw or "")
    segments = _strip_leading_cd(_split_on_any(tokens, _COMPOUND_SEPARATORS))
    # Multiple genuine (non-cd) segments left over from a compound command
    # (e.g. `make clean && make build`) — the LAST one is what the episode's
    # own rc/outcome actually reflects, so it defines the normalized command.
    primary_segment = segments[-1] if segments else []

    pipeline_stages = _strip_trailing_display_stages(
        _split_on_any(primary_segment, frozenset({"|"}))
    )
    primary_stage = pipeline_stages[0] if pipeline_stages else []
    downstream_stages = pipeline_stages[1:]
    if primary_stage and primary_stage[-1] == ARC_NORM_STDERR_MERGE_TOKEN:
        primary_stage = primary_stage[:-1]

    primary_stage, env_names = _strip_env_and_wrapper_prefixes(primary_stage)

    if not primary_stage and not downstream_stages:
        return NormalizedCommand(
            verb="",
            flags=(),
            args=(),
            arg_classes=(),
            env_prefix_names=tuple(env_names),
            cmd_raw=cmd_raw or "",
        )

    verb = ""
    j = 0
    if primary_stage:
        # verb = binary + immediate subcommand chain (`git commit`, `./mvnw
        # test`, `npm run build`): keep absorbing leading bareword tokens
        # (not flag-shaped, not arg-classified) up to the configured cap,
        # which bounds runaway absorption of positional arguments (e.g.
        # `cp src dest`) into the verb.
        verb_tokens = [primary_stage[0]]
        j = 1
        while j < len(primary_stage) and len(verb_tokens) < ARC_NORM_MAX_VERB_TOKENS:
            tok = primary_stage[j]
            if tok.startswith("-") or classify_arg(tok) != tok:
                break
            verb_tokens.append(tok)
            j += 1
        verb = " ".join(verb_tokens)

    flags: list[str] = []
    args: list[str] = []
    arg_classes: list[str] = []
    for tok in primary_stage[j:]:
        if tok.startswith("-"):
            flags.append(tok)
        else:
            args.append(tok)
            arg_classes.append(classify_arg(tok))

    # Non-trailing downstream pipeline stages (`| python train.py` in `cat
    # data.csv | python train.py`) must stay in the comparison set so
    # distinct pipelines never normalize identical — every one of their
    # tokens is folded into flags/args exactly like the primary stage's
    # own remainder, preserving order.
    for stage in downstream_stages:
        for tok in stage:
            if tok.startswith("-"):
                flags.append(tok)
            else:
                args.append(tok)
                arg_classes.append(classify_arg(tok))

    return NormalizedCommand(
        verb=verb,
        flags=tuple(sorted(flags)),
        args=tuple(args),
        arg_classes=tuple(arg_classes),
        env_prefix_names=tuple(env_names),
        cmd_raw=cmd_raw or "",
    )
