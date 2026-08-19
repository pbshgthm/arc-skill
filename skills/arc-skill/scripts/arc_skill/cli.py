from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .analysis import run_python
from .broker import (
    LOCAL_MODE,
    REMOTE_MODE,
    broker_matches_latest_event,
    broker_observe,
    broker_ping,
    is_remote_config,
    local_cache_root,
    reconcile_mutations,
    start_broker,
    stop_broker,
)
from .core import (
    ArcSkillError,
    RunPaths,
    append_event,
    append_jsonl,
    atomic_json,
    command_status,
    load_events,
    load_jsonl,
    make_event,
    normalize_game_id,
    now_iso,
    read_json,
    require_run,
    run_lock,
)
from .evidence import render_event
from .inspect import export_history, result_text, status_text, view_text
from .live import execute_action, execute_solve_plan, execute_steps, reset_level
from .perception import build_scene_dossier
from .predictions import CLAIMS_HELP
from .rules import RULES_HELP, init_rules, replay_rules, solve_rules


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ArcSkillError(message)


def _parser() -> Parser:
    parser = Parser(
        prog="arc", description="ARC-AGI-3 harness: look, predict, act, compare"
    )
    parser.add_argument("--run-dir", default=".", help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser(
        "start", help="start or resume the one persistent game run"
    )
    start.add_argument("game_id")
    start.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    start.add_argument("--adapter", help=argparse.SUPPRESS)
    start.add_argument(
        "--mode",
        choices=(LOCAL_MODE, REMOTE_MODE),
        help="local simulator (default), or expiring remote competition validation",
    )

    status = commands.add_parser(
        "status",
        help="full picture: progress, image, actions, recent results, notes",
    )
    status.add_argument("--history", type=int, default=8)

    view = commands.add_parser(
        "view", help="inspect the rendered board, exact pixels, diffs, and animation"
    )
    view.add_argument("--event", type=int)
    view.add_argument(
        "--grid", action="store_true", help="print the complete exact 0-f grid"
    )
    view.add_argument(
        "--frames", action="store_true", help="show causal animation frames"
    )
    view.add_argument(
        "--crop", metavar="R0:R1,C0:C1", help="print an exact half-open crop"
    )
    view.add_argument("--history", type=int, default=0)
    view.add_argument("--export", type=Path, metavar="FILE.npz")

    act = commands.add_parser(
        "act",
        help="take one action with a prediction; the result is graded against it",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=CLAIMS_HELP,
    )
    act.add_argument("action")
    act.add_argument("coordinates", nargs="*", help=argparse.SUPPRESS)
    act.add_argument(
        "--predict",
        required=True,
        help='what this action does, e.g. "move 12,5 1,0" (see below)',
    )
    act.add_argument("--because", help="short reason for choosing this action")
    act.add_argument("--at", type=int, dest="at_event")

    commit = commands.add_parser(
        "commit",
        help="run a prediction-checked batch or a solve-plan; halts on the first miss",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=CLAIMS_HELP,
    )
    commit.add_argument(
        "plan",
        nargs="?",
        help="solve-plan from `arc rules solve`, e.g. @.arc/plan.json",
    )
    commit.add_argument(
        "--step",
        action="append",
        default=[],
        metavar='"ACTION :: CLAIMS"',
        help="one action with its own prediction; repeat in execution order",
    )
    commit.add_argument("--at", type=int, dest="at_event")

    reset = commands.add_parser(
        "reset", help="pay one action to rewind only the current level"
    )
    reset.add_argument(
        "--because", help="why the current board is worth abandoning (required unless GAME_OVER)"
    )
    reset.add_argument("--at", type=int, dest="at_event")

    python = commands.add_parser(
        "python",
        help="run offline Python with grids, history, perception, BFS, and A* preloaded",
    )
    python.add_argument("source", nargs="?")
    python.add_argument("--file", type=Path)

    rules = commands.add_parser(
        "rules",
        help="optional executable-rules tier: write rules.py, replay-verify, search",
    )
    subcommands = rules.add_subparsers(dest="rules_command", required=True)
    subcommands.add_parser("help", help="print the compact rules.py contract")
    subcommands.add_parser("init", help="create a rules.py template without overwriting")
    subcommands.add_parser(
        "replay", help="check rules.py against every recorded transition"
    )
    solve = subcommands.add_parser(
        "solve", help="A* search rules.py for a level plan with per-step predictions"
    )
    solve.add_argument("--seconds", type=float, default=15.0)
    solve.add_argument("--max-nodes", type=int, default=250_000)
    return parser


def _crop(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    try:
        rows, columns = value.split(",", 1)
        top, bottom = (int(item) for item in rows.split(":", 1))
        left, right = (int(item) for item in columns.split(":", 1))
        return top, bottom, left, right
    except (ValueError, TypeError):
        raise ArcSkillError(
            "crop format is R0:R1,C0:C1, using half-open bounds"
        ) from None


def _action_token(action: str, coordinates: list[str]) -> str:
    token = action.upper()
    if coordinates:
        if token != "ACTION6" or len(coordinates) != 2:
            raise ArcSkillError(
                "coordinates are only accepted as `arc act ACTION6 X Y`"
            )
        try:
            token = f"ACTION6:{int(coordinates[0])},{int(coordinates[1])}"
        except ValueError:
            raise ArcSkillError(
                "ACTION6 coordinates must be integers: `arc act ACTION6 X Y`"
            ) from None
    return token


def _write_notes(paths: RunPaths, game_id: str) -> None:
    if paths.notes.exists():
        return
    paths.notes.write_text(
        f"# Notes — {game_id}\n"
        "\n"
        "## Verified (cite event ids)\n"
        "\n"
        "## Assumed / open questions\n"
        "\n"
        "## Plan\n"
    )


def _remote_idle_seconds(paths: RunPaths, config: dict[str, Any]) -> float:
    mutations = load_jsonl(paths.mutations)
    timestamp = (
        mutations[-1].get("timestamp") if mutations else config.get("created_at")
    )
    if not timestamp:
        return 0.0
    try:
        then = dt.datetime.fromisoformat(str(timestamp))
        if then.tzinfo is None:
            then = then.replace(tzinfo=dt.timezone.utc)
        return max(0.0, (dt.datetime.now(dt.timezone.utc) - then).total_seconds())
    except ValueError:
        return 0.0


def _start(paths: RunPaths, args: argparse.Namespace) -> None:
    requested = normalize_game_id(args.game_id)
    existing = read_json(paths.config)
    if isinstance(existing, dict):
        if existing.get("game_id") != requested:
            raise ArcSkillError(
                f"this directory already owns {existing.get('game_id')}; use a fresh directory for {requested}"
            )
        existing_mode = str(existing.get("mode", LOCAL_MODE)).lower()
        if args.mode is not None and args.mode != existing_mode:
            raise ArcSkillError(
                f"this directory already owns a {existing_mode} run; mode cannot be changed in place"
            )
        recovered = reconcile_mutations(paths)
        owner = read_json(paths.broker, {})
        if owner.get("status") == "FINISHED":
            print(f"RESUMED | {requested} | completed run")
            print(status_text(paths))
            return
        if is_remote_config(existing):
            idle = _remote_idle_seconds(paths, existing)
            if idle >= 15 * 60:
                raise ArcSkillError(
                    "REMOTE_LEASE_EXPIRED | no live action was recorded for at least 15 minutes. The remote competition run is not recoverable; preserve this directory and use a fresh one"
                )
            if not broker_ping(paths):
                raise ArcSkillError(
                    "the remote competition owner is unavailable and cannot be reconstructed; preserve this directory and use a fresh one"
                )
            if not broker_matches_latest_event(paths):
                raise ArcSkillError(
                    "REMOTE_STATE_DIVERGED | the live remote observation differs from the append-only timeline; stop using this run"
                )
            print(
                f"RESUMED | {requested} | REMOTE competition | action-idle lease about {max(0, 15 - int(idle // 60))}m"
            )
        else:
            restarted = False
            if not broker_ping(paths):
                stop_broker(paths)
                time.sleep(0.05)
                paths.socket.unlink(missing_ok=True)
                start_broker(paths)
                restarted = True
            if not broker_matches_latest_event(paths):
                raise ArcSkillError(
                    "LOCAL_REPLAY_DIVERGED | reconstructed simulator state differs from the latest timeline event"
                )
            verb = "RECOVERED" if restarted else "RESUMED"
            print(
                f"{verb} | {requested} | local simulator | replayed {len(load_events(paths)) - 1} paid actions"
            )
        if recovered:
            print(f"JOURNAL | recovered {recovered} paid action(s) into timeline")
        print(status_text(paths))
        return

    paths.root.mkdir(parents=True, exist_ok=True)
    mode = str(args.mode or os.getenv("ARC_SKILL_MODE", LOCAL_MODE)).lower()
    if mode not in {LOCAL_MODE, REMOTE_MODE}:
        raise ArcSkillError(
            f"ARC_SKILL_MODE must be {LOCAL_MODE!r} or {REMOTE_MODE!r}, got {mode!r}"
        )
    config: dict[str, Any] = {
        "game_id": requested,
        "seed": args.seed,
        "mode": mode,
        "adapter": args.adapter or "arc_agi",
        "created_at": now_iso(),
        "harness": "arc-skill",
    }
    if mode == LOCAL_MODE:
        config["cache_dir"] = str(local_cache_root())
    paths.state.mkdir(parents=True, exist_ok=True)
    atomic_json(paths.config, config)
    (paths.state / "python").write_text(sys.executable + "\n")
    try:
        start_broker(paths)
        observation, public_info = broker_observe(paths)
        config["public_info"] = public_info
        atomic_json(paths.config, config)
        event = append_event(
            paths,
            make_event(observation, "START", None, None, note="initial observation"),
        )
        _write_notes(paths, requested)
        render_event(paths, event)
        build_scene_dossier(paths, [event])
    except Exception:
        stop_broker(paths)
        shutil.rmtree(paths.state, ignore_errors=True)
        raise
    if mode == REMOTE_MODE:
        print(
            f"STARTED | {requested} | REMOTE competition | single run | ~15m action-idle lease | no replay recovery"
        )
    else:
        print(
            f"STARTED | {requested} | local simulator | competition accounting | replay recovery enabled"
        )
    print(status_text(paths))
    print(
        'USE | open IMAGE first; every `arc act` needs --predict "<claims>" '
        "(`arc act --help` lists the claim forms)"
    )


def _print_replay(result: dict[str, Any]) -> None:
    pixels = (
        "render() not defined"
        if result["pixel_checks"] == 0
        else f"exact pixels {result['pixel_matches']}/{result['pixel_checks']}"
    )
    print(
        f"REPLAY | {result['status']} | {result['explained']}/{result['transitions']} "
        f"transitions explained | {pixels}"
    )
    mismatch = result.get("first_mismatch")
    if mismatch:
        print(
            f"FIRST MISMATCH | e{mismatch['event']} {mismatch['action']} | {mismatch['detail']}"
        )
        if "expected" in mismatch:
            print(f"  expected observe: {mismatch['expected']}")
            print(f"  actual observe:   {mismatch['actual']}")
    for gap in result.get("gaps", ()):
        print(f"GAP | {gap}")
    if result["status"] == "HISTORY_FIT":
        print("NOTE | fits recorded history; unseen mechanics can still differ")


def _print_solve(result: dict[str, Any]) -> None:
    print(
        f"SOLVE | {result['status']} | nodes {result['nodes']} | frontier {result['frontier']} "
        f"| {result['elapsed_seconds']:.3f}s | unknown edges {result['unknown_edges']} "
        f"| dead pruned {result['dead_pruned']}"
    )
    if result.get("replay_status") != "HISTORY_FIT":
        print(
            f"REPLAY | {result['replay_status']} | {len(result.get('replay_gaps', ()))} gaps "
            "— the plan trusts rules that history has not verified"
        )
    if result.get("actions"):
        print("ACTIONS | " + " ".join(result["actions"]))
        print(f"PLAN | {result['plan']} — execute with `arc commit @.arc/plan.json`")
    elif result["status"] == "NO_PLAN_IN_MODEL":
        print(
            "NOTE | no goal state reachable inside rules.py; actions() or step() are "
            "too narrow, or the level needs something unmodeled"
        )
    if result["status"] in {"TIME_LIMIT", "NODE_LIMIT"}:
        print(
            "NOTE | a bound was reached; not evidence that no solution exists "
            "(--seconds / --max-nodes raise it)"
        )
    if result.get("unknown_edges") and not result.get("actions"):
        print(
            f"NOTE | {result['unknown_edges']} Unknown edge(s) were skipped during "
            "search; extending step() may open routes"
        )


def main() -> None:
    try:
        args = _parser().parse_args()
        paths = RunPaths(Path(args.run_dir).resolve())
        if args.command == "start":
            with run_lock(paths):
                _start(paths, args)
            raise SystemExit(0)

        require_run(paths)
        with (
            run_lock(paths),
            command_status(
                paths,
                f"rules {args.rules_command}"
                if args.command == "rules"
                else args.command,
            ),
        ):
            reconcile_mutations(paths)
            if args.command == "status":
                print(status_text(paths, history=args.history))
            elif args.command == "view":
                print(
                    view_text(
                        paths,
                        event_id=args.event,
                        grid=args.grid,
                        frames=args.frames,
                        crop=_crop(args.crop),
                        history=args.history,
                    )
                )
                if args.export:
                    destination = (
                        args.export
                        if args.export.is_absolute()
                        else paths.root / args.export
                    )
                    print(f"EXPORTED | {export_history(paths, destination)}")
            elif args.command == "act":
                receipt = execute_action(
                    paths,
                    _action_token(args.action, args.coordinates),
                    predict=args.predict,
                    because=args.because,
                    at_event=args.at_event,
                )
                print(result_text(paths, receipt))
            elif args.command == "commit":
                if bool(args.plan) == bool(args.step):
                    raise ArcSkillError(
                        "commit takes either @plan.json from `arc rules solve` or "
                        'one or more --step "ACTION :: claims"'
                    )
                receipt = (
                    execute_solve_plan(paths, args.plan, at_event=args.at_event)
                    if args.plan
                    else execute_steps(paths, args.step, at_event=args.at_event)
                )
                print(result_text(paths, receipt))
            elif args.command == "reset":
                receipt = reset_level(
                    paths,
                    because=args.because,
                    at_event=args.at_event,
                )
                print(result_text(paths, receipt))
            elif args.command == "python":
                if bool(args.source) == bool(args.file):
                    raise ArcSkillError(
                        "provide exactly one Python source argument or --file"
                    )
                source = (
                    args.source if args.source is not None else args.file.read_text()
                )
                run_python(paths, source)
            elif args.command == "rules":
                if args.rules_command == "help":
                    print(RULES_HELP)
                elif args.rules_command == "init":
                    print(f"CREATED | {init_rules(paths)}")
                    print(RULES_HELP)
                elif args.rules_command == "replay":
                    result = replay_rules(paths)
                    atomic_json(paths.verification, result)
                    append_jsonl(paths.activity, {"kind": "rules_replay", **result})
                    _print_replay(result)
                elif args.rules_command == "solve":
                    result = solve_rules(
                        paths, seconds=args.seconds, max_nodes=args.max_nodes
                    )
                    append_jsonl(paths.activity, {"kind": "rules_solve", **result})
                    _print_solve(result)
                else:
                    raise ArcSkillError(
                        f"unsupported rules command {args.rules_command}"
                    )
            else:
                raise ArcSkillError(f"unsupported command {args.command}")
        raise SystemExit(0)
    except ArcSkillError as error:
        print(f"ERROR | {error}", file=sys.stderr)
        raise SystemExit(2)
