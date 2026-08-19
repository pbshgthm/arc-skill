# arc-skill

An agent skill that plays [ARC-AGI-3](https://arcprize.org/arc-agi/3/) —
interactive, hidden-rule games on a 64×64 grid — from nothing but a game ID.

The repo contains one skill, `skills/arc-skill/`, which is two things in one
directory:

- **`SKILL.md`** — the playing doctrine the agent follows: predict before every
  action, read the grade, keep a one-page evidence file, batch only proven
  mechanics, and escalate to an executable-rules search only when a level
  resists.
- **`scripts/`** — the `arc` harness the agent drives: a crash-safe run
  directory, a local simulator with competition semantics, graded predictions,
  cell-exact diffs, rendered boards, an offline compute console, and an A\*
  planner over agent-written rules.

Doctrine and harness are co-designed: every rule in `SKILL.md` is one the
harness can enforce or grade, and every harness feature exists to make one rule
cheap to follow.

Provenance: this skill ran the 25 public ARC-AGI-3 games with Claude Code on
Claude Opus 5 and finished all 25 at RHAE 100.00 (7,645 actions, $728.17 at
Opus 5 list prices).

## Requirements

- **Python 3.12+**, or [**uv**](https://docs.astral.sh/uv/) (the launcher
  bootstraps a private runtime when the system Python is unsuitable — first use
  only)
- Runtime dependencies, installed automatically by the launcher:
  `arc-agi==0.9.9`, `numpy>=2,<3`, `pillow>=10,<13`
- **`ARC_API_KEY`** — an [ARC Prize](https://three.arcprize.org/) API key. Needed
  once per game to download it into the durable local cache, and always for
  `--mode competition`. Runs on an already-cached game need no key and no
  network.

## Install

### With `skills`

The repo is laid out for [`npx skills`](https://github.com/vercel-labs/skills),
which discovers the skill under `skills/`, asks which agents to target, and
symlinks it into each one:

```bash
npx skills add pbshgthm/arc-skill
```

Useful flags: `-g` global instead of project-scoped, `-a claude-code` to skip
the agent prompt, `-l` to list without installing, `--copy` if symlinks aren't
available. `pnpm dlx skills add ...` works identically and keeps npm out of it.

### By hand

Skills are auto-discovered from a per-platform directory. Copy or symlink
`skills/arc-skill/` into it — symlink if you want to keep pulling updates from
this repo.

**Claude Code** — one project (use `~/.claude/skills` for every project):

```bash
mkdir -p .claude/skills
ln -s /path/to/arc-skill/skills/arc-skill .claude/skills/arc-skill
```

**Codex**:

```bash
mkdir -p .agents/skills
ln -s /path/to/arc-skill/skills/arc-skill .agents/skills/arc-skill
```

**Anywhere else**: point the agent at `skills/arc-skill/SKILL.md` and tell it to
read the file completely before starting. The skill is agent-agnostic — nothing
in it depends on a particular harness.

### Then

```bash
export ARC_API_KEY=...
```

## Quickstart

Ask the agent to *"solve ARC-AGI-3 game `<GAME_ID>`"* and it takes over from
here. To drive the harness yourself:

```bash
ARC="/path/to/arc-skill/skills/arc-skill/scripts/arc"

mkdir -p ~/runs/ls20 && cd ~/runs/ls20   # one directory = one run
"$ARC" start ls20                        # local simulator, competition semantics
"$ARC" act ACTION1 --predict "move 12,5 0,-1"
"$ARC" status                            # full picture + notes
```

`start` is idempotent and crash-safe: rerun it after any interruption and the
run resumes, or is replayed exactly. Use `--mode competition` for the live
remote server (single run, ~15-minute idle lease, no replay recovery).

Commands: `start`, `status`, `view`, `act`, `commit`, `reset`, `python`, and
`rules {help,init,replay,solve}`. Every one has `--help`.

## Layout

```
skills/arc-skill/
├── SKILL.md                    # the doctrine — what the agent reads
└── scripts/
    ├── arc                     # launcher: resolves a usable Python, execs the CLI
    ├── arc_cli.py              # entry point (PEP 723 deps for `uv run`)
    ├── broker_server.py        # per-run game-session process
    └── arc_skill/
        ├── cli.py              # subcommands, argument surface, `start` bootstrap
        ├── core.py             # run paths, append-only event log, atomic writes, lock
        ├── broker.py           # arc-agi adapter, game cache, remote lease
        ├── live.py             # act / commit / reset, crash replay, level archives
        ├── predictions.py      # the claim vocabulary: parsing and grading
        ├── perception.py       # transition stories, components, lattices, motion
        ├── analysis.py         # the `arc python` console: namespace, BFS, A*
        ├── inspect.py          # status and view: grid text, crops, nudges, banners
        ├── evidence.py         # board and frame rendering to PNG, history lines
        └── rules.py            # rules tier: contract, history replay, A* search
```

A **run directory** is anywhere you `cd` to; the harness owns `.arc/` inside it
(config, `events.jsonl`, rendered images, recordings, `NOTES.md`, level
archives). Only `.arc/NOTES.md` is ever hand-edited.

## Environment variables

| Variable | Effect |
|---|---|
| `ARC_API_KEY` | ARC Prize API key — game downloads and competition mode |
| `ARC_SKILL_MODE` | Default mode when `--mode` is absent (`local` \| `competition`) |
| `ARC_SKILL_CACHE_DIR` | Override the durable game cache (default `~/.cache/arc-skill/arcade`) |
| `XDG_CACHE_HOME` | Cache base when `ARC_SKILL_CACHE_DIR` is unset |
| `AA3_CACHE_DIR` | Optional donor cache adopted on first use (migration from the older `aa3` skill) |
