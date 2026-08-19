# arc-skill

An agent skill that plays [ARC-AGI-3](https://arcprize.org/arc-agi/3/) — 25
video games that never explain their rules — from nothing but a game ID.

**[arc-skill.vercel.app](https://arc-skill.vercel.app/)** is the full write-up:
the method, the misses that mattered, and every run replayable board by board.

```bash
npx skills add pbshgthm/arc-skill

export ARC_API_KEY=...
claude "solve ARC-AGI-3 game lp85"
```

## The result

Claude Code on Claude Opus 5, unmodified, with this skill installed and nothing
else. Every recorded press was replayed through ARC's own servers afterwards;
all 25 games reproduced, none diverged.

| | |
|---|---|
| Games finished | **25 of 25** |
| Levels finished | **183 of 183** |
| RHAE | **100.00** — the benchmark's own score, and its ceiling |
| Actions | **7,645**, where the median human needs 17,135 |
| Verified by | ARC scorecard [`24ddb219`](https://arcprize.org/scorecards/24ddb219-987e-464f-9050-6398a29cf5ac) |

The skill is 129 lines of instruction plus a 4,343-line command line tool.
Neither one mentions a single one of those 25 games.

## The rule

Before the agent may press a button, it writes down what the press will do to
the grid — which cells change, and to what. The harness refuses a press that
arrives without a prediction, then grades it against the frame that comes back.

```bash
arc act ACTION6 58 32 \
  --predict "cell 22,17=b; cell 10,17=4" \
  --because "right arrow: test if cursor moves one tile clockwise"
```

A prediction that holds means the model of the game is still standing. A miss
is worth more: it dates the exact press where belief and reality came apart,
and hands over the grid that broke it.

Across the campaign the agent wrote **7,627 graded predictions**; **443
missed**. Every one of the 25 games contained at least one.

Eight claim forms, joined with `;`, each graded on its own — one wrong part is
a miss:

| Form | Meaning |
|---|---|
| `cell X,Y=V` | this cell will hold this colour |
| `move X,Y DX,DY` | this shape will shift by this much |
| `region X0:X1,Y0:Y1` | something in this box will change |
| `vanish X,Y` | this shape will be gone |
| `level+1` / `win` | this press finishes the level / the game |
| `change` / `noop` | something / nothing will change |

The grammar is deliberately small: every form describes something the next
frame can contradict.

## What the agent gets

Five kinds of instrument. None of them knows anything about any game.

| | |
|---|---|
| **See** | The board as an image, the exact grid, any crop, what changed, and the animation of any past press. |
| **Press** | One button, with a prediction attached. A press without one is refused, and the refusal is free. |
| **Plan** | A sequence where every step carries its own prediction. It halts at the first miss, so a wrong theory cannot burn the queue. |
| **Compute** | Offline Python over the grids already recorded — shape finding, lattice and line detection, motion tracing, pathfinding. It costs no actions. |
| **Model** | Optional. Write an executable model of the game, replay it against every recorded press, and A\* search it for a plan. |

## One page of notes

A long game does not fit in a context window — Claude Code compacted its own
context 115 times over this campaign. So each game keeps one short page:
what is verified, what is still assumed, what to test next. The agent rewrites
it as it plays. It is the only thing a compaction cannot take away.

Because the page cannot grow forever, observations have to become rules. The
pages stayed at a median of 60 lines, and grew their own headings — one run
kept a `REFUTED` section of eight dead beliefs next to the evidence that killed
each one.

## Tight rules, free thinking

Two rules pull against each other and both hold at once.

**The gate is hard.** A press without a falsifiable claim never reaches the
game. Not a suggestion in a prompt — the harness refuses it.

**The thinking is free.** No representation is required. The same unedited
skill produced a run that never wrote a line of Python and a run that built a
fluid simulator, searched 16 versions of the physics until one reproduced every
pour on record cell for cell, then placed 30 blocks in a single plan.

The escalation ladder the runs actually used:

1. **Sentences** — prose in the notes file. One game finished eight levels with
   one missed prediction and no Python at all.
2. **Python** — 24 of 25 games, 1,727 calls. Connected components, shortest
   paths, motion tracing. Free, because it runs on grids already recorded.
3. **A tool** — write a simulator for the one mechanic that resists.
4. **A model of the world** — the `rules` tier: an executable model of the whole
   game, fitted against every recorded press, searched with A\*. Taken once,
   and it never fitted. The game was won on the rung above.

The skill also never says when a mechanic counts as verified — only to batch
proven ones and never batch exploration. The agent drew that line itself, and
the two modes came out far apart: single test presses missed **37.1%** of the
time, planned sequences **2.9%**, and 91.6% of all presses went into plans.

## Requirements

- **Python 3.12+**, or [**uv**](https://docs.astral.sh/uv/) — the launcher
  bootstraps a private runtime when the system Python is unsuitable (first use
  only)
- Runtime dependencies, installed automatically: `arc-agi==0.9.9`,
  `numpy>=2,<3`, `pillow>=10,<13`
- **`ARC_API_KEY`** — an [ARC Prize](https://three.arcprize.org/) API key.
  Needed once per game to download it into the durable local cache, and always
  for `--mode competition`. Runs on an already-cached game need no key and no
  network.

## Install

### With `skills`

The repo is laid out for [`npx skills`](https://github.com/vercel-labs/skills),
which discovers the skill under `skills/`, asks which agents to target, and
links it into each one:

```bash
npx skills add pbshgthm/arc-skill
```

Useful flags: `-g` global instead of project-scoped, `-a claude-code` to skip
the agent prompt, `-l` to list without installing, `--copy` if symlinks aren't
available. `pnpm dlx skills add ...` works identically and keeps npm out of it.

### By hand

Skills are auto-discovered from a per-platform directory. Symlink
`skills/arc-skill/` into it — or copy, if you don't want updates from this repo.

```bash
# Claude Code — one project (use ~/.claude/skills for every project)
mkdir -p .claude/skills && ln -s "$PWD/skills/arc-skill" .claude/skills/arc-skill

# Codex
mkdir -p .agents/skills && ln -s "$PWD/skills/arc-skill" .agents/skills/arc-skill
```

**Anywhere else**: point the agent at `skills/arc-skill/SKILL.md` and tell it to
read the file completely before starting. Nothing in the skill depends on a
particular agent.

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

Doctrine and harness are co-designed: every rule in `SKILL.md` is one the
harness can enforce or grade, and every harness feature exists to make one rule
cheap to follow.

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
| `AA3_CACHE_DIR` | Optional donor cache adopted on first use |

---

By [Poobesh Gowtham](https://poobesh.com) · [@pbshgthm](https://x.com/pbshgthm).
August 2026. All 25 public games, Claude Code on Claude Opus 5, replayed and
verified by ARC on scorecard
[`24ddb219`](https://arcprize.org/scorecards/24ddb219-987e-464f-9050-6398a29cf5ac).
