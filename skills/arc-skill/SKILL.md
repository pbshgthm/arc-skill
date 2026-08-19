---
name: arc-skill
description: Solve an interactive hidden-rule ARC-AGI-3 game from its game ID. Contains the complete playing doctrine (predict before every action, graded claims, one-page notes, optional executable-rules search) plus the `arc` harness the agent drives. Use whenever asked to solve, play, or continue an ARC-AGI-3 game.
---

# Arc skill

Set the launcher once, work inside one directory per game, then start or resume:

```bash
ARC="<this-skill-directory>/scripts/arc"   # absolute path
mkdir -p <run-dir> && cd <run-dir>         # one directory = one run
"$ARC" start <GAME_ID>
```

`start` defaults to a local simulator with competition semantics: one logical
run, append-only history, paid resets, completed levels never lost. It is
crash-safe — after any interruption, rerun `"$ARC" start <GAME_ID>` and the run
is resumed or replayed exactly. Use `--mode competition` (live remote server,
~15-minute idle lease, no replay recovery) only when the user explicitly asks.
Never create a second run for the same game, never inspect the game's source or
private state, and never edit `.arc/` by hand except `NOTES.md`.

## The game

A 64×64 board of 16 colors, hidden rules, several levels. Interface priors
(not guarantees): ACTION1/2/3/4 = up/down/left/right, ACTION5 = interact,
ACTION6:x,y = click at x=column y=row, ACTION7 = undo. Availability can change
after every action.

**Finish first.** Historically every lost point came from unfinished games, not
from extra actions. Early levels are usually cheap tutorials: a wrong action
that teaches a mechanic beats a minute of deliberation. Act early, act often,
learn from every grade. Efficiency only matters once finishing is likely.

## The loop — look, predict, act, compare, note

1. **Look**: open the printed `IMAGE`; read the worded `TRANSITION` story.
2. **Predict + act**: every action requires `--predict`; the harness grades it:

```bash
"$ARC" act ACTION1 --predict "move 12,5 0,-1"
"$ARC" act ACTION6 3 14 --predict "cell 3,14=9; region 0:8,10:20" --because "test button"
```

3. **Compare**: read the ✓/✗ grade, the worded `TRANSITION` story, and the
   cell-exact `DIFF` before/after masks in the result (`arc view` re-renders
   them for any event). A ✗ is the most valuable thing that can happen —
   reality just corrected you for one action.
4. **Note**: keep `.arc/NOTES.md` to one page with three sections —
   `Verified (cite event ids)`, `Assumed / open questions`, `Plan`. After a ✗,
   fix the notes before the next action. `arc status` prints the file in full,
   so it is also your recovery story after any context loss.

Claim vocabulary (full reference: `"$ARC" act --help`): `noop`, `change`,
`cell X,Y=V`, `move X,Y DX,DY`, `vanish X,Y`, `region X0:X1,Y0:Y1`, `level+1`,
`win`; several separated by `;`. Coordinates are x=column, y=row, like ACTION6.
Free text is allowed and merely claims "something changes" — prefer one
specific claim; it grades sharper and teaches more.

## Batching proven mechanics

Once a mechanic is verified, stop paying one command per step — batch with a
claim on every step; execution halts at the first miss so a wrong theory cannot
burn the rest of the queue:

```bash
"$ARC" commit \
  --step "ACTION4 :: move 12,5 1,0" \
  --step "ACTION4 :: move 13,5 1,0" \
  --step "ACTION1 :: level+1"
```

Batch only movement you can predict cell-exactly or level-exactly; never batch
exploration.

## Levels, consumables, reset

- Completing a level archives your notes to `.arc/levels/`. The new level may
  reuse mechanics — treat every earlier `Verified` claim as `Assumed` until it
  survives one test on the new board (status reminds you until the notes change).
- An object that vanished and never came back is a consumable. Spend
  consumables last, after reversible probes; before an irreversible-looking
  action, prefer ACTION7 (undo) tests when available.
- `"$ARC" reset --because "<why this board is unrecoverable>"` rewinds only the
  current level, for the price of one action. After `GAME_OVER` the reason may
  be omitted. Completed levels and history are never lost.

## When a level resists — the rules tier (optional)

Status nudges you after many actions or repeated misses on one level. Then, and
only then, escalate from prose to executable rules: write a plain `rules.py`
(grounding, step, actions, goal), verify it against the entire recorded
history, and let A* search find the plan. Each plan step carries its own
prediction, so live execution still halts on the first surprise.

```bash
"$ARC" rules help      # the compact contract
"$ARC" rules init      # template; then edit rules.py
"$ARC" rules replay    # must fit or gap on every recorded transition
"$ARC" rules solve     # writes .arc/plan.json
"$ARC" commit @.arc/plan.json
```

Model only verified mechanics; mark everything else `Unknown("why")` — replay
reports gaps honestly instead of pretending a fit. This tier is never required
and never worth it before the game has taught you its mechanics.

## Evidence tools

```bash
"$ARC" status                       # full picture + notes; run after context loss
"$ARC" view --grid                  # exact 0-f pixels
"$ARC" view --crop 8:24,10:30       # exact half-open crop
"$ARC" view --event 12 --frames     # animation frames of a past action
"$ARC" python 'connected_components(grid)'
"$ARC" python 'shortest_path((r0,c0),(r1,c1), passable_mask)'
```

`arc python` preloads every settled board (`grid`, `previous`, `frames`,
`transitions`, `actions`), NumPy, perception helpers, BFS, and A* — free
offline computation; use it for parsing and pathfinding instead of paid probing.

## Install location

This folder is agent-agnostic. Copy it into the platform's auto-discovered
skill directory (Claude Code: `.claude/skills/arc-skill/`, Codex:
`.agents/skills/arc-skill/`). From an unrecognized location, instruct the
agent to read this `SKILL.md` completely before starting.
