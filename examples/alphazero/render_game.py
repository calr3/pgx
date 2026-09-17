# Play one game with a trained checkpoint and render it as a browsable HTML page.
#
# Writes one PNG per turn (not per ply: a Gess or Epaminondas move takes several
# actions, and only the completed position is interesting) plus an index.html
# that steps through them with the arrow keys.
#
# Usage:
#   python3 render_game.py model=../../checkpoints/<run>/000160.ckpt \
#       env_id=epaminondas out_dir=/tmp/game num_simulations=32
#
# PNG conversion needs cairosvg, which pulls in a stack that must not be
# installed next to JAX (see CLAUDE.md). Point `python_svg` at a throwaway
# virtualenv that has it:
#   python3 -m venv /tmp/svgvenv && /tmp/svgvenv/bin/pip install cairosvg
#   ... python_svg=/tmp/svgvenv/bin/python

import html
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import mctx
import pgx
from omegaconf import OmegaConf
from pydantic import BaseModel

from config import Config  # noqa: F401  (older checkpoints pickled it from __main__)
from model_tournament import load_from_checkpoint, make_recurrent_fn
from network import make_forward


class RenderConfig(BaseModel):
    env_id: pgx.EnvId = "epaminondas"
    model: str = ""
    out_dir: str = "game"
    # One engine per seat, comma separated: "model" or "negamax". The default
    # is self-play, which is what this script originally did.
    players: str = "model,model"
    # Per-action budget for a 'negamax' seat, and its depth ceiling.
    negamax_time_s: float = 0.5
    negamax_depth: int = 64
    seed: int = 0
    num_simulations: int = 32
    max_num_considered_actions: int = 16
    # 0.0 plays the search's argmax every time. A little noise gives a more
    # representative game than the single line the model always plays.
    gumbel_scale: float = 0.3
    max_turns: int = 400
    title: str = ""
    # Interpreter with cairosvg, for SVG -> PNG. Empty leaves the SVGs alone.
    python_svg: str = ""
    scale: float = 1.0

    class Config:
        extra = "forbid"


def build_choosers(cfg: RenderConfig, env: pgx.Env):
    """One action-chooser per seat, from the `players` spec.

    Each returns a batch-of-one action array, so the caller does not care which
    kind of engine is behind it.
    """
    kinds = [k.strip().lower() for k in cfg.players.split(",") if k.strip()]
    if len(kinds) != env.num_players:
        raise ValueError(
            f"players={cfg.players!r} gives {len(kinds)} seats, "
            f"but {env.id} has {env.num_players}"
        )

    choosers, names = [], []
    model_search = None
    for kind in kinds:
        if kind == "model":
            if model_search is None:
                model_search = make_model_search(cfg, env)
            choosers.append(model_search)
            names.append(f"{os.path.basename(os.path.dirname(cfg.model))} "
                         f"({cfg.num_simulations} sims)")
        elif kind in ("negamax", "alphabeta", "ab"):
            from negamax import NegamaxEngine, make_evaluator

            engine = NegamaxEngine(
                env,
                make_evaluator(env.id),
                max_depth=cfg.negamax_depth,
                time_limit_s=cfg.negamax_time_s or None,
            )

            def negamax_chooser(key, state, _engine=engine):
                del key
                unbatched = jax.tree_util.tree_map(lambda x: x[0], state)
                action, _ = _engine.select_action(unbatched)
                return jnp.int32([action])

            choosers.append(negamax_chooser)
            names.append(f"alpha-beta ({cfg.negamax_time_s}s per action)")
        else:
            raise ValueError(f"Unknown player {kind!r}; use 'model' or 'negamax'.")
    return choosers, names


def make_model_search(cfg: RenderConfig, env: pgx.Env):
    """The MCTS chooser for a 'model' seat."""
    model_cfg, model = load_from_checkpoint(cfg.model)
    forward = make_forward(env.num_actions, model_cfg)
    recurrent_fn = make_recurrent_fn(env, forward)
    params, model_state = model

    def search(key, state):
        (logits, value), _ = forward.apply(params, model_state, state.observation, is_eval=True)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        out = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=cfg.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=cfg.gumbel_scale,
            max_num_considered_actions=cfg.max_num_considered_actions,
        )
        return out.action

    return jax.jit(search)


def play(cfg: RenderConfig):
    """Play one game, returning a State per completed turn."""
    env = pgx.make(cfg.env_id)
    choosers, names = build_choosers(cfg, env)

    step = jax.jit(jax.vmap(env.step))
    key = jax.random.PRNGKey(cfg.seed)
    key, subkey = jax.random.split(key)
    state = jax.jit(jax.vmap(env.init))(jax.random.split(subkey, 1))

    # The search runs on a batch of one; frames are unbatched so each renders as
    # a single board rather than a one-cell grid.
    unbatch = lambda s: jax.tree_util.tree_map(lambda x: x[0], s)

    frames = [unbatch(state)]
    mover = int(state.current_player[0])
    for _ in range(cfg.max_turns * 8):
        if bool(state.terminated[0]):
            break
        key, key_search, key_step = jax.random.split(key, 3)
        # The seat to move picks the engine, so the two sides can differ.
        action = choosers[int(state.current_player[0])](key_search, state)
        state = step(state, action, jax.random.split(key_step, 1))
        # One frame per completed turn: current_player changing is the boundary.
        if int(state.current_player[0]) != mover or bool(state.terminated[0]):
            frames.append(unbatch(state))
            mover = int(state.current_player[0])
    return env, frames, names


def write_png(cfg: RenderConfig, svg_path: str, png_path: str) -> bool:
    if not cfg.python_svg:
        return False
    code = (
        "import cairosvg,sys; cairosvg.svg2png(url=sys.argv[1], write_to=sys.argv[2], scale=float(sys.argv[3]))"
    )
    subprocess.run([cfg.python_svg, "-c", code, svg_path, png_path, str(cfg.scale)], check=True)
    return True


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin: 0; font: 15px/1.5 system-ui, sans-serif; display: flex;
         flex-direction: column; align-items: center; gap: 12px; padding: 20px; }}
  h1 {{ font-size: 1.1rem; font-weight: 600; margin: 0; }}
  p.meta {{ margin: 0; opacity: 0.7; max-width: 46rem; text-align: center; }}
  img {{ max-width: min(90vw, 560px); image-rendering: auto; }}
  .controls {{ display: flex; align-items: center; gap: 12px; }}
  button {{ font: inherit; padding: 4px 14px; cursor: pointer; }}
  #turn {{ font-variant-numeric: tabular-nums; min-width: 12rem; text-align: center; }}
  input[type=range] {{ width: min(90vw, 560px); }}
</style>
<h1>{title}</h1>
<p class="meta">{meta}</p>
<img id="board" alt="board">
<div class="controls">
  <button id="prev">&larr;</button>
  <span id="turn"></span>
  <button id="next">&rarr;</button>
  <button id="playpause">Play</button>
</div>
<input type="range" id="scrub" min="0" max="{last}" value="0">
<script>
const frames = {frames};
const img = document.getElementById('board');
const scrub = document.getElementById('scrub');
const label = document.getElementById('turn');
let i = 0, timer = null;
function show(n) {{
  i = Math.max(0, Math.min(frames.length - 1, n));
  img.src = frames[i];
  scrub.value = i;
  label.textContent = i === 0 ? 'Start' : `Turn ${{i}} of ${{frames.length - 1}}`;
}}
document.getElementById('prev').onclick = () => show(i - 1);
document.getElementById('next').onclick = () => show(i + 1);
scrub.oninput = () => show(+scrub.value);
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft') show(i - 1);
  if (e.key === 'ArrowRight') show(i + 1);
}});
const btn = document.getElementById('playpause');
btn.onclick = () => {{
  if (timer) {{ clearInterval(timer); timer = null; btn.textContent = 'Play'; return; }}
  btn.textContent = 'Pause';
  timer = setInterval(() => {{
    if (i >= frames.length - 1) {{ clearInterval(timer); timer = null; btn.textContent = 'Play'; return; }}
    show(i + 1);
  }}, 400);
}};
show(0);
</script>
"""


def main():
    cfg = RenderConfig(**OmegaConf.from_cli())
    if not cfg.model:
        sys.exit("model=<checkpoint> is required")
    os.makedirs(os.path.join(cfg.out_dir, "frames"), exist_ok=True)

    # `names` below is the list of frame files; keep the seat labels separate.
    env, frames, player_names = play(cfg)
    print(f"{len(frames) - 1} turns")

    names = []
    for i, state in enumerate(frames):
        stem = os.path.join(cfg.out_dir, "frames", f"{i:04d}")
        state.save_svg(stem + ".svg")
        if write_png(cfg, stem + ".svg", stem + ".png"):
            os.remove(stem + ".svg")
            names.append(f"frames/{i:04d}.png")
        else:
            names.append(f"frames/{i:04d}.svg")

    final = frames[-1]
    rewards = [float(r) for r in final.rewards]
    if not bool(final.terminated):
        outcome = f"unfinished after {len(frames) - 1} turns"
    elif rewards[0] == rewards[1]:
        outcome = "drawn"
    else:
        winner = 0 if rewards[0] > rewards[1] else 1
        outcome = f"{player_names[winner]} won as player {winner}"
    seats = " vs ".join(f"player {i}: {n}" for i, n in enumerate(player_names))
    meta = f"{env.id} {env.version}, {len(frames) - 1} turns, {outcome}. {seats}."
    title = cfg.title or (
        f"{env.id} {env.version}: " + " vs ".join(dict.fromkeys(player_names))
    )

    with open(os.path.join(cfg.out_dir, "index.html"), "w") as f:
        f.write(PAGE.format(
            title=html.escape(title),
            meta=html.escape(meta),
            frames=repr(names).replace("'", '"'),
            last=len(names) - 1,
        ))
    print(f"wrote {cfg.out_dir}/index.html ({len(names)} frames)")


if __name__ == "__main__":
    main()
