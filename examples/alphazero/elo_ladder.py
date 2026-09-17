# Round-robin Elo ladder over AlphaZero checkpoints.
#
# Plays every pair of checkpoints with model_tournament.py's batched MCTS round
# runner and fits one Elo rating per checkpoint by maximum likelihood
# (Bradley-Terry with draws counted as half a win). Pair results are cached in a
# JSON file keyed by the two checkpoints and the tournament settings, so adding
# a checkpoint later only plays its new pairs.
#
# Usage:
#   python examples/alphazero/elo_ladder.py env_id=gess \
#     models=gf=checkpoints/a/000060.ckpt,rf=checkpoints/b/000060.ckpt,base=checkpoints/c/000125.ckpt \
#     games_per_pair=256 num_simulations=32 results_file=elo_gess.json
#
# Each model is "name=path" or just "path" (named by its path). The first model
# is anchored at Elo 0. Mirror games share openings, so the reported standard
# errors (which assume independent games) are optimistic.

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pgx
from omegaconf import OmegaConf
from pydantic import BaseModel

# Older checkpoints pickled Config from __main__; importing it here lets them load.
from config import Config  # noqa: F401
from model_tournament import TourneyConfig, build_round_runner, load_from_checkpoint
from network import make_forward


class LadderConfig(BaseModel):
    env_id: pgx.EnvId = "gess"
    seed: int = 49064405
    # Comma-separated "name=path" (or "path") entries; the first is anchored at 0.
    models: str = ""
    # Games per pair; must be a multiple of batch_size.
    games_per_pair: int = 256
    batch_size: int = 128
    num_simulations: int = 32
    max_num_considered_actions: int = 16
    gumbel_scale: float = 0.0
    random_opening_plies: int = 2
    max_num_steps: int = 256
    # JSON cache of pair results; empty = no cache.
    results_file: str = "elo_results.json"

    class Config:
        extra = "forbid"


def parse_models(spec: str) -> list[tuple[str, str]]:
    models = []
    for entry in (e.strip() for e in spec.split(",")):
        if not entry:
            continue
        name, sep, path = entry.partition("=")
        if not sep:
            name, path = entry, entry
        models.append((name.strip(), path.strip()))
    names = [n for n, _ in models]
    if len(models) < 2 or len(set(names)) != len(names):
        raise ValueError(f"models needs at least two entries with distinct names, got {names}")
    return models


def fit_elo(
    names: list[str], wins: dict[tuple[str, str], float], games: dict[tuple[str, str], int]
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood Elo ratings with the first model anchored at 0.

    wins[(a, b)] is a's score (wins + draws/2) over games[(a, b)] games against
    b, for a before b in `names`. One virtual drawn game is added per played
    pair so that perfect scores keep finite ratings. Returns (elo, stderr).
    """
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}
    W = np.zeros((n, n))  # W[i, j]: i's score against j
    N = np.zeros((n, n))
    for (a, b), g in games.items():
        i, j = idx[a], idx[b]
        s = wins[(a, b)] + 0.5
        W[i, j] += s
        W[j, i] += g + 1 - s
        N[i, j] += g + 1
        N[j, i] += g + 1

    scale = np.log(10.0) / 400.0  # Elo -> natural log-odds units
    theta = np.zeros(n)
    free = np.arange(1, n)
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(theta[None, :] - theta[:, None]))  # p[i, j] = P(i beats j)
        grad = (W - N * p).sum(axis=1)
        info = N * p * (1.0 - p)
        hess = np.diag(info.sum(axis=1)) - info  # negative Hessian
        step = np.linalg.solve(hess[np.ix_(free, free)], grad[free])
        theta[free] += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1.0 / (1.0 + np.exp(theta[None, :] - theta[:, None]))
    info = N * p * (1.0 - p)
    hess = np.diag(info.sum(axis=1)) - info
    stderr = np.zeros(n)
    stderr[free] = np.sqrt(np.diag(np.linalg.inv(hess[np.ix_(free, free)])))
    return theta / scale, stderr / scale


def play_pair(env, lcfg: LadderConfig, path_a: str, path_b: str, num_devices: int) -> tuple[float, int]:
    """Play games_per_pair seat-balanced games; return (A's score, games)."""
    tcfg = TourneyConfig(
        env_id=lcfg.env_id,
        seed=lcfg.seed,
        models=f"{path_a},{path_b}",
        games=lcfg.games_per_pair,
        batch_size=lcfg.batch_size,
        num_simulations=lcfg.num_simulations,
        max_num_considered_actions=lcfg.max_num_considered_actions,
        gumbel_scale=lcfg.gumbel_scale,
        random_opening_plies=lcfg.random_opening_plies,
        max_num_steps=lcfg.max_num_steps,
    )
    config_a, model_a = load_from_checkpoint(path_a)
    config_b, model_b = load_from_checkpoint(path_b)
    forward_a = make_forward(env.num_actions, config_a)
    forward_b = make_forward(env.num_actions, config_b)
    model_a, model_b = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (num_devices, *x.shape)), (model_a, model_b)
    )
    run_round = build_round_runner(env, forward_a, forward_b, tcfg, num_devices)

    rng_key = jax.random.PRNGKey(lcfg.seed)
    score = 0.0
    for _ in range(lcfg.games_per_pair // lcfg.batch_size):
        rng_key, subkey = jax.random.split(rng_key)
        result = run_round(model_a, model_b, jax.random.split(subkey, num_devices))
        score += float(((np.asarray(result.r) + 1.0) / 2.0).sum())
    return score, lcfg.games_per_pair


def main() -> None:
    lcfg = LadderConfig(**OmegaConf.from_cli())
    models = parse_models(lcfg.models)
    num_devices = len(jax.local_devices())
    if lcfg.batch_size % num_devices or (lcfg.batch_size // num_devices) % 2:
        raise ValueError(
            f"batch_size ({lcfg.batch_size}) must split evenly across {num_devices} devices "
            "with an even per-device share."
        )
    if lcfg.games_per_pair % lcfg.batch_size:
        raise ValueError(f"games_per_pair must be a multiple of batch_size ({lcfg.batch_size}).")

    # Cache entries are keyed by both checkpoint paths and every setting that
    # affects the games.
    settings = lcfg.model_dump(exclude={"models", "results_file"})
    # Bump when game generation changes, so older cached results are replayed.
    # 2: random openings avoid multi-stage choices whose every follow-up ends the
    #    game (previously ~13% of Gess games ended in the opening).
    # 3: Gess v1 rules - a captureless stalemate is decided on stone count.
    # 4: Epaminondas v1 rules - reaching the move cap is decided on piece count.
    # 5: Epaminondas v2 rules - that cap counts moves since the last capture.
    settings["game_version"] = 5
    cache: dict = {}
    if lcfg.results_file and os.path.exists(lcfg.results_file):
        with open(lcfg.results_file) as f:
            cache = json.load(f)

    def cache_key(path_a: str, path_b: str) -> str:
        return json.dumps(
            [os.path.abspath(path_a), os.path.abspath(path_b), settings], sort_keys=True
        )

    env = pgx.make(lcfg.env_id)
    names = [n for n, _ in models]
    wins: dict[tuple[str, str], float] = {}
    games: dict[tuple[str, str], int] = {}
    pairs = [(i, j) for i in range(len(models)) for j in range(i + 1, len(models))]
    for k, (i, j) in enumerate(pairs):
        (name_a, path_a), (name_b, path_b) = models[i], models[j]
        key = cache_key(path_a, path_b)
        if key in cache:
            score, n = cache[key]["score"], cache[key]["games"]
            print(f"[{k + 1}/{len(pairs)}] {name_a} vs {name_b}: cached {score:.1f}/{n}")
        else:
            st = time.perf_counter()
            score, n = play_pair(env, lcfg, path_a, path_b, num_devices)
            print(
                f"[{k + 1}/{len(pairs)}] {name_a} vs {name_b}: {score:.1f}/{n} "
                f"({time.perf_counter() - st:.0f}s)",
                flush=True,
            )
            cache[key] = {"score": score, "games": n}
            if lcfg.results_file:
                tmp = lcfg.results_file + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(cache, f, indent=1)
                os.replace(tmp, lcfg.results_file)
        wins[(name_a, name_b)] = score
        games[(name_a, name_b)] = n

    elo, stderr = fit_elo(names, wins, games)
    width = max(len(n) for n in names)
    print(f"\nElo (anchor {names[0]} = 0; errors assume independent games):")
    for i in np.argsort(-elo):
        played = sum(g for (a, b), g in games.items() if names[i] in (a, b))
        print(f"  {names[i]:<{width}}  {elo[i]:+7.0f} ± {stderr[i]:3.0f}   ({played} games)")


if __name__ == "__main__":
    main()
