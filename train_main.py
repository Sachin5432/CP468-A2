import csv
import datetime
import itertools
import os

from sim.agent import Agent
from sim.world import World


OUTPUT_DIR = "results"
MAX_EPISODE_STEPS = 200
RUNS_PER_CONFIG = 10

# Shared sim settings
START_NODE = 0
DEST_NODE = 15
MAX_WALKING_DIST = 3.0
BASE_SEED = 42

# Sweep 1
ALPHAS = [0.05, 0.1, 0.2]
GAMMAS = [0.8, 0.9, 0.95]
Q_ITERATIONS = [1, 5, 10]


FIXED_EPSILON = 0.9
FIXED_EPSILON_MIN = 0.05
FIXED_EPSILON_DECAY = 0.995

# Sweep 2: 
EPSILONS = [0.6, 0.8, 0.9]
EPSILON_MINS = [0.01, 0.05, 0.1]
EPSILON_DECAYS = [0.99, 0.995, 0.999]


FIXED_ALPHA = 0.1
FIXED_GAMMA = 0.9
FIXED_Q_ITERATIONS = 5


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_single_episode(
    *,
    alpha: float,
    gamma: float,
    q_iterations: int,
    epsilon: float,
    epsilon_min: float,
    epsilon_decay: float,
    seed: int,
) -> dict:
    world = World(seed=seed)
    agent = Agent(
        start=START_NODE,
        dest=DEST_NODE,
        max_walking_dist=MAX_WALKING_DIST,
        alpha=alpha,
        gamma=gamma,
        q_iterations=q_iterations,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    finished = False
    for _ in range(MAX_EPISODE_STEPS):
        finished = agent.act_q_learning(world)
        if finished or agent.finished():
            finished = True
            break
        world.update()

    steps_taken = agent.get_steps_taken()

    return {
        "seed": seed,
        "alpha": alpha,
        "gamma": gamma,
        "q_iterations": q_iterations,
        "epsilon": epsilon,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "steps_taken": steps_taken,
        "finished": finished,
    }


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_hyperparameter_sweep() -> list[dict]:
    rows = []

    for combo_idx, (alpha, gamma, q_iterations) in enumerate(
        itertools.product(ALPHAS, GAMMAS, Q_ITERATIONS),
        start=1,
    ):
        print(
            f"[{datetime.datetime.now().astimezone()}] "
            f"Hyper sweep config {combo_idx}: "
            f"alpha={alpha}, gamma={gamma}, q_iterations={q_iterations}"
        )

        for run_idx in range(RUNS_PER_CONFIG):
            seed = BASE_SEED + combo_idx * 1000 + run_idx
            result = run_single_episode(
                alpha=alpha,
                gamma=gamma,
                q_iterations=q_iterations,
                epsilon=FIXED_EPSILON,
                epsilon_min=FIXED_EPSILON_MIN,
                epsilon_decay=FIXED_EPSILON_DECAY,
                seed=seed,
            )
            result["sweep_type"] = "alpha_gamma_q_iterations"
            result["run_idx"] = run_idx
            rows.append(result)

    return rows


def run_epsilon_sweep() -> list[dict]:
    rows = []

    for combo_idx, (epsilon, epsilon_min, epsilon_decay) in enumerate(
        itertools.product(EPSILONS, EPSILON_MINS, EPSILON_DECAYS),
        start=1,
    ):
        print(
            f"[{datetime.datetime.now().astimezone()}] "
            f"Epsilon sweep config {combo_idx}: "
            f"epsilon={epsilon}, epsilon_min={epsilon_min}, epsilon_decay={epsilon_decay}"
        )

        for run_idx in range(RUNS_PER_CONFIG):
            seed = BASE_SEED + 100000 + combo_idx * 1000 + run_idx
            result = run_single_episode(
                alpha=FIXED_ALPHA,
                gamma=FIXED_GAMMA,
                q_iterations=FIXED_Q_ITERATIONS,
                epsilon=epsilon,
                epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
                seed=seed,
            )
            result["sweep_type"] = "epsilon_epsilon_min_epsilon_decay"
            result["run_idx"] = run_idx
            rows.append(result)

    return rows


if __name__ == "__main__":
    ensure_output_dir()

    hyper_rows = run_hyperparameter_sweep()
    epsilon_rows = run_epsilon_sweep()

    write_csv(os.path.join(OUTPUT_DIR, "grid_search_alpha_gamma_q_iterations.csv"), hyper_rows)
    write_csv(os.path.join(OUTPUT_DIR, "grid_search_epsilon_params.csv"), epsilon_rows)

    print(f"[{datetime.datetime.now().astimezone()}] Done. CSV files written to '{OUTPUT_DIR}/'")