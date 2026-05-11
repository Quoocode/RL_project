import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import run_first_fit, run_random


def test_first_fit_better_than_random():
    # Small smoke test: first-fit should not be worse than random on average
    num_nodes = 5
    num_services = 5
    n_episodes = 20
    seed = 0

    random_stats = run_random(num_nodes, num_services, n_episodes, seed)
    ff_stats = run_first_fit(num_nodes, num_services, n_episodes, seed)

    assert ff_stats["mean"] >= random_stats["mean"] - 1e-6, (
        f"First-Fit mean ({ff_stats['mean']}) < Random mean ({random_stats['mean']})"
    )


if __name__ == '__main__':
    test_first_fit_better_than_random()
    print('First-Fit unit test: PASS')
