"""Diagnostics for placement policy comparison and reward breakdown.

Usage:
    python diagnostics.py --seed 0 --nodes 5 --services 5 --episode 0
    python diagnostics.py --seed 0 --nodes 5 --services 5 --compare
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.k8s_env import K8sPlacementEnv
from evaluate import run_first_fit, run_ai_agent


@dataclass
class StepTrace:
    step: int
    service_id: int
    service_name: str
    node_id: int
    success: bool
    reward: float
    success_term: float
    overload_term: float
    latency_term: float
    node_cpu_util: float
    node_mem_util: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run placement diagnostics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=5)
    parser.add_argument("--services", type=int, default=5)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--compare", action="store_true")
    return parser.parse_args()


def make_env(num_nodes: int, num_services: int, seed: int) -> K8sPlacementEnv:
    env = K8sPlacementEnv(num_nodes=num_nodes, num_services=num_services)
    env.reset(seed=seed)
    return env


def first_fit_action(env: K8sPlacementEnv) -> int:
    service = env.service_chain.services[env.current_service_idx]
    nodes = env.topology.nodes

    for i, node in enumerate(nodes):
        if not node.is_active:
            continue
        if node.can_allocate(service.cpu_request, service.memory_request):
            exp_cpu_util = (node.cpu_used + service.cpu_request) / node.cpu_capacity
            exp_mem_util = (node.memory_used + service.memory_request) / node.memory_capacity
            if exp_cpu_util <= 0.8 and exp_mem_util <= 0.8:
                return i

    for i, node in enumerate(nodes):
        if node.is_active and node.can_allocate(service.cpu_request, service.memory_request):
            return i

    best_node = 0
    best_util = float("inf")
    for i, node in enumerate(nodes):
        if not node.is_active:
            continue
        util = (node.cpu_used / node.cpu_capacity + node.memory_used / node.memory_capacity) / 2
        if util < best_util:
            best_util = util
            best_node = i

    return best_node


def reward_breakdown(env: K8sPlacementEnv, service, node_id: int, success: bool) -> Tuple[float, float, float, float]:
    node = env.topology.get_node(node_id)

    if node is None or not node.is_active:
        return -50.0, 0.0, 0.0, 0.0
    if not success:
        return -20.0, 0.0, 0.0, 0.0

    success_term = 5.0
    cpu_util = node.cpu_used / node.cpu_capacity
    mem_util = node.memory_used / node.memory_capacity

    overload_term = 0.0
    if cpu_util > 0.8:
        overload_term -= ((cpu_util - 0.8) * 50) ** 2
    if mem_util > 0.8:
        overload_term -= ((mem_util - 0.8) * 50) ** 2

    latency_term = 0.0
    if service.id > 0:
        prev_service = env.service_chain.services[service.id - 1]
        if prev_service.placed_on >= 0:
            latency = env.topology.get_latency(prev_service.placed_on, node_id)
            latency_term -= min(latency * 0.5, 4.0)

    total = success_term + overload_term + latency_term
    return total, success_term, overload_term, latency_term


def trace_policy(
    name: str,
    env: K8sPlacementEnv,
    action_fn: Callable[[K8sPlacementEnv], int],
    seed: int,
) -> List[StepTrace]:
    obs, _ = env.reset(seed=int(seed))
    traces: List[StepTrace] = []

    done = False
    step = 0
    while not done:
        service = env.service_chain.services[env.current_service_idx]
        action = action_fn(env)
        obs, reward, terminated, truncated, info = env.step(action)
        _, success_term, overload_term, latency_term = reward_breakdown(
            env, service, action, info["success"]
        )
        node = env.topology.get_node(action)
        traces.append(
            StepTrace(
                step=step,
                service_id=service.id,
                service_name=service.name,
                node_id=action,
                success=info["success"],
                reward=reward,
                success_term=success_term,
                overload_term=overload_term,
                latency_term=latency_term,
                node_cpu_util=node.cpu_used / node.cpu_capacity,
                node_mem_util=node.memory_used / node.memory_capacity,
            )
        )
        done = terminated or truncated
        step += 1

    print(f"\n=== {name} TRACE ===")
    for row in traces:
        status = "OK" if row.success else "FAIL"
        print(
            f"Step {row.step:2d} | svc {row.service_id}:{row.service_name:<12} | "
            f"node {row.node_id} | {status} | reward {row.reward:+6.2f} | "
            f"succ {row.success_term:+5.1f} | over {row.overload_term:+6.1f} | "
            f"lat {row.latency_term:+4.1f} | util cpu/mem {row.node_cpu_util:.2f}/{row.node_mem_util:.2f}"
        )

    placed = sum(1 for svc in env.service_chain.services if svc.placed_on >= 0)
    total_reward = float(sum(row.reward for row in traces))
    print(f"Placed: {placed}/{env.num_services} | Total reward: {total_reward:.2f}")
    print(f"Placements: {[svc.placed_on for svc in env.service_chain.services]}")
    return traces


def load_agent_policy(agent_name: str, num_nodes: int, num_services: int, seed: int):
    model_path = os.path.join("models", f"{agent_name}_model.zip")
    if not os.path.exists(model_path):
        return None

    def action_fn(env: K8sPlacementEnv) -> int:
        stats = run_ai_agent(agent_name, model_path, num_nodes, num_services, 1, seed)
        raise RuntimeError(
            "run_ai_agent is summary-only; use compare mode with evaluate.py for aggregate metrics"
        )

    return action_fn


def compare_episodes(num_nodes: int, num_services: int, seed: int):
    print("\n=== Aggregate comparison (100 episodes) ===")
    print("Running First-Fit ...")
    ff = run_first_fit(num_nodes, num_services, 100, seed)
    print(f"First-Fit: mean={ff['mean']:.2f}, std={ff['std']:.2f}, placed={ff['placed_mean']*100:.1f}%")

    for agent_name in ["ppo", "dqn", "a2c"]:
        model_path = os.path.join("models", f"{agent_name}_model.zip")
        if not os.path.exists(model_path):
            print(f"{agent_name.upper()}: skipped (missing {model_path})")
            continue
        print(f"Running {agent_name.upper()} ...")
        stats = run_ai_agent(agent_name, model_path, num_nodes, num_services, 100, seed)
        print(
            f"{agent_name.upper()}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
            f"placed={stats['placed_mean']*100:.1f}%"
        )


def main() -> None:
    args = parse_args()
    env = make_env(args.nodes, args.services, args.seed)

    trace_policy("First-Fit", env, first_fit_action, args.seed + args.episode)

    if args.compare:
        compare_episodes(args.nodes, args.services, args.seed)


if __name__ == "__main__":
    main()