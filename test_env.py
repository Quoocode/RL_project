# test_env.py
"""
Kiểm tra sâu môi trường K8sPlacementEnv trước khi đưa vào training.
Chạy: python test_env.py

Nếu tất cả PASS → môi trường an toàn để train.
Nếu có FAIL     → đọc thông báo lỗi, fix trước khi train.
"""

import sys
import os
import traceback
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from envs.k8s_env import K8sPlacementEnv, K8sPlacementEnvDynamic

# ═════════════════════════════════════════════════════════════════════════════
# HELPER
# ═════════════════════════════════════════════════════════════════════════════

PASS  = "✅ PASS"
FAIL  = "❌ FAIL"
total_pass = 0
total_fail = 0
failed_tests = []

def run_test(name: str, fn):
    global total_pass, total_fail
    print(f"\n  [{name}]", end=" ")
    try:
        fn()
        print(PASS)
        total_pass += 1
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message)"
        print(f"{FAIL} — {msg}")
        total_fail += 1
        failed_tests.append((name, msg))
    except Exception as e:
        print(f"{FAIL} — Exception: {type(e).__name__}: {e}")
        total_fail += 1
        failed_tests.append((name, traceback.format_exc()))

def section(title: str):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

def make_env(seed=42, **kwargs) -> K8sPlacementEnv:
    env = K8sPlacementEnv(**kwargs)
    env.reset(seed=seed)
    return env

def make_dynamic_env(seed=42, **kwargs) -> K8sPlacementEnvDynamic:
    env = K8sPlacementEnvDynamic(**kwargs)
    env.reset(seed=seed)
    return env

def expected_obs_dim(num_nodes: int, num_services: int) -> int:
    # Env v2: mỗi node có 4 feature:
    # cpu_util, mem_util, cpu_capacity_norm, mem_capacity_norm
    return num_nodes * 4 + 2 + num_services


# ═════════════════════════════════════════════════════════════════════════════
# NHÓM 1 — CẤU TRÚC SPACE
# ═════════════════════════════════════════════════════════════════════════════
section("NHÓM 1 — Cấu trúc Observation & Action Space")

def t_obs_shape():
    env = make_env(num_nodes=5, num_services=5)
    obs, _ = env.reset()
    expected = expected_obs_dim(5, 5)  # 4 features/node + service cpu/mem + onehot services
    assert obs.shape == (expected,), \
        f"Shape sai: got {obs.shape}, expected ({expected},)"

def t_obs_dtype():
    env = make_env(num_nodes=5, num_services=5)
    obs, _ = env.reset()
    assert obs.dtype == np.float32, \
        f"dtype sai: got {obs.dtype}, expected float32"

def t_obs_bounds_reset():
    env = make_env(num_nodes=5, num_services=5)
    obs, _ = env.reset()
    assert obs.min() >= 0.0 and obs.max() <= 1.0, \
        f"Obs ngoài [0,1] sau reset: min={obs.min():.4f}, max={obs.max():.4f}"

def t_action_space():
    env = make_env(num_nodes=5, num_services=5)
    n = env.action_space.n
    assert n == 5, f"Action space size sai: got {n}, expected 5"

def t_custom_size():
    """Kiểm tra config khác mặc định."""
    env = K8sPlacementEnv(num_nodes=3, num_services=4)
    obs, _ = env.reset()
    expected = expected_obs_dim(3, 4)
    assert obs.shape == (expected,), \
        f"Shape sai với custom config: got {obs.shape}, expected ({expected},)"
    assert env.action_space.n == 3

run_test("obs_shape",       t_obs_shape)
run_test("obs_dtype",       t_obs_dtype)
run_test("obs_bounds_reset",t_obs_bounds_reset)
run_test("action_space_n",  t_action_space)
run_test("custom_size",     t_custom_size)


# ═════════════════════════════════════════════════════════════════════════════
# NHÓM 2 — OBS KHÔNG CÓ NaN / Inf (nguy hiểm nhất khi training)
# ═════════════════════════════════════════════════════════════════════════════
section("NHÓM 2 — Observation không có NaN / Inf")

def _check_obs_clean(obs, label=""):
    assert not np.any(np.isnan(obs)), \
        f"NaN trong obs{' (' + label + ')' if label else ''}: {obs}"
    assert not np.any(np.isinf(obs)), \
        f"Inf trong obs{' (' + label + ')' if label else ''}: {obs}"
    assert obs.min() >= 0.0 and obs.max() <= 1.0, \
        f"Obs ngoài [0,1]{' (' + label + ')' if label else ''}: " \
        f"min={obs.min():.4f}, max={obs.max():.4f}"

def t_obs_clean_full_episode():
    env = K8sPlacementEnv(num_nodes=5, num_services=5)
    obs, _ = env.reset(seed=0)
    _check_obs_clean(obs, "reset")
    done = False
    step = 0
    while not done:
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
        _check_obs_clean(obs, f"step {step}")

def t_obs_clean_with_dead_node():
    env = K8sPlacementEnv(num_nodes=5, num_services=5)
    obs, _ = env.reset(seed=1)
    env.topology.nodes[0].fail()
    env.topology.nodes[4].fail()
    done = False
    step = 0
    while not done:
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
        _check_obs_clean(obs, f"step {step} (2 dead nodes)")

def t_obs_clean_all_dead():
    """Tất cả node chết — obs phải vẫn hợp lệ (đầy 1.0)."""
    env = K8sPlacementEnv(num_nodes=5, num_services=5)
    obs, _ = env.reset(seed=2)
    for node in env.topology.nodes:
        node.fail()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        _check_obs_clean(obs, "all dead nodes")

run_test("obs_clean_full_episode", t_obs_clean_full_episode)
run_test("obs_clean_dead_nodes",   t_obs_clean_with_dead_node)
run_test("obs_clean_all_dead",     t_obs_clean_all_dead)


# ═════════════════════════════════════════════════════════════════════════════
# NHÓM 3 — REWARD KHÔNG CÓ NaN / Inf
# ═════════════════════════════════════════════════════════════════════════════
section("NHÓM 3 — Reward không có NaN / Inf")

def _check_reward_clean(reward, label=""):
    assert not np.isnan(reward), \
        f"Reward là NaN{' (' + label + ')' if label else ''}"
    assert not np.isinf(reward), \
        f"Reward là Inf{' (' + label + ')' if label else ''}"

def t_reward_clean_random():
    env = K8sPlacementEnv(num_nodes=5, num_services=5)
    for ep in range(50):
        env.reset(seed=ep)
        done = False
        step = 0
        while not done:
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1
            _check_reward_clean(reward, f"ep {ep} step {step}")

def t_reward_always_fail():
    """Agent luôn chọn node chết — reward phải là -50 không đổi."""
    env = K8sPlacementEnv(num_nodes=5, num_services=5)
    env.reset(seed=0)
    env.topology.nodes[3].fail()
    done = False
    while not done:
        _, reward, terminated, truncated, info = env.step(3)
        done = terminated or truncated
        _check_reward_clean(reward, "always dead node")
        assert reward == -50.0, \
            f"Reward sai khi chọn node chết: got {reward}, expected -50.0"

def t_reward_always_full():
    """Agent xếp vào node đầy — reward phải là -20."""
    env = K8sPlacementEnv(num_nodes=5, num_services=5)
    env.reset(seed=0)
    # Làm đầy node 0 thủ công
    n = env.topology.nodes[0]
    n.cpu_used    = n.cpu_capacity
    n.memory_used = n.memory_capacity
    _, reward, _, _, _ = env.step(0)
    _check_reward_clean(reward, "full node")
    assert reward == -20.0, \
        f"Reward sai khi node đầy: got {reward}, expected -20.0"

run_test("reward_clean_random",  t_reward_clean_random)
run_test("reward_always_fail",   t_reward_always_fail)
run_test("reward_always_full",   t_reward_always_full)


# ═════════════════════════════════════════════════════════════════════════════
# NHÓM 4 — TERMINATION & TRUNCATION
# ═════════════════════════════════════════════════════════════════════════════
section("NHÓM 4 — Termination & Truncation đúng logic")

def t_terminated_exactly_at_num_services():
    env = K8sPlacementEnv(num_nodes=5, num_services=5)
    env.reset(seed=0)
    terminated = False
    steps = 0
    while not terminated:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        assert not truncated, "truncated = True trước khi terminated — sai"
        assert steps <= env.num_services, \
            f"Episode chạy quá {env.num_services} bước mà chưa terminated"
    assert steps == env.num_services, \
        f"Terminated sau {steps} bước, expected {env.num_services}"

def t_truncated_triggers():
    """max_steps=3 với 5 services — phải truncated trước khi terminated."""
    env = K8sPlacementEnv(num_nodes=5, num_services=5, max_steps=3)
    env.reset(seed=0)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
    assert truncated and not terminated, \
        f"Kỳ vọng truncated=True, terminated=False, nhưng terminated={terminated}, truncated={truncated}"
    assert steps == 3, f"Truncated sau {steps} bước, expected 3"

def t_never_hangs():
    """Episode không được chạy quá max_steps dù agent tệ đến đâu."""
    env = K8sPlacementEnv(num_nodes=5, num_services=5, max_steps=20)
    for ep in range(20):
        env.reset(seed=ep)
        # Đánh sập tất cả node — agent không thể đặt được gì
        for node in env.topology.nodes:
            node.fail()
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1
            assert steps <= env.max_steps + 1, \
                f"Episode treo! Đã vượt max_steps ({env.max_steps})"

run_test("terminated_exactly",  t_terminated_exactly_at_num_services)
run_test("truncated_triggers",  t_truncated_triggers)
run_test("never_hangs",         t_never_hangs)


# ═════════════════════════════════════════════════════════════════════════════
# NHÓM 5 — SEED & REPRODUCIBILITY
# ═════════════════════════════════════════════════════════════════════════════
section("NHÓM 5 — Seed & Reproducibility")

def _run_episode_with_seed(seed: int) -> list:
    """Chạy 1 episode với seed cố định, trả về list reward."""
    # 1. Seed np.random TRƯỚC khi tạo env — topology dùng np.random trong __init__
    np.random.seed(seed)
    env = K8sPlacementEnv(num_nodes=5, num_services=5)
    # 2. Seed env (seeds env.np_random)
    env.reset(seed=seed)
    # 3. Seed action_space TƯỜNG MINH — gymnasium không tự đồng bộ
    #    action_space.np_random với env.np_random trong mọi version
    env.action_space.seed(seed)
    rewards = []
    done = False
    while not done:
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return rewards

def t_same_seed_same_result():
    r1 = _run_episode_with_seed(123)
    r2 = _run_episode_with_seed(123)
    assert r1 == r2, \
        f"Cùng seed nhưng reward khác nhau!\nr1={r1}\nr2={r2}"

def t_diff_seed_diff_result():
    r1 = _run_episode_with_seed(1)
    r2 = _run_episode_with_seed(2)
    # Không nhất thiết khác nhau 100% nhưng tổng reward phải khác
    assert sum(r1) != sum(r2), \
        "Hai seed khác nhau cho cùng tổng reward — có thể seed không hoạt động"

def t_reset_clears_state():
    """Sau reset, mọi service phải về placed_on = -1 và step_count = 0."""
    env = K8sPlacementEnv(num_nodes=5, num_services=5)
    env.reset(seed=0)
    # Chạy một ít rồi reset
    for _ in range(3):
        env.step(env.action_space.sample())
    env.reset(seed=99)
    for svc in env.service_chain.services:
        assert svc.placed_on == -1, \
            f"Service {svc.id} chưa được reset: placed_on={svc.placed_on}"
    assert env.current_service_idx == 0, \
        f"current_service_idx chưa reset: {env.current_service_idx}"
    assert env.step_count == 0, \
        f"step_count chưa reset: {env.step_count}"

run_test("same_seed_same_result", t_same_seed_same_result)
run_test("diff_seed_diff_result", t_diff_seed_diff_result)
run_test("reset_clears_state",    t_reset_clears_state)


# ═════════════════════════════════════════════════════════════════════════════
# NHÓM DYNAMIC — Tests cho K8sPlacementEnvDynamic
# ═════════════════════════════════════════════════════════════════════════════
section("NHÓM DYNAMIC — Observation shape và Invalid Actions")

def t_dynamic_obs_shape_consistent():
    e1 = make_dynamic_env(actual_num_nodes=5, actual_num_services=5)
    e2 = make_dynamic_env(actual_num_nodes=5, actual_num_services=8)
    e3 = make_dynamic_env(actual_num_nodes=8, actual_num_services=10)

    o1, _ = e1.reset()
    o2, _ = e2.reset()
    o3, _ = e3.reset()

    assert o1.shape == o2.shape == o3.shape, \
        f"Dynamic obs shape không giống nhau: {o1.shape}, {o2.shape}, {o3.shape}"

    # đồng thời so sánh với observation_space
    assert o1.shape == e1.observation_space.shape, \
        f"Obs shape không trùng với observation_space: {o1.shape} vs {e1.observation_space.shape}"

def t_dynamic_invalid_action_padding_node():
    env = make_dynamic_env(actual_num_nodes=5, actual_num_services=5)
    _, _ = env.reset()
    # chọn node padding (>=5), ví dụ 7
    obs, reward, terminated, truncated, info = env.step(7)
    assert info.get("invalid_action", False) is True, "invalid_action phải True khi chọn padding node"
    assert info.get("invalid_reason") == "padding_node", \
        f"invalid_reason sai: got {info.get('invalid_reason')}"
    assert reward == -50.0, f"Reward sai cho padding_node: got {reward}, expected -50.0"

def t_dynamic_invalid_action_inactive_node():
    env = make_dynamic_env(actual_num_nodes=5, actual_num_services=5)
    env.reset()
    # đánh sập node 2
    env.topology.nodes[2].fail()
    obs, reward, terminated, truncated, info = env.step(2)
    assert info.get("invalid_action", False) is True
    assert info.get("invalid_reason") == "inactive_node"
    assert reward == -50.0

run_test("dynamic_obs_shape_consistent", t_dynamic_obs_shape_consistent)
run_test("dynamic_invalid_padding_node", t_dynamic_invalid_action_padding_node)
run_test("dynamic_invalid_inactive_node", t_dynamic_invalid_action_inactive_node)


# ═════════════════════════════════════════════════════════════════════════════
# NHÓM 6 — INFO DICT & STRESS
# ═════════════════════════════════════════════════════════════════════════════
section("NHÓM 6 — Info dict & Stress test")

def t_info_keys():
    """Info dict phải có đủ key cần thiết cho training log."""
    required_keys = {"success", "service_idx", "placed_on_node",
                     "step_count", "placed_total"}
    env = K8sPlacementEnv(num_nodes=5, num_services=5)
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    missing = required_keys - set(info.keys())
    assert not missing, f"Thiếu key trong info: {missing}"

def t_placed_total_monotonic():
    """placed_total chỉ được tăng hoặc đứng yên, không bao giờ giảm."""
    env = K8sPlacementEnv(num_nodes=5, num_services=5)
    env.reset(seed=7)
    prev_placed = 0
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        assert info["placed_total"] >= prev_placed, \
            f"placed_total giảm! {prev_placed} → {info['placed_total']}"
        prev_placed = info["placed_total"]

def t_stress_1000_episodes():
    """1000 episodes không crash, không treo."""
    env = K8sPlacementEnv(num_nodes=5, num_services=5, max_steps=100)
    all_rewards = []
    for ep in range(1000):
        env.reset(seed=ep % 50)
        done = False
        ep_r = 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            ep_r += reward
            done = terminated or truncated
        all_rewards.append(ep_r)
    arr = np.array(all_rewards)
    assert not np.any(np.isnan(arr)), "NaN trong reward của stress test"
    assert not np.any(np.isinf(arr)), "Inf trong reward của stress test"

run_test("info_keys",             t_info_keys)
run_test("placed_total_monotonic",t_placed_total_monotonic)
run_test("stress_1000_episodes",  t_stress_1000_episodes)


# ═════════════════════════════════════════════════════════════════════════════
# KẾT QUẢ TỔNG HỢP
# ═════════════════════════════════════════════════════════════════════════════
total = total_pass + total_fail
print(f"\n{'═'*55}")
print(f"  KẾT QUẢ: {total_pass}/{total} test PASS", end="")

if total_fail == 0:
    print("  🎉")
    print(f"{'═'*55}")
    print("  Môi trường SẠCH — sẵn sàng để training! ✅")
else:
    print(f"  |  {total_fail} FAIL ❌")
    print(f"{'═'*55}")
    print("  Cần fix các lỗi sau trước khi train:\n")
    for name, msg in failed_tests:
        print(f"  • {name}")
        # Chỉ in dòng đầu của message để gọn
        first_line = msg.strip().split("\n")[0]
        print(f"    {first_line}")

print(f"{'═'*55}")
