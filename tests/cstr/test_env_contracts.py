"""Native CSTR threshold and deterministic-runtime tests."""

from __future__ import annotations

import numpy as np

from envs.cstr.env import CSTRConfig, CSTREnv


EPSILON = 1e-6


def make_env(**kwargs: object) -> CSTREnv:
    return CSTREnv(CSTRConfig(randomize_initial_state=False, randomize_setpoint=False, **kwargs))


def test_soak_temperature_band_boundaries_are_inclusive() -> None:
    env = make_env()

    env.temp = env.config.soak_band_low
    assert env._in_soak_temp_band()
    env.temp = env.config.soak_band_low - EPSILON
    assert not env._in_soak_temp_band()

    env.temp = env.config.soak_band_high
    assert env._in_soak_temp_band()
    env.temp = env.config.soak_band_high + EPSILON
    assert not env._in_soak_temp_band()


def test_soak_band_can_require_concentration_band() -> None:
    env = make_env(require_soak_concentration_band=True)

    env.temp = 345.0
    env.ca = env.config.soak_concentration_low
    assert env._in_soak_band()
    env.ca = env.config.soak_concentration_high
    assert env._in_soak_band()
    env.ca = env.config.soak_concentration_low - EPSILON
    assert not env._in_soak_band()
    env.ca = env.config.soak_concentration_high + EPSILON
    assert not env._in_soak_band()


def test_temperature_safety_and_critical_boundaries() -> None:
    env = make_env()

    env.temp = env.config.safe_temp_low
    assert env._temp_safe()
    env.temp = env.config.safe_temp_low - EPSILON
    assert not env._temp_safe()

    env.temp = env.config.safe_temp_high
    assert env._temp_safe()
    env.temp = env.config.safe_temp_high + EPSILON
    assert not env._temp_safe()

    env.temp = env.config.critical_temp_low
    assert env._temp_critical()
    env.temp = env.config.critical_temp_low + EPSILON
    assert not env._temp_critical()
    env.temp = env.config.critical_temp_high
    assert env._temp_critical()
    env.temp = env.config.critical_temp_high - EPSILON
    assert not env._temp_critical()


def test_stability_and_overshoot_thresholds_are_inclusive_where_expected() -> None:
    env = make_env(production_temp_low=346.0, production_temp_high=354.0, concentration_tolerance=0.04)

    env.temp = 350.0
    env.ca_setpoint = 0.5
    env.ca = 0.46
    assert env._production_stable()
    env.ca = 0.459999
    assert not env._production_stable()

    env.ca = env.config.ca_overshoot_low
    assert not env._overshoot()
    env.ca = env.config.ca_overshoot_low - EPSILON
    assert env._overshoot()


def test_reset_seed_determinism_under_fixed_action_sequence() -> None:
    config = CSTRConfig(
        max_episode_steps=8,
        randomize_initial_state=True,
        randomize_setpoint=True,
        enable_disturbance=True,
        disturbance_probability=0.5,
    )
    actions = [np.array([value], dtype=np.float32) for value in (-0.5, 0.0, 0.25, 1.0, -1.0)]

    env_a = CSTREnv(config)
    env_b = CSTREnv(config)
    obs_a, info_a = env_a.reset(seed=123)
    obs_b, info_b = env_b.reset(seed=123)

    np.testing.assert_allclose(obs_a, obs_b)
    assert info_a["ca"] == info_b["ca"]
    assert info_a["reactor_temperature"] == info_b["reactor_temperature"]
    assert info_a["target_concentration"] == info_b["target_concentration"]

    for action in actions:
        step_a = env_a.step(action)
        step_b = env_b.step(action)
        np.testing.assert_allclose(step_a[0], step_b[0])
        assert step_a[1] == step_b[1]
        assert step_a[2] == step_b[2]
        assert step_a[3] == step_b[3]
        assert step_a[4]["ca"] == step_b[4]["ca"]
        assert step_a[4]["reactor_temperature"] == step_b[4]["reactor_temperature"]
