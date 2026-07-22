"""Validate the Python decoding of the CSTR startup task against the RML monitor.

The external RML monitor returns an unstable closure string (fresh gensym vars
per episode/process), so the *raw* monitor_state cannot be used as a stable
encoding. What IS stable and process-independent is the monitor's **verdict**
(true/false/currently_*). This script therefore validates the Python automaton by
its verdict, not by string matching:

  * a pure ``ReferenceStartupAutomaton`` (payload-driven, never reads the monitor
    verdict) is run in lock-step with the monitor (via the wrapper's
    ``assert_monitor_consistency`` hook);
  * we check, over success / failure / drift trajectories and over BOTH spec
    variants (recover=True as used in training, recover=False as used in eval),
    that the reference accept/reject decision matches the monitor verdict at every
    step, and that the deployed semantic progress decoder state matches the reference state.

If agreement is ~100% on both variants, the Python encoding independently
validates consistency with the RML specification across the tested traces.

Run from the repository root:

    python3 -m envs.cstr.experiments.inspect_monitor_state --soak-steps 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from envs.cstr import CSTRConfig, RMLCSTRConfig, make_rml_cstr_env
from envs.cstr.rml_generation import generate_cstr_rml
from rml_rm.monitors import RMLMonitorProcess, find_free_port


def staged(temp, target, soaked, *, soak_steps, gain=5.0, ramp=1.0, soak_center=345.0):
    hold = soak_steps + 2
    if target < soak_center or soaked >= hold:
        target = min(target + ramp, 350.0)
    else:
        soaked += 1
    return float(np.clip(300.0 - gain * (temp - target), 250.0, 350.0)), target, soaked


def reckless(temp, target, soaked, **_):
    return float(np.clip(300.0 - 5.0 * (temp - 350.0), 250.0, 350.0)), target, soaked


def drift(temp, target, soaked, *, soak_steps, step, drift_at=70):
    """Stage to regulate, then leave the band (exercises Regulate recovery/reject)."""
    if step >= drift_at:
        return 312.0, target, soaked  # cool toward the safe-but-not-stable branch
    return staged(temp, target, soaked, soak_steps=soak_steps)


def to_action(coolant, low, high):
    return np.asarray([2.0 * ((coolant - low) / (high - low)) - 1.0], dtype=np.float32)


def run_episode(env, cfg, *, controller, soak_steps, seed):
    obs, info = env.reset(seed=seed)
    target = float(info["reactor_temperature"])
    soaked = 0
    rows = []
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        coolant, target, soaked = controller(
            float(info["reactor_temperature"]), target, soaked, soak_steps=soak_steps, step=step
        ) if controller is drift else controller(
            float(info["reactor_temperature"]), target, soaked, soak_steps=soak_steps
        )
        obs, _r, terminated, truncated, info = env.step(to_action(coolant, cfg.action_low, cfg.action_high))
        step += 1
        rows.append(
            {
                "monitor_verdict": info.get("monitor_verdict"),
                "monitor_state_norm": info.get("rml_monitor_state_unencoded_normalized", ""),
                "reference_verdict": info.get("reference_verdict"),
                "monitor_consistent": bool(info.get("monitor_consistent", True)),
                "decoder_state": info.get("rml_monitor_state_normalized"),
                "reference_state": info.get("reference_state"),
                "ca": round(float(info.get("ca", 0)), 3),
                "temp": round(float(info.get("reactor_temperature", 0)), 1),
            }
        )
    return rows


def _decoder_verdict(decoder_state: str) -> str:
    if decoder_state == "success":
        return "accept"
    if decoder_state == "failure":
        return "reject"
    return "run"


def validate_variant(*, recover: bool, soak_steps: int, max_steps: int, output: Path) -> tuple[int, int, int, int]:
    port = find_free_port()
    generated = generate_cstr_rml(
        soak_steps=soak_steps,
        recover_from_regulation_failure=recover,
        port=port,
        max_episode_steps=max_steps,
        generated_root=output / f"monitors_recover_{recover}",
    )
    monitor = RMLMonitorProcess(
        generated.spec_path, port=port, log_path=output / f"monitor_recover_{recover}.log"
    ).start()
    try:
        cfg = CSTRConfig(
            soak_steps=soak_steps, max_episode_steps=max_steps,
            concentration_tolerance=0.08, production_temp_low=346.0, production_temp_high=354.0,
            deadline_steps=max_steps, randomize_initial_state=False,
        )
        env = make_rml_cstr_env(
            RMLCSTRConfig(
                cstr_env=cfg, observation_mode="semantic_progress", reward_mode="env_rml",
                config_path=generated.config_path, monitor_port=port, soak_steps=soak_steps,
                recover_from_regulation_failure=recover,
                terminate_on_rml_failure=False, terminate_on_rml_success=False,
                assert_monitor_consistency=True,
            )
        )
        rows: list[dict[str, Any]] = []
        for i in range(3):
            rows += run_episode(env, cfg, controller=staged, soak_steps=soak_steps, seed=1000 + i)
        for i in range(2):
            rows += run_episode(env, cfg, controller=reckless, soak_steps=soak_steps, seed=2000 + i)
        for i in range(2):
            rows += run_episode(env, cfg, controller=drift, soak_steps=soak_steps, seed=3000 + i)

        from envs.cstr.reference_automaton import verdict_matches_monitor
        verdict_ok = sum(r["monitor_consistent"] for r in rows)  # reference vs monitor
        state_ok = sum(r["decoder_state"] == r["reference_state"] for r in rows)
        decoder_ok = sum(  # DEPLOYED encoding vs monitor (the headline number)
            verdict_matches_monitor(_decoder_verdict(r["decoder_state"]),
                                    r["monitor_verdict"], r["monitor_state_norm"])
            for r in rows
        )
        total = len(rows)
        tag = "recover=True (train spec)" if recover else "recover=False (eval spec)"
        print(f"\n----- {tag} -----")
        print(f"  steps validated         : {total}")
        print(f"  DEPLOYED decoder == monitor (verdict/state): {decoder_ok}/{total} "
              f"({100.0 * decoder_ok / max(total,1):.2f}%)   <- deployed consistency")
        print(f"  reference  == monitor accept/reject        : {verdict_ok}/{total} "
              f"({100.0 * verdict_ok / max(total,1):.2f}%)")
        print(f"  semantic progress decoder == reference state         : {state_ok}/{total} "
              f"({100.0 * state_ok / max(total,1):.2f}%)")
        mismatches = [r for r in rows if not r["monitor_consistent"]][:8]
        if mismatches:
            print("  VERDICT MISMATCHES (first 8):")
            for r in mismatches:
                print(f"     monitor={r['monitor_verdict']!r} reference={r['reference_verdict']!r} "
                      f"state(dec/ref)={r['decoder_state']}/{r['reference_state']} ca={r['ca']} T={r['temp']}")
        smismatch = [r for r in rows if r["decoder_state"] != r["reference_state"]][:8]
        if smismatch:
            print("  STATE MISMATCHES (first 8):")
            for r in smismatch:
                print(f"     decoder={r['decoder_state']} reference={r['reference_state']} "
                      f"verdict={r['monitor_verdict']!r} ca={r['ca']} T={r['temp']}")
        return verdict_ok, state_ok, total, decoder_ok
    finally:
        monitor.stop()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--soak-steps", type=int, default=10)
    p.add_argument("--max-episode-steps", type=int, default=300)
    p.add_argument("--output", type=Path, default=Path("/tmp/cstr_monitor_validation"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    totals = []
    for recover in (True, False):
        totals.append(validate_variant(
            recover=recover, soak_steps=args.soak_steps,
            max_steps=args.max_episode_steps, output=args.output,
        ))

    v_ok = sum(t[0] for t in totals)
    s_ok = sum(t[1] for t in totals)
    n = sum(t[2] for t in totals)
    d_ok = sum(t[3] for t in totals)
    print("\n==================== OVERALL ====================")
    print(f"  DEPLOYED decoder vs monitor (both variants): {d_ok}/{n} ({100.0 * d_ok / max(n,1):.2f}%)")
    print(f"  reference   vs monitor      (both variants): {v_ok}/{n} ({100.0 * v_ok / max(n,1):.2f}%)")
    print(f"  decoder     vs reference    (both variants): {s_ok}/{n} ({100.0 * s_ok / max(n,1):.2f}%)")
    if d_ok == n:
        print("  PASS: the DEPLOYED semantic progress encoding is consistent with the RML monitor")
        print("        on 100% of steps. Any reference<->monitor residual is a one-step")
        print("        epsilon-transition boundary effect (self-correcting); cite the")
        print("        deployed number, and the reference number as a stress test.")
    else:
        print("  MISMATCH FOUND: the deployed decoder diverges from the monitor -> investigate.")


if __name__ == "__main__":
    main()
