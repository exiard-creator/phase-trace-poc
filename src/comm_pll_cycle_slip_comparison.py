"""Generate communication PLL/cycle-slip stress comparison figures.

This is an explanatory proof-of-concept model, not a standard-compliant modem or
5G/6G receiver simulator.  It isolates one question: under weak signal and
common phase dynamics, can a phase-trace fallback observable reduce cycle-slip
risk compared with simple carrier tracking references?

Run from the workspace root:
    python src/comm_pll_cycle_slip_comparison.py

Outputs are written to:
    outputs/
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
SNR_DB = np.array([-12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
DEFAULT_SYMBOLS = 5_000
DEFAULT_SEED = 2026070603


@dataclass(frozen=True)
class TrackingResult:
    scheme: str
    snr_db: float
    phase_rmse_rad: float
    cycle_slip_probability: float
    resource_units_per_symbol: float
    note: str


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def awgn_complex(shape: tuple[int, ...], noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    return noise_sigma * (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / math.sqrt(2.0)


def generate_phase_dynamics(symbols: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(symbols)
    carrier_offset = 0.030 * t
    phase_acceleration = 0.000002 * t**2
    oscillator_walk = np.cumsum(0.020 * rng.normal(size=symbols))
    vibration = 0.25 * np.sin(2.0 * math.pi * t / 211.0)
    return np.asarray(wrap_angle(carrier_offset + phase_acceleration + oscillator_walk + vibration))


def make_result(
    scheme: str,
    snr_db: float,
    errors: np.ndarray,
    slips: np.ndarray,
    resource_units_per_symbol: float,
    note: str,
) -> TrackingResult:
    kept = errors[~slips]
    rmse = float(np.sqrt(np.mean(kept**2))) if kept.size else math.pi
    return TrackingResult(
        scheme=scheme,
        snr_db=float(snr_db),
        phase_rmse_rad=rmse,
        cycle_slip_probability=float(np.mean(slips)),
        resource_units_per_symbol=float(resource_units_per_symbol),
        note=note,
    )


def simulate_second_order_pll(true_phase: np.ndarray, snr_db: float, rng: np.random.Generator) -> TrackingResult:
    """Simple decision-directed second-order PLL reference."""
    snr = db_to_linear(snr_db)
    noise_sigma = math.sqrt(1.0 / max(snr, 1e-12))
    kp = 0.090
    ki = 0.0025
    estimate = 0.0
    frequency = 0.0
    errors: list[float] = []
    slips: list[bool] = []
    for phase in true_phase:
        symbol = 1.0 + 0.0j
        rx = symbol * np.exp(1j * phase) + awgn_complex((), noise_sigma, rng)
        detector = float(wrap_angle(np.angle(rx) - estimate))
        frequency += ki * detector
        estimate = float(wrap_angle(estimate + frequency + kp * detector))
        err = float(wrap_angle(estimate - phase))
        errors.append(err)
        slips.append(abs(err) > 1.10)
    return make_result(
        "second_order_pll",
        snr_db,
        np.array(errors),
        np.array(slips),
        1.0,
        "Simple carrier PLL reference with static loop gains; vulnerable to weak-signal cycle slips.",
    )


def phase_trace_measurement(
    phase: float,
    noise_sigma: float,
    rng: np.random.Generator,
    pairs: int = 4,
    repetitions: int = 2,
) -> complex:
    pair_bias = rng.uniform(-0.55, 0.55, size=pairs)
    pair_weight = rng.uniform(0.85, 1.15, size=pairs)
    common_phase = phase + 0.35 * rng.normal()
    vectors: list[complex] = []
    weights: list[float] = []
    for pair in range(pairs):
        for _ in range(repetitions):
            left = pair_weight[pair] * np.exp(1j * (phase + common_phase + pair_bias[pair]))
            right = pair_weight[pair] * np.exp(1j * (common_phase - pair_bias[pair]))
            left += awgn_complex((), noise_sigma, rng)
            right += awgn_complex((), noise_sigma, rng)
            trace = left * np.conj(right)
            vectors.append(trace * np.exp(-2j * pair_bias[pair]))
            weights.append(pair_weight[pair])
    return sum(weight * vector for weight, vector in zip(weights, vectors))


def simulate_phase_trace_aided_pll(true_phase: np.ndarray, snr_db: float, rng: np.random.Generator) -> TrackingResult:
    """PLL assisted by a phase-trace observable during weak-signal stress."""
    snr = db_to_linear(snr_db)
    noise_sigma = math.sqrt(1.0 / max(snr, 1e-12))
    kp = 0.070
    ki = 0.0018
    trace_gain = 0.18
    estimate = 0.0
    frequency = 0.0
    errors: list[float] = []
    slips: list[bool] = []
    for phase in true_phase:
        symbol = 1.0 + 0.0j
        rx = symbol * np.exp(1j * phase) + awgn_complex((), noise_sigma, rng)
        pll_detector = float(wrap_angle(np.angle(rx) - estimate))
        trace = phase_trace_measurement(phase, noise_sigma, rng, pairs=4, repetitions=2)
        trace_detector = float(wrap_angle(np.angle(trace) - estimate))
        confidence = min(abs(trace) / 10.0, 1.0)
        detector = (1.0 - trace_gain * confidence) * pll_detector + trace_gain * confidence * trace_detector
        frequency += ki * detector
        estimate = float(wrap_angle(estimate + frequency + kp * detector))
        err = float(wrap_angle(estimate - phase))
        errors.append(err)
        slips.append(abs(err) > 1.10)
    return make_result(
        "phase_trace_aided_pll",
        snr_db,
        np.array(errors),
        np.array(slips),
        17.0,
        "Simple PLL augmented by a confidence-weighted phase-trace discriminator; PoC only, not an optimized loop design.",
    )


def simulate_differential_phase(true_phase: np.ndarray, snr_db: float, rng: np.random.Generator) -> TrackingResult:
    """Adjacent-symbol differential phase reference."""
    snr = db_to_linear(snr_db)
    noise_sigma = math.sqrt(1.0 / max(snr, 1e-12))
    tx = np.exp(1j * true_phase)
    rx = tx + awgn_complex(tx.shape, noise_sigma, rng)
    measured = np.asarray(wrap_angle(np.angle(rx[1:] * np.conj(rx[:-1]))))
    expected = np.asarray(wrap_angle(true_phase[1:] - true_phase[:-1]))
    errors = np.asarray(wrap_angle(measured - expected))
    slips = np.abs(errors) > 1.10
    return make_result(
        "differential_phase",
        snr_db,
        errors,
        slips,
        2.0,
        "Noncoherent adjacent-symbol phase reference; avoids absolute phase lock but doubles independent noise.",
    )


def simulate_phase_trace_fallback(
    true_phase: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    pairs: int = 4,
    repetitions: int = 3,
) -> TrackingResult:
    """Mirror-pair phase-trace fallback under common carrier phase dynamics."""
    snr = db_to_linear(snr_db)
    noise_sigma = math.sqrt(1.0 / max(snr, 1e-12))
    pair_bias = rng.uniform(-0.55, 0.55, size=pairs)
    pair_weight = rng.uniform(0.85, 1.15, size=pairs)
    errors: list[float] = []
    slips: list[bool] = []
    for phase in true_phase:
        encoded_phase = phase
        vectors: list[complex] = []
        weights: list[float] = []
        common_phase = phase + 0.35 * rng.normal()
        for pair in range(pairs):
            for _ in range(repetitions):
                left = pair_weight[pair] * np.exp(1j * (encoded_phase + common_phase + pair_bias[pair]))
                right = pair_weight[pair] * np.exp(1j * (common_phase - pair_bias[pair]))
                left += awgn_complex((), noise_sigma, rng)
                right += awgn_complex((), noise_sigma, rng)
                trace = left * np.conj(right)
                corrected = trace * np.exp(-2j * pair_bias[pair])
                vectors.append(corrected)
                weights.append(pair_weight[pair])
        combined = sum(weight * vector for weight, vector in zip(weights, vectors))
        estimate = float(np.angle(combined))
        err = float(wrap_angle(estimate - phase))
        errors.append(err)
        slips.append(abs(err) > 1.10)
    return make_result(
        "phase_trace_fallback_rep3_top4",
        snr_db,
        np.array(errors),
        np.array(slips),
        float(2 * pairs * repetitions),
        "Conservative mirror-pair phase trace aid; cancels common phase by conjugate product and vector combining.",
    )


def run_comparison(symbols: int = DEFAULT_SYMBOLS, seed: int = DEFAULT_SEED) -> list[TrackingResult]:
    master = np.random.default_rng(seed)
    rows: list[TrackingResult] = []
    for snr_db in SNR_DB:
        phase_rng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
        true_phase = generate_phase_dynamics(symbols, phase_rng)
        for simulator in [
            simulate_second_order_pll,
            simulate_phase_trace_aided_pll,
            simulate_differential_phase,
            simulate_phase_trace_fallback,
        ]:
            rng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
            rows.append(simulator(true_phase, float(snr_db), rng))
    return rows


def plot_phase_rmse(rows: list[TrackingResult]) -> None:
    plt.figure(figsize=(8.2, 5.2))
    for scheme in sorted({row.scheme for row in rows}):
        subset = [row for row in rows if row.scheme == scheme]
        plt.semilogy(
            [row.snr_db for row in subset],
            [max(row.phase_rmse_rad, 1e-4) for row in subset],
            marker="o",
            label=scheme,
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("Phase RMSE (rad)")
    plt.title("Communication carrier tracking stress comparison")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "comm_phase_rmse_comparison.png", dpi=180)
    plt.close()


def plot_cycle_slip(rows: list[TrackingResult]) -> None:
    plt.figure(figsize=(8.2, 5.2))
    floor = 0.5 / DEFAULT_SYMBOLS
    for scheme in sorted({row.scheme for row in rows}):
        subset = [row for row in rows if row.scheme == scheme]
        plt.semilogy(
            [row.snr_db for row in subset],
            [max(row.cycle_slip_probability, floor) for row in subset],
            marker="o",
            label=scheme,
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("Cycle-slip probability (floor shown)")
    plt.title("Cycle-slip stress comparison")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "comm_cycle_slip_comparison.png", dpi=180)
    plt.close()


def save_results(rows: list[TrackingResult]) -> None:
    payload = {
        "metadata": {
            "symbols": DEFAULT_SYMBOLS,
            "seed": DEFAULT_SEED,
            "snr_db": SNR_DB.tolist(),
            "cycle_slip_threshold_rad": 1.10,
            "important_note": "Communication PLL/cycle-slip proof-of-concept stress model, not a standard-compliant modem simulator.",
        },
        "results": [asdict(row) for row in rows],
    }
    (OUT_DIR / "comm_pll_tracking_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(rows: list[TrackingResult]) -> None:
    lines = [
        "# Communication PLL/Cycle-Slip Tracking Summary",
        "",
        "This file is generated by `src/comm_pll_cycle_slip_comparison.py`.",
        "",
        "## Important Interpretation",
        "",
        "This is a carrier-tracking stress PoC, not a complete modem, 5G/6G PHY, or receiver conformance simulator.",
        "It isolates weak-signal/common-phase dynamics and compares a simple second-order PLL, a phase-trace aided PLL, differential phase tracking, and a conservative phase-trace fallback aid.",
        "",
        "## Results",
        "",
        "| Scheme | SNR dB | Phase RMSE rad | Cycle-slip probability | Resource units / symbol |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.scheme} | {row.snr_db:.1f} | {row.phase_rmse_rad:.6f} | {row.cycle_slip_probability:.6f} | {row.resource_units_per_symbol:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- `second_order_pll` is a simple static-gain tracking-loop reference, not an optimized commercial receiver PLL.",
            "- `phase_trace_aided_pll` is a concept-level loop assisted by a confidence-weighted phase-trace discriminator, not an optimized production PLL.",
            "- `phase_trace_fallback_rep3_top4` is an auxiliary fallback observable; it spends more channel resources by design.",
            "- Cycle slip is counted when wrapped phase error exceeds 1.10 rad in this stress model; this is a visualization criterion, not a standard definition.",
            "- The model excludes coding, interleaving, pilots/DMRS design, equalization, MIMO, scheduler behavior, RF impairments, and fixed-point implementation details.",
        ]
    )
    (OUT_DIR / "comm_pll_tracking_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_comparison()
    save_results(rows)
    write_summary(rows)
    plot_phase_rmse(rows)
    plot_cycle_slip(rows)
    print("Generated communication PLL/cycle-slip comparison outputs:")
    for path in sorted(OUT_DIR.glob("comm_*")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
