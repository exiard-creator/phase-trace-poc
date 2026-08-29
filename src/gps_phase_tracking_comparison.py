"""Generate GNSS/GPS-like phase tracking stress comparison figures.

This is a public-facing explanatory model, not a GPS L1 C/A receiver simulator.
It compares simple known GNSS receiver observables with a phase-trace fallback
style observable under weak-signal and high-dynamics stress.

Run from the workspace root:
    python src/gps_phase_tracking_comparison.py

Outputs:
    outputs/gps_phase_rmse_comparison.png
    outputs/gps_outage_comparison.png
    outputs/gps_phase_tracking_results.json
    outputs/gps_phase_tracking_summary.md
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
CN0_DBHZ = np.array([18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0, 45.0])
DEFAULT_EPOCHS = 2_000
DEFAULT_SEED = 2026070602
COHERENT_TIME_S = 0.02
L1_WAVELENGTH_M = 0.190293672798365
C_M_PER_S = 299_792_458.0
CA_CHIP_RATE_HZ = 1.023e6


@dataclass(frozen=True)
class GpsSchemeResult:
    scheme: str
    cn0_dbhz: float
    phase_rmse_rad: float
    outage_probability: float
    note: str


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def awgn_complex(shape: tuple[int, ...], noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    return noise_sigma * (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / math.sqrt(2.0)


def generate_phase_dynamics(epochs: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(epochs)
    carrier_offset = 0.012 * t
    phase_acceleration = 0.000001 * t**2
    oscillator_walk = np.cumsum(0.015 * rng.normal(size=epochs))
    vibration = 0.20 * np.sin(2.0 * math.pi * t / 181.0)
    return np.asarray(wrap_angle(carrier_offset + phase_acceleration + oscillator_walk + vibration))


def make_result(
    scheme: str,
    cn0_dbhz: float,
    errors: np.ndarray,
    outage: np.ndarray,
    note: str,
) -> GpsSchemeResult:
    rmse = float(np.sqrt(np.mean(errors**2))) if errors.size else math.pi
    return GpsSchemeResult(
        scheme=scheme,
        cn0_dbhz=float(cn0_dbhz),
        phase_rmse_rad=rmse,
        outage_probability=float(np.mean(outage)),
        note=note,
    )


def simulate_legacy_code_tracking(true_phase: np.ndarray, cn0_dbhz: float, rng: np.random.Generator) -> GpsSchemeResult:
    cn0 = db_to_linear(cn0_dbhz)
    noise_sigma = math.sqrt(1.0 / max(cn0, 1e-12))
    errors: list[float] = []
    outages: list[bool] = []
    for phase in true_phase:
        rx = np.exp(1j * phase) + awgn_complex((), noise_sigma, rng)
        estimate = float(np.angle(rx))
        err = float(wrap_angle(estimate - phase))
        errors.append(err)
        outages.append(abs(err) > 0.90)
    return make_result(
        "legacy_code_tracking",
        cn0_dbhz,
        np.array(errors),
        np.array(outages),
        "Coarse legacy-style code tracking reference; robust but slow to react.",
    )


def simulate_carrier_pll_tracking(true_phase: np.ndarray, cn0_dbhz: float, rng: np.random.Generator) -> GpsSchemeResult:
    cn0 = db_to_linear(cn0_dbhz)
    noise_sigma = math.sqrt(1.0 / max(cn0, 1e-12))
    estimate = 0.0
    errors: list[float] = []
    outages: list[bool] = []
    for phase in true_phase:
        rx = np.exp(1j * phase) + awgn_complex((), noise_sigma, rng)
        detector = float(wrap_angle(np.angle(rx) - estimate))
        estimate = float(wrap_angle(estimate + 0.08 * detector))
        err = float(wrap_angle(estimate - phase))
        errors.append(err)
        outages.append(abs(err) > 1.10)
    return make_result(
        "carrier_pll_tracking",
        cn0_dbhz,
        np.array(errors),
        np.array(outages),
        "Simple carrier PLL reference; vulnerable to weak-signal outages.",
    )


def simulate_differential_carrier_phase(true_phase: np.ndarray, cn0_dbhz: float, rng: np.random.Generator) -> GpsSchemeResult:
    cn0 = db_to_linear(cn0_dbhz)
    noise_sigma = math.sqrt(1.0 / max(cn0, 1e-12))
    tx = np.exp(1j * true_phase)
    rx = tx + awgn_complex(tx.shape, noise_sigma, rng)
    measured = np.asarray(wrap_angle(np.angle(rx[1:] * np.conj(rx[:-1]))))
    expected = np.asarray(wrap_angle(true_phase[1:] - true_phase[:-1]))
    errors = np.asarray(wrap_angle(measured - expected))
    outages = np.abs(errors) > 1.10
    return make_result(
        "differential_carrier_phase",
        cn0_dbhz,
        errors,
        outages,
        "Adjacent-epoch differential carrier reference; avoids absolute phase lock.",
    )


def simulate_phase_trace_fallback_aid(true_phase: np.ndarray, cn0_dbhz: float, rng: np.random.Generator) -> GpsSchemeResult:
    cn0 = db_to_linear(cn0_dbhz)
    noise_sigma = math.sqrt(1.0 / max(cn0, 1e-12))
    errors: list[float] = []
    outages: list[bool] = []
    for phase in true_phase:
        common_phase = phase + 0.25 * rng.normal()
        left = np.exp(1j * (phase + common_phase)) + awgn_complex((), noise_sigma, rng)
        right = np.exp(1j * (common_phase)) + awgn_complex((), noise_sigma, rng)
        trace = left * np.conj(right)
        estimate = float(np.angle(trace))
        err = float(wrap_angle(estimate - phase))
        errors.append(err)
        outages.append(abs(err) > 1.10)
    return make_result(
        "phase_trace_fallback_aid",
        cn0_dbhz,
        np.array(errors),
        np.array(outages),
        "Simple phase-trace fallback aid under common phase rotation; PoC only.",
    )


def run_comparison(epochs: int = DEFAULT_EPOCHS, seed: int = DEFAULT_SEED) -> list[GpsSchemeResult]:
    master = np.random.default_rng(seed)
    rows: list[GpsSchemeResult] = []
    for cn0_dbhz in CN0_DBHZ:
        phase_rng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
        true_phase = generate_phase_dynamics(epochs, phase_rng)
        for simulator in [
            simulate_legacy_code_tracking,
            simulate_carrier_pll_tracking,
            simulate_differential_carrier_phase,
            simulate_phase_trace_fallback_aid,
        ]:
            rng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
            rows.append(simulator(true_phase, float(cn0_dbhz), rng))
    return rows


def plot_phase_rmse(rows: list[GpsSchemeResult]) -> None:
    plt.figure(figsize=(8.2, 5.2))
    for scheme in sorted({row.scheme for row in rows}):
        subset = [row for row in rows if row.scheme == scheme]
        plt.semilogy(
            [row.cn0_dbhz for row in subset],
            [max(row.phase_rmse_rad, 1e-4) for row in subset],
            marker="o",
            label=scheme,
        )
    plt.xlabel("C/N0 (dB-Hz)")
    plt.ylabel("Phase RMSE (rad)")
    plt.title("GPS-like phase observable stress comparison")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "gps_phase_rmse_comparison.png", dpi=180)
    plt.close()


def plot_outage(rows: list[GpsSchemeResult]) -> None:
    plt.figure(figsize=(8.2, 5.2))
    floor = 0.5 / DEFAULT_EPOCHS
    for scheme in sorted({row.scheme for row in rows}):
        subset = [row for row in rows if row.scheme == scheme]
        plt.semilogy(
            [row.cn0_dbhz for row in subset],
            [max(row.outage_probability, floor) for row in subset],
            marker="o",
            label=scheme,
        )
    plt.xlabel("C/N0 (dB-Hz)")
    plt.ylabel("Outage probability (floor shown)")
    plt.title("GPS-like weak-signal outage comparison")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "gps_outage_comparison.png", dpi=180)
    plt.close()


def save_results(rows: list[GpsSchemeResult]) -> None:
    payload = {
        "metadata": {
            "epochs": DEFAULT_EPOCHS,
            "seed": DEFAULT_SEED,
            "cn0_dbhz": CN0_DBHZ.tolist(),
            "important_note": "GNSS-like phase-observable proof of concept, not a full GNSS receiver simulator.",
        },
        "results": [asdict(row) for row in rows],
    }
    (OUT_DIR / "gps_phase_tracking_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(rows: list[GpsSchemeResult]) -> None:
    lines = [
        "# GPS-like Phase Tracking Summary",
        "",
        "This file is generated by `src/gps_phase_tracking_comparison.py`.",
        "",
        "## Results",
        "",
        "| Scheme | C/N0 dB-Hz | Phase RMSE rad | Outage probability |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.scheme} | {row.cn0_dbhz:.1f} | {row.phase_rmse_rad:.6f} | {row.outage_probability:.6f} |"
        )
    (OUT_DIR / "gps_phase_tracking_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_comparison()
    save_results(rows)
    write_summary(rows)
    plot_phase_rmse(rows)
    plot_outage(rows)
    print("Generated GPS-like phase tracking outputs:")
    for path in sorted(OUT_DIR.glob("gps_*")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
