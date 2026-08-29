"""Generate public comparison figures for mirror phase trace fallback.

The purpose of this script is not to claim replacement of 5G/6G PHY.  It shows a
small, reproducible comparison between known low-order communication baselines
and a mirror-subcarrier phase-trace fallback model under common phase rotation.

Run from the workspace root:
    python src/known_tech_comparison.py

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
SNR_DB = np.array([-20.0, -16.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
DEFAULT_BITS = 30_000
DEFAULT_SEED = 20260706


@dataclass(frozen=True)
class SchemeResult:
    scheme: str
    ebn0_db: float
    bit_errors: int
    bit_count: int
    ber: float
    channel_uses_per_bit: float
    useful_bits_per_channel_use: float
    note: str


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


def awgn_complex(shape: tuple[int, ...], noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    return noise_sigma * (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / math.sqrt(2.0)


def bits_to_bpsk(bits: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * bits.astype(float)


def make_result(
    scheme: str,
    ebn0_db: float,
    errors: int,
    bit_count: int,
    channel_uses_per_bit: float,
    note: str,
) -> SchemeResult:
    ber = errors / max(bit_count, 1)
    useful = (1.0 - ber) / channel_uses_per_bit
    return SchemeResult(
        scheme=scheme,
        ebn0_db=float(ebn0_db),
        bit_errors=int(errors),
        bit_count=int(bit_count),
        ber=float(ber),
        channel_uses_per_bit=float(channel_uses_per_bit),
        useful_bits_per_channel_use=float(useful),
        note=note,
    )


def simulate_coherent_bpsk(bits: np.ndarray, ebn0_db: float, rng: np.random.Generator) -> SchemeResult:
    """Ideal coherent BPSK reference with known carrier phase."""
    ebn0 = db_to_linear(ebn0_db)
    noise_sigma = math.sqrt(1.0 / (2.0 * ebn0))
    tx = bits_to_bpsk(bits)
    rx = tx + noise_sigma * rng.normal(size=bits.shape)
    decoded = (rx < 0.0).astype(np.int8)
    errors = int(np.count_nonzero(decoded != bits))
    return make_result(
        "ideal_coherent_bpsk",
        ebn0_db,
        errors,
        len(bits),
        channel_uses_per_bit=1.0,
        note="Known carrier phase; best-case low-order reference.",
    )


def simulate_dbpsk(bits: np.ndarray, ebn0_db: float, rng: np.random.Generator) -> SchemeResult:
    """Differential BPSK reference using adjacent-symbol phase difference."""
    ebn0 = db_to_linear(ebn0_db)
    noise_sigma = math.sqrt(1.0 / ebn0)
    phase = np.zeros(len(bits) + 1)
    phase[1:] = np.cumsum(np.where(bits == 0, 0.0, math.pi))
    common_phase = rng.uniform(-math.pi, math.pi)
    tx = np.exp(1j * (phase + common_phase))
    rx = tx + awgn_complex(tx.shape, noise_sigma, rng)
    diff = rx[1:] * np.conj(rx[:-1])
    decoded = (np.real(diff) < 0.0).astype(np.int8)
    errors = int(np.count_nonzero(decoded != bits))
    return make_result(
        "differential_bpsk",
        ebn0_db,
        errors,
        len(bits),
        channel_uses_per_bit=(len(bits) + 1) / len(bits),
        note="Known noncoherent baseline; avoids absolute phase tracking but has a BER penalty.",
    )


def simulate_repetition3_bpsk(bits: np.ndarray, ebn0_db: float, rng: np.random.Generator) -> SchemeResult:
    """Simple repetition-3 majority vote reference."""
    ebn0 = db_to_linear(ebn0_db)
    noise_sigma = math.sqrt(1.0 / (2.0 * ebn0))
    tx = np.repeat(bits_to_bpsk(bits), 3).reshape(len(bits), 3)
    rx = tx + noise_sigma * rng.normal(size=tx.shape)
    hard = rx < 0.0
    decoded = (np.sum(hard, axis=1) >= 2).astype(np.int8)
    errors = int(np.count_nonzero(decoded != bits))
    return make_result(
        "repetition3_bpsk_majority",
        ebn0_db,
        errors,
        len(bits),
        channel_uses_per_bit=3.0,
        note="Reliability by repetition; intentionally rate-expensive.",
    )


def build_mirror_channel(num_subcarriers: int, rng: np.random.Generator) -> np.ndarray:
    bins = np.arange(num_subcarriers)
    amplitude = 0.75 + 0.20 * np.cos(2.0 * math.pi * bins / num_subcarriers)
    amplitude += 0.06 * rng.normal(size=num_subcarriers)
    amplitude = np.clip(amplitude, 0.20, 1.25)
    phase = 0.45 * np.sin(2.0 * math.pi * bins / num_subcarriers) + 0.08 * rng.normal(size=num_subcarriers)
    return amplitude * np.exp(1j * phase)


def ranked_mirror_pairs(channel: np.ndarray) -> list[tuple[int, int, float]]:
    num_subcarriers = len(channel)
    pairs: list[tuple[int, int, float]] = []
    for k in range(1, num_subcarriers // 2):
        mirror = num_subcarriers - k
        amp = min(abs(channel[k]), abs(channel[mirror]))
        static_pair_phase = abs(np.angle(channel[k] * np.conj(channel[mirror])))
        stability = 1.0 / (1.0 + static_pair_phase)
        score = 0.72 * amp + 0.28 * stability
        pairs.append((k, mirror, float(score)))
    return sorted(pairs, key=lambda item: item[2], reverse=True)


def simulate_mirror_phase_trace(
    bits: np.ndarray,
    ebn0_db: float,
    rng: np.random.Generator,
    num_subcarriers: int = 64,
    top_pairs: int = 4,
    repetitions: int = 3,
    scheme_name: str | None = None,
) -> SchemeResult:
    """Mirror-subcarrier phase trace fallback under common phase rotation.

    Information is encoded as phase 0/pi on one side of a mirror pair.  The
    receiver computes y[k] * conj(y[N-k]), removes static pair bias, and combines
    all pair/repetition vectors before a hard decision.  The vector combining is
    the main robustness mechanism: random phase/noise components tend to average
    down, while the intended 0/pi phase trace adds coherently.
    """
    ebn0 = db_to_linear(ebn0_db)
    noise_sigma = math.sqrt(1.0 / max(ebn0, 1e-12))
    channel = build_mirror_channel(num_subcarriers, rng)
    pairs = ranked_mirror_pairs(channel)[:top_pairs]
    decoded = np.empty_like(bits)

    for idx, bit in enumerate(bits):
        vectors: list[complex] = []
        weights: list[float] = []
        data_phase = 0.0 if bit == 0 else math.pi
        for rep in range(repetitions):
            common_phase = rng.uniform(-math.pi, math.pi)
            slow_cfo_phase = 0.05 * (idx * repetitions + rep)
            common = np.exp(1j * (common_phase + slow_cfo_phase))
            for k, mirror, score in pairs:
                noise_k = awgn_complex((), noise_sigma, rng)
                noise_m = awgn_complex((), noise_sigma, rng)
                y_k = channel[k] * np.exp(1j * data_phase) * common + noise_k
                y_m = channel[mirror] * common + noise_m
                trace = y_k * np.conj(y_m)
                pair_bias = channel[k] * np.conj(channel[mirror])
                normalized = trace / max(abs(pair_bias), 1e-12) * np.exp(-1j * np.angle(pair_bias))
                vectors.append(normalized)
                weights.append(score)
        combined = sum(w * v for w, v in zip(weights, vectors))
        decoded[idx] = 0 if np.real(combined) >= 0.0 else 1

    errors = int(np.count_nonzero(decoded != bits))
    channel_uses = 2.0 * top_pairs * repetitions
    name = scheme_name or f"mirror_phase_trace_rep{repetitions}_top{top_pairs}"
    return make_result(
        name,
        ebn0_db,
        errors,
        len(bits),
        channel_uses_per_bit=channel_uses,
        note=(
            f"Mirror-pair conjugate product with vector combining; "
            f"top_pairs={top_pairs}, repetitions={repetitions}."
        ),
    )


def simulate_mirror_trace_rep1_top1(bits: np.ndarray, ebn0_db: float, rng: np.random.Generator) -> SchemeResult:
    return simulate_mirror_phase_trace(
        bits,
        ebn0_db,
        rng,
        top_pairs=1,
        repetitions=1,
        scheme_name="mirror_phase_trace_rep1_top1",
    )


def simulate_mirror_trace_rep1_top4(bits: np.ndarray, ebn0_db: float, rng: np.random.Generator) -> SchemeResult:
    return simulate_mirror_phase_trace(
        bits,
        ebn0_db,
        rng,
        top_pairs=4,
        repetitions=1,
        scheme_name="mirror_phase_trace_rep1_top4",
    )


def simulate_mirror_trace_rep3_top4(bits: np.ndarray, ebn0_db: float, rng: np.random.Generator) -> SchemeResult:
    return simulate_mirror_phase_trace(
        bits,
        ebn0_db,
        rng,
        top_pairs=4,
        repetitions=3,
        scheme_name="mirror_phase_trace_rep3_top4",
    )


def run_comparison(num_bits: int = DEFAULT_BITS, seed: int = DEFAULT_SEED) -> list[SchemeResult]:
    rng_master = np.random.default_rng(seed)
    bits = rng_master.integers(0, 2, size=num_bits, dtype=np.int8)
    rows: list[SchemeResult] = []
    for ebn0_db in SNR_DB:
        for simulator in [
            simulate_coherent_bpsk,
            simulate_dbpsk,
            simulate_repetition3_bpsk,
            simulate_mirror_trace_rep1_top1,
            simulate_mirror_trace_rep1_top4,
            simulate_mirror_trace_rep3_top4,
        ]:
            local_seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(local_seed)
            rows.append(simulator(bits, float(ebn0_db), rng))
    return rows


def plot_ber(rows: list[SchemeResult]) -> None:
    plt.figure(figsize=(8.2, 5.2))
    for scheme in sorted({row.scheme for row in rows}):
        subset = [row for row in rows if row.scheme == scheme]
        xs = [row.ebn0_db for row in subset]
        ys = [max(row.ber, 0.5 / row.bit_count) for row in subset]
        plt.semilogy(xs, ys, marker="o", label=scheme)
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER (floor shown at 0.5/N)")
    plt.title("BER comparison: known baselines vs mirror phase trace fallback")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ber_comparison.png", dpi=180)
    plt.close()


def plot_efficiency(rows: list[SchemeResult]) -> None:
    plt.figure(figsize=(8.2, 5.2))
    for scheme in sorted({row.scheme for row in rows}):
        subset = [row for row in rows if row.scheme == scheme]
        xs = [row.ebn0_db for row in subset]
        ys = [row.useful_bits_per_channel_use for row in subset]
        plt.plot(xs, ys, marker="o", label=scheme)
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Useful bits per channel use")
    plt.title("Reliability vs resource-efficiency tradeoff")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "resource_efficiency_comparison.png", dpi=180)
    plt.close()


def plot_tradeoff_dashboard(rows: list[SchemeResult], target_ebn0_db: float = 0.0) -> None:
    subset = [row for row in rows if row.ebn0_db == target_ebn0_db]
    subset = sorted(subset, key=lambda row: row.scheme)
    schemes = [row.scheme for row in subset]
    x = np.arange(len(schemes))

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.8), sharex=True)

    axes[0].bar(x, [max(row.ber, 0.5 / row.bit_count) for row in subset], color="#4C78A8")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("BER\n(lower is better)")
    axes[0].grid(True, axis="y", which="both", alpha=0.30)

    axes[1].bar(x, [row.useful_bits_per_channel_use for row in subset], color="#59A14F")
    axes[1].set_ylabel("Useful bits /\nchannel use")
    axes[1].grid(True, axis="y", alpha=0.30)

    axes[2].bar(x, [row.channel_uses_per_bit for row in subset], color="#F28E2B")
    axes[2].set_ylabel("Channel uses / bit\n(lower is better)")
    axes[2].grid(True, axis="y", alpha=0.30)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(schemes, rotation=28, ha="right", fontsize=8)

    fig.suptitle(f"Multi-axis tradeoff dashboard at Eb/N0 = {target_ebn0_db:.0f} dB")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(OUT_DIR / "tradeoff_dashboard_0db.png", dpi=180)
    plt.close(fig)


def write_summary(rows: list[SchemeResult]) -> None:
    lines = [
        "# Known Technique Comparison Summary",
        "",
        "This file is generated by `src/known_tech_comparison.py`.",
        "",
        "## Important Interpretation",
        "",
        "The mirror phase trace row is a fallback-oriented method, not a replacement for coherent BPSK or a complete 5G/6G PHY.",
        "It intentionally spends more resource units per information bit to keep low-rate control information decodable under common phase rotation.",
        "Therefore, read the BER plot together with the resource-efficiency plot.",
        "A displayed BER of 0 means no error occurred in this finite Monte Carlo run; it is not a mathematical proof of zero error probability.",
        "",
        "## Results",
        "",
        "| Scheme | Eb/N0 dB | BER | Channel uses / bit | Useful bits / channel use |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.scheme} | {row.ebn0_db:.1f} | {row.ber:.6f} | {row.channel_uses_per_bit:.2f} | {row.useful_bits_per_channel_use:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Schemes",
            "",
            "- `ideal_coherent_bpsk`: ideal carrier-phase reference.",
            "- `differential_bpsk`: known noncoherent adjacent-symbol differential reference.",
            "- `repetition3_bpsk_majority`: simple robust repetition baseline.",
            "- `mirror_phase_trace_rep1_top1`: minimum-resource mirror trace, 1 pair and 1 repetition.",
            "- `mirror_phase_trace_rep1_top4`: intermediate mirror trace, 4 pairs and 1 repetition.",
            "- `mirror_phase_trace_rep3_top4`: conservative robust mirror trace, 4 pairs and 3 repetitions.",
            "",
            "## Caveats",
            "",
            "- The channel model is intentionally compact and reproducible, not a standard-compliant 5G NR link simulator.",
            "- The mirror trace method can incur noise-product and signal-noise cross terms because it multiplies received samples.",
            "- High-SNR coherent detection remains the appropriate primary path when synchronization is reliable.",
            "- The intended use case is fallback control/low-rate signaling under degraded synchronization or channel conditions.",
            "- This simulation uses static conservative settings for reproducibility. Practical implementations may use confidence-based adaptive processing to reduce average computation and resource use, but any such controller must be validated against the target BER and outage requirements.",
        ]
    )
    (OUT_DIR / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_results(rows: list[SchemeResult]) -> None:
    payload = {
        "metadata": {
            "num_bits": DEFAULT_BITS,
            "seed": DEFAULT_SEED,
            "snr_db": SNR_DB.tolist(),
            "zero_ber_note": "BER=0 means zero observed errors in DEFAULT_BITS bits, not a proof of zero error probability.",
            "purpose": "public comparison figures for known baselines and mirror phase trace fallback",
        },
        "results": [asdict(row) for row in rows],
    }
    (OUT_DIR / "comparison_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_comparison()
    save_results(rows)
    write_summary(rows)
    plot_ber(rows)
    plot_efficiency(rows)
    plot_tradeoff_dashboard(rows)
    print("Generated public comparison package outputs:")
    for path in sorted(OUT_DIR.glob("*")):
        print(f"  {path}")


if __name__ == "__main__":
    main()