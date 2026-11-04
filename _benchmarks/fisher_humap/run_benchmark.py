"""Benchmark Fisher humap against posterior humap and a raw-feature baseline.

Run from the repository root (the ``viz`` extra provides matplotlib)::

    PYTHONPATH=.:src python _benchmarks/fisher_humap/run_benchmark.py quality
    PYTHONPATH=.:src python _benchmarks/fisher_humap/run_benchmark.py approximation
    PYTHONPATH=.:src python _benchmarks/fisher_humap/run_benchmark.py scaling
    NUMBA_DISABLE_JIT=1 PYTHONPATH=.:src mpiexec -n 4 \\
        python _benchmarks/fisher_humap/run_benchmark.py mpi --n 1000000
    PYTHONPATH=.:src python _benchmarks/fisher_humap/run_benchmark.py figures
    PYTHONPATH=.:src python _benchmarks/fisher_humap/run_benchmark.py tables

Results are written to ``results/*.json`` and figures to ``figures/*.png``
next to this file. The data generator, reference metric and quality metrics
live in ``tests/utils/fisher_benchmark.py``.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

warnings.filterwarnings("ignore", message=".*n_jobs value.*")


def _save(name: str, payload: Any) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")


def _load(name: str) -> Any:
    return json.loads((RESULTS / f"{name}.json").read_text())


def _environment() -> Dict[str, Any]:
    import umap  # type: ignore[import-untyped]

    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        "numpy": np.__version__,
        "umap": umap.__version__,
    }


# --- Quality ------------------------------------------------------------------


def _methods() -> Dict[str, Dict[str, Any]]:
    from dmx.utils.fisher import FisherScoreOptions

    balanced = FisherScoreOptions(balance_fields=True)
    rv: Dict[str, Dict[str, Any]] = {"posterior": {}}
    for label, scale in (("default", None), ("0.02", 0.02), ("1.0", 1.0)):
        rv[f"fisher c={label}"] = {"representation": "fisher", "fisher_scale": scale}
    for label, scale in (("default", None), ("0.02", 0.02), ("1.0", 1.0)):
        rv[f"fisher balanced c={label}"] = {
            "representation": "fisher",
            "fisher_scale": scale,
            "fisher_options": balanced,
        }
    return rv


def run_quality(count: int, seeds: List[int]) -> None:
    """Compare embeddings on ground-truth metrics for two kinds of model."""
    import umap  # type: ignore[import-untyped]

    from dmx.utils.humap import humap
    from tests.utils.fisher_benchmark import (
        embedding_report,
        generate_ring_clusters,
        oracle_mixture,
        raw_features,
    )

    rows: List[Dict[str, Any]] = []
    layouts: Dict[str, Any] = {}
    for seed in seeds:
        data = generate_ring_clusters(count, seed=seed)
        _, dpm, _, _ = humap(
            data.records,
            seed=seed,
            max_components=20,
            max_its=200,
            verbose=False,
            umap_kwargs={"random_state": seed},
        )
        models = {"oracle": oracle_mixture(data), "dpm": dpm}
        for model_name, model in models.items():
            for method, options in _methods().items():
                start = time.perf_counter()
                embedding, *_ = humap(
                    data.records,
                    mix_model=model,
                    verbose=False,
                    umap_kwargs={"random_state": seed},
                    **options,
                )
                rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "components": int(model.num_components),
                        "method": method,
                        "seconds": time.perf_counter() - start,
                        **embedding_report(embedding, data),
                    }
                )
                if seed == seeds[0] and model_name == "oracle":
                    layouts[method] = np.round(embedding, 3).tolist()
            print(f"seed {seed} {model_name} done", flush=True)
        raw = umap.UMAP(random_state=seed).fit_transform(
            raw_features(data.records, data.spec.levels)
        )
        for model_name in models:
            rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "components": 0,
                    "method": "raw UMAP",
                    "seconds": float("nan"),
                    **embedding_report(raw, data),
                }
            )
        if seed == seeds[0]:
            layouts["raw UMAP"] = np.round(raw, 3).tolist()
            layouts["_angle"] = np.round(data.angle, 4).tolist()
            layouts["_cluster"] = data.cluster.tolist()
    _save(
        "quality",
        {"count": count, "seeds": seeds, "rows": rows, "env": _environment()},
    )
    _save("layouts", layouts)


# --- Approximation ------------------------------------------------------------


def run_approximation(count: int, seed: int) -> None:
    """Compare the Euclidean features with the exact Hellinger reference."""
    from dmx.utils.fisher import fisher_embedding
    from tests.utils.fisher_benchmark import (
        fisher_hellinger_reference,
        generate_ring_clusters,
        knn_graph_agreement,
        oracle_mixture,
    )

    data = generate_ring_clusters(count, seed=seed)
    model = oracle_mixture(data)
    default = fisher_embedding(model, data.records).scale
    same = (data.cluster[:, None] == data.cluster[None, :]) & ~np.eye(count, dtype=bool)
    rows = []
    for scale in (1e-4, 1e-3, 1e-2, 0.05, default, 0.2):
        reference = fisher_hellinger_reference(model, data.records, scale=scale)
        for top_m in (None, 3, 1):
            phi = fisher_embedding(
                model, data.records, scale=scale, top_m=top_m, dtype=np.float64
            ).matrix.toarray()
            squared = ((phi[:, None, :] - phi[None, :, :]) ** 2).sum(axis=-1)
            ratio = 2.0 * reference[same] / squared[same]
            rows.append(
                {
                    "scale": scale,
                    "is_default": scale == default,
                    "top_m": "all" if top_m is None else top_m,
                    "knn_agreement": knn_graph_agreement(phi, reference),
                    "ratio_median": float(np.median(ratio)),
                    "ratio_q05": float(np.quantile(ratio, 0.05)),
                    "ratio_q95": float(np.quantile(ratio, 0.95)),
                }
            )
        print(f"scale {scale:.3g} done", flush=True)
    _save("approximation", {"count": count, "seed": seed, "rows": rows})


# --- Scaling ------------------------------------------------------------------


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def _measure_stage(stage: str, count: int) -> Dict[str, Any]:
    """Run one stage in this process and report time and peak memory."""
    import umap  # type: ignore[import-untyped]

    from dmx.utils.fisher import fisher_embedding
    from tests.utils.fisher_benchmark import generate_ring_clusters, oracle_mixture

    model = oracle_mixture(generate_ring_clusters(10_000, seed=0))
    data = generate_ring_clusters(count, seed=1)
    features: Any = None
    if stage.startswith("umap"):
        if stage == "umap-fisher":
            features = fisher_embedding(model, data.records, chunk_size=50_000).matrix
        else:
            encoded = model.dist_to_encoder().seq_encode(data.records)
            # float64, as humap passes posteriors to UMAP.
            features = model.seq_posterior(encoded)
    baseline = _peak_rss_mb()
    start = time.perf_counter()
    if stage == "embed":
        result = fisher_embedding(model, data.records, chunk_size=50_000)
        extra = {
            "feature_dim": int(result.matrix.shape[1]),
            "nnz": int(result.matrix.nnz),
        }
    elif stage == "posterior":
        encoded = model.dist_to_encoder().seq_encode(data.records)
        model.seq_posterior(encoded)
        extra = {}
    else:
        metric = "euclidean" if stage == "umap-fisher" else "hellinger"
        # The spectral initialization fails on near-duplicate posteriors;
        # "umap-posterior-random-init" times the fit without it.
        init = "random" if stage.endswith("random-init") else "spectral"
        umap.UMAP(metric=metric, init=init, random_state=None).fit_transform(features)
        extra = {}
    seconds = time.perf_counter() - start
    return {
        "stage": stage,
        "count": count,
        "seconds": seconds,
        "peak_rss_mb": _peak_rss_mb(),
        "stage_peak_increase_mb": max(_peak_rss_mb() - baseline, 0.0),
        **extra,
    }


STAGES = (
    "posterior",
    "embed",
    "umap-posterior",
    "umap-posterior-random-init",
    "umap-fisher",
)


def run_scaling(
    counts: List[int], umap_budget: float, stages: Optional[List[str]] = None
) -> None:
    """Measure each stage at each size in a fresh subprocess.

    With ``stages``, only those stages are measured and their rows replace
    any earlier rows for the same stage and size in ``results/scaling.json``.
    """
    previous = (
        _load("scaling")["rows"]
        if stages and (RESULTS / "scaling.json").exists()
        else []
    )
    rows = []
    for count in counts:
        for stage in stages or STAGES:
            command = [sys.executable, __file__, "stage", stage, "--n", str(count)]
            timeout = umap_budget if stage.startswith("umap") else None
            try:
                output = subprocess.run(
                    command, capture_output=True, text=True, check=True, timeout=timeout
                ).stdout
                row = json.loads(output.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                row = {"stage": stage, "count": count, "timed_out_after": umap_budget}
            except subprocess.CalledProcessError as err:
                lines = [line for line in err.stderr.splitlines() if line.strip()]
                row = {"stage": stage, "count": count, "error": lines[-1][:200]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    done = {(r["stage"], r["count"]) for r in rows}
    kept = [r for r in previous if (r["stage"], r["count"]) not in done]
    order = {stage: i for i, stage in enumerate(STAGES)}
    rows = sorted(kept + rows, key=lambda r: (r["count"], order.get(r["stage"], 99)))
    _save("scaling", {"rows": rows, "env": _environment()})


def run_mpi(count: int) -> None:
    """Time fisher_embedding_mpi with records split across ranks."""
    from mpi4py import MPI  # pylint: disable=no-name-in-module

    from dmx.mpi4py.utils.fisher import fisher_embedding_mpi
    from tests.utils.fisher_benchmark import generate_ring_clusters, oracle_mixture

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    model = oracle_mixture(generate_ring_clusters(10_000, seed=0))
    share = count // size + (1 if rank < count % size else 0)
    local = generate_ring_clusters(share, seed=100 + rank).records
    comm.Barrier()
    start = time.perf_counter()
    result = fisher_embedding_mpi(model, local, comm, chunk_size=50_000)
    seconds = comm.reduce(time.perf_counter() - start, op=MPI.MAX, root=0)
    peak = comm.reduce(_peak_rss_mb(), op=MPI.MAX, root=0)
    if rank == 0:
        assert result is not None
        row = {
            "stage": "embed-mpi",
            "count": count,
            "ranks": size,
            "seconds": seconds,
            "max_rank_peak_rss_mb": peak,
            "rows": int(result.matrix.shape[0]),
        }
        _save(f"mpi_{size}", row)
        print(json.dumps(row))


# --- Figures and tables -------------------------------------------------------

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )


def _summary(rows: List[Dict[str, Any]], model: str, metric: str) -> Dict[str, Any]:
    by_method: Dict[str, List[float]] = {}
    for row in rows:
        if row["model"] == model:
            by_method.setdefault(row["method"], []).append(row[metric])
    return {m: (float(np.mean(v)), float(np.std(v))) for m, v in by_method.items()}


METRICS = (
    ("within_cluster_rank_corr", "Within-cluster rank corr."),
    ("knn_recall", "kNN recall (k=15)"),
    ("trustworthiness", "Trustworthiness"),
    ("continuity", "Continuity"),
)


def make_figures() -> None:
    """Draw one figure per comparison from the saved results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    FIGURES.mkdir(exist_ok=True)

    # 1. Layouts: one panel per method, colored by the true ring angle.
    layouts = _load("layouts")
    angle = np.asarray(layouts["_angle"])
    panels = ["posterior", "fisher c=default", "fisher balanced c=default", "raw UMAP"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for axis, name in zip(axes, panels):
        points = np.asarray(layouts[name])
        scatter = axis.scatter(
            points[:, 0],
            points[:, 1],
            c=angle,
            cmap="twilight",
            vmin=0,
            vmax=2 * np.pi,
            s=8,
            linewidths=0,
        )
        axis.set_title(name, loc="left")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
    bar = fig.colorbar(scatter, ax=axes, fraction=0.015, pad=0.01)
    bar.set_label("true angle on the ring (radians)")
    bar.outline.set_visible(False)
    fig.suptitle(
        "Embeddings of the same records (oracle model, seed 1)",
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.savefig(FIGURES / "layouts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Quality: dot plot of mean +- sd per method, one panel per metric.
    quality = _load("quality")
    for model in ("oracle", "dpm"):
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
        order = list(_methods()) + ["raw UMAP"]
        positions = np.arange(len(order))[::-1]
        for axis, (metric, title) in zip(axes, METRICS):
            summary = _summary(quality["rows"], model, metric)
            means = [summary[m][0] for m in order]
            sds = [summary[m][1] for m in order]
            axis.errorbar(
                means,
                positions,
                xerr=sds,
                fmt="o",
                color=SERIES[0],
                ecolor=INK_2,
                elinewidth=1,
                markersize=6,
                capsize=0,
            )
            axis.axvline(
                summary["posterior"][0], color=INK_2, linewidth=1, linestyle=":"
            )
            axis.set_title(title, loc="left")
        axes[0].set_yticks(positions)
        axes[0].set_yticklabels(order)
        fig.suptitle(
            f"Embedding quality, {'oracle' if model == 'oracle' else 'fitted DPM'} "
            f"model (mean ± sd over {len(quality['seeds'])} seeds; "
            "dotted line = posterior humap)",
            x=0.01,
            ha="left",
            color=INK,
        )
        fig.savefig(FIGURES / f"quality_{model}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. Approximation: kNN agreement with the exact reference vs scale.
    approximation = _load("approximation")
    rows = [r for r in approximation["rows"] if r["top_m"] == "all"]
    same_for_all_top_m = all(
        r["knn_agreement"] == other["knn_agreement"]
        for r in rows
        for other in approximation["rows"]
        if other["scale"] == r["scale"]
    )
    scales = [r["scale"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(
        scales,
        [r["knn_agreement"] for r in rows],
        marker="o",
        markersize=5,
        linewidth=2,
        color=SERIES[0],
    )
    axes[1].plot(
        scales,
        [r["ratio_median"] for r in rows],
        marker="o",
        markersize=5,
        linewidth=2,
        color=SERIES[0],
    )
    axes[1].fill_between(
        scales,
        [r["ratio_q05"] for r in rows],
        [r["ratio_q95"] for r in rows],
        color=SERIES[0],
        alpha=0.15,
        linewidth=0,
    )
    default = next(r["scale"] for r in rows if r["is_default"])
    for axis in axes:
        axis.set_xscale("log")
        axis.axvline(default, color=INK_2, linewidth=1, linestyle=":")
        axis.set_xlabel("scale c (dotted line = default 1/p_eff)")
    note = (
        "\n(identical for top_m = all, 3, 1)"
        if same_for_all_top_m
        else "\n(top_m = all)"
    )
    axes[0].set_title("kNN agreement of phi with exact Hellinger" + note, loc="left")
    axes[0].set_ylabel("shared neighbours (k=15)")
    axes[1].set_title(
        "2H² / ||Δphi||² within clusters\n(median and 5–95% band)", loc="left"
    )
    fig.savefig(FIGURES / "approximation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Scaling: time and memory vs N, as two separate charts.
    if not (RESULTS / "scaling.json").exists():
        print(f"figures written to {FIGURES} (no scaling results yet)")
        return
    scaling = _load("scaling")
    rows = [r for r in scaling["rows"] if "seconds" in r]
    stages = (
        ("embed", "Fisher features"),
        ("umap-fisher", "UMAP, Fisher"),
        ("umap-posterior-random-init", "UMAP, posteriors\n(random init)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.subplots_adjust(wspace=0.45)
    for index, (stage, label) in enumerate(stages):
        points = sorted((r["count"], r) for r in rows if r["stage"] == stage)
        if not points:
            continue
        counts = [p[0] for p in points]
        for axis, key in zip(axes, ("seconds", "peak_rss_mb")):
            values = [p[1][key] for p in points]
            axis.plot(
                counts,
                values,
                marker="o",
                markersize=5,
                linewidth=2,
                color=SERIES[index],
            )
            axis.annotate(
                label,
                (counts[-1], values[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=INK_2,
                fontsize=8,
            )
    for axis, title in zip(axes, ("Wall time (s)", "Peak process memory (MB)")):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("records N")
        axis.set_title(title, loc="left")
        axis.margins(x=0.35)
    fig.savefig(FIGURES / "scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figures written to {FIGURES}")


def print_tables() -> None:
    """Print markdown tables of the saved results."""
    quality = _load("quality")
    order = list(_methods()) + ["raw UMAP"]
    for model in ("oracle", "dpm"):
        components = sorted(
            {r["components"] for r in quality["rows"] if r["model"] == model} - {0}
        )
        print(f"\n### {model} model (components: {components})\n")
        print("| method | " + " | ".join(t for _, t in METRICS) + " |")
        print("|---|" + "---:|" * len(METRICS))
        summaries = {m: _summary(quality["rows"], model, m) for m, _ in METRICS}
        for method in order:
            cells = [
                f"{summaries[m][method][0]:.3f} ± {summaries[m][method][1]:.3f}"
                for m, _ in METRICS
            ]
            print(f"| {method} | " + " | ".join(cells) + " |")
    approximation = _load("approximation")
    print("\n### approximation\n")
    print("| scale c | top_m | kNN agreement | median 2H²/‖Δφ‖² | 5–95% |")
    print("|---:|---:|---:|---:|---|")
    for r in approximation["rows"]:
        mark = " (default)" if r["is_default"] else ""
        print(
            f"| {r['scale']:.3g}{mark} | {r['top_m']} | {r['knn_agreement']:.3f} | "
            f"{r['ratio_median']:.3f} | {r['ratio_q05']:.3f}–{r['ratio_q95']:.3f} |"
        )
    scaling = _load("scaling")
    print("\n### scaling\n")
    print("| N | stage | wall time (s) | peak RSS (MB) | stage peak increase (MB) |")
    print("|---:|---|---:|---:|---:|")
    for r in scaling["rows"]:
        if "seconds" in r:
            print(
                f"| {r['count']:,} | {r['stage']} | {r['seconds']:.1f} | "
                f"{r['peak_rss_mb']:.0f} | {r['stage_peak_increase_mb']:.0f} |"
            )
        elif "error" in r:
            reason = "NaN in spectral init" if "NaN" in r["error"] else r["error"][:60]
            print(f"| {r['count']:,} | {r['stage']} | failed ({reason}) | – | – |")
        else:
            print(
                f"| {r['count']:,} | {r['stage']} | > {r['timed_out_after']:.0f} "
                "(stopped) | – | – |"
            )
    for path in sorted(RESULTS.glob("mpi_*.json")):
        r = json.loads(path.read_text())
        print(
            f"| {r['count']:,} | Fisher features, MPI {r['ranks']} ranks | "
            f"{r['seconds']:.1f} | {r['max_rank_peak_rss_mb']:.0f} (max rank) | – |"
        )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=[
            "quality",
            "approximation",
            "scaling",
            "stage",
            "mpi",
            "figures",
            "tables",
        ],
    )
    parser.add_argument("stage", nargs="?", help="Stage name for 'stage'.")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--counts", type=int, nargs="+", default=[10_000, 100_000, 1_000_000]
    )
    parser.add_argument("--umap-budget", type=float, default=1800.0)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=None)
    args = parser.parse_args(argv)
    commands: Dict[str, Callable[[], None]] = {
        "quality": lambda: run_quality(args.n or 2000, args.seeds),
        "approximation": lambda: run_approximation(args.n or 1000, args.seeds[0]),
        "scaling": lambda: run_scaling(args.counts, args.umap_budget, args.stages),
        "stage": lambda: print(json.dumps(_measure_stage(args.stage, args.n))),
        "mpi": lambda: run_mpi(args.n or 1_000_000),
        "figures": make_figures,
        "tables": print_tables,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
