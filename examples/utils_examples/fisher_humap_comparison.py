"""Compare Fisher humap, posterior humap and htsne on a known mixture.

Draws heterogeneous records from a 5-component composite mixture and keeps
each record's true component as the ground-truth group. One mixture is
fitted with ``humap``'s default Dirichlet-process path and reused by all
three embeddings (never refit), so they differ only in the representation:

- ``humap(..., representation="fisher")``: posteriors plus whitened Fisher
  scores, so records also spread out inside their cluster;
- ``humap(...)``: posteriors only (Hellinger metric);
- ``htsne(...)``: t-SNE with mixture-derived affinities.

Each record is ``(x, y, level, count, tokens)``: two Gaussian fields, a
categorical level, a Poisson count, and a variable-length list of tokens.
The script prints label-agreement metrics and writes a self-contained
interactive page, ``fisher_humap_comparison.html``, that opens in any
browser without a network connection. Hover a point to see the record and
the same record highlighted in all three panels; click a group in the legend
to hide or show it. Run it from the repository root::

    python examples/utils_examples/fisher_humap_comparison.py

It takes about a minute on a laptop.
"""

import argparse
import html
import json
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from dmx.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    GaussianDistribution,
    PoissonDistribution,
    SequenceDistribution,
)
from dmx.utils.automatic import encode_mixture_data
from dmx.utils.htsne import htsne
from dmx.utils.humap import humap

NUM_GROUPS = 5
LEVELS = ("red", "green", "blue", "amber", "violet", "gray")
TOKENS = ("login", "read", "write", "delete", "admin", "query", "export", "sync")
UMAP_KWARGS = {"n_neighbors": 15, "min_dist": 0.1}


def make_components() -> List[CompositeDistribution]:
    """Return the five composite components that generate the records.

    Neighbouring components overlap in the Gaussian fields, so the groups are
    not trivially separable; the categorical, count and token fields carry
    extra, partly shared, evidence.
    """
    centers = [(-6.0, 0.0), (-2.0, 3.0), (2.0, 3.0), (6.0, 0.0), (0.0, -3.5)]
    rates = [1.0, 4.0, 8.0, 14.0, 22.0]
    components = []
    for k, (cx, cy) in enumerate(centers):
        level_probs = np.full(len(LEVELS), 0.4 / (len(LEVELS) - 1))
        level_probs[k] = 0.6
        token_probs = np.full(len(TOKENS), 0.25 / (len(TOKENS) - 2))
        token_probs[k] = token_probs[(k + 3) % len(TOKENS)] = 0.375
        components.append(
            CompositeDistribution(
                (
                    GaussianDistribution(cx, 1.6),
                    GaussianDistribution(cy, 1.0),
                    CategoricalDistribution(dict(zip(LEVELS, level_probs))),
                    PoissonDistribution(rates[k]),
                    SequenceDistribution(
                        CategoricalDistribution(dict(zip(TOKENS, token_probs))),
                        len_dist=PoissonDistribution(2.0 + k),
                    ),
                )
            )
        )
    return components


def sample_with_labels(
    count: int, seed: int
) -> Tuple[List[Tuple[Any, ...]], np.ndarray]:
    """Draw ``count`` records and the true component of each one.

    Args:
        count: Number of records.
        seed: Seed for the component choice and the component samplers.

    Returns:
        The records, shuffled, and their true group labels ``0..4``.
    """
    rng = np.random.RandomState(seed)
    components = make_components()
    weights = np.array([0.15, 0.2, 0.2, 0.25, 0.2])
    labels = rng.choice(NUM_GROUPS, size=count, p=weights)
    samplers = [c.sampler(seed=int(rng.randint(2**31))) for c in components]
    records: List[Tuple[Any, ...]] = [()] * count
    for k in range(NUM_GROUPS):
        rows = np.flatnonzero(labels == k)
        for row, record in zip(rows, samplers[k].sample(len(rows))):
            records[row] = tuple(record)
    return records, labels


def label_agreement(coords: np.ndarray, labels: np.ndarray, k: int = 15) -> float:
    """Return the mean fraction of each point's ``k`` neighbours in its group."""
    distances = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(distances, np.inf)
    neighbours = np.argsort(distances, axis=1)[:, :k]
    return float(np.mean(labels[neighbours] == labels[:, None]))


def silhouette(coords: np.ndarray, labels: np.ndarray) -> float:
    """Return the silhouette score of the true groups in an embedding."""
    distances = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    scores = []
    for i, label in enumerate(labels):
        same = labels == label
        same[i] = False
        if not same.any():
            continue
        inside = distances[i, same].mean()
        outside = min(
            distances[i, labels == other].mean()
            for other in np.unique(labels)
            if other != label
        )
        scores.append((outside - inside) / max(inside, outside))
    return float(np.mean(scores))


def run_embeddings(
    records: Sequence[Any], seed: int
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, float]]:
    """Fit one mixture and embed the records three ways.

    Args:
        records: Heterogeneous records.
        seed: Seed for mixture fitting, UMAP and t-SNE.

    Returns:
        Coordinates by method, each record's most probable fitted component,
        and the seconds each step took.
    """
    seconds: Dict[str, float] = {}
    start = time.perf_counter()
    posterior_coords, model, _, posteriors = humap(
        records,
        seed=seed,
        max_components=20,
        max_its=300,
        verbose=False,
        umap_kwargs={**UMAP_KWARGS, "random_state": seed},
    )
    seconds["mixture fit + posterior humap"] = time.perf_counter() - start

    start = time.perf_counter()
    fisher_coords, _, _, _ = humap(
        records,
        mix_model=model,
        verbose=False,
        umap_kwargs={**UMAP_KWARGS, "random_state": seed},
        representation="fisher",
    )
    seconds["Fisher humap"] = time.perf_counter() - start

    start = time.perf_counter()
    htsne_coords = htsne(records, mix_model=model, seed=seed, verbose=False)
    seconds["htsne"] = time.perf_counter() - start

    fitted = np.argmax(
        np.asarray(model.seq_posterior(encode_mixture_data(records, model))), axis=1
    )
    print(
        f"Fitted mixture: {model.num_components} components "
        f"(true groups: {NUM_GROUPS}); posterior matrix {posteriors.shape}"
    )
    coords = {
        "Fisher humap": np.asarray(fisher_coords),
        "Posterior humap": np.asarray(posterior_coords),
        "htsne": np.asarray(htsne_coords),
    }
    return coords, fitted, seconds


def _record_text(record: Tuple[Any, ...]) -> List[str]:
    """Return a record's fields as short display strings."""
    x, y, level, count, tokens = record
    return [
        f"{x:.2f}",
        f"{y:.2f}",
        str(level),
        str(count),
        " ".join(tokens) if tokens else "(none)",
    ]


def write_page(
    path: str,
    records: Sequence[Tuple[Any, ...]],
    labels: np.ndarray,
    fitted: np.ndarray,
    coords: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
) -> None:
    """Write the interactive comparison page.

    Args:
        path: Output HTML file.
        records: Records, for tooltips and the data table.
        labels: True group of each record.
        fitted: Most probable fitted component of each record.
        coords: 2-D coordinates by method.
        metrics: Label-agreement metrics by method.
    """
    payload = {
        "groups": [f"Group {k + 1}" for k in range(NUM_GROUPS)],
        "fields": ["x", "y", "level", "count", "tokens"],
        "labels": labels.tolist(),
        "fitted": fitted.tolist(),
        "records": [_record_text(record) for record in records],
        "panels": [
            {
                "title": name,
                "metrics": metrics[name],
                "points": np.round(points, 4).tolist(),
            }
            for name, points in coords.items()
        ],
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    page = PAGE.replace("__DATA__", data).replace(
        "__TITLE__", html.escape("Fisher humap vs posterior humap vs htsne")
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)


def main() -> None:
    """Generate data, embed it three ways, report metrics and write the page."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1000, help="number of records")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output", default="fisher_humap_comparison.html")
    args = parser.parse_args()

    records, labels = sample_with_labels(args.n, args.seed)
    print(f"Drew {len(records)} records from {NUM_GROUPS} groups.")
    coords, fitted, seconds = run_embeddings(records, args.seed)

    metrics = {
        name: {
            "kNN label agreement": label_agreement(points, labels),
            "silhouette (true groups)": silhouette(points, labels),
        }
        for name, points in coords.items()
    }
    print(f"\n{'method':18s} {'kNN agreement':>14s} {'silhouette':>11s}")
    for name, values in metrics.items():
        print(
            f"{name:18s} {values['kNN label agreement']:14.3f} "
            f"{values['silhouette (true groups)']:11.3f}"
        )
    print("\n" + ", ".join(f"{step}: {sec:.1f} s" for step, sec in seconds.items()))

    write_page(args.output, records, labels, fitted, coords, metrics)
    print(f"Wrote {args.output}; open it in a browser.")


# The page is plain HTML, CSS and JavaScript with the data inlined, so it
# needs no plotting library and no network access.
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --surface-2: #f3f2ef; --grid: #e4e3df;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #8a8984;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #242423; --grid: #383835;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --surface-2: #242423; --grid: #383835;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
}
body { margin: 0; }
.viz-root {
  background: var(--surface-1); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  padding: 20px 24px 32px; min-height: 100vh; box-sizing: border-box;
}
h1 { font-size: 20px; margin: 0 0 4px; font-weight: 600; }
.lede { color: var(--text-secondary); margin: 0 0 14px; max-width: 900px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  margin-bottom: 14px; }
.legend-item { display: inline-flex; align-items: center; gap: 6px;
  border: 1px solid var(--grid); background: var(--surface-1);
  color: var(--text-primary); border-radius: 999px; padding: 4px 10px;
  cursor: pointer; font: inherit; }
.legend-item[aria-pressed="false"] { opacity: 0.4; }
.legend-item svg { display: block; }
.hint { color: var(--text-muted); font-size: 12px; margin-left: 6px; }
.panels { display: grid; gap: 16px;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
.panel { background: var(--surface-1); border: 1px solid var(--grid);
  border-radius: 8px; padding: 12px 12px 6px; }
.panel h2 { font-size: 15px; margin: 0; font-weight: 600; }
.panel .sub { color: var(--text-secondary); font-size: 12px; margin: 2px 0 6px; }
.panel svg.plot { width: 100%; height: auto; display: block;
  touch-action: none; }
.mark { stroke: var(--surface-1); stroke-width: 1; }
.mark.dim { opacity: 0.12; }
.ring { fill: none; stroke: var(--text-primary); stroke-width: 2; }
.tooltip { position: fixed; pointer-events: none; z-index: 10;
  background: var(--surface-2); color: var(--text-primary);
  border: 1px solid var(--grid); border-radius: 6px; padding: 8px 10px;
  font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,.15); display: none;
  max-width: 280px; }
.tooltip b { font-weight: 600; }
.tooltip .muted { color: var(--text-secondary); }
table { border-collapse: collapse; margin-top: 8px; font-size: 13px; }
th, td { text-align: right; padding: 4px 10px;
  border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
details { margin-top: 18px; }
summary { cursor: pointer; color: var(--text-secondary); }
.scroll { max-height: 360px; overflow: auto; }
</style>
</head>
<body>
<div class="viz-root">
  <h1>__TITLE__</h1>
  <p class="lede">The same records, embedded three ways from one fitted mixture.
  Colors and shapes show the <b>true</b> mixture component each record was
  drawn from; the embeddings never saw these labels. Hover a point to see
  the record and the same record in every panel.</p>
  <div class="controls" id="legend" role="group"
       aria-label="Show or hide groups"></div>
  <div class="panels" id="panels"></div>
  <details open>
    <summary>Metrics (higher is better)</summary>
    <table id="metrics"></table>
  </details>
  <details>
    <summary>Data table</summary>
    <div class="scroll"><table id="records"></table></div>
  </details>
  <div class="tooltip" id="tooltip" role="status"></div>
</div>
<script>
const DATA = __DATA__;
const SHAPES = ["circle", "square", "diamond", "triangle", "cross"];
const SIZE = 340, PAD = 14, R = 4.5, HIT = 14;
const visible = DATA.groups.map(() => true);
const svgNS = "http://www.w3.org/2000/svg";
const tooltip = document.getElementById("tooltip");

function shapePath(shape, x, y, r) {
  const s = r * 0.9, d = r * 1.25, t = r * 1.3;
  switch (shape) {
    case "square":
      return `M${x - s},${y - s}h${2 * s}v${2 * s}h${-2 * s}Z`;
    case "diamond":
      return `M${x},${y - d}L${x + d},${y}L${x},${y + d}L${x - d},${y}Z`;
    case "triangle":
      return `M${x},${y - t}L${x + t},${y + t * 0.8}` +
             `L${x - t},${y + t * 0.8}Z`;
    case "cross": {
      const a = r * 0.45, b = r * 1.15;
      return `M${x - a},${y - b}h${2 * a}v${b - a}h${b - a}v${2 * a}` +
             `h${a - b}v${b - a}h${-2 * a}v${a - b}h${a - b}v${-2 * a}` +
             `h${b - a}Z`;
    }
    default:
      return `M${x - r},${y}a${r},${r} 0 1,0 ${2 * r},0` +
             `a${r},${r} 0 1,0 ${-2 * r},0`;
  }
}

function robustRange(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const at = (q) => sorted[Math.round(q * (sorted.length - 1))];
  const lo = at(0.005), hi = at(0.995), pad = (hi - lo) * 0.04 || 1;
  return [lo - pad, hi + pad];
}

function el(name, attrs, parent) {
  const node = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, value);
  }
  if (parent) parent.appendChild(node);
  return node;
}

function legendSwatch(group) {
  const svg = el("svg", {width: 14, height: 14, viewBox: "0 0 14 14",
                         "aria-hidden": "true"});
  el("path", {d: shapePath(SHAPES[group], 7, 7, 4.5),
              fill: `var(--s${group + 1})`}, svg);
  return svg;
}

const legend = document.getElementById("legend");
DATA.groups.forEach((name, group) => {
  const button = document.createElement("button");
  button.className = "legend-item";
  button.setAttribute("aria-pressed", "true");
  button.appendChild(legendSwatch(group));
  button.appendChild(document.createTextNode(name));
  button.addEventListener("click", () => {
    visible[group] = !visible[group];
    button.setAttribute("aria-pressed", String(visible[group]));
    panels.forEach((panel) => panel.update());
  });
  legend.appendChild(button);
});
const hint = document.createElement("span");
hint.className = "hint";
hint.textContent = "Click a group to hide or show it.";
legend.appendChild(hint);

function makePanel(spec) {
  const card = document.createElement("section");
  card.className = "panel";
  const title = document.createElement("h2");
  title.textContent = spec.title;
  const sub = document.createElement("div");
  sub.className = "sub";
  sub.textContent = Object.entries(spec.metrics)
    .map(([key, value]) => `${key} ${value.toFixed(3)}`).join(" \\u00b7 ");
  card.append(title, sub);
  const svg = el("svg", {class: "plot", viewBox: `0 0 ${SIZE} ${SIZE}`,
                         role: "img", "aria-label": spec.title});
  card.appendChild(svg);
  document.getElementById("panels").appendChild(card);

  // Scale to the central 99% of points so a few far outliers do not squash
  // the rest; outliers are pinned to the frame edge and drawn hollow.
  const xs = spec.points.map((p) => p[0]), ys = spec.points.map((p) => p[1]);
  const [x0, x1] = robustRange(xs), [y0, y1] = robustRange(ys);
  const span = Math.max(x1 - x0, y1 - y0) || 1;
  const ox = (SIZE - 2 * PAD - (x1 - x0) / span * (SIZE - 2 * PAD)) / 2;
  const oy = (SIZE - 2 * PAD - (y1 - y0) / span * (SIZE - 2 * PAD)) / 2;
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const outside = spec.points.map(([x, y]) =>
    x < x0 || x > x1 || y < y0 || y > y1);
  const pos = spec.points.map(([x, y]) => [
    PAD + ox + (clamp(x, x0, x1) - x0) / span * (SIZE - 2 * PAD),
    SIZE - PAD - oy - (clamp(y, y0, y1) - y0) / span * (SIZE - 2 * PAD),
  ]);
  const pinned = outside.filter(Boolean).length;
  if (pinned) {
    sub.textContent += ` \u00b7 ${pinned} outlying point` +
      `${pinned > 1 ? "s" : ""} pinned to the edge (hollow)`;
  }
  el("rect", {x: 0.5, y: 0.5, width: SIZE - 1, height: SIZE - 1,
              fill: "none", stroke: "var(--grid)"}, svg);
  const layer = el("g", {}, svg);
  const marks = pos.map(([x, y], i) => {
    const group = DATA.labels[i];
    const color = `var(--s${group + 1})`;
    return el("path", outside[i]
      ? {d: shapePath(SHAPES[group], x, y, R), class: "mark", fill: "none",
         style: `stroke: ${color}; stroke-width: 1.5`}
      : {d: shapePath(SHAPES[group], x, y, R), class: "mark", fill: color},
      layer);
  });
  const ring = el("circle", {class: "ring", r: R + 4, visibility: "hidden"},
                  svg);

  function toLocal(event) {
    const box = svg.getBoundingClientRect();
    return [(event.clientX - box.left) * SIZE / box.width,
            (event.clientY - box.top) * SIZE / box.height,
            box.width / SIZE];
  }
  function nearest(event) {
    const [mx, my, scale] = toLocal(event);
    let best = -1, bestD = (HIT / scale) ** 2;
    pos.forEach(([x, y], i) => {
      if (!visible[DATA.labels[i]]) return;
      const d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    });
    return best;
  }
  svg.addEventListener("pointermove", (event) => {
    const i = nearest(event);
    if (i < 0) { highlight(-1); return; }
    highlight(i);
    showTooltip(i, event);
  });
  svg.addEventListener("pointerleave", () => highlight(-1));

  return {
    update() {
      marks.forEach((mark, i) => {
        mark.style.display = visible[DATA.labels[i]] ? "" : "none";
      });
    },
    focus(i) {
      if (i < 0 || !visible[DATA.labels[i]]) {
        ring.setAttribute("visibility", "hidden");
        return;
      }
      ring.setAttribute("cx", pos[i][0]);
      ring.setAttribute("cy", pos[i][1]);
      ring.setAttribute("visibility", "visible");
    },
  };
}

const panels = DATA.panels.map(makePanel);

function highlight(i) {
  panels.forEach((panel) => panel.focus(i));
  if (i < 0) tooltip.style.display = "none";
}

function showTooltip(i, event) {
  const fields = DATA.fields.map((name, j) =>
    `<div><span class="muted">${name}</span> ${escape(DATA.records[i][j])}` +
    `</div>`).join("");
  tooltip.innerHTML =
    `<div><b>${DATA.groups[DATA.labels[i]]}</b> ` +
    `<span class="muted">(record ${i}, fitted component ` +
    `${DATA.fitted[i] + 1})</span></div>${fields}`;
  tooltip.style.display = "block";
  const box = tooltip.getBoundingClientRect();
  let left = event.clientX + 14, top = event.clientY + 14;
  if (left + box.width > window.innerWidth - 8) {
    left = event.clientX - box.width - 14;
  }
  if (top + box.height > window.innerHeight - 8) {
    top = event.clientY - box.height - 14;
  }
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

function escape(text) {
  return String(text).replace(/[&<>"]/g, (c) =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[c]);
}

const metricNames = Object.keys(DATA.panels[0].metrics);
document.getElementById("metrics").innerHTML =
  `<tr><th>method</th>${metricNames.map((m) => `<th>${m}</th>`).join("")}` +
  `</tr>` + DATA.panels.map((panel) =>
    `<tr><td>${panel.title}</td>` + metricNames.map((m) =>
      `<td>${panel.metrics[m].toFixed(3)}</td>`).join("") + `</tr>`
  ).join("");

const header = ["record", "true group", "fitted", ...DATA.fields,
  ...DATA.panels.flatMap((p) => [`${p.title} x`, `${p.title} y`])];
document.getElementById("records").innerHTML =
  `<tr>${header.map((h) => `<th>${h}</th>`).join("")}</tr>` +
  DATA.records.map((record, i) => `<tr><td>${i}</td>` +
    `<td>${DATA.labels[i] + 1}</td><td>${DATA.fitted[i] + 1}</td>` +
    record.map((v) => `<td>${escape(v)}</td>`).join("") +
    DATA.panels.map((p) => `<td>${p.points[i][0].toFixed(2)}</td>` +
      `<td>${p.points[i][1].toFixed(2)}</td>`).join("") + `</tr>`
  ).join("");
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
