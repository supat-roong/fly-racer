"""Training curves: episode score vs training episodes, one line per run, as a standalone HTML page.

    python -m flyracer.plot runs/fly-real-1 runs/fly-synthetic-1 --out runs/training_curves.html --png

Uses each run's episodes.csv (every episode) when present, else the rolling
mean of 20 episodes from progress.csv. Note that training episodes stop early
when the car stalls, so they score lower than full evaluation episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

ROLL = 20


def load_run(run_dir: Path) -> dict:
    ep_file = run_dir / "episodes.csv"
    raw = []
    if ep_file.exists():
        with open(ep_file) as f:
            raw = [[int(r["episode"]), float(r["score"])] for r in csv.DictReader(f)]
        mean = []
        for i in range(len(raw)):
            window = raw[max(0, i - ROLL + 1): i + 1]
            mean.append([raw[i][0], round(sum(s for _, s in window) / len(window), 1)])
    else:
        by_ep = {}
        with open(run_dir / "progress.csv") as f:
            for r in csv.DictReader(f):
                if r["return_mean"] != "nan" and int(r["episodes"]) > 0:
                    by_ep[int(r["episodes"])] = float(r["return_mean"])
        mean = [[e, round(v, 1)] for e, v in sorted(by_ep.items())]
    args = json.loads((run_dir / "args.json").read_text())
    conn = args.get("connectome")
    label = f"{run_dir.name} ({args['core']}, {'real MaleCNS' if conn else 'synthetic'} wiring)" \
        if args["core"] != "mlp" else f"{run_dir.name} (mlp)"
    return {"name": run_dir.name, "label": label, "mean": mean, "raw": [[e, round(s, 1)] for e, s in raw]}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fly Racer Training</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f3f0; --surface-1: #fcfcfb;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7974;
  --grid: #e6e5e1; --axis: #b9b8b3; --ref: #8a8984;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
    --grid: #2c2c2a; --axis: #4a4a46; --ref: #77766f;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
  --grid: #2c2c2a; --axis: #4a4a46; --ref: #77766f;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 24px 16px 40px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0 0 16px; }
.card { background: var(--surface-1); border-radius: 12px; padding: 16px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 0 0 8px; color: var(--text-secondary); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
.legend i.dash { height: 0; border-top: 2px dashed var(--ref); }
.chart { position: relative; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--text-muted); font-size: 12px; }
.tip { position: absolute; pointer-events: none; background: var(--surface-0); color: var(--text-primary);
  border: 1px solid var(--grid); border-radius: 8px; padding: 8px 10px; font-size: 13px;
  box-shadow: 0 4px 14px rgba(0,0,0,.12); white-space: nowrap; display: none; }
.tip .row { display: flex; align-items: center; gap: 6px; }
.tip i { width: 10px; height: 3px; border-radius: 2px; display: inline-block; }
.note { color: var(--text-muted); font-size: 12.5px; margin: 10px 0 0; }
details { margin-top: 14px; }
summary { cursor: pointer; color: var(--text-secondary); }
.tablewrap { overflow-x: auto; max-height: 320px; margin-top: 8px; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
th, td { padding: 4px 12px; text-align: right; border-bottom: 1px solid var(--grid); }
th { color: var(--text-secondary); font-weight: 600; position: sticky; top: 0; background: var(--surface-1); }
html.static details { display: none; }
</style></head>
<body><main>
<h1>Fly Racer: training score</h1>
<p class="sub">CarRacing-v3 score per training episode, rolling mean of 20 episodes</p>
<div class="card">
  <div class="legend" id="legend"></div>
  <div class="chart" id="chart"><div class="tip" id="tip"></div></div>
  <p class="note">Training episodes end early when the car stalls for 50 decisions, so they score below full evaluation episodes. The dashed line marks 900, the score at which CarRacing counts as solved.</p>
  <details><summary>Show data table</summary><div class="tablewrap" id="table"></div></details>
</div>
</main>
<script>
if (location.hash === "#static") document.documentElement.classList.add("static");
const RUNS = __DATA__;
const SOLVED = 900;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const W = 900, H = 440, M = {l: 52, r: 150, t: 12, b: 40};
const iw = W - M.l - M.r, ih = H - M.t - M.b;
const xmax = Math.max(...RUNS.flatMap(r => r.mean.map(p => p[0])));
const allY = RUNS.flatMap(r => r.mean.map(p => p[1])).concat(RUNS.flatMap(r => r.raw.map(p => p[1])));
const ymin = Math.min(-100, Math.floor(Math.min(...allY) / 100) * 100), ymax = 1000;
const X = v => M.l + v / xmax * iw, Y = v => M.t + (1 - (v - ymin) / (ymax - ymin)) * ih;
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); parent && parent.appendChild(e); return e; };

const chart = document.getElementById("chart"), tip = document.getElementById("tip");
const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img",
  "aria-label": "Line chart of training score versus episodes for each run"});
chart.insertBefore(svg, tip);

function draw() {
  svg.replaceChildren();
  const g = el("g", {}, svg);
  for (let v = ymin; v <= ymax; v += 100) {
    el("line", {x1: M.l, x2: M.l + iw, y1: Y(v), y2: Y(v), stroke: css(v === 0 ? "--axis" : "--grid"), "stroke-width": 1}, g);
    if (v % 200 === 0) { const t = el("text", {x: M.l - 8, y: Y(v) + 4, "text-anchor": "end"}, g); t.textContent = v; }
  }
  const step = xmax > 2000 ? 500 : xmax > 800 ? 250 : 100;
  for (let v = 0; v <= xmax; v += step) {
    const t = el("text", {x: X(v), y: M.t + ih + 20, "text-anchor": "middle"}, g); t.textContent = v.toLocaleString();
  }
  const xl = el("text", {x: M.l + iw / 2, y: H - 2, "text-anchor": "middle"}, g); xl.textContent = "training episodes";
  el("line", {x1: M.l, x2: M.l + iw, y1: Y(SOLVED), y2: Y(SOLVED), stroke: css("--ref"), "stroke-width": 1.5, "stroke-dasharray": "5 4"}, g);
  const st = el("text", {x: M.l + 6, y: Y(SOLVED) - 6}, g); st.textContent = "solved (900)";

  const ends = [];
  RUNS.forEach((r, i) => {
    const c = css(`--series-${i + 1}`);
    r.raw.forEach(p => el("circle", {cx: X(p[0]), cy: Y(p[1]), r: 1.6, fill: c, "fill-opacity": 0.18}, g));
    el("path", {d: r.mean.map((p, j) => (j ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join(""),
      fill: "none", stroke: c, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round"}, g);
    const last = r.mean[r.mean.length - 1];
    ends.push({x: X(last[0]), y: Y(last[1]), c, name: r.name, v: last[1]});
  });
  // Direct labels at the line ends, nudged apart so they never overlap.
  // Labels sit at each line's own end (runs differ in length); a surface-colored halo
  // keeps them legible over other lines. Labels ending at the same x are nudged apart.
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++)
    for (let j = 0; j < i; j++)
      if (Math.abs(ends[i].x - ends[j].x) < 120 && ends[i].y - ends[j].y < 18) ends[i].y = ends[j].y + 18;
  ends.forEach(e => { const t = el("text", {x: e.x + 8, y: e.y + 4, "paint-order": "stroke",
      stroke: css("--surface-1"), "stroke-width": 4, "stroke-linejoin": "round"}, g);
    t.style.fill = css("--text-primary"); t.style.fontWeight = 600; t.textContent = `${e.name} ${e.v.toFixed(0)}`; });

  const cross = el("line", {y1: M.t, y2: M.t + ih, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden"}, g);
  const dots = RUNS.map((r, i) => el("circle", {r: 5, fill: css(`--series-${i + 1}`), stroke: css("--surface-1"),
    "stroke-width": 2, visibility: "hidden"}, g));
  const hit = el("rect", {x: M.l, y: M.t, width: iw, height: ih, fill: "transparent"}, g);
  const nearest = (arr, x) => { let lo = 0, hi = arr.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m][0] < x) lo = m + 1; else hi = m; }
    return lo > 0 && Math.abs(arr[lo - 1][0] - x) < Math.abs(arr[lo][0] - x) ? arr[lo - 1] : arr[lo]; };
  hit.addEventListener("pointermove", ev => {
    const box = svg.getBoundingClientRect(), sx = W / box.width;
    const ep = Math.max(0, Math.min(xmax, ((ev.clientX - box.left) * sx - M.l) / iw * xmax));
    cross.setAttribute("x1", X(ep)); cross.setAttribute("x2", X(ep)); cross.setAttribute("visibility", "visible");
    let rows = `<div style="color:var(--text-secondary);margin-bottom:4px">episode ${Math.round(ep).toLocaleString()}</div>`;
    RUNS.forEach((r, i) => { const p = nearest(r.mean, ep);
      dots[i].setAttribute("cx", X(p[0])); dots[i].setAttribute("cy", Y(p[1])); dots[i].setAttribute("visibility", "visible");
      rows += `<div class="row"><i style="background:var(--series-${i + 1})"></i>${r.name}: <b>${p[1].toFixed(1)}</b></div>`; });
    tip.innerHTML = rows; tip.style.display = "block";
    const px = (ev.clientX - chart.getBoundingClientRect().left);
    tip.style.left = Math.min(px + 14, chart.clientWidth - tip.offsetWidth - 4) + "px";
    tip.style.top = "8px";
  });
  hit.addEventListener("pointerleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden");
    dots.forEach(d => d.setAttribute("visibility", "hidden")); });
}

document.getElementById("legend").innerHTML = RUNS.map((r, i) =>
  `<span><i style="background:var(--series-${i + 1})"></i>${r.label}</span>`).join("") +
  `<span><i class="dash"></i>solved (900)</span>`;

const eps = [...new Set(RUNS.flatMap(r => r.mean.map(p => p[0])))].sort((a, b) => a - b);
const look = RUNS.map(r => new Map(r.mean.map(p => [p[0], p[1]])));
document.getElementById("table").innerHTML = "<table><thead><tr><th>episode</th>" +
  RUNS.map(r => `<th>${r.name}</th>`).join("") + "</tr></thead><tbody>" +
  eps.map(e => `<tr><td>${e}</td>` + look.map(m => `<td>${m.has(e) ? m.get(e).toFixed(1) : ""}</td>`).join("") + "</tr>").join("") +
  "</tbody></table>";

draw();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
</script></body></html>
"""


CHROME_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser",
)


def save_png(html: Path, png: Path) -> None:
    """Screenshot the chart page at 2x with headless Chrome."""
    chrome = next((c for c in CHROME_PATHS if Path(c).exists() or shutil.which(c)), None)
    if chrome is None:
        raise SystemExit("--png needs Google Chrome or Chromium installed")
    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
         "--window-size=960,720", f"--screenshot={png.resolve()}", html.resolve().as_uri() + "#static"],
        check=True, capture_output=True,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories (max 4)")
    p.add_argument("--out", default="runs/training_curves.html")
    p.add_argument("--png", action="store_true", help="also save a PNG next to the HTML (needs Chrome)")
    args = p.parse_args(argv)
    if len(args.runs) > 4:
        p.error("at most 4 runs per chart")
    runs = [load_run(Path(r)) for r in args.runs]
    Path(args.out).write_text(PAGE.replace("__DATA__", json.dumps(runs)))
    print(f"saved {args.out}")
    if args.png:
        png = Path(args.out).with_suffix(".png")
        save_png(Path(args.out), png)
        print(f"saved {png}")


if __name__ == "__main__":
    main()
