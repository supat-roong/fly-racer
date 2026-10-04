"""Extract a Tier S subgraph of the MaleCNS connectome from neuPrint.

Needs a neuPrint token: log in at https://neuprint.janelia.org, open Account,
copy the auth token, and put it in the project's .env (gitignored):

    NEUPRINT_APPLICATION_CREDENTIALS=<token>

or export it in the shell. Then

    python -m flyracer.connectome.extract --config configs/tier_s.yaml
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
import yaml

from .schema import CORE, INPUT, OUTPUT, Connectome

INHIBITORY_NT = ("gaba", "glu")  # glutamate is mostly inhibitory in the fly CNS (GluCl)
NT_COLUMNS = ("consensusNt", "predictedNt", "celltypePredictedNt")
FALLBACK_TYPE = "?"


def nt_sign(nt) -> int:
    if isinstance(nt, str) and nt.lower().startswith(INHIBITORY_NT):
        return -1
    return 1


def assemble(neurons: pd.DataFrame, conns: pd.DataFrame, meta: dict) -> Connectome:
    """Build a Connectome from neuPrint-style dataframes.

    neurons: bodyId, type, group (INPUT/CORE/OUTPUT), and optionally somaSide,
             somaLocation ([x, y, z] or NaN) and one of NT_COLUMNS.
    conns:   bodyId_pre, bodyId_post, weight (may repeat per ROI; summed here).
    """
    neurons = neurons.drop_duplicates("bodyId").reset_index(drop=True)
    index = pd.Series(np.arange(len(neurons)), index=neurons["bodyId"].to_numpy())

    conns = conns[conns["bodyId_pre"].isin(index.index) & conns["bodyId_post"].isin(index.index)]
    conns = conns.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    conns = conns[conns["bodyId_pre"] != conns["bodyId_post"]]

    nt_col = next((c for c in NT_COLUMNS if c in neurons), None)
    sign = neurons[nt_col].map(nt_sign) if nt_col else pd.Series(1, index=neurons.index)

    side_col = neurons.get("somaSide", pd.Series(index=neurons.index, dtype=object))
    side = side_col.map({"L": -1, "R": 1}).fillna(0)

    pos = np.full((len(neurons), 3), np.nan, dtype=np.float32)
    if "somaLocation" in neurons:
        for i, loc in enumerate(neurons["somaLocation"]):
            if isinstance(loc, dict):  # raw neo4j Point
                loc = loc.get("coordinates")
            if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) == 3:
                pos[i] = loc
    # Neurons without a soma (e.g. truncated) sit at the mean of their cell type.
    types = neurons["type"].fillna(FALLBACK_TYPE).astype(str)
    missing = np.isnan(pos).any(axis=1)
    for t in types[missing].unique():
        same = (types == t).to_numpy()
        known = same & ~missing
        fill = pos[known].mean(axis=0) if known.any() else np.nanmean(pos, axis=0)
        pos[same & missing] = fill
    pos = np.nan_to_num(pos)

    return Connectome(
        pre=index[conns["bodyId_pre"].to_numpy()].to_numpy().astype(np.int32),
        post=index[conns["bodyId_post"].to_numpy()].to_numpy().astype(np.int32),
        count=conns["weight"].to_numpy(dtype=np.float32),
        sign=sign.to_numpy(dtype=np.int8),
        group=neurons["group"].to_numpy(dtype=np.int8),
        side=side.to_numpy(dtype=np.int8),
        pos=pos,
        body_id=neurons["bodyId"].to_numpy(dtype=np.int64),
        cell_type=types.to_numpy(),
        meta={**meta, "nt_column": nt_col},
    )


def _fetch_types(patterns, group, cap, rng):
    from neuprint import NeuronCriteria as NC
    from neuprint import fetch_neurons

    frames = []
    for pat in patterns:
        df, _ = fetch_neurons(NC(type=pat, regex=True))
        print(f"  {pat!r:14} -> {len(df):5d} neurons")
        if df.empty:
            print(f"  WARNING: no neurons match {pat!r}; check the type name in neuPrint")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("bodyId")
    if cap and len(df) > cap:
        df = df.sample(cap, random_state=rng.integers(1 << 31))
    return df.assign(group=group)


def _bridges(sel_up, sel_down, k, min_weight):
    """Neurons with the strongest two-way link between the upstream and downstream selections."""
    from neuprint import NeuronCriteria as NC
    from neuprint import fetch_adjacencies

    if k <= 0:
        return []
    _, down = fetch_adjacencies(NC(bodyId=list(sel_up)), None, min_total_weight=min_weight)
    _, up = fetch_adjacencies(None, NC(bodyId=list(sel_down)), min_total_weight=min_weight)
    inflow = down.groupby("bodyId_post")["weight"].sum()
    outflow = up.groupby("bodyId_pre")["weight"].sum()
    score = pd.concat([inflow, outflow], axis=1, join="inner").min(axis=1)
    score = score.drop(labels=list(set(sel_up) | set(sel_down)), errors="ignore")
    return score.nlargest(k).index.tolist()


def load_dotenv(path: str = ".env") -> None:
    """Set KEY=VALUE lines from .env into os.environ (shell exports take precedence)."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _top_targets(source_ids, superclass, k, exclude, min_weight):
    """The k neurons of `superclass` receiving the most synapses from source_ids."""
    from neuprint import NeuronCriteria as NC
    from neuprint import fetch_adjacencies

    if k <= 0:
        return []
    _, conns = fetch_adjacencies(NC(bodyId=list(source_ids)), NC(superclass=superclass),
                                 min_total_weight=min_weight)
    inflow = conns.groupby("bodyId_post")["weight"].sum().drop(labels=list(exclude), errors="ignore")
    return inflow.nlargest(k).index.tolist()


def extract(cfg: dict) -> Connectome:
    from neuprint import Client
    from neuprint import NeuronCriteria as NC
    from neuprint import fetch_adjacencies, fetch_neurons

    load_dotenv()
    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token:
        raise SystemExit("Set NEUPRINT_APPLICATION_CREDENTIALS (see module docstring).")
    # neuprint holds its default client by weak reference: keep this one alive.
    client = Client(cfg["server"], dataset=cfg["dataset"], token=token, progress=False)  # noqa: F841
    rng = np.random.default_rng(cfg.get("seed", 0))
    caps = cfg.get("max_per_group", {})

    print("inputs:")
    inp = _fetch_types(cfg["input_types"], INPUT, caps.get("input"), rng)
    print("core:")
    core = _fetch_types(cfg["core_types"], CORE, caps.get("core"), rng)
    print("outputs:")
    out = _fetch_types(cfg["output_types"], OUTPUT, caps.get("output"), rng)

    up = set(inp.bodyId) | set(core.bodyId)
    down = set(core.bodyId) | set(out.bodyId)
    bridge_ids = _bridges(up, down, cfg.get("bridge_neurons", 0), cfg.get("min_weight", 1))
    print(f"bridges: {len(bridge_ids)} neurons")
    if bridge_ids:
        bridge, _ = fetch_neurons(NC(bodyId=bridge_ids))
        core = pd.concat([core, bridge.assign(group=CORE)], ignore_index=True)

    fill_sc = cfg.get("output_fill_superclass")
    n_fill = caps.get("output", 0) - len(out)
    if fill_sc and n_fill > 0:
        fill_ids = _top_targets(set(core.bodyId), fill_sc, n_fill, set(out.bodyId), cfg.get("min_weight", 1))
        print(f"output fill: {len(fill_ids)} {fill_sc} neurons")
        if fill_ids:
            fill, _ = fetch_neurons(NC(bodyId=fill_ids))
            out = pd.concat([out, fill.assign(group=OUTPUT)], ignore_index=True)

    # Group priority when a neuron matched twice: output > input > core.
    neurons = pd.concat([out, inp, core], ignore_index=True).drop_duplicates("bodyId")
    ids = neurons["bodyId"].tolist()
    _, conns = fetch_adjacencies(NC(bodyId=ids), NC(bodyId=ids), min_total_weight=cfg.get("min_weight", 1))
    meta = {"source": "neuprint", "dataset": cfg["dataset"], "config": cfg}
    return assemble(neurons, conns, meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/tier_s.yaml")
    ap.add_argument("--out", help="override output path from config")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    c = extract(cfg)
    out = args.out or cfg["out"]
    c.save(out)
    print(c.summary())
    print(f"saved {out}")


if __name__ == "__main__":
    main()
