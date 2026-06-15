"""Run the full data analysis and write results/analysis/ + a verdict report.

Tests five hypotheses about what distinguishes the cod-sound classes — and whether
the CLAUDE.md claim ("a tight vocal pulse is a click to Perch") is actually true:

  H1  vocals are pulse trains, clicks single transients        (signal: pulses)
  H2  click vs vocal differ spectrally, not only temporally     (signal: flatness)
  H3  click<->vocal is the closest pair in Perch space          (embedding: centroids)
  H4  cropping a vocal tight migrates it into the click cluster (embedding: scale)
  H5  the real _all-file events reproduce H1/H4                 (selection tables)

    uv run python -m src.data.analysis.run                 # test set + all 3 files
    uv run python -m src.data.analysis.run --quick         # fewer migration samples

TensorFlow (Perch) only — never imports PyTorch, so no segfault.
"""
import argparse
import glob
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, silhouette_score

from src.data.analysis import embedding_features as efeat
from src.data.analysis import signal_features as sfeat
from src.data.config import CLASSES, RESULTS_DIR
from src.inference.evaluate_detection import find_selection_table, load_selection_table
from src.model.perch import load_perch_model

TEST_DIR = Path("data/annotated/test")
OUT = RESULTS_DIR / "analysis"
WIDTHS_MS = [2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]
ALL_FILES = [
    Path("data/unannotated/01-220412_1221_Ch6_all.wav"),  # quiet vocals (breaks classify)
    Path("data/unannotated/01-220301_1434_Ch6_all.wav"),  # vocal-heavy
    Path("data/unannotated/01-220224_1200_Ch4_all.wav"),  # click-heavy
]
COLORS = {"click": "#d62728", "vocal": "#1f77b4", "water": "#2ca02c",
          "other": "#ff7f0e", "silence": "#9467bd"}


# ----------------------------------------------------------------- gather data
def test_clips() -> dict[str, list[str]]:
    return {c: sorted(glob.glob(str(TEST_DIR / c / "*.wav"))) for c in CLASSES}


def acoustic_row(path, max_s: float = 6.0) -> dict:
    """Signal features for one clip, truncated to max_s to bound cost on long clips."""
    audio, sr = sfeat.load_clip(path)
    if len(audio) > int(max_s * sr):
        audio = audio[: int(max_s * sr)]
    return {"path": str(path), **sfeat.features_from_audio(audio, sr)}


# ----------------------------------------------------------------------- plots
def grouped_box(df, col, classes, path, log=False, title=None):
    data = [df[df["class"] == c][col].dropna().values for c in classes if (df["class"] == c).any()]
    labels = [c for c in classes if (df["class"] == c).any()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, labels=labels, showfliers=False)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(col)
    ax.set_title(title or col)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def example_panel(paths, cls, out):
    n = len(paths)
    fig, axes = plt.subplots(n, 3, figsize=(13, 2.6 * n))
    if n == 1:
        axes = axes[None, :]
    for r, p in enumerate(paths):
        audio, sr = sfeat.load_clip(p)
        filt = sfeat.bandpass(audio, sr)
        s, e, _ = sfeat.sound_extent(sfeat.amplitude_envelope(filt, sr, 2.0), sr)
        core = filt[s:e] if e > s else filt
        cenv = sfeat.amplitude_envelope(core, sr, 1.0)
        npulse, ptimes = sfeat.detect_pulses(cenv, sr)
        t = np.arange(len(core)) / sr * 1000
        axes[r, 0].plot(t, core, lw=0.5, color=COLORS.get(cls, "k"))
        axes[r, 0].set_title(f"{cls} waveform ({len(core)/sr*1000:.0f} ms)")
        axes[r, 0].set_xlabel("ms")
        axes[r, 1].plot(t, cenv, color="k", lw=0.9)
        for x in ptimes * 1000:
            axes[r, 1].axvline(x, color="r", ls="--", lw=0.7)
        axes[r, 1].set_title(f"envelope — {npulse} pulse(s)")
        axes[r, 1].set_xlabel("ms")
        nfft = int(min(256, max(16, 2 ** int(np.log2(max(16, len(core)))))))
        axes[r, 2].specgram(core, NFFT=nfft, Fs=sr, noverlap=nfft // 2, cmap="magma")
        axes[r, 2].set_ylim(0, 2000)
        axes[r, 2].set_title("spectrogram (0-2 kHz)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def annotated_matrix(M, labels, path, title, fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                    color="w", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def tsne_plot(emb, labels, path, n=1000):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(emb), min(n, len(emb)), replace=False)
    xy = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(
        efeat.l2norm(emb[idx]))
    fig, ax = plt.subplots(figsize=(7, 6))
    lab = np.asarray(labels)[idx]
    for c in CLASSES:
        m = lab == c
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.6, label=c, color=COLORS[c])
    ax.legend()
    ax.set_title("Perch embeddings (t-SNE, native clips)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_migration_dist(curves, path):
    focus = list(curves)
    fig, axes = plt.subplots(1, len(focus), figsize=(4.2 * len(focus), 4), sharey=True)
    if len(focus) == 1:
        axes = [axes]
    for ax, cls in zip(axes, focus):
        to_click, to_vocal, to_self = curves[cls]
        ax.plot(WIDTHS_MS, to_click, "-o", color=COLORS["click"], label="→ click centroid")
        ax.plot(WIDTHS_MS, to_vocal, "-o", color=COLORS["vocal"], label="→ vocal centroid")
        ax.plot(WIDTHS_MS, to_self, "--", color="gray", label=f"→ {cls} centroid")
        ax.set_xscale("log")
        ax.set_xlabel("crop width (ms, log)")
        ax.set_title(f"true {cls}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean cosine distance")
    axes[0].legend(fontsize=8)
    fig.suptitle("Scale-migration: where does a tight crop land? (H4)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_migration_fracs(nearest, focus, path):
    fig, axes = plt.subplots(1, len(focus), figsize=(4.2 * len(focus), 4), sharey=True)
    if len(focus) == 1:
        axes = [axes]
    for ax, cls in zip(axes, focus):
        sub = nearest[nearest["true"] == cls]
        for c in CLASSES:
            cc = sub[sub["pred"] == c]
            if cc["frac"].sum() > 0:
                series = [cc[cc["width_ms"] == w]["frac"].sum() for w in WIDTHS_MS]
                ax.plot(WIDTHS_MS, series, "-o", color=COLORS[c], label=c)
        ax.set_xscale("log")
        ax.set_xlabel("crop width (ms, log)")
        ax.set_title(f"true {cls} — predicted class")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("fraction predicted (nearest centroid)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Scale-migration: nearest-centroid class vs crop width (H4)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# -------------------------------------------------------------------- sections
def run_acoustic(clips):
    print("== acoustic features (signal domain) ==")
    rows = []
    for c, paths in clips.items():
        print(f"  {c}: {len(paths)} clips")
        for i, p in enumerate(paths):
            try:
                r = acoustic_row(p)
            except Exception as ex:  # noqa: BLE001
                print(f"    skip {p}: {ex}")
                continue
            r["class"] = c
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "acoustic_per_clip.csv", index=False)
    feats = ["sound_duration_ms", "n_pulses", "pulse_rate_hz", "pulse_strength",
             "centroid_hz", "bandwidth_hz", "dominant_hz", "flatness", "rms"]
    summary = df.groupby("class")[feats].median().reindex(CLASSES)
    summary.to_csv(OUT / "acoustic_summary.csv")
    for col, log in [("n_pulses", False), ("pulse_strength", False), ("flatness", False),
                     ("centroid_hz", False), ("sound_duration_ms", True)]:
        grouped_box(df, col, CLASSES, OUT / f"acoustic_{col}.png", log=log)
    for c, paths in clips.items():
        if paths:
            example_panel(paths[:2], c, OUT / f"example_{c}.png")
    print(summary.round(2).to_string())
    return df, summary


def run_embeddings(model, clips):
    print("== embedding features (Perch space) ==")
    paths, labels = [], []
    for c, ps in clips.items():
        paths += ps
        labels += [c] * len(ps)
    emb = efeat.embed_paths(model, paths)
    labels = np.array(labels)
    centroids = efeat.class_centroids(emb, labels, CLASSES)
    M, present = efeat.centroid_distance_matrix(centroids, CLASSES)
    annotated_matrix(M, present, OUT / "centroid_distance.png",
                     "Class-centroid cosine distance (native clips)")
    sil = float(silhouette_score(efeat.l2norm(emb), labels, metric="cosine"))
    pred, _, _ = efeat.nearest_centroid_pred(emb, centroids, CLASSES)
    idx = {c: i for i, c in enumerate(CLASSES)}
    cm = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(labels, pred):
        cm[idx[t], idx[p]] += 1
    annotated_matrix(cm.astype(float), CLASSES, OUT / "embedding_confusion.png",
                     "Nearest-centroid confusion (native clips)", fmt="{:.0f}")
    tsne_plot(emb, labels, OUT / "tsne.png")
    print(f"  silhouette (cosine) = {sil:.3f}")
    return centroids, M, present, sil, cm


def run_migration(model, clips, centroids, n_migrate, focus=("vocal", "click", "water", "other")):
    print("== scale-migration (H4) ==")
    rng = np.random.default_rng(0)
    curves, nearest_rows = {}, []
    for cls in focus:
        ps = clips.get(cls, [])
        if not ps:
            continue
        if len(ps) > n_migrate:
            ps = list(rng.choice(ps, n_migrate, replace=False))
        print(f"  {cls}: {len(ps)} clips × {len(WIDTHS_MS)} widths")
        audios = [efeat.load_32k(p) for p in ps]
        to_click, to_vocal, to_self = [], [], []
        for w in WIDTHS_MS:
            emb = efeat.embed_crops(model, audios, w)
            pred, present, _ = efeat.nearest_centroid_pred(emb, centroids, CLASSES)
            frac = Counter(pred)
            for k in present:
                nearest_rows.append(dict(true=cls, width_ms=w, pred=k,
                                         frac=frac.get(k, 0) / len(pred)))
            to_click.append(float(efeat.cosine_dist_to(emb, centroids["click"]).mean()))
            to_vocal.append(float(efeat.cosine_dist_to(emb, centroids["vocal"]).mean()))
            to_self.append(float(efeat.cosine_dist_to(emb, centroids[cls]).mean()))
        curves[cls] = (np.array(to_click), np.array(to_vocal), np.array(to_self))
    nearest = pd.DataFrame(nearest_rows)
    nearest.to_csv(OUT / "scale_migration_nearest.csv", index=False)
    plot_migration_dist(curves, OUT / "scale_migration.png")
    plot_migration_fracs(nearest, [c for c in focus if c in curves],
                         OUT / "scale_migration_fracs.png")
    return nearest, curves


def run_all_file(model, wav, centroids, n_migrate):
    table = find_selection_table(wav)
    if not table:
        print(f"  no selection table for {wav.name} — skipping")
        return None, None
    gt = load_selection_table(table)
    stem = wav.stem
    d = OUT / "events" / stem
    d.mkdir(parents=True, exist_ok=True)
    print(f"== {stem}: {len(gt)} annotated events ==")
    rows, audios_by_class = [], {}
    with sf.SoundFile(str(wav)) as f:
        sr, total = f.samplerate, len(f)
        pad = int(sr * 0.3)
        for b, e, label in gt:
            start = max(0, int(b * sr) - pad)
            stop = min(total, int(e * sr) + pad)
            f.seek(start)
            seg = f.read(stop - start, dtype="float32", always_2d=False)
            if seg.ndim > 1:
                seg = seg.mean(axis=1)
            rows.append({**sfeat.features_from_audio(seg, sr), "class": label})
            audios_by_class.setdefault(label, []).append(efeat.to_32k(seg, sr))
    df = pd.DataFrame(rows)
    df.to_csv(d / "acoustic.csv", index=False)
    df.groupby("class")[["sound_duration_ms", "n_pulses", "pulse_strength",
                         "flatness", "centroid_hz"]].median().to_csv(d / "acoustic_summary.csv")
    nearest_rows = []
    for cls in ("vocal", "click"):
        audios = audios_by_class.get(cls, [])
        if not audios:
            continue
        if len(audios) > n_migrate:
            audios = audios[:n_migrate]
        for w in WIDTHS_MS:
            emb = efeat.embed_crops(model, audios, w)
            pred, _, _ = efeat.nearest_centroid_pred(emb, centroids, CLASSES)
            frac = Counter(pred)
            for k, v in frac.items():
                nearest_rows.append(dict(true=cls, width_ms=w, pred=k, frac=v / len(pred)))
    nd = pd.DataFrame(nearest_rows)
    if not nd.empty:
        nd.to_csv(d / "migration_nearest.csv", index=False)
        plot_migration_fracs(nd, [c for c in ("vocal", "click") if c in nd["true"].values],
                             d / "migration_fracs.png")
    return df, nd


# --------------------------------------------------------------------- report
def df_to_md(df):
    """Minimal DataFrame→markdown table (avoids the optional `tabulate` dep)."""
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def _med(df, cls, col):
    s = df[df["class"] == cls][col]
    return float(s.median()) if len(s) else float("nan")


def _frac_at(nearest, true, width, pred):
    sub = nearest[(nearest["true"] == true) & (nearest["width_ms"] == width) & (nearest["pred"] == pred)]
    return float(sub["frac"].sum())


def build_report(adf, summary, M, present, sil, nearest, per_file):
    L = ["# Data analysis — what distinguishes the cod-sound classes\n"]
    L.append("Generated by `src/data/analysis/run.py`. Figures + CSVs in this folder.\n")

    # H1
    L.append("## H1 — temporal structure (vocal = pulse train, click = single transient)\n")
    L.append("Median by class (sound region only):\n")
    L.append(df_to_md(summary[["sound_duration_ms", "n_pulses", "pulse_rate_hz", "pulse_strength"]]
             .round(2)) + "\n")
    vp, cp = _med(adf, "vocal", "n_pulses"), _med(adf, "click", "n_pulses")
    vs, cs = _med(adf, "vocal", "pulse_strength"), _med(adf, "click", "pulse_strength")
    h1 = vp >= 2 and cp <= 1.5 and vs > cs
    L.append(f"**Verdict H1: {'SUPPORTED' if h1 else 'NOT clearly supported'}** — "
             f"vocal median pulses={vp:.0f} (strength {vs:.2f}), click={cp:.0f} (strength {cs:.2f}).\n")

    # H2
    L.append("## H2 — spectral difference (do click & vocal differ in frequency, not just timing?)\n")
    L.append(df_to_md(summary[["centroid_hz", "bandwidth_hz", "dominant_hz", "flatness"]]
             .round(2)) + "\n")
    try:
        sub = adf[adf["class"].isin(["click", "vocal"])]
        y = (sub["class"] == "vocal").astype(int).values
        auc_flat = roc_auc_score(y, -sub["flatness"].values)  # click expected flatter
        auc_cent = roc_auc_score(y, sub["centroid_hz"].values)
        auc = max(auc_flat, 1 - auc_flat, auc_cent, 1 - auc_cent)
    except Exception:  # noqa: BLE001
        auc = float("nan")
    h2 = auc >= 0.75
    L.append(f"**Verdict H2: {'SUPPORTED' if h2 else 'WEAK'}** — best single-spectral-feature "
             f"separability click-vs-vocal AUC ≈ {auc:.2f} "
             f"(flatness click {_med(adf,'click','flatness'):.3f} vs vocal {_med(adf,'vocal','flatness'):.3f}). "
             f"High AUC ⇒ they differ spectrally ⇒ hope for separation even at tiny crops.\n")

    # H3
    L.append("## H3 — separability in Perch embedding space\n")
    pairs = [(present[i], present[j], M[i, j]) for i in range(len(present))
             for j in range(i + 1, len(present))]
    pairs.sort(key=lambda x: x[2])
    L.append(f"Silhouette (cosine) = **{sil:.3f}**. Closest class pairs (centroid cosine distance):\n")
    for a, b, dpair in pairs[:4]:
        L.append(f"- {a} ↔ {b}: {dpair:.3f}")
    cv = next((d for a, b, d in pairs if {a, b} == {"click", "vocal"}), float("nan"))
    rank = next((k for k, (a, b, _) in enumerate(pairs, 1) if {a, b} == {"click", "vocal"}), -1)
    h3 = rank == 1
    L.append(f"\n**Verdict H3: click↔vocal is the {rank}{'st' if rank==1 else 'th'}-closest pair "
             f"(distance {cv:.3f}).** {'They are the most confusable pair.' if h3 else ''}\n")

    # H4
    L.append("## H4 — scale-migration: does a tight vocal crop become a click?\n")
    L.append("Fraction of clips whose *nearest centroid* is `click` vs `vocal`, by crop width:\n")
    hdr = "| true | " + " | ".join(f"{w:g}ms" for w in WIDTHS_MS) + " |"
    sep = "|" + "---|" * (len(WIDTHS_MS) + 1)
    L.append(hdr); L.append(sep)
    for true in ("vocal", "click"):
        cells = [f"{_frac_at(nearest, true, w, 'click'):.2f}→clk / {_frac_at(nearest, true, w, 'vocal'):.2f}→voc"
                 for w in WIDTHS_MS]
        L.append(f"| {true} | " + " | ".join(cells) + " |")
    v10, v50 = _frac_at(nearest, "vocal", 10.0, "click"), _frac_at(nearest, "vocal", 50.0, "click")
    h4 = v10 > v50 and v10 > 0.3
    L.append(f"\n**Verdict H4: {'SUPPORTED' if h4 else 'NOT supported'}** — at 10 ms, "
             f"{v10:.0%} of *vocals* land nearest the click centroid; at 50 ms only {v50:.0%}. "
             f"{'A tight crop does migrate vocals → click.' if h4 else ''}\n")

    # H5
    L.append("## H5 — do the real _all-file events reproduce this?\n")
    for stem, (df, nd) in per_file.items():
        if df is None:
            continue
        vp = _med(df, "vocal", "n_pulses") if (df["class"] == "vocal").any() else float("nan")
        line = f"- **{stem}**: vocal median pulses={vp:.0f}"
        if nd is not None and not nd.empty and (nd["true"] == "vocal").any():
            r10 = nd[(nd["true"] == "vocal") & (nd["width_ms"] == 10.0) & (nd["pred"] == "click")]["frac"].sum()
            r50 = nd[(nd["true"] == "vocal") & (nd["width_ms"] == 50.0) & (nd["pred"] == "click")]["frac"].sum()
            line += f"; real vocals → click: {r10:.0%} @10ms vs {r50:.0%} @50ms"
        L.append(line)
    L.append("")

    (OUT / "analysis_report.md").write_text("\n".join(L))
    print("\nWrote report → " + str(OUT / "analysis_report.md"))


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Full data analysis (signal + embedding domains)")
    ap.add_argument("--quick", action="store_true", help="fewer migration samples (faster)")
    ap.add_argument("--no-files", action="store_true", help="skip the long _all files")
    args = ap.parse_args()
    n_migrate = 40 if args.quick else 120
    OUT.mkdir(parents=True, exist_ok=True)

    clips = test_clips()
    adf, summary = run_acoustic(clips)

    model = load_perch_model()
    centroids, M, present, sil, _cm = run_embeddings(model, clips)
    nearest, _curves = run_migration(model, clips, centroids, n_migrate)

    per_file = {}
    if not args.no_files:
        for wav in ALL_FILES:
            if wav.exists():
                df, nd = run_all_file(model, wav, centroids, n_migrate)
                per_file[wav.stem] = (df, nd)
            else:
                print(f"  missing {wav} — skipping")

    build_report(adf, summary, M, present, sil, nearest, per_file)
    print(f"\nAll analysis outputs in {OUT}/")


if __name__ == "__main__":
    main()
