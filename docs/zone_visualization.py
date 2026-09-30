"""3 zone (zone_A, zone_B, empty) icin toplanan CSI verisinin gorsellestirilmesi.

Uretir:
  1) Her anten (SOL/ORTA/SAĞ) icin zone'lara gore ortalama normalize CSI sekli
  2) 75-boyutlu ozellik uzayinin PCA ile 2B izdusumu (zone'lar ne kadar ayrisiyor?)
  3) Zone basina log-enerji dagilimi (anten bazinda)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.csi_features import (
    N_SUBCARRIERS, normalize_row, find_dead_subcarriers,
    build_multi_features_matrix,
)
from core.esp32_multi_reader import CHANNEL_LABELS

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "csi_fingerprints_multi.csv")
OUT_DIR = os.path.dirname(__file__)

NEON, CYAN, ORANGE = "#39ff14", "#00d8ff", "#ff9100"
DIM, BG, FG, PANEL = "#888888", "#0d1117", "#e6edf3", "#161b22"
ZONE_COLORS = {"zone_A": CYAN, "zone_B": ORANGE, "empty": NEON}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": FG, "axes.edgecolor": DIM, "axes.labelcolor": FG,
    "xtick.color": DIM, "ytick.color": DIM, "grid.color": "#30363d",
})


def main():
    df = pd.read_csv(DATA_FILE)
    print(f"Veri: {len(df)} satır  |  {df['label'].value_counts().to_dict()}")
    zones = ["zone_A", "zone_B", "empty"]

    # ── 1) Anten başına ortalama normalize CSI şekli ────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("Anten Başına Ortalama Normalize CSI Şekli (zone'lara göre)",
                 fontsize=13, color=NEON, fontweight="bold")
    for ch in range(3):
        cols = [f"amp_ch{ch}_{i}" for i in range(N_SUBCARRIERS)]
        X_raw = df[cols].values.astype(np.float32)
        ax = axes[ch]
        for zone in zones:
            mask = (df["label"] == zone).values
            shapes = np.array([normalize_row(row) for row in X_raw[mask]])
            mean_shape = shapes.mean(axis=0)
            std_shape = shapes.std(axis=0)
            x = np.arange(len(mean_shape))
            ax.plot(x, mean_shape, color=ZONE_COLORS[zone], label=zone, linewidth=1.6)
            ax.fill_between(x, mean_shape - std_shape, mean_shape + std_shape,
                            color=ZONE_COLORS[zone], alpha=0.12)
        ax.set_title(CHANNEL_LABELS[ch], fontsize=10, color=FG)
        ax.set_xlabel("Subcarrier index")
        ax.set_ylim(0, 2.6)   # SC#0'daki DC-tepe ölçeği domine ediyor; şekil farkına odaklan
        if ch == 0:
            ax.set_ylabel("Normalize genlik (ortalama=1)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor=DIM, labelcolor=FG)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p1 = os.path.join(OUT_DIR, "zone_csi_shapes.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    print(f"Kaydedildi: {p1}")

    # ── 2) 75-boyutlu özellik uzayının PCA izdüşümü ─────────────────────────
    raw_channels, channel_dead = [], []
    for ch in range(3):
        cols = [f"amp_ch{ch}_{i}" for i in range(N_SUBCARRIERS)]
        X_raw_ch = df[cols].values.astype(np.float32)
        dead_sc = find_dead_subcarriers(X_raw_ch, var_threshold=1e-3)
        raw_channels.append(X_raw_ch)
        channel_dead.append(dead_sc)

    X_multi = build_multi_features_matrix(raw_channels, channel_dead)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(X_multi)
    var_ratio = pca.explained_variance_ratio_

    fig2, ax2 = plt.subplots(figsize=(7.5, 6.5))
    fig2.suptitle("75-Boyutlu Özellik Uzayının PCA İzdüşümü\n(zone'lar ne kadar ayrışıyor?)",
                  fontsize=12, color=NEON, fontweight="bold")
    rng = np.random.default_rng(42)
    for zone in zones:
        mask = (df["label"] == zone).values
        idx = np.where(mask)[0]
        sample = rng.choice(idx, size=min(800, len(idx)), replace=False)
        ax2.scatter(proj[sample, 0], proj[sample, 1], s=6, alpha=0.45,
                    color=ZONE_COLORS[zone], label=zone, edgecolors="none")
    ax2.set_xlabel(f"PC1  ({var_ratio[0]*100:.1f}% varyans)")
    ax2.set_ylabel(f"PC2  ({var_ratio[1]*100:.1f}% varyans)")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, facecolor=PANEL, edgecolor=DIM, labelcolor=FG)
    fig2.tight_layout(rect=[0, 0, 1, 0.90])
    p2 = os.path.join(OUT_DIR, "zone_pca_projection.png")
    fig2.savefig(p2, dpi=130)
    plt.close(fig2)
    print(f"Kaydedildi: {p2}")

    # ── 3) Anten başına log-enerji dağılımı (kutu grafiği) ──────────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.6))
    fig3.suptitle("Anten Başına Log-Enerji Dağılımı (zone'lara göre)",
                  fontsize=13, color=NEON, fontweight="bold")
    for ch in range(3):
        cols = [f"amp_ch{ch}_{i}" for i in range(N_SUBCARRIERS)]
        X_raw = df[cols].values.astype(np.float32)
        log_energy = np.log1p(np.abs(X_raw).mean(axis=1))
        ax = axes3[ch]
        data = [log_energy[(df["label"] == zone).values] for zone in zones]
        bp = ax.boxplot(data, labels=zones, patch_artist=True, widths=0.55,
                        medianprops=dict(color=BG, linewidth=1.6),
                        flierprops=dict(marker='.', markersize=2, alpha=0.3, markerfacecolor=DIM))
        for patch, zone in zip(bp["boxes"], zones):
            patch.set_facecolor(ZONE_COLORS[zone])
            patch.set_alpha(0.65)
        ax.set_title(CHANNEL_LABELS[ch], fontsize=10, color=FG)
        if ch == 0:
            ax.set_ylabel("log(1 + ortalama |CSI|)")
        ax.grid(alpha=0.25, axis="y")
        ax.tick_params(colors=FG)
    fig3.tight_layout(rect=[0, 0, 1, 0.93])
    p3 = os.path.join(OUT_DIR, "zone_log_energy.png")
    fig3.savefig(p3, dpi=130)
    plt.close(fig3)
    print(f"Kaydedildi: {p3}")


if __name__ == "__main__":
    main()
