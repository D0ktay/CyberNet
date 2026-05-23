"""
CSI feature engineering — eğitim ve canlı inference için tek kaynak.

Tasarım kararları:
  - Per-packet normalize (mean=1) → mutlak güç değişimine dayanıklı
  - Ölü/sabit subcarrier'lar fit sırasında otomatik bulunur, drop edilir
  - Şekil odaklı feature'lar (grup ortalamaları, spektral tilt, log enerji)
  - Toplam ~25 feature → küçük veri + RBF için sağlıklı
"""

import numpy as np
from typing import List


N_SUBCARRIERS = 64
_EPS = 1e-6


def normalize_row(row: np.ndarray) -> np.ndarray:
    """Per-packet normalize: nonzero ortalamayı 1'e çek."""
    nz = row[row > 0]
    if len(nz) == 0:
        return row.copy()
    return row / (float(nz.mean()) + _EPS)


def find_dead_subcarriers(X_raw: np.ndarray, var_threshold: float = 1e-3) -> List[int]:
    """Eğitim setinde varyansı pratik olarak sıfır olan SC indekslerini döndür.
    Bunlar pilot/DC/guard subcarrier'larıdır — bilgi taşımazlar."""
    variances = np.var(X_raw, axis=0)
    return [int(i) for i in np.where(variances < var_threshold)[0]]


def build_feature_vector(row: np.ndarray, dead_sc: List[int]) -> np.ndarray:
    """Tek bir CSI paketinden ~25 boyutlu feature vektörü.

    Adımlar:
      1) Ölü SC'leri at
      2) Per-packet normalize (mean=1)
      3) Şekil özetleri çıkar
    """
    # 1) Ölü SC drop
    if dead_sc:
        mask = np.ones(len(row), dtype=bool)
        for i in dead_sc:
            if 0 <= i < len(row):
                mask[i] = False
        row = row[mask]

    # 2) Normalize (per-packet)
    norm = normalize_row(row)

    # 3) Grup istatistikleri (kaç grup yapabiliyoruz, ona göre)
    n = len(norm)
    n_groups = 8
    group_size = n // n_groups
    if group_size < 1:
        group_size = 1
        n_groups = n
    usable = group_size * n_groups
    groups = norm[:usable].reshape(n_groups, group_size)
    group_means = groups.mean(axis=1)                  # 8

    # Grup farkları (komşu farklar, monotonluk bilgisi)
    group_diffs = np.diff(group_means)                 # 7

    # Spektral şekil
    x = np.arange(n)
    # Lineer fit eğimi (tilt)
    if n > 1:
        slope = float(np.polyfit(x, norm, 1)[0])
    else:
        slope = 0.0
    centroid = float((x * norm).sum() / (norm.sum() + _EPS))
    flatness = float(norm.std() / (norm.mean() + _EPS))

    # Mutlak güç bilgisi tek bir özet olarak: log enerji
    log_energy = float(np.log1p(np.abs(row).mean()))

    # Global özetler (normalize edilmiş — boyutsuz)
    g_min = float(norm.min())
    g_max = float(norm.max())
    g_p25 = float(np.percentile(norm, 25))
    g_p75 = float(np.percentile(norm, 75))

    feat = np.concatenate([
        group_means,                                   # 8
        group_diffs,                                   # 7
        [slope, centroid, flatness],                   # 3
        [log_energy],                                  # 1
        [g_min, g_max, g_p25, g_p75],                  # 4
    ]).astype(np.float32)                              # toplam 23

    return feat


def build_features_matrix(X_raw: np.ndarray, dead_sc: List[int]) -> np.ndarray:
    """Birden çok paket için feature matrisi."""
    return np.array(
        [build_feature_vector(row, dead_sc) for row in X_raw],
        dtype=np.float32,
    )
