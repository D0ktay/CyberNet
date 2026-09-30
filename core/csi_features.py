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


# ─────────────────────────────────────────────────────────────────────────────
# ÇOK ANTENLİ (FAN DİZİLİMİ) ÖZNİTELİKLER
#
# Tek tek antenlerin kendi CSI şekli konum hakkında bilgi taşır — ama fan
# dizilimi (-45° / 0° / +45°) asıl gücünü antenler ARASINDAKİ FARKTAN alır:
# kişi bir antene diğerlerinden daha yakın/doğrudan ise o antenin enerjisi ve
# yansıma şekli farklılaşır — bu örtük bir YÖN (AoA / açısal çeşitlilik)
# bilgisidir. Sadece kanalları yan yana koymak (3x23 = 69) bu örtük yön
# bilgisini AÇIKÇA bir özniteliğe çevirmez; model bunu kendi keşfetmek
# zorunda kalır. Burada çıkarılan "çapraz kanal" öznitelikler bu yön
# bilgisini doğrudan modele sunar.
# ─────────────────────────────────────────────────────────────────────────────

def _drop_dead(row: np.ndarray, dead_sc: List[int]) -> np.ndarray:
    if not dead_sc:
        return row
    mask = np.ones(len(row), dtype=bool)
    for i in dead_sc:
        if 0 <= i < len(row):
            mask[i] = False
    return row[mask]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[:n], b[:n]
    if a.std() < _EPS or b.std() < _EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def cross_channel_features(rows: List[np.ndarray], dead_sc_list: List[List[int]]) -> np.ndarray:
    """3 antenden gelen ham CSI satırları arasındaki yön/çeşitlilik bilgisini çıkarır.

    Çıktı (6 boyut):
      - log-enerji farkları   (SOL-ORTA, ORTA-SAĞ, SOL-SAĞ): hangi anten "daha güçlü görüyor"
      - şekil korelasyonları  (SOL-ORTA, ORTA-SAĞ, SOL-SAĞ): yansıma örüntüleri ne kadar benzer
    Kişi bir yöne yakınsa o yöndeki antenin enerjisi artar / şekli farklılaşır
    -> bu farklar konumla doğrudan ilişkilidir (üçgenleme/AoA'nın basitleştirilmiş hali).
    """
    log_energies = []
    norm_shapes = []
    for row, dead_sc in zip(rows, dead_sc_list):
        clean = _drop_dead(row, dead_sc)
        log_energies.append(float(np.log1p(np.abs(clean).mean())))
        norm_shapes.append(normalize_row(clean))

    energy_diffs = np.array([
        log_energies[0] - log_energies[1],   # SOL  - ORTA
        log_energies[1] - log_energies[2],   # ORTA - SAĞ
        log_energies[0] - log_energies[2],   # SOL  - SAĞ
    ], dtype=np.float32)

    shape_corrs = np.array([
        _safe_corr(norm_shapes[0], norm_shapes[1]),
        _safe_corr(norm_shapes[1], norm_shapes[2]),
        _safe_corr(norm_shapes[0], norm_shapes[2]),
    ], dtype=np.float32)

    return np.concatenate([energy_diffs, shape_corrs])   # 6 öznitelik


def build_multi_feature_vector(rows: List[np.ndarray], dead_sc_list: List[List[int]]) -> np.ndarray:
    """3 antenden gelen birleşik CSI örneği için tam feature vektörü.

    [kanal0 (23) | kanal1 (23) | kanal2 (23) | çapraz-kanal yön bilgisi (6)]
    Toplam 75 boyut.
    """
    per_channel = [build_feature_vector(row, dead_sc) for row, dead_sc in zip(rows, dead_sc_list)]
    cross = cross_channel_features(rows, dead_sc_list)
    return np.concatenate(per_channel + [cross]).astype(np.float32)


def build_multi_features_matrix(X_raw_channels: List[np.ndarray], dead_sc_list: List[List[int]]) -> np.ndarray:
    """Birden çok birleşik örnek için feature matrisi.
    X_raw_channels: her biri (n_samples, 64) olan, kanal sırasına göre 3 elemanlı liste."""
    n = X_raw_channels[0].shape[0]
    return np.array([
        build_multi_feature_vector([X_raw_channels[ch][i] for ch in range(3)], dead_sc_list)
        for i in range(n)
    ], dtype=np.float32)
