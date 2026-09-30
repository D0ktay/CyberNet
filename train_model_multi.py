"""
3-anten birleşik CSI fingerprint eğitimi + tek-anten ile adil karşılaştırma.

Adil karşılaştırma mantığı:
  Eski tek-anten modeli farklı bir oturumda/ortamda toplandığı için onunla
  kıyaslamak elma-armut olur. Bunun yerine AYNI 3-antenli oturumdan gelen
  veriden iki model çıkarıyoruz:
    (A) SADECE ORTA (0°) anten  -> 23 öznitelik   (mevcut tek-anten boru hattıyla birebir aynı)
    (B) 3 ANTEN BİRLEŞİK        -> 69 öznitelik   (3 x 23, her kanal kendi feature setiyle)
  İkisi de aynı satırlardan, aynı zaman dilimlerinden, aynı GroupKFold/holdout
  şemasıyla değerlendirilir -> "3 anten ne kadar katkı sağlıyor" sorusuna
  doğrudan cevap.

Kullanım:  python train_model_multi.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from core.csi_features import (
    N_SUBCARRIERS,
    find_dead_subcarriers,
    build_features_matrix,
    build_multi_features_matrix,
)
from core.esp32_multi_reader import CHANNEL_LABELS

DATA_FILE  = os.path.join(os.path.dirname(__file__), "data", "csi_fingerprints_multi.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "core", "wifi_radar_model_multi.pkl")

BLOCK_SIZE = 200   # ardışık paketleri aynı grup altında topla (GroupKFold için)
CENTER_CH  = 1     # "tek anten" karşılaştırması için ORTA (0°) kanal indeksi


def make_groups(df: pd.DataFrame, block_size: int = BLOCK_SIZE) -> np.ndarray:
    groups = np.zeros(len(df), dtype=np.int64)
    gid = 0
    for label in df["label"].unique():
        idx = df.index[df["label"] == label].tolist()
        for i, row_idx in enumerate(idx):
            if i > 0 and i % block_size == 0:
                gid += 1
            groups[row_idx] = gid
        gid += 1
    return groups


def get_models():
    return {
        "LogReg (L2)":     Pipeline([("scaler", StandardScaler()),
                                     ("clf", LogisticRegression(C=0.5, max_iter=2000))]),
        "Linear SVM":      Pipeline([("scaler", StandardScaler()),
                                     ("clf", CalibratedClassifierCV(
                                         SVC(kernel="linear", C=1.0), cv=3))]),
        "RBF SVM":         Pipeline([("scaler", StandardScaler()),
                                     ("clf", CalibratedClassifierCV(
                                         SVC(kernel="rbf", C=2.0, gamma="scale"), cv=3))]),
        "Random Forest":   RandomForestClassifier(
                              n_estimators=300, max_depth=10,
                              min_samples_leaf=5, random_state=42, n_jobs=-1),
        "Gradient Boost":  GradientBoostingClassifier(
                              n_estimators=200, max_depth=3, learning_rate=0.05,
                              random_state=42),
        "KNN-7":           Pipeline([("scaler", StandardScaler()),
                                     ("clf", KNeighborsClassifier(n_neighbors=7))]),
    }


def split_holdout(df: pd.DataFrame):
    """Her label için son %20'lik zaman bloğunu test olarak ayır."""
    train_idx, test_idx = [], []
    for label in df["label"].unique():
        idx = df.index[df["label"] == label].tolist()
        split = int(len(idx) * 0.8)
        train_idx.extend(idx[:split])
        test_idx.extend(idx[split:])
    return np.array(train_idx), np.array(test_idx)


def evaluate(name_tag, X, y, groups, train_idx, test_idx):
    """Verilen feature matrisi için tüm modelleri yarıştır, şampiyonu döndür."""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train    = groups[train_idx]

    n_cv_splits = min(5, len(np.unique(groups_train)))
    gkf = GroupKFold(n_splits=n_cv_splits)

    print(f"\n### {name_tag}  (öznitelik boyutu: {X.shape[1]}) ###")
    print(f"{'Model':<20}  {'Test Acc':>8}  {'GroupKF Mean':>13}  {'GKF Std':>8}")
    print("-" * 60)

    best_name, best_cv, best_model, best_test = None, -1.0, None, -1.0
    for mname, model in get_models().items():
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        cv_scores = cross_val_score(
            model, X_train, y_train,
            groups=groups_train, cv=gkf,
            scoring="accuracy", n_jobs=-1,
        )
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        print(f"{mname:<20}  {test_acc*100:>7.1f}%  {cv_mean*100:>12.1f}%  {cv_std*100:>7.1f}%")
        if cv_mean > best_cv:
            best_cv, best_name, best_model, best_test = cv_mean, mname, model, test_acc

    print("-" * 60)
    print(f"ŞAMPİYON [{name_tag}]: {best_name}  (GroupKFold CV: {best_cv*100:.1f}%  |  Holdout: {best_test*100:.1f}%)")
    return {
        "tag": name_tag, "name": best_name, "model": best_model,
        "cv": best_cv, "test": best_test, "X_test": X_test, "y_test": y_test,
    }


def main():
    if not os.path.exists(DATA_FILE):
        print(f"HATA: {DATA_FILE} bulunamadi. Once collect_data_multi.py ile veri topla.")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    print(f"Veri: {len(df)} satir  |  Labellar: {df['label'].value_counts().to_dict()}")
    if len(df) < 150:
        print("UYARI: Cok az veri. Her label icin en az ~300 satir topla.")
        sys.exit(1)

    y = df["label"].values
    groups = make_groups(df)
    train_idx, test_idx = split_holdout(df)
    print(f"Eğitim: {len(train_idx)}  |  Test (son %20 zaman bloğu): {len(test_idx)}\n")

    # ── Her kanal için ayrı ayrı ham matris + ölü SC + feature ───────────────
    raw_channels  = []
    channel_dead  = []
    channel_feats = []
    for ch in range(3):
        cols = [f"amp_ch{ch}_{i}" for i in range(N_SUBCARRIERS)]
        X_raw_ch = df[cols].values.astype(np.float32)
        dead_sc = find_dead_subcarriers(X_raw_ch, var_threshold=1e-3)
        feats = build_features_matrix(X_raw_ch, dead_sc)
        raw_channels.append(X_raw_ch)
        channel_dead.append(dead_sc)
        channel_feats.append(feats)
        print(f"  {CHANNEL_LABELS[ch]:18s}  ölü/sabit subcarrier: {len(dead_sc):2d}  ->  feature: {feats.shape[1]}")

    # (A) SADECE ORTA anten — tek-anten boru hattının birebir eşdeğeri
    X_single = channel_feats[CENTER_CH]
    result_single = evaluate(f"TEK ANTEN ({CHANNEL_LABELS[CENTER_CH]})", X_single, y, groups, train_idx, test_idx)

    # (B) 3 anten birleşik — kanal feature'ları + ÇAPRAZ KANAL yön/çeşitlilik özellikleri
    #     (sadece 3x23'ü yan yana koymak antenler arasındaki göreceli farkı,
    #      yani fan diziliminin asıl avantajı olan örtük yön bilgisini, kullanmaz)
    X_multi = build_multi_features_matrix(raw_channels, channel_dead)
    result_multi = evaluate("3 ANTEN BİRLEŞİK + ÇAPRAZ-KANAL YÖN ÖZELLİKLERİ", X_multi, y, groups, train_idx, test_idx)

    # ── Karşılaştırma özeti ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KARŞILAŞTIRMA — aynı oturum, aynı satırlar, aynı değerlendirme")
    print("=" * 60)
    print(f"{'':25s}{'GroupKFold CV':>15}{'Holdout Test':>15}")
    print(f"{'Tek anten (ORTA)':25s}{result_single['cv']*100:>14.1f}%{result_single['test']*100:>14.1f}%")
    print(f"{'3 anten birleşik':25s}{result_multi['cv']*100:>14.1f}%{result_multi['test']*100:>14.1f}%")
    delta_cv   = (result_multi['cv']   - result_single['cv'])   * 100
    delta_test = (result_multi['test'] - result_single['test']) * 100
    print(f"{'FARK (3anten - tek)':25s}{delta_cv:>+14.1f}%{delta_test:>+14.1f}%")
    print("=" * 60)

    # ── Şampiyon (3 anten) için detaylı rapor ────────────────────────────────
    best = result_multi
    preds = best["model"].predict(best["X_test"])
    print(f"\n[3 ANTEN BİRLEŞİK] Şampiyon: {best['name']}  — Holdout Raporu:")
    print(classification_report(best["y_test"], preds))

    labels = sorted(df["label"].unique())
    cm = confusion_matrix(best["y_test"], preds, labels=labels)
    print("Confusion Matrix (holdout, 3 anten birleşik):")
    print(f"{'':>10}", "  ".join(f"{l:>8}" for l in labels))
    for i, row in enumerate(cm):
        print(f"{labels[i]:>10}", "  ".join(f"{v:>8}" for v in row))

    # ── Kaydet ────────────────────────────────────────────────────────────────
    joblib.dump({
        "model":         best["model"],
        "name":          best["name"],
        "labels":        labels,
        "channel_dead":  channel_dead,   # her kanal için ölü SC listesi
        "n_channels":    3,
        "version":       1,
    }, MODEL_FILE)
    print(f"\nModel kaydedildi: {MODEL_FILE}")


if __name__ == "__main__":
    main()
