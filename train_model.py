"""
CSI fingerprint eğitimi — gerçek genelleme odaklı.

Kritik değişiklikler:
  - Per-packet normalize + ölü SC drop (core/csi_features.py)
  - GroupKFold: ardışık paketleri aynı fold'a topluyor → sızıntısız CV
  - Probability calibration (isotonic) → zone_B'nin canlıda %0 çıkma problemini düzeltir
  - Holdout: son %20'lik zaman bloğu test, ilk %80 eğitim
  - 5 model yarışıyor, en iyi GroupKFold ortalaması kazanıyor

Kullanım:  python train_model.py
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
)

DATA_FILE  = os.path.join(os.path.dirname(__file__), "data", "csi_fingerprints.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), "core", "wifi_radar_model.pkl")

# Ardışık paketleri aynı blok'a topla — GroupKFold için
BLOCK_SIZE = 200   # ~6-7 saniye @ 30Hz


def make_groups(df: pd.DataFrame, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """Her label içinde ardışık BLOCK_SIZE paketi tek group_id altında topla.
    Bu sayede GroupKFold ardışık (yüksek korelasyonlu) paketleri ayırmaz."""
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
    """Calibrated wrapper'lar zone_B'nin canlıda %0 çıkma problemini çözer."""
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


def main():
    if not os.path.exists(DATA_FILE):
        print(f"HATA: {DATA_FILE} bulunamadi. Once collect_data.py ile veri topla.")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    print(f"\nVeri: {len(df)} satir  |  Labellar: {df['label'].value_counts().to_dict()}")

    if len(df) < 50:
        print("UYARI: Cok az veri. Her label icin en az 100 paket topla.")
        sys.exit(1)

    # ── Ham amplitüd matrisi ──────────────────────────────────────────────────
    amp_cols = [f"amp_{i}" for i in range(N_SUBCARRIERS)]
    X_raw = df[amp_cols].values.astype(np.float32)
    y     = df["label"].values

    # ── Ölü subcarrier'ları bul ────────────────────────────────────────────────
    dead_sc = find_dead_subcarriers(X_raw, var_threshold=1e-3)
    print(f"\nOlu/sabit subcarrier sayisi: {len(dead_sc)}  -> indeksler: {dead_sc[:10]}{'...' if len(dead_sc) > 10 else ''}")

    # ── Feature engineering ────────────────────────────────────────────────────
    X = build_features_matrix(X_raw, dead_sc)
    print(f"Feature boyutu: {X.shape[1]} (her paket basina)")

    # ── Group'lar (ardisik paketleri ayni fold'a topla) ───────────────────────
    groups = make_groups(df)
    n_groups = len(np.unique(groups))
    print(f"Toplam grup (blok) sayisi: {n_groups}  |  Block size: {BLOCK_SIZE}")

    # ── Holdout: son %20'lik zaman blogu test ──────────────────────────────────
    # Her label icin son bloklari test'e ayir
    test_idx = []
    train_idx = []
    for label in df["label"].unique():
        idx = df.index[df["label"] == label].tolist()
        split = int(len(idx) * 0.8)
        train_idx.extend(idx[:split])
        test_idx.extend(idx[split:])
    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train    = groups[train_idx]

    print(f"\nEgitim: {len(X_train)}  |  Test (son %20 zaman blogu): {len(X_test)}")
    print(f"Test label dagilimi: {dict(zip(*np.unique(y_test, return_counts=True)))}\n")

    n_cv_splits = min(5, len(np.unique(groups_train)))
    gkf = GroupKFold(n_splits=n_cv_splits)

    print(f"{'Model':<20}  {'Test Acc':>8}  {'GroupKF Mean':>13}  {'GKF Std':>8}")
    print("-" * 60)

    best_name, best_cv, best_model = None, -1.0, None

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))

        cv_scores = cross_val_score(
            model, X_train, y_train,
            groups=groups_train, cv=gkf,
            scoring="accuracy", n_jobs=-1,
        )
        cv_mean = cv_scores.mean()
        cv_std  = cv_scores.std()
        print(f"{name:<20}  {test_acc*100:>7.1f}%  {cv_mean*100:>12.1f}%  {cv_std*100:>7.1f}%")

        # En iyi: GroupKFold cv ortalamasi (gercek genelleme metrigi)
        if cv_mean > best_cv:
            best_cv, best_name, best_model = cv_mean, name, model

    print("-" * 60)
    print(f"\nSAMPIYON: {best_name}  (GroupKFold CV: {best_cv*100:.1f}%)\n")

    # ── Sampiyonun raporu (holdout uzerinde) ──────────────────────────────────
    preds = best_model.predict(X_test)
    print("Holdout Test Raporu:")
    print(classification_report(y_test, preds))

    labels = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, preds, labels=labels)
    print("Confusion Matrix (holdout):")
    print(f"{'':>10}", "  ".join(f"{l:>8}" for l in labels))
    for i, row in enumerate(cm):
        print(f"{labels[i]:>10}", "  ".join(f"{v:>8}" for v in row))

    # Probability sanity check
    if hasattr(best_model, "predict_proba"):
        probs = best_model.predict_proba(X_test)
        proba_labels = best_model.classes_
        print("\nOrtalama predict_proba (holdout, label basina):")
        for i, lbl in enumerate(proba_labels):
            mean_p = probs[:, i].mean()
            print(f"  {lbl:<10}: {mean_p:.3f}")

    # ── Kaydet: model + ölü SC listesi ────────────────────────────────────────
    joblib.dump({
        "model":    best_model,
        "name":     best_name,
        "labels":   labels,
        "dead_sc":  dead_sc,
        "version":  2,
    }, MODEL_FILE)
    print(f"\nModel kaydedildi: {MODEL_FILE}")
    print("Simdi 'python ui/radar_ui.py' ile UI'i ac.")


if __name__ == "__main__":
    main()
