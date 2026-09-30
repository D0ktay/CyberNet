import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# Modellerimizi çağırıyoruz
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

MODEL_PATH = 'core/wifi_radar_model.pkl'
DATA_PATH = 'data/csi_fingerprints.csv'

class RadarML:
    def __init__(self):
        # Deneyeceğimiz tüm modelleri bir sözlükte topluyoruz
        self.candidate_models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Linear SVM": SVC(kernel='linear', C=1.0, random_state=42),
            "RBF SVM": SVC(kernel='rbf', C=1.0, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
        }
        
    def train_model(self):
        """Tüm modelleri eğitir, yarıştırır ve EN İYİSİNİ seçip kaydeder"""
        if not os.path.exists(DATA_PATH):
            print("\n❌ [EĞİTİM HATASI]: 'data/csi_fingerprints.csv' bulunamadı. Önce veri toplayın!")
            return False
            
        print("\n" + "="*50)
        print("🧠 YAPAY ZEKA MODEL ARENASI BAŞLIYOR")
        print("="*50)
        
        # 1. Veriyi Oku
        df = pd.read_csv(DATA_PATH)

        # Etiket sütunu farklı isimlerde olabilir; birkaç yaygın alternatifi dene
        possible_label_cols = ['Bölge_Etiketi', 'label', 'zone', 'Zone', 'Bölge']
        label_col = None
        for col in possible_label_cols:
            if col in df.columns:
                label_col = col
                break

        if label_col is None:
            print(f"\n❌ [EĞİTİM HATASI]: Beklenen etiket sütunu bulunamadı. Mevcut sütunlar: {list(df.columns)}")
            return False

        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # 2. Veriyi Parçala (%80 Eğitim, %20 Test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        best_model_name = None
        best_accuracy = -1
        best_model_object = None
        
        # 3. Modelleri Yarıştır
        print(f"📊 Toplam Veri Seti: {len(df)} Örnek Satırı. Modeller eğitiliyor...\n")
        
        for name, model in self.candidate_models.items():
            # Modeli eğit
            model.fit(X_train, y_train)
            # Test verisiyle tahmin et
            preds = model.predict(X_test)
            # Skoru hesapla
            acc = accuracy_score(y_test, preds)
            
            print(f"➡️ {name:<25} -> Başarı Oranı: %{acc*100:.2f}")
            
            # En iyisini bul ve hafızaya al
            if acc > best_accuracy:
                best_accuracy = acc
                best_model_name = name
                best_model_object = model

        print("\n" + "="*50)
        print(f"🏆 ŞAMPİYON MODEL: {best_model_name} (Başarı: %{best_accuracy*100:.2f})")
        print("="*50)
        
        # 4. Detaylı Rapor (Hangi odada ne kadar karıştırıyor?)
        final_preds = best_model_object.predict(X_test)
        print("\n📝 ŞAMPİYON MODELİN DETAYLI ANALİZ RAPORU:")
        print(classification_report(y_test, final_preds))
        
        # 5. Şampiyonu Bilgisayara Kaydet
        joblib.dump(best_model_object, MODEL_PATH)
        # Hangi modelin seçildiğini bir metin dosyasına da yazalım (UI okuyabilsin diye)
        with open('core/selected_model.txt', 'w') as f:
            f.write(best_model_name)
            
        print(f"💾 {best_model_name} modeli '{MODEL_PATH}' olarak sisteme entegre edildi.")
        return True