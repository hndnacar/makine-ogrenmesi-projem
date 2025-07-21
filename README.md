# Makine Öğrenmesi ile Kredi Başvuru Tahmini (PyQt5 GUI)

Bu proje, kullanıcı dostu bir PyQt5 arayüzü üzerinden kredi başvurusunun onaylanıp onaylanmayacağını tahmin etmek amacıyla oluşturulmuştur. Projede veri işleme, model eğitimi ve değerlendirme süreçleri grafiksel arayüz ile entegre edilmiştir.

## 🧠 Kullanılan Makine Öğrenmesi Modelleri
- K-Nearest Neighbors (KNN)
- Lojistik Regresyon
- Karar Ağacı (Decision Tree)

## 📊 Uygulama Özellikleri

### 🔹 Orijinal Veri Sayfası
- Kategorik verilerin sayısal verilere dönüştürülmesi
- Model seçimi ve eğitimi
- K-Fold Cross Validation ile performans değerlendirmesi
- Doğruluk, hassasiyet, F1 skoru gibi metriklerin hesaplanması

### 🔹 Gürültülü Veri Sayfası
- Eksik verilerin doldurulması
- Gürültülü veri ile model eğitimi ve performans analizi
- Görselleştirmeler ile model çıktılarının sunulması

### 🔹 Normalizasyon Sayfası
- MinMaxScaler ve StandardScaler ile veri ölçekleme
- Eğitim ve test işlemleri
- Aşırı uyum/yetersiz uyum analizi
- Karmaşıklık matrisi görselleştirmesi

### 🔹 Dengesizlik Sayfası
- Sınıf dengesizliğini gösteren grafikler
- ROS (Random OverSampling) ve RUS (Random UnderSampling) yöntemleriyle veri dengelenmesi
- Dengesizlik giderildikten sonra model eğitimi
- İlk 10 örnek için tahmin ve olasılık gösterimi

### 🔹 Tahmin Sayfası (Test Paneli)
- Kullanıcıdan veri girişi
- Model tahmin sonucu mesaj kutusunda gösterilir
- Hataların kullanıcıya bildirilmesi

## 🖥️ Arayüz Teknolojisi
- **PyQt5** kullanılarak modern ve sezgisel bir masaüstü uygulaması geliştirilmiştir.

## 📁 Proje Yapısı
```bash
├── main.py                   # PyQt5 uygulama giriş noktası
├── gui/                      # Farklı sayfaları içeren PyQt5 ekranları
│   ├── orijinal_page.py
│   ├── gurultulu_page.py
│   ├── normalizasyon_page.py
│   ├── dengesizlik_page.py
│   └── tahmin_page.py
├── utils/
│   ├── preprocessing.py      # Eksik veri doldurma, kategorik dönüşüm
│   ├── normalization.py
│   ├── imbalance.py
│   ├── model_utils.py        # Model eğitim, test, KFold, metrikler
│   └── visualizations.py     # Grafiksel gösterimler
├── dataset/
│   └── loan_data.csv
└── README.md

📌 Notlar
Uygulama, kullanıcıya modelin nasıl eğitildiğini ve sonuçların nasıl analiz edildiğini görsel olarak sunmayı amaçlamaktadır.

Eğitim ve tahmin süreçleri tek ekran üzerinden kolayca yönetilebilir.