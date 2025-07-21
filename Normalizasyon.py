from PyQt5.QtWidgets import *
from Normalizasyon_python import Ui_Normalizasyon
from EgitimSonuc import EgitimSonucuPage
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTextEdit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class NormalizasyonPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.Normalizasyonform = Ui_Normalizasyon()
        self.Normalizasyonform.setupUi(self)# ana metodu self ile çağırıcam.

        self.veri_seti = None  # Veri setini burada saklayacağız
        self.model = None  # Modeli burada saklayacağız
        self.scaler = None  # Başlangıçta scaler None

        # "Veri Setini Yükle" butonunu tıklama olayına bağla
        self.Normalizasyonform.pushButton_veriSetiYkle.clicked.connect(self.veri_seti_yukle)

        # "Veri Setini Göster" butonunu tıklama olayına bağla
        self.Normalizasyonform.pushButton_veriSetiGster.clicked.connect(self.veri_setini_goster)

        # "Sayısal Olmayan Sütunları Kategorize Et" butonunu tıklama olayına bağla
        self.Normalizasyonform.pushButton_onslemeHazrlk.clicked.connect(self.kategorize_et)

        self.Normalizasyonform.pushButtonNormalizeEt.clicked.connect(self.normalize_veri_seti)

        self.Normalizasyonform.pushButtonNormalizeEt_2.clicked.connect(self.normalize_veri_seti2)

        self.Normalizasyonform.pushButton_veriSetiniEgit.clicked.connect(self.train_model)

        self.Normalizasyonform.pushButton_predict_proba.clicked.connect(self.predict_methods)

        self.Normalizasyonform.pushButtonMetrikler.clicked.connect(self.show_metrics)

        # "Model Eğitimini Göster" butonunu tıklama olayına bağla
        self.Normalizasyonform.pushButtonModelOrnek.clicked.connect(self.open_egitim_sonucu)

    def set_model(self, model):
        self.model = model  # Eğitilmiş modeli kaydediyoruz

    def set_scaler(self, scaler):
        self.scaler = scaler  # Scaler'ı kaydediyoruz

    def open_egitim_sonucu(self):
        if not hasattr(self, 'model'):
            QMessageBox.warning(self, "Error", "No trained model found. Please train the model first.")
            return
        self.egitimsonuc_window = EgitimSonucuPage(self.model, self.scaler)
        self.egitimsonuc_window.show()

    def veri_seti_yukle(self):
        # Dosya seçim penceresi aç
        dosya_ismi, _ = QFileDialog.getOpenFileName(
            self,
            "Veri Setini Yükle",  # Pencere başlığı
            "",  # Varsayılan başlangıç dizini
            "Veri Dosyaları (*.csv *.xlsx);;Tüm Dosyalar (*.*)"  # Filtreler
        )

        if dosya_ismi:  # Kullanıcı bir dosya seçtiyse
            try:
                # Dosya uzantısını kontrol et
                if dosya_ismi.endswith('.csv'):
                    try:
                        self.veri_seti = pd.read_csv(dosya_ismi, encoding='utf-8')
                    except UnicodeDecodeError:
                        self.veri_seti = pd.read_csv(dosya_ismi, encoding='ISO-8859-1')

                elif dosya_ismi.endswith('.xlsx'):
                    self.veri_seti = pd.read_excel(dosya_ismi)
                else:
                    raise ValueError("Desteklenmeyen dosya formatı!")

                # Veri setini başarıyla yüklendi mesajı göster
                QMessageBox.information(self, "Başarılı", "Veri seti başarıyla yüklendi!")

            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Veri seti yüklenirken bir hata oluştu:\n{str(e)}")

        else:
            QMessageBox.warning(self, "Uyarı", "Dosya seçimi iptal edildi. Lütfen bir dosya seçin.")

    def veri_setini_goster(self):
        if self.veri_seti is None:
            QMessageBox.warning(self, "Uyarı", "Henüz bir veri seti yüklenmedi!")
            return

        try:

            veri_goruntu = self.veri_seti.head(10)

            # QTableWidget ile tablo oluştur
            table_widget = QTableWidget()
            table_widget.setRowCount(len(veri_goruntu))  # Satır sayısı
            table_widget.setColumnCount(len(veri_goruntu.columns))  # Sütun sayısı
            table_widget.setHorizontalHeaderLabels(veri_goruntu.columns)  # Sütun başlıklarını ayarla

            # Verileri tabloya yerleştir
            for row_index, row in veri_goruntu.iterrows():
                for col_index, value in enumerate(row):
                    table_widget.setItem(row_index, col_index, QTableWidgetItem(str(value)))

            # Yeni bir pencere açarak tabloyu göster
            table_window = QMainWindow(self)
            table_window.setWindowTitle("Veri Seti Görüntüleme")
            table_window.setCentralWidget(table_widget)
            table_window.resize(800, 400)
            table_window.show()

        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri seti gösterilirken bir hata oluştu:\n{str(e)}")

    def kategorize_et(self):
        if self.veri_seti is None:
            QMessageBox.warning(self, "Uyarı", "Henüz bir veri seti yüklenmedi!")
            return

        try:
            # Cinsiyet sütununu sayısal değerlere dönüştür
            def convert_gender_to_numeric(gender):
                if gender == "male":
                    return 1
                elif gender == "female":
                    return 0
                return None

            self.veri_seti['kisinin_cinsiyeti'] = self.veri_seti['kisinin_cinsiyeti'].apply(convert_gender_to_numeric)

            # Yes/No verilerini 1 ve 0'a dönüştürme
            boolean_columns = ['önceki_kredi_temerrütlerinin_göstergesi']
            for col in boolean_columns:
                self.veri_seti[col] = self.veri_seti[col].astype(str).str.strip().str.lower()
                self.veri_seti[col] = self.veri_seti[col].map({"yes": 1, "no": 0}).astype(int)

            # Eğitim seviyesi sütununu sayısal verilere dönüştür
            def convert_education_to_numeric(education_level):
                if education_level == "Bachelor":
                    return 1
                elif education_level == "Associate":
                    return 2
                elif education_level == "High School":
                    return 3
                elif education_level == "Master":
                    return 4
                elif education_level == "Doctorate":
                    return 5
                return None

            self.veri_seti['kisinin_egitim_seviyesi'] = self.veri_seti['kisinin_egitim_seviyesi'].apply(
                convert_education_to_numeric)

            # Ev sahipliği durumunu sayısal verilere dönüştürmek için fonksiyon
            def convert_housing_status_to_numeric(housing_status):
                if housing_status == "RENT":
                    return 1
                elif housing_status == "MORTGAGE":
                    return 2
                elif housing_status == "OWN":
                    return 3
                elif housing_status == "OTHER":
                    return 4
                return None

            self.veri_seti['ev_sahipligi_durumu'] = self.veri_seti['ev_sahipligi_durumu'].apply(
                convert_housing_status_to_numeric)

            # Kredi amacı sütununu sayısal verilere dönüştürmek için fonksiyon
            def convert_loan_purpose_to_numeric(loan_purpose):
                if loan_purpose == "EDUCATION":
                    return 1
                elif loan_purpose == "MEDICAL":
                    return 2
                elif loan_purpose == "VENTURE":
                    return 3
                elif loan_purpose == "PERSONAL":
                    return 4
                elif loan_purpose == "DEBTCONSOLIDATION":
                    return 5
                elif loan_purpose == "HOMEIMPROVEMENT":
                    return 6
                return None

            self.veri_seti['kredinin_amaci'] = self.veri_seti['kredinin_amaci'].apply(convert_loan_purpose_to_numeric)

            # Başarılı mesajı
            QMessageBox.information(self, "Başarılı", "Sayısal Olmayan Sütunlar Başarıyla Kategorize Edildi!")

        except Exception as e:
            QMessageBox.critical(self, "Hata",
                                 f"Sayısal Olmayan Sütunlar Kategorize Edilirken Bir Hata Oluştu:\n{str(e)}")

    def normalize_veri_seti(self):
        if self.veri_seti is None:
            QMessageBox.warning(self, "Uyarı", "Henüz bir veri seti yüklenmedi!")
            return

        try:
            # Sadece sayısal sütunları seç
            sayisal_sutunlar = self.veri_seti.select_dtypes(include=['float64', 'int64']).columns

            # Normalize edici MinMaxScaler kullanımı
            scaler = MinMaxScaler()

            # Sayısal sütunları normalize et
            self.veri_seti[sayisal_sutunlar] = scaler.fit_transform(self.veri_seti[sayisal_sutunlar])

            # Başarı mesajı
            QMessageBox.information(self, "Başarılı", "Veri seti başarıyla normalize edildi!")

        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri seti normalize edilirken bir hata oluştu:\n{str(e)}")

    def normalize_veri_seti2(self):
        if self.veri_seti is None:
            QMessageBox.warning(self, "Uyarı", "Henüz bir veri seti yüklenmedi!")
            return

        try:
            # Sadece sayısal sütunları seç
            sayisal_sutunlar = self.veri_seti.select_dtypes(include=['float64', 'int64']).columns

            # Normalize edici StandardScaler kullanımı
            scaler = StandardScaler()

            # Sayısal sütunları normalize et
            self.veri_seti[sayisal_sutunlar] = scaler.fit_transform(self.veri_seti[sayisal_sutunlar])

            # Başarı mesajı
            QMessageBox.information(self, "Başarılı", "Veri seti başarıyla Z-skoru ile normalize edildi!")

        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri seti normalize edilirken bir hata oluştu:\n{str(e)}")

        # Modeli eğitme ve scaler'ı kaydetme
    def train_model(self):
        if self.veri_seti is None:
            QMessageBox.warning(self, "Error", "Dataset is not loaded. Please load a dataset first.")
            return

        if self.veri_seti.isnull().values.any():
            QMessageBox.warning(self, "Error", "Dataset contains missing values. Please clean the data first.")
            return

        try:
            # Veri ayrımı
            X = self.veri_seti.iloc[:, :-1]
            y = self.veri_seti.iloc[:, -1]

            # Verileri ölçeklendir
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # X_scaled'i pandas DataFrame'e çevirerek sütun isimlerini koruyun
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

            # Eğitim ve test verisi olarak ayır
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

            # Kullanıcı tarafından seçilen modeli belirle
            if self.Normalizasyonform.radioButton_knn.isChecked():
                self.model = KNeighborsClassifier(n_neighbors=5)  # Varsayılan olarak 5 komşu
                model_name = "KNN"
            elif self.Normalizasyonform.radioButton_lr.isChecked():
                self.model = LogisticRegression(max_iter=1000, solver='saga')  # Solver ve max_iter ayarlandı
                model_name = "Logistic Regression"
            elif self.Normalizasyonform.radioButton_dt.isChecked():
                self.model = DecisionTreeClassifier(random_state=42)
                model_name = "Decision Tree"
            else:
                QMessageBox.warning(self, "Error", "No model selected. Please select a model.")
                return

            # Modeli eğit
            self.model.fit(X_train, y_train)

            # Model ve scaler'ı ana sayfada kaydediyoruz
            self.set_model(self.model)
            self.set_scaler(scaler)

            # Test verisi ile doğruluk hesapla
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Eğitim ve test doğruluklarını hesapla
            train_acc = round(self.model.score(X_train, y_train) * 100, 2)
            test_acc = round(self.model.score(X_test, y_test) * 100, 2)

            # Performans mesajı oluştur
            if abs(train_acc - test_acc) < 5:
                evaluation_msg = "The model is well-trained!"
            elif train_acc > test_acc:
                evaluation_msg = "The model might be overfitting!"
            else:
                evaluation_msg = "The model might be underfitting!"

            # Sonuçları GUI'de göster
            QMessageBox.information(
                self,
                "Model Training Results",
                f"Model: {model_name}\n"
                f"Training Accuracy: {train_acc:.2f}%\n"
                f"Testing Accuracy: {test_acc:.2f}%\n\n"
                f"Evaluation: {evaluation_msg}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during training: {str(e)}")

    def predict_methods(self):
        if not hasattr(self, "model") or self.model is None:
            QMessageBox.warning(self, "Error", "No trained model found. Please train a model first.")
            return

        try:
            # Test setini kontrol edin
            X_test = self.veri_seti.iloc[:, :-1].values  # Özellikler
            y_test = self.veri_seti.iloc[:, -1].values  # Etiketler

            # Modelin ölçeklendirilmiş test verisine erişmesi gerekebilir
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_test)
            self.scaler = scaler  # scaler'ı kaydediyoruz

            # Predict (tahmin) işlemi
            predictions = self.model.predict(X_scaled)

            # Predict proba (olasılık tahmini) işlemi (eğer destekleniyorsa)
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X_scaled)
            else:
                probabilities = "Bu model olasılık tahmini yapamaz."

            # Sonuçları GUI'de göster
            if isinstance(probabilities, str):
                QMessageBox.information(
                    self,
                    "Prediction Results",
                    f"Predictions: {predictions}\n\n"
                    f"Probability Support: {probabilities}"
                )
            else:
                QMessageBox.information(
                    self,
                    "Prediction Results",
                    f"Predictions: {predictions}\n\n"
                    f"Prediction Probabilities:\n{probabilities}"
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during prediction: {str(e)}")

    def show_metrics(self):
        if self.veri_seti is None:
            QMessageBox.warning(self, "Error", "Dataset is not loaded. Please load a dataset first.")
            return

        try:
            # Özellikler ve hedef değişkeni ayır
            X = self.veri_seti.iloc[:, :-1]
            y = self.veri_seti.iloc[:, -1]

            # Verileri ölçeklendir
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Eğitim ve test setlerine ayır
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Modelleri tanımla
            models = [
                ("KNN", KNeighborsClassifier(n_neighbors=5)),
                ("Decision Tree", DecisionTreeClassifier(random_state=42)),
                ("Logistic Regression", LogisticRegression(max_iter=1000, solver='saga'))
            ]

            for model_name, model in models:
                # Modeli eğit
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Karmaşıklık matrisi oluştur
                conf_matrix = confusion_matrix(y_test, y_pred)
                TN, FP, FN, TP = conf_matrix.ravel()

                # Metrikleri hesapla
                accuracy = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                specificity = TN / (TN + FP)
                f1 = f1_score(y_test, y_pred)

                # Karmaşıklık matrisini görselleştir
                fig, ax = plt.subplots(figsize=(8, 6))  # Yeni bir figür oluştur
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"],
                            yticklabels=["Class 0", "Class 1"], ax=ax)
                ax.set_title(f"Confusion Matrix - {model_name}")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")

                # Metrikleri grafik dışında, alt kısımda göstereceğiz
                metrics_text = (
                    f"Accuracy: {accuracy:.4f}\n"
                    f"Recall: {recall:.4f}\n"
                    f"Precision: {precision:.4f}\n"
                    f"Specificity: {specificity:.4f}\n"
                    f"F1 Score: {f1:.4f}"
                )

                # Grafik alanını alt kısma kaydır
                plt.subplots_adjust(bottom=0.3)  # Alt kısmı boşaltalım

                # Metrikleri alt kısma ekleyelim
                plt.figtext(0.5, 0.05, metrics_text, wrap=True, horizontalalignment='center', fontsize=12,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))

                # Grafiği göster
                plt.show()

        except Exception as e:
            # Hata durumunda daha detaylı bilgi yazdır
            print(f"Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error during metrics calculation: {str(e)}")
