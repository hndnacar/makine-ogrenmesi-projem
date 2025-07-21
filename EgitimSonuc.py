from PyQt5.QtWidgets import QWidget, QMessageBox
from EgitimSonuc_python import Ui_Form9

class EgitimSonucuPage(QWidget):
    def __init__(self, model, scaler) -> None:
        super().__init__()
        self.ui = Ui_Form9()  # Arayüzü burada başlatıyoruz
        self.ui.setupUi(self)

        self.model = model  # Eğitilmiş modeli burada saklıyoruz
        self.scaler = scaler  # Scaler'ı burada saklıyoruz
        self.ui.pushButtonSonuc.clicked.connect(self.predict_result)

    def predict_result(self):
        try:
            # Kullanıcıdan alınan girdileri topla
            kisi_yasi = float(self.ui.lineEditYas.text())
            kisi_cinsiyeti = float(self.ui.lineEditCinsiyet.text())
            kisi_egitim_seviyesi = float(self.ui.lineEditEgitimSeviye.text())  # İsim düzeltmesi
            yillik_gelir = float(self.ui.lineEditYillikGelir.text())
            yillik_calisma_denemimi = float(self.ui.lineEditDeneyim.text())
            ev_sahipligi_durumu = float(self.ui.lineEditEvDurumu.text())
            talep_edilen_kredi = float(self.ui.lineEditKrediTutari.text())
            kredi_amaci = float(self.ui.lineEditKrediAmaci.text())
            kredi_faiz_orani = float(self.ui.lineEditKrediAmaci_2.text())
            kredi_tutar_yillik_gelir = float(self.ui.lineEditKrediYillikGelir.text())
            kredi_gecmis_uzunluk = float(self.ui.lineEditKrediGecmisi.text())
            kisi_kredi_notu = float(self.ui.lineEditKrediNot.text())
            kredi_temerrut = float(self.ui.lineEditOncekiKredi.text())

            # Girdileri bir listeye koy
            features = [[
                kisi_yasi, kisi_cinsiyeti, kisi_egitim_seviyesi, yillik_gelir,
                yillik_calisma_denemimi, ev_sahipligi_durumu, talep_edilen_kredi,
                kredi_amaci, kredi_faiz_orani, kredi_tutar_yillik_gelir,
                kredi_gecmis_uzunluk, kisi_kredi_notu, kredi_temerrut
            ]]

            # Eğitimde kullanılan scaler ile veriyi ölçeklendir
            features_scaled = self.scaler.transform(features)  # Burada scaler kullanarak özellikleri ölçeklendiriyoruz

            # Modeli kullanarak tahmin yap
            prediction = self.model.predict(features_scaled)[0]
            tahmin_sonucu = "Kredi Onaylandı" if prediction == 1 else "Kredi Onaylanmadı"

            # Sonucu kullanıcıya göster
            QMessageBox.information(self, "Tahmin Sonucu", tahmin_sonucu)

        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Girdiler yanlış veya eksik: {str(e)}")
