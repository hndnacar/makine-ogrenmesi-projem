from PyQt5.QtWidgets import *
from AnaEkran_python import Ui_MainWindow
from Orijinal import OrijinalPage
from Gürültülü import GurultuluPage
from Dengesizlik import DengesizlikPage
from Normalizasyon import NormalizasyonPage

class AnaEkranPage(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.AnaEkranform = Ui_MainWindow()
        self.AnaEkranform.setupUi(self)

        # Orijinal veri seti butonu bağlantısı
        self.AnaEkranform.pushButton_Original.clicked.connect(self.GirisYap)
        self.AnaEkranform.pushButton_Gurultulu.clicked.connect(self.GirisYap2)
        self.AnaEkranform.pushButton_Dengesizlik.clicked.connect(self.GirisYap3)
        self.AnaEkranform.pushButton_Normalize.clicked.connect(self.GirisYap5)

    def GirisYap(self):# butona basıldığında neler yapılacağını yaptır
        self.OrijinalEkranac = OrijinalPage()
        self.OrijinalEkranac.show()

    def GirisYap2(self):
        self.GürültüEkranAc = GurultuluPage()
        self.GürültüEkranAc.show()

    def GirisYap3(self):
        self.DengesizlikEkranAc = DengesizlikPage()
        self.DengesizlikEkranAc.show()

    def GirisYap5(self):
        self.NormalizasyonEkranAc = NormalizasyonPage()
        self.NormalizasyonEkranAc.show()









