import sys
from browser import BrowserDialog
from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets  import QWebEngineView
 
class MyBrowser(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        QWebEngineView.__init__(self)
        self.ui = BrowserDialog()
        self.ui.setupUi(self)
        self.ui.lineEdit.returnPressed.connect(self.loadURL)
 
    def loadURL(self):
        url = self.ui.lineEdit.text()
        self.ui.qwebview.load(QUrl(url))
        self.show()  
 
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyBrowser()
    myapp.ui.qwebview.load(QUrl('http://www.pythonspot.com'))
    myapp.show()
    sys.exit(app.exec_())