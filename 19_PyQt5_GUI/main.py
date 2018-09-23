import sys
from winForm1 import winForm1
from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets  import QWebEngineView
 
class MyWinForm(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        QWebEngineView.__init__(self)
        self.ui = winForm1()
        self.ui.setupUi(self)
 
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWinForm()
    myapp.show()
    app.exec_()