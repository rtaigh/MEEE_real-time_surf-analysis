from PyQt5 import QtWidgets, QtGui,QtCore

import sys


def Window():
    my_app = QtWidgets.QApplication(sys.argv)

    w=QtWidgets.QWidget()
    w.setStyleSheet("background-color: black;")
    
    
    nom_plan_label = QtGui.QLabel()
    nom_plan_label.setText(nom_plan_vignette)
    nom_plan_label.setStyleSheet('color: yellow')



    l1 = QtWidgets.QLabel(w)
    l1.setText('Testing')
    l1.setFont(QtGui.QFont('SansSerif', 13))
   # l1.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    
    l1.setFont(QtGui.QFont('SansSerif', 10))
 #   l1.set
    w.setWindowTitle('draft 1')
   

    w.setGeometry(1000,500,800,600)
    
    w.show()
    sys.exit(my_app.exec_())
   
    
Window()

