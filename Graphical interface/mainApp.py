import logging
import os.path
import subprocess
import sys


from PyQt6.QtWidgets import QApplication, QMainWindow, QDialog, QWidget, QLineEdit
from PyQt6.QtCore import QCoreApplication, pyqtSignal
from data_acquisition import StockObj
from minorClasses import CandlestickGraph

UI_files_path = fr"UI files"

class Application(QApplication):
    def __init__(self, *args, **kwargs):
        super(Application, self).__init__(*args, **kwargs)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        self.setApplicationName("Stock prediction App")
        self.setStyle("Windows")
        self.mainWindow = MainWindow()
        self.dialogs = []

        self.mainWindow.show()
        self.exec()

    def run(self):
        self.dialogs[0].show()
        self.exec()


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui_file_name = fr"mainWindow.ui"
        subprocess.run(["pyuic6", "-x", os.path.join(UI_files_path, self.ui_file_name), "-o",
                        self.ui_file_name.replace('.ui', '.py')], check=True)
        from mainWindow import Ui_MainWindow
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.stockObject = StockObj()
        self.stockObject.get_stooq_ohlcv('WIG20.PL')
        self.CandlestickGraph = CandlestickGraph(self.stockObject)

        s = TickerSelector(self.stockObject, self.ui.searchLine)
        s.tickerChanged.connect(lambda x: self.changeTicker(x))
        self.ui.graphBoxLayout.addWidget(self.CandlestickGraph)

    def changeTicker(self, stockObject: StockObj):
        self.stockObject = stockObject
        ticker = self.stockObject.stock_name
        self.stockObject.get_stooq_ohlcv(ticker)

        previous_graph = self.CandlestickGraph
        self.ui.graphBoxLayout.removeWidget(previous_graph)
        previous_graph.destroy()
        self.CandlestickGraph = CandlestickGraph(self.stockObject)
        self.ui.graphBoxLayout.addWidget(self.CandlestickGraph)


class TickerSelector(QLineEdit):
    from data_acquisition import StockObj
    from minorClasses import CustomItemModel, MyItemDelegate, CustomListView, CustomCompleter
    tickerChanged = pyqtSignal(object)

    def __init__(self, initial_stockObject=None, lineEditInstance: QLineEdit = None):
        super(TickerSelector, self).__init__()
        self._selected_stock = None

        self.stockObject = self.StockObj() if initial_stockObject is None else initial_stockObject
        self.stockObject.get_stock_info()

        self.completerModel = self.CustomItemModel(self.stockObject.stock_info)
        self.completer = self.CustomCompleter(self.completerModel, self)
        self.completerListView = self.CustomListView()

        itemDelegateParent = self if lineEditInstance is None else lineEditInstance
        self.completerItemDelegate = self.MyItemDelegate(itemDelegateParent)

        self.completer.setPopup(self.completerListView)
        self.completerListView.setItemDelegate(self.completerItemDelegate)

        self.completer.activated.connect(lambda x: self.selected_stock(x))

        if lineEditInstance is None:
            self.setCompleter(self.completer)
            self.editingFinished.connect(self.selected_stock)
        else:
            lineEditInstance.setCompleter(self.completer)
            lineEditInstance.editingFinished.connect(self.selected_stock)

    def selected_stock(self, stock_ticker):
        self._selected_stock = stock_ticker
        self.stockObject.stock_name = stock_ticker
        self.tickerChanged.emit(self.stockObject)

        print(f"Stock -> {self._selected_stock} has been selected.")


if __name__ == "__main__":
    print(f"Application starting...")
    app = Application(sys.argv)
