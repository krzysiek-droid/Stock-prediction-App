import os
import subprocess
import sys

import pandas as pd
from PyQt6.QtCore import Qt, QModelIndex, QVariant, QAbstractItemModel, QSize
from PyQt6.QtGui import QFontMetrics, QPen
from PyQt6.QtWidgets import QStyledItemDelegate, QListView, QCompleter, QWidget, QApplication, QVBoxLayout

from matplotlib import pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import numpy as np
from matplotlib.patches import Rectangle


class CustomItemModel(QAbstractItemModel):
    def __init__(self, data_frame, parent=None):
        super().__init__(parent)
        self.data_frame = data_frame.copy()  # Avoid modifying original DataFrame

    def rowCount(self, parent=QModelIndex()):
        if not parent.isValid():  # For the top level
            return len(self.data_frame)  # Return the total number of rows in the DataFrame
        else:
            return 0  # No hierarchical structure

    def columnCount(self, parent=QModelIndex()):
        return 1

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            return f"{self.data_frame.loc[index.row(), 'ticker']}"
        else:
            # Handle other roles as needed (optional)
            # return a QVariant() for unsupported roles
            return QVariant()

    def index(self, row, column, parent=QModelIndex()):
        if not parent.isValid():  # For the top level items
            if 0 <= row < len(self.data_frame):
                return self.createIndex(row, column, None)  # Return a valid index
            else:
                return QModelIndex()  # Invalid index for out-of-bounds rows
        else:
            # You can implement hierarchical structure logic here if needed
            # In this case, we're assuming a flat structure, so return an invalid index
            return QModelIndex()

    def parent(self, index):
        # In this case, we're assuming a flat structure with no parent-child relationships
        # You can modify this logic if you introduce hierarchy in the future
        return QModelIndex()

    def getData(self):
        return self.data_frame

    def filter(self, filter_string):
        filter_string = filter_string.lower()
        self.filtered_data = [item for item in self.data_frame if filter_string in item.lower()]
        self.layoutChanged.emit()


class MyItemDelegate(QStyledItemDelegate):
    data_frame = None

    def __init__(self, parent=None):
        super(MyItemDelegate, self).__init__(parent)
        self.indent = 15
        self.padding = 10

    def paint(self, painter, option, index):
        if self.data_frame is None:
            model = index.model()
            while model and hasattr(model, 'sourceModel'):
                model = model.sourceModel()
            self.data_frame = model.getData()
        # getting font to be customized
        font = painter.font()

        # painting ticker
        font.setPointSize(12)  # Adjust font size
        font.setBold(True)  # Set bold font
        painter.setFont(font)
        # Call the base class paint method to draw default item appearance
        painter.save()
        painter.setFont(font)
        # option.rect.adjusted(left, top, right, bottom)
        indented_rect = option.rect.adjusted(self.padding, self.padding, -self.padding, -self.padding)
        # painting text (drawing)
        painter.drawText(indented_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         index.data().replace('.PL', ''))

        # painting the full name of a stock
        font.setPointSize(9)
        font.setBold(True)
        font.setItalic(True)
        painter.setFont(font)
        painter.save()
        # option.rect.adjusted(left, top, right, bottom)
        indented_rect = option.rect.adjusted(self.padding, self.padding, -self.padding, -self.padding)
        additional_info = self.data_frame.loc[self.data_frame['ticker'] == index.data()]
        painter.drawText(indented_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                         f"{additional_info.iloc[0]['full_name']}")
        # painting the stock group
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        painter.setFont(font)
        painter.save()
        additional_info = self.data_frame.loc[self.data_frame['ticker'] == index.data()]
        # option.rect.adjusted(left, top, right, bottom)
        indented_rect = option.rect.adjusted(2 * self.padding, 0, -0, 0)
        # Set color of painted text
        color_pen = QPen(Qt.GlobalColor.red)
        painter.setPen(color_pen)
        painter.drawText(indented_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
                         f"{additional_info.iloc[0]['group'].split('_')[0].upper()}")
        painter.restore()

        # painting the border of a listItem
        pen = QPen(Qt.GlobalColor.white)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(option.rect.left(), option.rect.bottom() + self.padding, option.rect.right(),
                         option.rect.bottom() + self.padding)
        painter.restore()

    def sizeHint(self, option, index):
        # Set desired font properties
        metrics = QFontMetrics(option.font)
        size = metrics.size(Qt.TextFlag.TextSingleLine, index.data())
        return QSize(size.width(), size.height() + 40)  # Add some padding


class CustomListView(QListView):
    def __init__(self):
        super(CustomListView, self).__init__()
        self.setStyleSheet(
            "background-color: rgb(40, 40, 40);"
            "color: white;"
            "border: 0px;"
            "border-radius: 5px; "
        )

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.verticalScrollBar().hide()


class CustomCompleter(QCompleter):
    def __init__(self, model, parent=None):
        super().__init__(model, parent)
        self.setCaseSensitivity(Qt.CaseSensitivity(0))


class CandlestickGraph(QWidget):
    UI_files_path = fr"UI files"

    def __init__(self, stockObject, looking_past=150, extra_days=5):
        super().__init__()
        self.ui_file_name = fr"chartWidget.ui"
        subprocess.run(["pyuic6", "-x", os.path.join(self.UI_files_path, self.ui_file_name), "-o",
                        self.ui_file_name.replace('.ui', '.py')], check=True)
        from chartWidget import Ui_Form
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.rectangles = []
        self.stock_name = stockObject.stock_name
        self.mouse_btn_pressed = None
        self.max_future_days = 10
        self.df = stockObject.ohlcv.copy().head(looking_past)
        self.extra_days = extra_days
        self.x_range_right = 0
        self.x_range_left = 30

        self.ui.stockNameLbl.setText(f"{self.stock_name}")
        self.ui.stockDataInfoLbl.setText(f"(Past {looking_past} days)")
        self.initUI()

    def initUI(self):
        _ = plt.subplots()
        self.figure: plt.Figure = _[0]
        self.ax_candlestick: plt.Axes = _[1]
        self.canvas = FigureCanvas(self.figure)
        self.ui.chartLayout.addWidget(self.canvas)

        self.ax_volume: plt.Axes = self.ax_candlestick.twinx()  # Copy the y-axis that shares the X-axis

        self.plot_candlestick()

    def plot_candlestick(self):

        self.ui.dateLbl.setText(f"{self.df['Date'][0].strftime('%Y-%m-%d')}")
        self.ui.openPriceLbl.setText(f"{self.df['open'][0]}")
        self.ui.closePriceLbl.setText(f"{self.df['close'][0]}")
        self.ui.highPriceLbl.setText(f"{self.df['high'][0]}")
        self.ui.lowPriceLbl.setText(f"{self.df['low'][0]}")
        self.ui.volumeLbl.setText(f"{format_number(self.df['volume'][0])}")

        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Date_Num'] = mdates.date2num(self.df['Date'])

        quotes = [tuple(x) for x in self.df[['Date_Num', 'open', 'high', 'low', 'close']].to_numpy()]

        # Draw candlesticks on the main axis
        candlestick_ohlc(self.ax_candlestick, quotes, width=0.6, colorup='g', colordown='r')

        # Draw the volume bars on the secondary axis
        self.ax_volume.bar(self.df['Date_Num'], self.df['volume'], width=0.6, color='white', alpha=0.3, align='center')

        self.ax_candlestick.xaxis_date()

        # self.ax_candlestick.set_xlabel(f"Date")
        self.x_lim_left = self.ax_candlestick.get_xlim()[0]
        self.x_lim_right = self.ax_candlestick.get_xlim()[1]
        self.ax_candlestick.set_xlim(self.df['Date_Num'].to_list()[self.x_range_left],
                                     self.df['Date_Num'].to_list()[self.x_range_right] + self.max_future_days)

        # Set the volume axis label and adjust the y-axis limits
        self.ax_volume.set_ylim(0, self.df['volume'].max() * 1.1)


        self.ax_candlestick.xaxis.remove_overlapping_locs = False

        major_locator = mdates.AutoDateLocator()
        self.ax_candlestick.xaxis.set_major_locator(major_locator)
        self.ax_candlestick.xaxis.set_major_formatter(mdates.DateFormatter("%d"))

        minor_locator = mdates.MonthLocator(bymonthday=15)
        self.ax_candlestick.xaxis.set_minor_locator(minor_locator)
        self.ax_candlestick.xaxis.set_minor_formatter(mdates.DateFormatter('\n%b %y'))

        self.ax_candlestick.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        self.ax_candlestick.tick_params('x', length=4, width=1, which='both', colors='white')
        self.ax_candlestick.tick_params(axis='x', which='major', labelsize=10)

        self.figure.subplots_adjust(bottom=0.15)

        self.customize_plot()

        self.canvas.draw()

        self.connect_events()

    def connect_events(self):
        self.canvas.mpl_connect('scroll_event', self.on_zoom)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.press_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_pan)

    def on_zoom(self, event):
        self.x_range_left = self.ax_candlestick.get_xlim()[0]
        self.x_range_right = self.ax_candlestick.get_xlim()[1]

        if event.button == 'up':  # Zooming in
            print(self.x_range_right - self.x_range_left)
            if self.x_range_right - self.x_range_left > 10:
                self.x_range_right -= 20
                self.x_range_left += 20
        elif event.button == 'down':  # Zooming out
            self.x_range_right += 10 if self.x_range_right <= (self.df['Date_Num'].to_numpy()[0] + self.max_future_days) \
                else 0
            self.x_range_left -= 10 if self.x_range_left >= (self.df['Date_Num'].to_numpy()[-1]) else 0

        self.ax_candlestick.set_xlim(self.x_range_left, self.x_range_right)
        self.canvas.draw()

    def on_press(self, event):
        if event.button == 1:
            self.mouse_btn_pressed = event
        elif event.button == 3:
            self.ax_candlestick.set_xlim(self.df['Date_Num'].to_numpy()[30],
                                         self.df['Date_Num'].to_numpy()[0] + self.max_future_days)
            self.canvas.draw()

    def press_release(self, event):
        self.mouse_btn_pressed = None
        self.button_released = event

    def on_pan(self, event):
        self.draw_crosshair(event)
        if self.mouse_btn_pressed is None or event.xdata is None:
            return
        dx = event.xdata - self.mouse_btn_pressed.xdata

        if dx != 0:
            dx_days = int(dx)
            self.x_range_right = self.ax_candlestick.get_xlim()[1] - dx_days
            self.x_range_left = self.ax_candlestick.get_xlim()[0] - dx_days
            self.ax_candlestick.set_xlim(self.x_range_left, self.x_range_right)
            self.canvas.draw()

    def draw_crosshair(self, event):
        if event.inaxes != self.ax_volume:
            return

        # Clear previous crosshair lines
        lines = self.ax_volume.get_lines()
        for line in lines:
            line.remove()

        for rectangle in self.rectangles:
            rectangle.remove()
        self.rectangles.clear()

        # Find nearest candlestick
        xdata = self.df['Date_Num']
        nearest_index = np.abs(xdata - event.xdata).argmin()
        nearest_row = self.df.iloc[nearest_index]
        nearest_x_tick = self.df['Date_Num'][nearest_index]

        # Draw new horizontal and vertical lines
        self.ax_volume.axhline(y=event.ydata, color='gray', linestyle='--', zorder=10)
        self.ax_volume.axvline(x=nearest_x_tick, color='gray', linestyle='--', zorder=10)

        # Convert mouse coordinates to data coordinates for both axes
        data_x, data_y_volume = self.ax_volume.transData.inverted().transform((event.x, event.y))
        data_x, data_y_candlestick = self.ax_candlestick.transData.inverted().transform((event.x, event.y))

        # Calculate the y-values corresponding to the horizontal line
        y_value_volume = data_y_volume
        y_value_candlestick = data_y_candlestick

        # Create rectangle annotations at the ends of the horizontal line
        rectangle_left = Rectangle((nearest_row['Date_Num'] - 0.3, y_value_candlestick - 0.1), 0.1, 0.2, color='green',
                                   alpha=0.5)
        rectangle_right = Rectangle((nearest_row['Date_Num'] + 0.2, y_value_volume - 0.1), 0.1, 0.2, color='green',
                                    alpha=0.5)
        self.ax_candlestick.add_patch(rectangle_left)
        self.ax_volume.add_patch(rectangle_right)
        self.rectangles.extend([rectangle_left, rectangle_right])

        self.ui.dateLbl.setText(f"{nearest_row['Date'].strftime('%Y-%m-%d')}")
        self.ui.openPriceLbl.setText(f"{nearest_row['open']}")
        self.ui.closePriceLbl.setText(f"{nearest_row['close']}")
        self.ui.highPriceLbl.setText(f"{nearest_row['high']}")
        self.ui.lowPriceLbl.setText(f"{nearest_row['low']}")
        self.ui.volumeLbl.setText(f"{format_number(nearest_row['volume'])}")

        # Redraw the canvas
        self.canvas.draw()

    def customize_plot(self):
        # Set the figure and axes background color
        self.figure.patch.set_facecolor((40 / 255, 40 / 255, 40 / 255))
        self.ax_candlestick.set_facecolor((60 / 255, 60 / 255, 60 / 255))
        self.ax_volume.set_facecolor((60 / 255, 60 / 255, 60 / 255))

        # Set the color of the axis labels and ticks
        self.ax_candlestick.tick_params(axis='x', colors='white')
        self.ax_candlestick.tick_params(axis='y', colors='white')
        self.ax_volume.tick_params(axis='y', colors='white')

        # Set the color of the spines (the box around the plot)
        self.ax_candlestick.spines['bottom'].set_color('white')
        self.ax_candlestick.spines['top'].set_color('white')
        self.ax_candlestick.spines['left'].set_color('white')
        self.ax_candlestick.spines['right'].set_color('white')
        self.ax_volume.spines['bottom'].set_color('white')
        self.ax_volume.spines['top'].set_color('white')
        self.ax_volume.spines['left'].set_color('white')
        self.ax_volume.spines['right'].set_color('white')

        # Set the color and font properties of the axis labels
        self.ax_candlestick.set_xlabel('Date', color='white', fontsize=14, fontweight='bold')
        self.ax_candlestick.set_ylabel('Price', color='white', fontsize=14, fontweight='bold')
        self.ax_volume.set_ylabel('Volume', color='white', fontsize=14, fontweight='bold')


def format_number(num):
    if num >= 1e9:
        return f'{num / 1e9:.2f}B'
    elif num >= 1e6:
        return f'{num / 1e6:.2f}M'
    elif num >= 1e3:
        return f'{num / 1e3:.2f}K'
    else:
        return f'{num:.2f}'


if __name__ == "__main__":
    import data_acquisition

    app = QApplication(sys.argv)
    s = data_acquisition.StockObj()
    s.stock_name = 'WIG20.PL'
    s.get_stooq_ohlcv(s.stock_name)
    stock_info = s.get_stock_info(s.stock_name)
    # graph = CandlestickChart(s.ohlcv, stock_info)
    graph = CandlestickGraph(s)
    graph.show()
    app.exec()
