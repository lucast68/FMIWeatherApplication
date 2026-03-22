#Imports necessary packages for GUI
try:
    #Qt components
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
        QPushButton, QPlainTextEdit, QLabel, QSplitter, QMessageBox
    )
    #Functions for API requests
    from api import get_weather, get_prediction
    #Matplotlib for imbedded graph
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.dates as mdates
    #Data processing
    import pandas as pd
except ImportError:
    #Raise error if packages are missing
    raise RuntimeError(
        "Missing Python packages.\n"
        "Run: pip install -r requirements.txt to install the needed packages"
    )

#Creates human readable columns
COLUMN_NAMES = {
    "timestamp": "Timestamp",
    "temperature": "Temperature",
    "wind_speed": "Wind Speed",
    "wind_direction": "Wind Direction",
    "humidity": "Humidity",
    "pressure": "Pressure",
    "winddir_sin": "Wind Direction (sin)",
    "winddir_cos": "Wind Direction (cos)",
    "temperature_prediction_next_hour": "Predicted temperature",
    "abs_error": "AbsoluteError"
}


#Creates main window for Qt application
class WeatherWindow(QWidget):
    def __init__(self):
        super().__init__()
        #Main window configurations
        self.setWindowTitle("Weather and prediction application")
        self.setGeometry(100, 100, 1200, 600)

        #Main split: left = text data, right = graph
        splitter = QSplitter(Qt.Orientation.Horizontal)

        #Left panel: JSON text data + buttons
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        place = "Helsinki, Kaisaniemi"
        API = "Meteorologiska institutet (opendata.fmi.fi)"

        #Highes row with location and API info
        header_row = QHBoxLayout()
        self.location_label = QLabel(f"Plats: {place}")
        self.location_label.setStyleSheet("font-size:12px;")
        self.api_label = QLabel(f"Väder API: {API}")
        self.api_label.setStyleSheet("font-size:12px;")

        header_row.addWidget(self.location_label)
        header_row.addStretch(1) #Moves API label to the left
        header_row.addWidget(self.api_label)

        left_layout.addLayout(header_row)

        #Text field for JSON related data
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumSize(500,400)
        left_layout.addWidget(self.output)

        #Buttons to fetch weather data or predictions
        btn_layout = QHBoxLayout()
        btn1 = QPushButton("Hämta väder")
        btn1.clicked.connect(self.load_weather)
        btn2 = QPushButton("Hämta prediktion")
        btn2.clicked.connect(self.load_prediction)

        btn_layout.addWidget(btn1)
        btn_layout.addWidget(btn2)

        left_layout.addLayout(btn_layout)

        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)

        #Right panel: Graph for Matplotlib
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)

        #Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)


    #Fetches and shows actual weather data
    def load_weather(self):
        data = get_weather()
        units = data.get("units", {})
        rows = data.get("data", [])

        text = ""
        for row in rows:
            for k, v in row.items():
                sv_name = COLUMN_NAMES_SV.get(k, k)
                unit = units.get(k, "")
                text += f"{sv_name}: {v} {unit}\n"
            text += "-" * 40 + "\n"

        self.output.setPlainText(text)

    #Fetches and shows predictions and updates graph
    def load_prediction(self):
        
        try:
            data = get_prediction()
        except Exception as e:
            #Error handling if user doesn't run Flask before starting Qt client
            QMessageBox.critical(
                self,
                "error",
                "Couldn't fetch prediction data\n\n"
                "Make sure the Flask server (backend/app.py) is running before starting Qt.\n\n"
                "Details:\n"
                f"{str(e)}"
            )
            return
        units = data.get("units", {})
        rows = data.get("data", [])

        #Shows text data
        text = ""
        for row in rows:
            for k, v in row.items():
                name = COLUMN_NAME.get(k, k)
                unit = units.get(k, "")
                text += f"{name}: {v} {unit}\n"
            text += "-" * 40 + "\n"

        self.output.setPlainText(text)

        #Draws graph: Actual and predicted temperature
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        timestamps = [pd.to_datetime(r["timestamp"]) for r in rows]
        temps = [r["temperature_prediction_next_hour"] for r in rows]
        actuals = [r["temperature"] for r in rows]

        ax.plot(timestamps, actuals, marker="o", label="Faktisk temperatur")
        ax.plot(timestamps, temps, marker="o", label="Predikterad temperatur")
        ax.set_title("Prediktion vs faktisk temperatur nästa 60 min")
        ax.set_xlabel("Tidpunkt")
        ax.set_ylabel(f"Temperatur ({units.get('temperature_prediction_next_hour', '°C')})")
        ax.legend()
        ax.grid(True)

        #Formats time axis for readability
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))

        self.figure.autofmt_xdate(rotation=45)
        self.canvas.draw() #Updates graph

if __name__ == "__main__":
    app = QApplication([])
    #Sets consistent theme to platform
    app.setStyle("Fusion")
    #Styles for buttons and text fields
    app.setStyleSheet("""
        QPushButton { font-size: 14px; padding: 6px; }
        QPlainTextEdit { font-family: Consolas, monospace; font-size: 12px; }
    """)
    window = WeatherWindow()
    window.show()
    app.exec()
