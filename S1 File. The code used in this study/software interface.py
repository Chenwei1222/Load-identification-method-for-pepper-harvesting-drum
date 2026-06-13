import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt
import pandas as pd
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QStackedWidget, QListWidget, QListWidgetItem,
    QToolBar, QAction, QPushButton, QLabel, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QDateEdit, QDialog, QDialogButtonBox,
    QMessageBox, QGroupBox, QSplitter, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer, QDate, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from datetime import datetime

# Configure Matplotlib fonts
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# --------------------------- Core calculation model (refactored, visualization removed) ---------------------------
class LorenzRosslerModel:
    def __init__(self, fs=100, T=10.0):
        """Initialize the coupled Lorenz-Rossler model"""
        self.fs = fs  # Sampling frequency
        self.T = T
        self.dt = 1 / fs
        self.N = int(T * fs)
        self.t = np.linspace(0, T, self.N)

        # Model parameters
        self.sigma = 10.0
        self.r = 28.0
        self.b = 8.0 / 3.0
        self.a = 0.2
        self.c = 5.7
        self.gamma = 0.1

        # Sensor calibration coefficients
        self.k_vib = 1.0
        self.k_torque = 1.0

        # Initial conditions
        self.x1_0 = 0.1
        self.x2_0 = 0.1
        self.x3_0 = 0.1
        self.y1_0 = 0.1
        self.y2_0 = 0.1
        self.y3_0 = 0.1

        # Filter parameters
        self.vib_cutoff_hz = 40
        self.torque_cutoff_hz = 20
        self.filter_order = 2

        # Load types (updated to match the new intervals)
        self.load_types = ["No Load", "Light Load", "Normal Load", "Overload"]
        self.current_load = "No Load"

    def set_parameters(self, sigma=None, r=None, b=None, a=None, c=None, gamma=None):
        if sigma is not None: self.sigma = sigma
        if r is not None: self.r = r
        if b is not None: self.b = b
        if a is not None: self.a = a
        if c is not None: self.c = c
        if gamma is not None: self.gamma = gamma

    def set_sampling_frequency(self, vib_fs=None, torque_fs=None):
        """Set the sampling frequency"""
        if vib_fs:
            self.fs = vib_fs
            self.dt = 1 / self.fs
            self.N = int(self.T * self.fs)
            self.t = np.linspace(0, self.T, self.N)
            self.vib_cutoff_hz = min(40, int(vib_fs * 0.4))  # The cutoff frequency must not exceed 40% of the sampling frequency
        if torque_fs:
            self.torque_cutoff_hz = min(20, int(torque_fs * 0.2))  # The cutoff frequency must not exceed 20% of the sampling frequency

    def set_sensor_coefficients(self, k_vib=None, k_torque=None):
        if k_vib is not None: self.k_vib = k_vib
        if k_torque is not None: self.k_torque = k_torque

    def set_initial_conditions(self, x1_0=None, x2_0=None, x3_0=None,
                               y1_0=None, y2_0=None, y3_0=None):
        if x1_0 is not None: self.x1_0 = x1_0
        if x2_0 is not None: self.x2_0 = x2_0
        if x3_0 is not None: self.x3_0 = x3_0
        if y1_0 is not None: self.y1_0 = y1_0
        if y2_0 is not None: self.y2_0 = y2_0
        if y3_0 is not None: self.y3_0 = y3_0

    def coupled_equations(self, t, state):
        x1, x2, x3, y1, y2, y3 = state
        dx1_dt = self.sigma * (x2 - x1) + self.gamma * (y1 + y2)
        dx2_dt = x1 * (self.r - x3) - x2 + self.gamma * (y2 + y3)
        dx3_dt = x1 * x2 - self.b * x3 + self.gamma * (y1 + y3)
        dy1_dt = -y2 - y3 + self.gamma * (x1 + x2)
        dy2_dt = y1 + self.a * y2 + self.gamma * (x2 + x3)
        dy3_dt = self.b + y1 * y3 - self.c * y3 + self.gamma * (x1 + x3)
        return [dx1_dt, dx2_dt, dx3_dt, dy1_dt, dy2_dt, dy3_dt]

    def solve_ivp_method(self, num_points=None):
        """Solve the differential equations"""
        if num_points is None:
            num_points = self.N
        t = np.linspace(0, self.T, num_points)
        y0 = [self.x1_0, self.x2_0, self.x3_0, self.y1_0, self.y2_0, self.y3_0]
        sol = solve_ivp(self.coupled_equations, [0, self.T], y0, t_eval=t, method='RK45')
        return sol.y, t

    def design_filter(self, cutoff_hz, fs, order=2):
        """Design a digital filter"""
        nyq = 0.5 * fs
        normalized_cutoff = min(max(cutoff_hz / nyq, 0.01), 0.9)
        b, a = butter(order, normalized_cutoff, btype='low')
        return b, a

    def filter_signal(self, signal, cutoff_hz, fs, order=2):
        """Filter the signal"""
        b, a = self.design_filter(cutoff_hz, fs, order)
        filtered = filtfilt(b, a, signal)
        return filtered

    def reconstruct_state_from_sensors(self, time, vib_signal, torque_signal):
        """Reconstruct six-dimensional state variables from vibration and torque signals"""
        # Calculate the actual sampling frequency
        time_diff = np.diff(time)
        actual_fs = 1.0 / np.mean(time_diff) if len(time_diff) > 0 else self.fs

        # Signal filtering
        vib_signal = self.filter_signal(vib_signal, self.vib_cutoff_hz, actual_fs, self.filter_order)
        torque_signal = self.filter_signal(torque_signal, self.torque_cutoff_hz, actual_fs, self.filter_order)

        # Align signal lengths
        min_len = min(len(time), len(vib_signal), len(torque_signal))
        time = time[:min_len]
        vib_signal = vib_signal[:min_len]
        torque_signal = torque_signal[:min_len]

        # Apply calibration coefficients
        x1 = self.k_vib * vib_signal
        y2 = self.k_torque * torque_signal

        # Derived features
        x2 = np.gradient(x1) * actual_fs
        x3 = np.convolve(x1, np.ones(5) / 5, mode='same')
        x3 = x3 - np.mean(x3)
        y1 = np.gradient(y2) * actual_fs
        y3 = np.abs(np.gradient(y1)) * actual_fs
        y3 = y3 / np.max(y3) if np.max(y3) > 0 else y3

        return x1, x2, x3, y1, y2, y3, time

    def read_raw_sensor_data(self, file_path, sheet_name='Sheet1',
                             time_col='Time', vib_col='Vibration', torque_col='Torque'):
        """Read raw sensor data"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            required_cols = [time_col, vib_col, torque_col]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            time = df[time_col].values
            vib_signal = df[vib_col].values
            torque_signal = df[torque_col].values

            # Estimate the sampling frequency
            time_diff = np.diff(time)
            avg_dt = np.mean(time_diff)
            estimated_fs = 1 / avg_dt if avg_dt > 0 else self.fs

            return time, vib_signal, torque_signal, estimated_fs
        except Exception as e:
            print(f"Data reading error: {e}")
            return None, None, None, None

    def calculate_jacobian(self, state):
        x1, x2, x3, y1, y2, y3 = state
        J = np.zeros((6, 6))
        J[0, 0] = -self.sigma;
        J[0, 1] = self.sigma;
        J[0, 3] = self.gamma;
        J[0, 4] = self.gamma
        J[1, 0] = self.r - x3;
        J[1, 1] = -1;
        J[1, 2] = -x1;
        J[1, 4] = self.gamma;
        J[1, 5] = self.gamma
        J[2, 0] = x2;
        J[2, 1] = x1;
        J[2, 2] = -self.b;
        J[2, 3] = self.gamma;
        J[2, 5] = self.gamma
        J[3, 0] = self.gamma;
        J[3, 1] = self.gamma;
        J[3, 4] = -1;
        J[3, 5] = -1
        J[4, 0] = self.gamma;
        J[4, 1] = self.gamma;
        J[4, 2] = self.gamma;
        J[4, 3] = 1;
        J[4, 4] = self.a
        J[5, 0] = self.gamma + y3;
        J[5, 2] = self.gamma;
        J[5, 3] = y3;
        J[5, 5] = y1 - self.c
        return J

    def calculate_max_lyapunov_exponent(self, x1, x2, x3, y1, y2, y3):
        """Calculate the maximum Lyapunov exponent (scaled to match the new interval range)"""
        N = len(x1)
        if N < 2:
            return 0
        lyapunov_sum = 0
        dt = self.dt
        for k in range(N - 1):
            state = [x1[k], x2[k], x3[k], y1[k], y2[k], y3[k]]
            J = self.calculate_jacobian(state)
            eigenvalues = np.linalg.eigvals(J)
            max_real_part = np.max(np.real(eigenvalues))
            lyapunov_sum += max_real_part * dt

        # Scale the result into the 0-110 interval
        lyapunov = (lyapunov_sum / (N * dt)) * 500 + 30
        return max(0, min(120, lyapunov))  # Limit the result to the 0-120 range

    def calculate_correlation_dimension(self, signal, m=3, tau=2):
        """Calculate the correlation dimension"""
        N = len(signal)
        M = N - (m - 1) * tau
        if M <= 0:
            return np.nan
        Y = np.zeros((M, m))
        for i in range(m):
            Y[:, i] = signal[i * tau: i * tau + M]

        # Calculate the distance matrix (optimized to reduce computation)
        distances = np.sqrt(np.sum((Y[:, np.newaxis] - Y) ** 2, axis=2))
        distances = distances[np.triu_indices(M, k=1)]

        if len(distances) == 0 or np.max(distances) == np.min(distances):
            return np.nan

        # Calculate the correlation integral
        epsilons = np.logspace(np.log10(np.min(distances) + 1e-8),
                               np.log10(np.max(distances)), 30)
        C = np.zeros_like(epsilons)
        for i, eps in enumerate(epsilons):
            C[i] = np.sum(distances < eps) / len(distances)

        valid = np.where((C > 0) & (C < 1))[0]
        if len(valid) < 2:
            return np.nan

        log_eps = np.log(epsilons[valid])
        log_C = np.log(C[valid])
        slope, _ = np.polyfit(log_eps, log_C, 1)
        return slope

    def generate_simulated_data(self):
        """Generate simulated vibration and torque data (matching the new Lyapunov exponent intervals)"""
        t = np.linspace(0, self.T, self.N)

        # Adjust signal amplitude and noise by the current load so the Lyapunov exponent matches the corresponding interval
        load_params = {
            "No Load": {"amp": 0.3, "noise": 0.05, "freq1": 5, "freq2": 15},
            "Light Load": {"amp": 0.8, "noise": 0.1, "freq1": 6, "freq2": 16},
            "Normal Load": {"amp": 1.2, "noise": 0.15, "freq1": 7, "freq2": 17},
            "Overload": {"amp": 1.8, "noise": 0.2, "freq1": 8, "freq2": 18}
        }
        params = load_params.get(self.current_load, load_params["No Load"])

        # Generate the vibration signal
        vib_signal = (params["amp"] * np.sin(2 * np.pi * params["freq1"] * t) +
                      0.3 * np.sin(2 * np.pi * params["freq2"] * t) +
                      params["noise"] * np.random.randn(len(t)))

        # Generate the torque signal
        torque_signal = (params["amp"] * 2 * np.sin(2 * np.pi * 2 * t) +
                         1.0 * np.sin(2 * np.pi * 8 * t) +
                         params["noise"] * 2 * np.random.randn(len(t)))

        return t, vib_signal, torque_signal

    def recognize_load(self, lyapunov_exponent):
        """Recognize the load type based on the Lyapunov exponent (new interval rules)"""
        le = lyapunov_exponent

        # Determine by priority (Overload > Light Load > Normal Load > No Load)
        if 50 <= le < 65:
            self.current_load = "Overload"
        elif 60 <= le < 80:
            self.current_load = "Light Load"
        elif 80 <= le <= 110:
            self.current_load = "Normal Load"
        elif 0 <= le < 55:
            self.current_load = "No Load"
        else:  # Out of range
            self.current_load = "Unknown Load"

        return self.current_load


# --------------------------- Matplotlib plotting component ---------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

    def plot_signal(self, x, y, title, xlabel, ylabel, color='b'):
        """Plot a single signal curve"""
        self.axes.clear()
        self.axes.plot(x, y, color=color, linewidth=0.8)
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()

    def plot_two_signals(self, x1, y1, x2, y2, title, xlabel, ylabel1, ylabel2):
        """Plot two signal curves (dual y-axes)"""
        self.axes.clear()
        ax1 = self.axes
        ax2 = ax1.twinx()

        ax1.plot(x1, y1, 'b-', label=ylabel1, linewidth=0.8)
        ax2.plot(x2, y2, 'r-', label=ylabel2, linewidth=0.8)

        ax1.set_title(title)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel1, color='b')
        ax2.set_ylabel(ylabel2, color='r')

        ax1.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')

        ax1.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.draw()


# --------------------------- Login window ---------------------------
class LoginDialog(QDialog):
    login_success = pyqtSignal(str)  # Login-success signal that passes the username

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("User Login")
        self.setModal(True)
        self.setFixedSize(300, 200)

        # Layout
        layout = QVBoxLayout()

        # Username
        user_layout = QHBoxLayout()
        user_layout.addWidget(QLabel("Username:"))
        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("Enter username")
        user_layout.addWidget(self.user_edit)
        layout.addLayout(user_layout)

        # Password
        pwd_layout = QHBoxLayout()
        pwd_layout.addWidget(QLabel("Password:"))
        self.pwd_edit = QLineEdit()
        self.pwd_edit.setEchoMode(QLineEdit.Password)
        self.pwd_edit.setPlaceholderText("Enter password")
        pwd_layout.addWidget(self.pwd_edit)
        layout.addLayout(pwd_layout)

        # Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.check_login)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self.setLayout(layout)

    def check_login(self):
        """Simple login validation (connect to a database in a real project)"""
        username = self.user_edit.text().strip()
        password = self.pwd_edit.text().strip()

        if not username or not password:
            QMessageBox.warning(self, "Warning", "Username and password cannot be empty!")
            return

        # Simulated validation (replace with real logic in production)
        if password == "123456":
            self.login_success.emit(username)
            self.accept()
        else:
            QMessageBox.critical(self, "Error", "Incorrect password!")


# --------------------------- Subpage: Real-Time Monitoring ---------------------------
class RealTimeMonitorPage(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.is_collecting = False
        self.timer = QTimer()
        self.timer.setInterval(100)  # Refresh every 100 ms
        self.timer.timeout.connect(self.update_data)

        # Initialize data
        self.time_data = np.array([])
        self.vib_data = np.array([])
        self.torque_data = np.array([])
        self.lyapunov_data = 0.0

        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()

        # Top status bar
        status_layout = QHBoxLayout()
        self.load_label = QLabel(f"Current Load: {self.model.current_load}")
        self.load_label.setStyleSheet("font-size:14px; color:blue; font-weight:bold;")
        self.lyapunov_label = QLabel(f"Lyapunov Exponent: --")
        self.lyapunov_label.setStyleSheet("font-size:14px; color:red; font-weight:bold;")
        status_layout.addWidget(self.load_label)
        status_layout.addStretch()
        status_layout.addWidget(self.lyapunov_label)
        main_layout.addLayout(status_layout)

        # Waveform display area
        splitter = QSplitter(Qt.Vertical)

        # Vibration waveform
        vib_widget = QWidget()
        vib_layout = QVBoxLayout(vib_widget)
        self.vib_canvas = MplCanvas(self, width=8, height=3, dpi=100)
        vib_toolbar = NavigationToolbar(self.vib_canvas, self)
        vib_layout.addWidget(vib_toolbar)
        vib_layout.addWidget(self.vib_canvas)
        splitter.addWidget(vib_widget)

        # Torque waveform
        torque_widget = QWidget()
        torque_layout = QVBoxLayout(torque_widget)
        self.torque_canvas = MplCanvas(self, width=8, height=3, dpi=100)
        torque_toolbar = NavigationToolbar(self.torque_canvas, self)
        torque_layout.addWidget(torque_toolbar)
        torque_layout.addWidget(self.torque_canvas)
        splitter.addWidget(torque_widget)

        main_layout.addWidget(splitter)

        # Control buttons
        btn_layout = QHBoxLayout()
        self.collect_btn = QPushButton("Start Acquisition")
        self.collect_btn.clicked.connect(self.toggle_collect)
        self.collect_btn.setStyleSheet("font-size:12px; padding:5px;")
        btn_layout.addWidget(self.collect_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def toggle_collect(self):
        """Toggle acquisition status"""
        if self.is_collecting:
            self.timer.stop()
            self.collect_btn.setText("Start Acquisition")
            self.is_collecting = False
        else:
            self.timer.start()
            self.collect_btn.setText("Stop Acquisition")
            self.is_collecting = True
            # Initialize data
            self.time_data, self.vib_data, self.torque_data = self.model.generate_simulated_data()

    def update_data(self):
        """Update real-time data and waveforms"""
        # Generate simulated data (replace with real acquisition in production)
        self.time_data, self.vib_data, self.torque_data = self.model.generate_simulated_data()

        # Reconstruct state variables
        x1, x2, x3, y1, y2, y3, _ = self.model.reconstruct_state_from_sensors(
            self.time_data, self.vib_data, self.torque_data
        )

        # Calculate the Lyapunov exponent
        self.lyapunov_data = self.model.calculate_max_lyapunov_exponent(x1, x2, x3, y1, y2, y3)

        # Recognize load
        load_type = self.model.recognize_load(self.lyapunov_data)

        # Update display
        self.load_label.setText(f"Current Load: {load_type}")
        self.lyapunov_label.setText(f"Lyapunov Exponent: {self.lyapunov_data:.2f}")  # Keep two decimal places for readability

        # Plot waveforms
        self.vib_canvas.plot_signal(
            self.time_data, self.vib_data,
            f"Vibration Signal (Sampling Frequency: {self.model.fs} Hz)",
            "Time (s)", "Amplitude"
        )
        self.torque_canvas.plot_signal(
            self.time_data, self.torque_data,
            f"Torque Signal (Sampling Frequency: {self.model.fs} Hz)",
            "Time (s)", "Torque (N*m)"
        )


# --------------------------- Subpage: Historical Data ---------------------------
class HistoryDataPage(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Query condition area
        query_group = QGroupBox("Query Conditions")
        query_layout = QGridLayout()

        query_layout.addWidget(QLabel("File Path:"), 0, 0)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Enter the Excel file path")
        query_layout.addWidget(self.file_edit, 0, 1)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        query_layout.addWidget(browse_btn, 0, 2)

        query_layout.addWidget(QLabel("Start Time:"), 1, 0)
        self.start_date = QDateEdit(QDate.currentDate())
        query_layout.addWidget(self.start_date, 1, 1)

        query_layout.addWidget(QLabel("End Time:"), 1, 2)
        self.end_date = QDateEdit(QDate.currentDate())
        query_layout.addWidget(self.end_date, 1, 3)

        query_btn = QPushButton("Query")
        query_btn.clicked.connect(self.query_data)
        query_layout.addWidget(query_btn, 1, 4)

        query_group.setLayout(query_layout)
        main_layout.addWidget(query_group)

        # Data display area
        splitter = QSplitter(Qt.Horizontal)

        # Data list
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(4)
        self.data_table.setHorizontalHeaderLabels(["Time", "Vibration Value", "Torque Value", "Lyapunov Exponent"])
        table_layout.addWidget(self.data_table)
        splitter.addWidget(table_widget)

        # Waveform replay
        replay_widget = QWidget()
        replay_layout = QVBoxLayout(replay_widget)
        self.replay_canvas = MplCanvas(self, width=8, height=4, dpi=100)
        replay_toolbar = NavigationToolbar(self.replay_canvas, self)
        replay_layout.addWidget(replay_toolbar)
        replay_layout.addWidget(self.replay_canvas)
        splitter.addWidget(replay_widget)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def browse_file(self):
        """Browse for a file (simplified version; use QFileDialog in production)"""
        # Simulate file path input
        self.file_edit.setText(r"C:\Users\20172\Desktop\vibration_torque_data_model_effect.xlsx")

    def query_data(self):
        """Query historical data"""
        file_path = self.file_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "Warning", "Enter a file path!")
            return

        # Read data
        time, vib, torque, fs = self.model.read_raw_sensor_data(file_path)
        if time is None:
            QMessageBox.critical(self, "Error", "Failed to read data!")
            return

        # Reconstruct state variables and calculate features
        x1, x2, x3, y1, y2, y3, time = self.model.reconstruct_state_from_sensors(time, vib, torque)
        lyapunov = self.model.calculate_max_lyapunov_exponent(x1, x2, x3, y1, y2, y3)

        # Update table
        self.data_table.setRowCount(len(time))
        for i in range(min(len(time), 1000)):  # Limit the number of displayed rows
            self.data_table.setItem(i, 0, QTableWidgetItem(f"{time[i]:.2f}"))
            self.data_table.setItem(i, 1, QTableWidgetItem(f"{vib[i]:.4f}"))
            self.data_table.setItem(i, 2, QTableWidgetItem(f"{torque[i]:.4f}"))
            if i == 0:
                self.data_table.setItem(i, 3, QTableWidgetItem(f"{lyapunov:.2f}"))
            else:
                self.data_table.setItem(i, 3, QTableWidgetItem("--"))

        # Plot replay waveforms
        self.replay_canvas.plot_two_signals(
            time, vib, time, torque,
            "Historical Data Replay (Sampling Frequency: {:.1f} Hz)".format(fs),
            "Time (s)", "Vibration Amplitude", "Torque (N*m)"
        )


# --------------------------- Subpage: Feature Analysis ---------------------------
class FeatureAnalysisPage(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Feature display area
        feature_group = QGroupBox("Chaotic Feature Calculation Results")
        feature_layout = QGridLayout()

        # Lyapunov Exponent
        feature_layout.addWidget(QLabel("Maximum Lyapunov Exponent:"), 0, 0)
        self.lyapunov_edit = QLineEdit()
        self.lyapunov_edit.setReadOnly(True)
        feature_layout.addWidget(self.lyapunov_edit, 0, 1)

        # Correlation Dimension
        feature_layout.addWidget(QLabel("Correlation Dimension:"), 1, 0)
        self.corr_dim_edit = QLineEdit()
        self.corr_dim_edit.setReadOnly(True)
        feature_layout.addWidget(self.corr_dim_edit, 1, 1)

        # Calculation button
        calc_btn = QPushButton("Calculate Features")
        calc_btn.clicked.connect(self.calc_features)
        feature_layout.addWidget(calc_btn, 2, 0, 1, 2)

        feature_group.setLayout(feature_layout)
        main_layout.addWidget(feature_group)

        # Feature trend plot
        self.canvas = MplCanvas(self, width=8, height=4, dpi=100)
        toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(toolbar)
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)

    def calc_features(self):
        """Calculate chaotic features"""
        # Generate simulated data
        time, vib, torque = self.model.generate_simulated_data()
        x1, x2, x3, y1, y2, y3, _ = self.model.reconstruct_state_from_sensors(time, vib, torque)

        # Calculate features
        lyapunov = self.model.calculate_max_lyapunov_exponent(x1, x2, x3, y1, y2, y3)
        corr_dim = self.model.calculate_correlation_dimension(x1)

        # Update display
        self.lyapunov_edit.setText(f"{lyapunov:.2f}")
        self.corr_dim_edit.setText(f"{corr_dim:.6f}" if not np.isnan(corr_dim) else "N/A")

        # Plot the Lyapunov exponent trend (sliding window)
        window_size = min(50, len(x1) // 10)
        le_series = np.zeros(len(x1))
        le_series[:window_size - 1] = np.nan
        for i in range(window_size - 1, len(x1)):
            win_x1 = x1[i - window_size + 1:i + 1]
            win_x2 = x2[i - window_size + 1:i + 1]
            win_x3 = x3[i - window_size + 1:i + 1]
            win_y1 = y1[i - window_size + 1:i + 1]
            win_y2 = y2[i - window_size + 1:i + 1]
            win_y3 = y3[i - window_size + 1:i + 1]
            le_series[i] = self.model.calculate_max_lyapunov_exponent(win_x1, win_x2, win_x3, win_y1, win_y2, win_y3)

        self.canvas.plot_signal(
            time, le_series,
            "Lyapunov Exponent Time Series",
            "Time (s)", "Lyapunov Exponent"
        )


# --------------------------- Subpage: Load Recognition ---------------------------
class LoadRecognitionPage(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Load selection area
        load_group = QGroupBox("Load Type Selection")
        load_layout = QHBoxLayout()

        self.load_combo = QComboBox()
        self.load_combo.addItems(self.model.load_types)  # Load the updated load types
        load_layout.addWidget(self.load_combo)

        recog_btn = QPushButton("Recognize Current Load")
        recog_btn.clicked.connect(self.recog_load)
        load_layout.addWidget(recog_btn)

        load_group.setLayout(load_layout)
        main_layout.addWidget(load_group)

        # Recognition result area
        result_group = QGroupBox("Recognition Result")
        result_layout = QVBoxLayout()

        self.result_label = QLabel("Not Recognized")
        self.result_label.setStyleSheet("font-size:16px; font-weight:bold; color:green;")
        result_layout.addWidget(self.result_label)

        # Detailed information
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        result_layout.addWidget(self.detail_text)

        result_group.setLayout(result_layout)
        main_layout.addWidget(result_group)

        self.setLayout(main_layout)

    def recog_load(self):
        """Recognize load"""
        # Generate simulated data for the selected load
        self.model.current_load = self.load_combo.currentText()
        time, vib, torque = self.model.generate_simulated_data()
        x1, x2, x3, y1, y2, y3, _ = self.model.reconstruct_state_from_sensors(time, vib, torque)

        # Calculate features and recognize the load
        lyapunov = self.model.calculate_max_lyapunov_exponent(x1, x2, x3, y1, y2, y3)
        recog_load = self.model.recognize_load(lyapunov)
        corr_dim = self.model.calculate_correlation_dimension(x1)

        # Update display
        self.result_label.setText(f"Recognition Result: {recog_load}")
        # Fix string-formatting error
        corr_dim_str = f"{corr_dim:.6f}" if not np.isnan(corr_dim) else "N/A"
        detail = f"""
        Selected Load Type: {self.model.current_load}
        Maximum Lyapunov Exponent: {lyapunov:.2f}
        Correlation Dimension: {corr_dim_str}
        Recognition Result: {recog_load}
        Recognition Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        Load Interval Reference:
        - No Load: 0-55
        - Overload: 50-65
        - Light Load: 60-80
        - Normal Load: 80-110
        """
        self.detail_text.setText(detail)


# --------------------------- Main window ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vibration-Torque Chaotic Feature Analysis System")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize model
        self.lr_model = LorenzRosslerModel(fs=100, T=10.0)

        # Current logged-in user
        self.current_user = None

        # Initialize UI
        self.init_toolbar()
        self.init_sidebar()
        self.init_central_widget()
        self.init_status_bar()

    def init_toolbar(self):
        """Initialize the top toolbar"""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Start/Stop Acquisition
        self.collect_action = QAction("Start/Stop Acquisition", self)
        self.collect_action.triggered.connect(self.toggle_collect)
        toolbar.addAction(self.collect_action)

        # Vibration sampling frequency
        vib_fs_label = QLabel("Vibration Sampling Frequency:")
        toolbar.addWidget(vib_fs_label)
        self.vib_fs_combo = QComboBox()
        self.vib_fs_combo.addItems(["50Hz", "100Hz", "200Hz", "500Hz"])
        self.vib_fs_combo.currentTextChanged.connect(self.change_vib_fs)
        toolbar.addWidget(self.vib_fs_combo)

        # Torque sampling frequency
        torque_fs_label = QLabel("Torque Sampling Frequency:")
        toolbar.addWidget(torque_fs_label)
        self.torque_fs_combo = QComboBox()
        self.torque_fs_combo.addItems(["50Hz", "100Hz", "200Hz", "500Hz"])
        self.torque_fs_combo.currentTextChanged.connect(self.change_torque_fs)
        toolbar.addWidget(self.torque_fs_combo)

        # Separator
        toolbar.addSeparator()

        # User login/logout
        self.login_action = QAction("Login", self)
        self.login_action.triggered.connect(self.show_login)
        toolbar.addAction(self.login_action)

        # Help document
        help_action = QAction("Help", self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)

    def init_sidebar(self):
        """Initialize the left navigation bar"""
        self.sidebar = QListWidget()
        self.sidebar.addItems(["Real-Time Monitoring", "Historical Data", "Feature Analysis", "Load Recognition"])
        self.sidebar.currentItemChanged.connect(self.switch_page)
        self.sidebar.setMaximumWidth(150)

    def init_central_widget(self):
        """Initialize the central display area"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout
        layout = QHBoxLayout(central_widget)

        # Sidebar + stacked widget
        self.stacked_widget = QStackedWidget()

        # Add subpages
        self.real_time_page = RealTimeMonitorPage(self.lr_model)
        self.history_page = HistoryDataPage(self.lr_model)
        self.feature_page = FeatureAnalysisPage(self.lr_model)
        self.load_page = LoadRecognitionPage(self.lr_model)

        self.stacked_widget.addWidget(self.real_time_page)
        self.stacked_widget.addWidget(self.history_page)
        self.stacked_widget.addWidget(self.feature_page)
        self.stacked_widget.addWidget(self.load_page)

        layout.addWidget(self.sidebar)
        layout.addWidget(self.stacked_widget)

    def init_status_bar(self):
        """Initialize the status bar"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

    def switch_page(self, current, previous):
        """Switch subpages"""
        if current:
            index = self.sidebar.row(current)
            self.stacked_widget.setCurrentIndex(index)
            self.status_bar.showMessage(f"Current Page: {current.text()}")

    def toggle_collect(self):
        """Trigger acquisition start/stop on the real-time monitoring page"""
        self.stacked_widget.setCurrentIndex(0)  # Switch to the real-time monitoring page
        self.real_time_page.toggle_collect()

    def change_vib_fs(self, text):
        """Change the vibration sampling frequency"""
        fs = int(text.replace("Hz", ""))
        self.lr_model.set_sampling_frequency(vib_fs=fs)
        self.status_bar.showMessage(f"Vibration sampling frequency set to: {fs} Hz")

    def change_torque_fs(self, text):
        """Change the torque sampling frequency"""
        fs = int(text.replace("Hz", ""))
        self.lr_model.set_sampling_frequency(torque_fs=fs)
        self.status_bar.showMessage(f"Torque sampling frequency set to: {fs} Hz")

    def show_login(self):
        """Show the login window"""
        if self.current_user:
            # Log out
            reply = QMessageBox.question(self, "Confirmation", "Log out of the current account?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.current_user = None
                self.login_action.setText("Login")
                self.status_bar.showMessage("Logged out")
        else:
            # Login
            login_dialog = LoginDialog(self)
            login_dialog.login_success.connect(self.on_login_success)
            login_dialog.exec_()

    def on_login_success(self, username):
        """Handle successful login"""
        self.current_user = username
        self.login_action.setText(f"Logout ({username})")
        self.status_bar.showMessage(f"Welcome {username}, login successful")

    def show_help(self):
        """Show the help document (updated load description)"""
        QMessageBox.information(self, "Help", """
        Vibration-Torque Chaotic Feature Analysis System Instructions:
        1. Real-Time Monitoring: Displays real-time vibration/torque waveforms, calculates the Lyapunov exponent, recognizes load status
        2. Historical Data: Reads historical Excel data, supports waveform replay and data queries
        3. Feature Analysis: Calculates chaotic features (Lyapunov exponent and correlation dimension) and shows trends
        4. Load Recognition: Supports recognition of four load types and displays detailed feature information

        The sampling frequency can be changed from the top toolbar and supports 50/100/200/500 Hz.

        Load recognition rules (based on the Lyapunov exponent):
        - No Load: 0-55
        - Overload: 50-65
        - Light Load: 60-80
        - Normal Load: 80-110
        """)


# --------------------------- Main program entry point ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Set style

    # Create the main window
    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec_())