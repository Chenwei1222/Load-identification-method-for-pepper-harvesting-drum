import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random
import pandas as pd
from scipy.signal import butter, filtfilt
import os  # Create the output folder.

# Font settings for English-only figures.
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Vector-figure export settings.
OUTPUT_DIR = "vector_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
VECTOR_FORMATS = ['svg', 'eps']  # Vector formats: SVG for general use and EPS for academic publishing.
DPI = 300  # Fallback high-resolution raster output.

# Font-size enhancement settings.
AXIS_LABEL_FONT_SIZE_OFFSET = 6  # Increase axis-label font size by 6 points.
TICK_LABEL_FONT_SIZE_OFFSET = 3  # Increase tick-label font size by 3 points.
TITLE_FONT_SIZE = 12 + AXIS_LABEL_FONT_SIZE_OFFSET  # Match title size with axis-label size.


class LorenzRosslerModel:
    def __init__(self, fs=100, T=10.0, sensor_type='vibration_torque'):
        """Initialize the coupled Lorenz-Rossler model with a 100 Hz sampling rate."""
        self.fs = fs  # Sampling frequency: 100 Hz.
        self.T = T
        self.dt = 1 / fs  # Time interval: 0.01 s.
        self.N = int(T * fs)  # Number of data points: 100 * T.
        self.t = np.linspace(0, T, self.N)
        self.sensor_type = sensor_type

        # Initial model parameters.
        self.sigma = 10.0
        self.r = 28.0
        self.b = 8.0 / 3.0
        self.a = 0.2
        self.c = 5.7
        self.gamma = 0.1

        # Sensor calibration coefficients.
        self.k_vib = 1.0  # Vibration-signal calibration coefficient.
        self.k_torque = 1.0  # Torque-signal calibration coefficient.

        # Initial state variables.
        self.x1_0 = 0.1
        self.x2_0 = 0.1
        self.x3_0 = 0.1
        self.y1_0 = 0.1
        self.y2_0 = 0.1
        self.y3_0 = 0.1

        # Filter parameters adjusted for the 100 Hz sampling rate.
        self.vib_cutoff_hz = 40  # Vibration-signal cutoff frequency: 40 Hz.
        self.torque_cutoff_hz = 20  # Torque-signal cutoff frequency: 20 Hz.
        self.filter_order = 2

    def set_parameters(self, sigma=None, r=None, b=None, a=None, c=None, gamma=None):
        if sigma is not None:
            self.sigma = sigma
        if r is not None:
            self.r = r
        if b is not None:
            self.b = b
        if a is not None:
            self.a = a
        if c is not None:
            self.c = c
        if gamma is not None:
            self.gamma = gamma

    def set_sensor_coefficients(self, k_vib=None, k_torque=None):
        if k_vib is not None:
            self.k_vib = k_vib
        if k_torque is not None:
            self.k_torque = k_torque

    def set_initial_conditions(self, x1_0=None, x2_0=None, x3_0=None,
                               y1_0=None, y2_0=None, y3_0=None):
        if x1_0 is not None:
            self.x1_0 = x1_0
        if x2_0 is not None:
            self.x2_0 = x2_0
        if x3_0 is not None:
            self.x3_0 = x3_0
        if y1_0 is not None:
            self.y1_0 = y1_0
        if y2_0 is not None:
            self.y2_0 = y2_0
        if y3_0 is not None:
            self.y3_0 = y3_0

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
        """Solve the differential equations and match the default 100 Hz sampling rate."""
        if num_points is None:
            num_points = self.N
        t = np.linspace(0, self.T, num_points)
        y0 = [self.x1_0, self.x2_0, self.x3_0, self.y1_0, self.y2_0, self.y3_0]
        sol = solve_ivp(self.coupled_equations, [0, self.T], y0, t_eval=t, method='RK45')
        return sol.y, t

    def design_filter(self, cutoff_hz, fs, order=2):
        """Design a digital filter for the 100 Hz sampling rate."""
        nyq = 0.5 * fs  # Nyquist frequency is 50 Hz when the sampling rate is 100 Hz.
        normalized_cutoff = min(max(cutoff_hz / nyq, 0.01), 0.9)  # Keep the cutoff below 45 Hz.
        if normalized_cutoff != cutoff_hz / nyq:
            print(f"Warning: cutoff frequency was adjusted to {normalized_cutoff * nyq:.2f} Hz "
                  f"(normalized value: {normalized_cutoff:.4f})")
        b, a = butter(order, normalized_cutoff, btype='low')
        return b, a

    def filter_signal(self, signal, cutoff_hz, fs, order=2):
        """Filter a signal using settings adapted to the 100 Hz sampling rate."""
        b, a = self.design_filter(cutoff_hz, fs, order)
        filtered = filtfilt(b, a, signal)
        return filtered

    def reconstruct_state_from_sensors(self, time, vib_signal, torque_signal):
        """Reconstruct six-dimensional state variables from vibration and torque signals."""
        # Calculate the actual sampling frequency.
        time_diff = np.diff(time)
        if len(time_diff) > 0:
            actual_fs = 1.0 / np.mean(time_diff)
        else:
            actual_fs = self.fs
        print(f"Actual sampling frequency: {actual_fs:.2f} Hz")

        # Signal preprocessing.
        vib_signal = self.filter_signal(vib_signal, self.vib_cutoff_hz, actual_fs, self.filter_order)
        torque_signal = self.filter_signal(torque_signal, self.torque_cutoff_hz, actual_fs, self.filter_order)

        # Ensure consistent signal lengths.
        min_len = min(len(time), len(vib_signal), len(torque_signal))
        time = time[:min_len]
        vib_signal = vib_signal[:min_len]
        torque_signal = torque_signal[:min_len]

        # Apply calibration coefficients.
        x1 = self.k_vib * vib_signal  # Vibration velocity feature.
        y2 = self.k_torque * torque_signal  # Mean torque level.

        # Derived vibration-signal features.
        x2 = np.gradient(x1) * actual_fs  # Vibration displacement gradient.
        x3 = np.convolve(x1, np.ones(5) / 5, mode='same')  # Vibration energy dissipation term.
        x3 = x3 - np.mean(x3)  # Remove the mean value.

        # Derived torque-signal features.
        y1 = np.gradient(y2) * actual_fs  # Torque rate of change.
        y3 = np.abs(np.gradient(y1)) * actual_fs  # Torque-change acceleration.
        y3 = y3 / np.max(y3) if np.max(y3) > 0 else y3  # Normalize.

        return x1, x2, x3, y1, y2, y3, time

    def read_raw_sensor_data(self, file_path, sheet_name='Sheet1',
                             time_col='Time', vib_col='Vibration', torque_col='Torque'):
        """Read raw time, vibration, and torque data."""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            required_cols = [time_col, vib_col, torque_col]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            time = df[time_col].values
            vib_signal = df[vib_col].values
            torque_signal = df[torque_col].values

            # Check sampling-frequency consistency.
            time_diff = np.diff(time)
            avg_dt = np.mean(time_diff)
            estimated_fs = 1 / avg_dt if avg_dt > 0 else self.fs
            print(f"Estimated sampling frequency: {estimated_fs:.2f} Hz")

            return time, vib_signal, torque_signal
        except Exception as e:
            print(f"Data reading error: {e}")
            return None, None, None

    def visualize_sensor_data(self, time, vib_signal, torque_signal,
                              save_name="01_vibration_torque_time_domain_waveforms"):
        """Plot vibration and torque time-domain waveforms."""
        fig = plt.figure(figsize=(12, 8))

        # Subplot 1: vibration signal.
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(time, vib_signal, linewidth=2)
        ax1.set_title('Vibration Signal Time-Domain Waveform (100 Hz Sampling)', fontsize=TITLE_FONT_SIZE)
        ax1.set_xlabel('Time (s)', fontsize=12 + AXIS_LABEL_FONT_SIZE_OFFSET)
        ax1.set_ylabel('Amplitude (g)', fontsize=12 + AXIS_LABEL_FONT_SIZE_OFFSET)
        ax1.tick_params(axis='both', labelsize=10 + TICK_LABEL_FONT_SIZE_OFFSET)

        # Subplot 2: torque signal.
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(time, torque_signal, linewidth=2)
        ax2.set_title('Torque Signal Time-Domain Waveform (100 Hz Sampling)', fontsize=TITLE_FONT_SIZE)
        ax2.set_xlabel('Time (s)', fontsize=12 + AXIS_LABEL_FONT_SIZE_OFFSET)
        ax2.set_ylabel('Torque (N*m)', fontsize=12 + AXIS_LABEL_FONT_SIZE_OFFSET)
        ax2.tick_params(axis='both', labelsize=10 + TICK_LABEL_FONT_SIZE_OFFSET)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.show()

    def visualize_coupled_system(self, x1, x2, x3, y1, y2, y3, time=None,
                                 save_name="02_vibration_torque_coupled_feature_plots"):
        """Visualize two-dimensional phase plots for the vibration-torque coupled system."""
        if time is not None:
            min_len = min(len(time), len(x1), len(y1))
            x1 = x1[:min_len]
            x2 = x2[:min_len]
            x3 = x3[:min_len]
            y1 = y1[:min_len]
            y2 = y2[:min_len]
            y3 = y3[:min_len]

        fig = plt.figure(figsize=(12, 8))

        # Subplot 1: x1 vs y1.
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(x1, y1, linewidth=0.7)
        ax1.set_title('Coupled Feature: Vibration Velocity vs Torque Rate of Change', fontsize=TITLE_FONT_SIZE)
        ax1.set_xlabel('$x_1$ (Vibration Velocity)', fontsize=12 + AXIS_LABEL_FONT_SIZE_OFFSET)
        ax1.set_ylabel('$y_1$ (Torque Rate of Change)', fontsize=12 + AXIS_LABEL_FONT_SIZE_OFFSET)
        ax1.tick_params(axis='both', labelsize=10 + TICK_LABEL_FONT_SIZE_OFFSET)

        # Subplot 2: x1 vs y2.
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(x1, y2, linewidth=0.7)
        ax2.set_title('Coupled Feature: Vibration Velocity vs Mean Torque Level', fontsize=TITLE_FONT_SIZE)
        ax2.set_xlabel('$x_1$ (Vibration Velocity)', fontsize=12 + AXIS_LABEL_FONT_SIZE_OFFSET)
        ax2.set_ylabel('$y_2$ (Mean Torque Level)', fontsize=12 + AXIS_LABEL_FONT_SIZE_OFFSET)
        ax2.tick_params(axis='both', labelsize=10 + TICK_LABEL_FONT_SIZE_OFFSET)

        # Subplot 3: x2 vs y1.
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(x2, y1, linewidth=0.7)
        ax3.set_title('Coupled Feature: Vibration Displacement Gradient vs Torque Rate of Change',
                      fontsize=TITLE_FONT_SIZE)
        ax3.set_xlabel('$x_2$ (Vibration Displacement Gradient)', fontsize=12 + AXIS_LABEL_FONT_SIZE_OFFSET)
        ax3.set_ylabel('$y_1$ (Torque Rate of Change)', fontsize=12 + AXIS_LABEL_FONT_SIZE_OFFSET)
        ax3.tick_params(axis='both', labelsize=10 + TICK_LABEL_FONT_SIZE_OFFSET)

        # Subplot 4: x3 vs y2.
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(x3, y2, linewidth=0.7)
        ax4.set_title('Coupled Feature: Vibration Energy Dissipation vs Mean Torque Level',
                      fontsize=TITLE_FONT_SIZE)
        ax4.set_xlabel('$x_3$ (Vibration Energy Dissipation)', fontsize=12 + AXIS_LABEL_FONT_SIZE_OFFSET)
        ax4.set_ylabel('$y_2$ (Mean Torque Level)', fontsize=12 + AXIS_LABEL_FONT_SIZE_OFFSET)
        ax4.tick_params(axis='both', labelsize=10 + TICK_LABEL_FONT_SIZE_OFFSET)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.show()

    def _save_figure(self, fig, filename):
        """
        Save a figure in vector formats.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The Matplotlib figure object.
        filename : str
            Output filename without extension.
        """
        # Clean special characters in filenames.
        filename = filename.replace(':', '').replace(' ', '_').replace('/', '-')

        # Save all specified vector formats.
        for fmt in VECTOR_FORMATS:
            try:
                save_path = os.path.join(OUTPUT_DIR, f"{filename}.{fmt}")
                fig.savefig(
                    save_path,
                    format=fmt,
                    dpi=DPI,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    transparent=True
                )
                print(f"Vector figure saved: {save_path}")
            except Exception as e:
                print(f"Failed to save {fmt} format: {e}")

        # Fallback high-resolution PNG output.
        try:
            png_path = os.path.join(OUTPUT_DIR, f"{filename}_300DPI.png")
            fig.savefig(
                png_path,
                dpi=DPI,
                bbox_inches='tight',
                pad_inches=0.1,
                transparent=True
            )
            print(f"High-resolution raster figure saved: {png_path}")
        except Exception as e:
            print(f"Failed to save PNG format: {e}")


if __name__ == "__main__":
    # Create a model instance with a 100 Hz sampling rate and a 10 s sampling duration.
    model = LorenzRosslerModel(fs=100, T=10.0)

    # Read raw sensor data.
    excel_path = r"C:\Users\20172\Desktop\vibration_torque_data_model_effect.xlsx"
    time, vib_signal, torque_signal = model.read_raw_sensor_data(excel_path)

    if time is None:
        print("Data reading failed. Using simulated data for demonstration (100 Hz sampling).")
        # Generate simulated data adapted to 100 Hz sampling.
        t = np.linspace(0, 10, 1000)  # 100 Hz * 10 s = 1000 points.
        vib_signal = (
            0.5 * np.sin(2 * np.pi * 5 * t)
            + 0.3 * np.sin(2 * np.pi * 15 * t)
            + 0.1 * np.random.randn(len(t))
        )
        torque_signal = (
            2.0 * np.sin(2 * np.pi * 2 * t)
            + 1.0 * np.sin(2 * np.pi * 8 * t)
            + 0.2 * np.random.randn(len(t))
        )
        time = t
    else:
        print(f"Successfully read {len(time)} data points (100 Hz sampling).")

    # Plot vibration and torque time-domain waveforms.
    model.visualize_sensor_data(time, vib_signal, torque_signal)

    # Reconstruct state variables from sensor data for the coupled-feature plots.
    x1, x2, x3, y1, y2, y3, time = model.reconstruct_state_from_sensors(time, vib_signal, torque_signal)

    # Plot coupled-feature phase diagrams.
    model.visualize_coupled_system(x1, x2, x3, y1, y2, y3, time)
