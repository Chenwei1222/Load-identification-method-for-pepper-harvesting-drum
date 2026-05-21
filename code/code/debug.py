import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import pandas as pd
from scipy.signal import butter, filtfilt
import os  # Added: used to create folders

# ========== Font and font-size settings (38 pt) ==========
# Set font family
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# Set global font size to 38
plt.rcParams.update({
    'font.size': 38,  # Global font size
    'axes.labelsize': 38,  # Axis label size
    'axes.titlesize': 38,  # Title size
    'xtick.labelsize': 38,  # X-axis tick label size
    'ytick.labelsize': 38,  # Y-axis tick label size
    'legend.fontsize': 38,  # Legend size
    'figure.titlesize': 38  # Figure title size
})
# ========== End of font-size settings ==========

# ========== Added: vector figure saving configuration ==========
# Create output folder
OUTPUT_DIR = "vector_figure_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Vector format configuration (priority: SVG > EPS > EMF, compatible with different use cases)
VECTOR_FORMATS = ['svg', 'eps']  # Can add 'emf'（Windows）/ 'pdf'
DPI = 300  # Fallback bitmap resolution


class LorenzRosslerModel:
    def __init__(self, fs=1000, T=10.0, sensor_type='vibration_torque'):
        """
        Initialize the Lorenz-Rossler coupled model (including 7 parameters)
        Lorenz system parameters: sigma, r, b
        Rossler system parameters: a, d, c
        Coupling parameter: gamma
        """
        self.fs = fs
        self.T = T
        self.dt = 1 / fs
        self.N = int(T * fs)
        self.t = np.linspace(0, T, self.N)
        self.sensor_type = sensor_type

        # Initial model parameter values (including the d parameter for the Rossler system)
        self.sigma = 10.0  # Lorenz vibration-mode coupling coefficient
        self.r = 28.0  # Lorenz nonlinear excitation intensity
        self.b = 8.0 / 3.0  # Lorenz vibration damping factor
        self.a = 0.2  # Rossler drive-system damping coefficient
        self.d = 1.0  # Rossler constant-load bias term (new parameter)
        self.c = 5.7  # Rossler torque feedback gain
        self.gamma = 0.1  # Coupling strength coefficient

        # Sensor calibration coefficients
        self.k_vib = 1.0  # Vibration signal calibration coefficient
        self.k_torque = 1.0  # Torque signal calibration coefficient

        # Initial state-variable values
        self.x1_0 = 0.1
        self.x2_0 = 0.1
        self.x3_0 = 0.1
        self.y1_0 = 0.1
        self.y2_0 = 0.1
        self.y3_0 = 0.1

        # Filter parameters
        self.vib_cutoff_hz = 500
        self.torque_cutoff_hz = 200
        self.filter_order = 2

    def set_parameters(self, sigma=None, r=None, b=None, a=None, d=None, c=None, gamma=None):
        """Update model parameters (including d parameter setting)"""
        if sigma is not None: self.sigma = sigma
        if r is not None: self.r = r
        if b is not None: self.b = b
        if a is not None: self.a = a
        if d is not None: self.d = d  # Added d parameter update
        if c is not None: self.c = c
        if gamma is not None: self.gamma = gamma

    def coupled_equations(self, t, state):
        """Corrected coupled equations: the Rossler y3 equation uses d (the original program incorrectly used b)"""
        x1, x2, x3, y1, y2, y3 = state
        # Lorenz system (vibration subsystem)
        dx1_dt = self.sigma * (x2 - x1) + self.gamma * (y1 + y2)  # Coupling term: torque feedback on vibration velocity
        dx2_dt = x1 * (self.r - x3) - x2 + self.gamma * (y2 + y3)  # Coupling term: torque feedback on displacement gradient
        dx3_dt = x1 * x2 - self.b * x3 + self.gamma * (y1 + y3)  # Coupling term: torque feedback on energy dissipation

        # Rossler system (torque subsystem) - corrected the constant-load term in the y3 equation to d
        dy1_dt = -y2 - y3 + self.gamma * (x1 + x2)  # Coupling term: vibration excitation of torque change rate
        dy2_dt = y1 + self.a * y2 + self.gamma * (x2 + x3)  # Coupling term: vibration excitation of mean torque
        dy3_dt = self.d + y1 * y3 - self.c * y3 + self.gamma * (x1 + x3)  # Correction: use d as the constant-load bias term
        return [dx1_dt, dx2_dt, dx3_dt, dy1_dt, dy2_dt, dy3_dt]

    def solve_ivp_method(self, num_points=None):
        if num_points is None:
            num_points = self.N
        t = np.linspace(0, self.T, num_points)
        y0 = [self.x1_0, self.x2_0, self.x3_0, self.y1_0, self.y2_0, self.y3_0]
        sol = solve_ivp(self.coupled_equations, [0, self.T], y0, t_eval=t, method='RK45')
        return sol.y, t

    def design_filter(self, cutoff_hz, fs, order=2):
        nyq = 0.5 * fs
        normalized_cutoff = min(max(cutoff_hz / nyq, 0.01), 0.99)
        if normalized_cutoff != cutoff_hz / nyq:
            print(f"Warning: cutoff frequency has been adjusted to {normalized_cutoff * nyq:.2f} Hz (Normalization: {normalized_cutoff:.4f})")
        b, a = butter(order, normalized_cutoff, btype='low')
        return b, a

    def filter_signal(self, signal, cutoff_hz, fs, order=2):
        b, a = self.design_filter(cutoff_hz, fs, order)
        filtered = filtfilt(b, a, signal)
        return filtered

    def reconstruct_state_from_sensors(self, time, vib_signal, torque_signal):
        time_diff = np.diff(time)
        if len(time_diff) > 0:
            actual_fs = 1.0 / np.mean(time_diff)
        else:
            actual_fs = self.fs
        print(f"Actual sampling frequency: {actual_fs:.2f} Hz")

        # Signal filtering
        vib_signal = self.filter_signal(vib_signal, self.vib_cutoff_hz, actual_fs, self.filter_order)
        torque_signal = self.filter_signal(torque_signal, self.torque_cutoff_hz, actual_fs, self.filter_order)

        # Length alignment
        min_len = min(len(time), len(vib_signal), len(torque_signal))
        time = time[:min_len]
        vib_signal = vib_signal[:min_len]
        torque_signal = torque_signal[:min_len]

        # State-variable reconstruction
        x1 = self.k_vib * vib_signal  # Vibration velocity feature
        y2 = self.k_torque * torque_signal  # Mean torque level

        x2 = np.gradient(x1) * actual_fs  # Vibration displacement gradient
        x3 = np.convolve(x1, np.ones(10) / 10, mode='same') - np.mean(x1)  # Vibration energy dissipation term

        y1 = np.gradient(y2) * actual_fs  # Torque change rate
        y3 = np.abs(np.gradient(y1)) * actual_fs  # Torque fluctuation amplitude
        y3 = y3 / np.max(y3) if np.max(y3) > 0 else y3  # Normalization

        return x1, x2, x3, y1, y2, y3, time

    def read_raw_sensor_data(self, file_path, sheet_name='Sheet1',
                             time_col='Time', vib_col='Vibration', torque_col='Torque'):
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            required_cols = [time_col, vib_col, torque_col]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            time = df[time_col].values
            vib_signal = df[vib_col].values
            torque_signal = df[torque_col].values

            # Estimated sampling frequency
            time_diff = np.diff(time)
            avg_dt = np.mean(time_diff)
            estimated_fs = 1 / avg_dt if avg_dt > 0 else self.fs
            print(f"Estimated sampling frequency: {estimated_fs:.2f} Hz")

            return time, vib_signal, torque_signal
        except Exception as e:
            print(f"Data reading error: {e}")
            return None, None, None

    def calculate_jacobian(self, state):
        """Jacobian matrix calculation (adapted to the 7-parameter model)"""
        x1, x2, x3, y1, y2, y3 = state
        J = np.zeros((6, 6))
        # Partial derivatives with respect to x1
        J[0, 0] = -self.sigma
        J[0, 1] = self.sigma
        J[0, 3] = self.gamma
        J[0, 4] = self.gamma
        # Partial derivatives with respect to x2
        J[1, 0] = self.r - x3
        J[1, 1] = -1
        J[1, 2] = -x1
        J[1, 4] = self.gamma
        J[1, 5] = self.gamma
        # Partial derivatives with respect to x3
        J[2, 0] = x2
        J[2, 1] = x1
        J[2, 2] = -self.b
        J[2, 3] = self.gamma
        J[2, 5] = self.gamma
        # Partial derivatives with respect to y1
        J[3, 0] = self.gamma
        J[3, 1] = self.gamma
        J[3, 4] = -1
        J[3, 5] = -1
        # Partial derivatives with respect to y2
        J[4, 0] = self.gamma
        J[4, 1] = self.gamma
        J[4, 2] = self.gamma
        J[4, 3] = 1
        J[4, 4] = self.a
        # Partial derivatives with respect to y3
        J[5, 0] = self.gamma + y3
        J[5, 2] = self.gamma
        J[5, 3] = y3
        J[5, 5] = y1 - self.c
        return J

    def calculate_max_lyapunov_exponent(self, x1, x2, x3, y1, y2, y3, eps=1e-8):
        N = len(x1)
        lyapunov_sum = 0
        for k in range(N - 1):
            state = [x1[k], x2[k], x3[k], y1[k], y2[k], y3[k]]
            J = self.calculate_jacobian(state)
            eigenvalues = np.linalg.eigvals(J)
            max_real_part = np.max(np.real(eigenvalues))
            lyapunov_sum += max_real_part * self.dt
        return lyapunov_sum / (N * self.dt)

    def objective_function(self, params, target_signals):
        """Objective function: adapted to 7 parameters"""
        sigma, r, b, a, d, c, gamma = params  # Added d parameter
        self.set_parameters(sigma, r, b, a, d, c, gamma)
        model_signals, _ = self.solve_ivp_method(len(target_signals[0]))
        rmse = 0
        for model, target in zip(model_signals, target_signals):
            min_len = min(len(model), len(target))
            rmse += np.mean((model[:min_len] - target[:min_len]) ** 2)
        return np.sqrt(rmse / len(model_signals))

    def parameter_estimation_ga(self, target_signals, pop_size=20, generations=30):
        """Genetic algorithm parameter estimation: adjusted to 7 parameters"""
        # Parameter ranges: added the range of d (constant-load bias term, set to positive values based on physical meaning)
        param_ranges = [
            (5.0, 20.0),  # sigma: Vibration-mode coupling coefficient
            (10.0, 40.0),  # r: Nonlinear excitation intensity
            (1.0, 5.0),  # b: Vibration damping factor
            (0.1, 1.0),  # a: Drive-system damping coefficient
            (0.5, 5.0),  # d: Constant-load bias term (new)
            (2.0, 10.0),  # c: Torque feedback gain
            (0.01, 0.5)  # gamma: Coupling strength coefficient
        ]

        # Initialize population (7 parameters)
        population = [[random.uniform(low, high) for low, high in param_ranges]
                      for _ in range(pop_size)]
        best_fitness = float('-inf')
        best_params = None

        # Record optimization process
        generations_rmse = []
        generations_best_params = []

        for gen in range(generations):
            # Calculate fitness (the objective function is RMSE, and fitness is its reciprocal)
            fitness = [1 / (self.objective_function(ind, target_signals) + 1e-8)
                       for ind in population]
            current_best_rmse = 1 / max(fitness) - 1e-8
            generations_rmse.append(current_best_rmse)

            # Record the best parameters of the current generation
            best_idx = fitness.index(max(fitness))
            generations_best_params.append(population[best_idx])

            # Selection operation
            total_fitness = sum(fitness)
            if total_fitness == 0:
                probs = [1.0 / pop_size] * pop_size
            else:
                probs = [f / total_fitness for f in fitness]

            # Crossover and mutation (7 parameters)
            new_population = []
            for _ in range(pop_size):
                # Select parents
                parent1 = population[np.random.choice(pop_size, p=probs)]
                parent2 = population[np.random.choice(pop_size, p=probs)]
                # Crossover
                child = [parent1[j] if random.random() < 0.5 else parent2[j]
                         for j in range(7)]  # 7 parameters
                # Mutation
                for j in range(7):
                    if random.random() < 0.1:  # 10%mutation probability
                        child[j] = random.uniform(param_ranges[j][0], param_ranges[j][1])
                new_population.append(child)
            population = new_population

            # Update the global optimum
            curr_fitness = max(fitness)
            if curr_fitness > best_fitness:
                best_fitness = curr_fitness
                best_params = population[fitness.index(curr_fitness)]

            if gen % 5 == 0:
                print(f"Generation {gen}, Best RMSE: {current_best_rmse:.6f}")

        return best_params, generations_rmse, generations_best_params

    # ========== Core modification 1: reconstructed state-variable visualization (with vector saving) ==========
    def visualize_results(self, x1, x2, x3, y1, y2, y3, time=None, save_name="reconstructed_state_variables"):
        if time is None:
            time = np.linspace(0, self.T, len(x1))
        else:
            min_len = min(len(time), len(x1))
            time = time[:min_len]
            x1 = x1[:min_len]
            x2 = x2[:min_len]
            x3 = x3[:min_len]
            y1 = y1[:min_len]
            y2 = y2[:min_len]
            y3 = y3[:min_len]

        # Remove prefixes such as x1: and x2:, keeping only the physical meaning
        param_names = [
            'Vibration velocity feature',
            'Vibration displacement gradient',
            'Vibration energy dissipation',
            'Torque change rate',
            'Mean torque level',
            'Torque fluctuation amplitude'
        ]

        # 2 x 3 layout; canvas adjusted to width 36 and height 24 (suitable for landscape layout)
        fig = plt.figure(figsize=(36, 24))
        for i, (signal, name) in enumerate(zip([x1, x2, x3, y1, y2, y3], param_names)):
            plt.subplot(2, 3, i + 1)  # 2-row by 3-column layout
            plt.plot(time, signal, linewidth=3)
            plt.title(name)
            plt.xlabel('Time (s)')
            plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save as vector formats
        self._save_figure(fig, save_name)
        plt.show()

    # ========== Core modification 2: GA optimization-process visualization (with vector saving) ==========
    def visualize_ga_optimization(self, generations_rmse, save_name="GA_optimization_RMSE_change"):
        fig = plt.figure(figsize=(36, 18))  # Widen the canvas to fit 38-pt font
        plt.plot(range(1, len(generations_rmse) + 1), generations_rmse, 'b-', linewidth=4)
        plt.xlabel('Number of generations')
        plt.ylabel('Best RMSE')
        plt.title('Genetic Algorithm Optimization Process: Best RMSE vs. Number of Generations')
        plt.grid(True, alpha=0.3)

        if len(generations_rmse) > 0:
            final_rmse = generations_rmse[-1]
            plt.axhline(y=final_rmse, color='r', linestyle='--', alpha=0.7,
                        label=f'Final RMSE: {final_rmse:.6f}')
            plt.annotate(f'{final_rmse:.6f}',
                         xy=(len(generations_rmse), final_rmse),
                         xytext=(len(generations_rmse) + 1, final_rmse + 0.001),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))
        plt.legend()
        plt.tight_layout()

        # Save as vector formats
        self._save_figure(fig, save_name)
        plt.show()

    # ========== Core modification 3: signal-comparison visualization (with vector saving) ==========
    def visualize_signal_comparison(self, time, target_signals, optimized_signals, save_name="signal_comparison_before_and_after_optimization"):
        # 2 x 3 layout; canvas adjusted to width 36 and height 24
        fig, axes = plt.subplots(2, 3, figsize=(36, 24))
        param_names = [
            'Vibration velocity feature', 'Vibration displacement gradient', 'Vibration energy dissipation',
            'Torque change rate', 'Mean torque level', 'Torque fluctuation amplitude'
        ]

        for i in range(6):
            # Index calculation changed to: row = i // 3, col = i % 3 (for the 2 x 3 layout)
            row, col = i // 3, i % 3
            axes[row, col].plot(time, target_signals[i], 'b-', label='Reconstructed signal', linewidth=3)
            axes[row, col].plot(time, optimized_signals[i], 'r--', label='Model output', linewidth=3)
            axes[row, col].set_title(f'{param_names[i]}Comparison')
            axes[row, col].set_xlabel('Time (s)')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        plt.tight_layout()

        # Save as vector formats
        self._save_figure(fig, save_name)
        plt.show()

    # ========== Added: General vector-figure saving function ==========
    def _save_figure(self, fig, filename):
        """
        Save the figure in vector formats
        :param fig: matplotlib figure object
        :param filename: output filename (without extension)
        """
        # Clean special characters in the filename
        filename = filename.replace(':', '').replace(' ', '_').replace('/', '-')

        # Save all specified vector formats
        for fmt in VECTOR_FORMATS:
            try:
                save_path = os.path.join(OUTPUT_DIR, f"{filename}.{fmt}")
                fig.savefig(
                    save_path,
                    format=fmt,
                    dpi=DPI,
                    bbox_inches='tight',  # Remove white margins
                    pad_inches=0.1,  # Fine-tune margins
                    transparent=True  # Transparent background (optional)
                )
                print(f"✅ Vector figure saved: {save_path}")
            except Exception as e:
                print(f"❌ Save{fmt}format failed: {e}")

        # Fallback save as high-resolution PNG (compatible with all use cases)
        try:
            png_path = os.path.join(OUTPUT_DIR, f"{filename}_300DPI.png")
            fig.savefig(
                png_path,
                dpi=DPI,
                bbox_inches='tight',
                pad_inches=0.1,
                transparent=True
            )
            print(f"✅ High-resolution bitmap saved: {png_path}")
        except Exception as e:
            print(f"❌ PNG save failed: {e}")


if __name__ == "__main__":
    # Create a model instance (sampling frequency 1000 Hz, sampling duration 10 seconds)
    model = LorenzRosslerModel(fs=1000, T=10.0)

    # Read sensor data (specified file path)
    excel_path = r"C:\Users\20172\Desktop\vibration_data_model_effect_part1.xlsx"
    time, vib_signal, torque_signal = model.read_raw_sensor_data(excel_path)

    # Use simulated data if data reading fails
    if time is None:
        print("Data reading failed; using simulated data for demonstration")
        t = np.linspace(0, 10, 10000)
        # Simulated vibration signal (with multiple frequency components + noise)
        vib_signal = 0.5 * np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 30 * t) + 0.1 * np.random.randn(len(t))
        # Simulated torque signal (with low-frequency components + noise)
        torque_signal = 2.0 * np.sin(2 * np.pi * 5 * t) + 1.0 * np.sin(2 * np.pi * 15 * t) + 0.2 * np.random.randn(
            len(t))
        time = t
    else:
        print(f"Successfully read{len(time)}data points")

    # Reconstruct state variables from sensor data
    x1, x2, x3, y1, y2, y3, time = model.reconstruct_state_from_sensors(time, vib_signal, torque_signal)

    # Calculate chaotic characteristics (only necessary calculation retained)
    lambda_max = model.calculate_max_lyapunov_exponent(x1, x2, x3, y1, y2, y3)
    print(f"Before optimization - maximum Lyapunov exponent: {lambda_max:.6f}")

    # Visualization 1: reconstructed state variables (2 x 3 layout, without x1/y1 prefixes)
    model.visualize_results(x1, x2, x3, y1, y2, y3, time, save_name="01_reconstructed_state_variables")

    # Parameter optimization (genetic algorithm)
    print("Starting parameter optimization (7 parameters)...")
    target_signals = [x1, x2, x3, y1, y2, y3]
    best_params, generations_rmse, generations_best_params = model.parameter_estimation_ga(
        target_signals, pop_size=15, generations=20)  # population size 15, 20 generations

    # Output optimized parameters
    sigma, r, b, a, d, c, gamma = best_params
    print(f"Optimized parameters: "
          f"sigma={sigma:.2f}, r={r:.2f}, b={b:.2f}, "
          f"a={a:.2f}, d={d:.2f}, c={c:.2f}, gamma={gamma:.3f}")

    # Visualization 2: GA optimization process
    model.visualize_ga_optimization(generations_rmse, save_name="02_GA_optimization_RMSE_change")

    # Apply optimized parameters and solve the model
    model.set_parameters(sigma, r, b, a, d, c, gamma)
    optimized_signals, _ = model.solve_ivp_method(len(target_signals[0]))

    # Visualization 3: signal comparison before and after optimization (2 x 3 layout)
    model.visualize_signal_comparison(time, target_signals, optimized_signals, save_name="03_signal_comparison_before_and_after_optimization")