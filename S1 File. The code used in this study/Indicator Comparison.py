import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             precision_score, recall_score, f1_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import joblib
from itertools import cycle
import tensorflow as tf
from tensorflow import keras

# Configure default fonts for final visualization
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Suppress redundant TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


# ========================
# Fixed module: KPCA dimensionality-reduction class (keeps the previously fixed version)
# ========================
class KPCA:
    def __init__(self, kernel='rbf', tau=10.0):
        self.kernel = kernel
        self.tau = tau  # Gaussian-kernel parameter (set to 10 according to Reference 2)
        self.alpha = None  # Eigenvectors
        self.lambdas = None  # Eigenvalues
        self.X_train = None  # Cached training data
        self.k = None  # Number of effective dimensions

    def _rbf_kernel(self, X, Y):
        """Compute the Gaussian kernel matrix."""
        pairwise_sq_dists = np.sum(X ** 2, axis=1)[:, np.newaxis] + np.sum(Y ** 2, axis=1) - 2 * np.dot(X, Y.T)
        return np.exp(-pairwise_sq_dists / self.tau)

    def fit_transform(self, X):
        """Train KPCA and return the reduced-dimensional data (core bug fixed)."""
        self.X_train = X.copy()
        n_samples = X.shape[0]

        # 1. Compute the kernel matrix
        K = self._rbf_kernel(X, X)

        # 2. Center the kernel matrix
        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        # 3. Eigenvalue decomposition (keep only positive eigenvalues)
        eigvals, eigvecs = np.linalg.eigh(K_centered)

        # Filter out non-positive eigenvalues (fixes invalid sqrt warnings)
        positive_mask = eigvals > 1e-8
        eigvals = eigvals[positive_mask]
        eigvecs = eigvecs[:, positive_mask]

        # 4. Sort values (take the first k principal components; cumulative contribution rate >= 90%)
        idx = eigvals.argsort()[::-1]
        self.lambdas = eigvals[idx]
        self.alpha = eigvecs[:, idx]

        # 5. Normalize eigenvectors (with division-by-zero protection)
        if len(self.lambdas) > 0 and np.max(self.lambdas) > 1e-8:
            self.alpha = self.alpha / np.sqrt(self.lambdas[np.newaxis, :] + 1e-10)
        else:
            self.alpha = self.alpha

        # 6. Determine the effective number of dimensions
        if len(self.lambdas) == 0:
            self.k = 1
        else:
            cumulative_var = np.cumsum(self.lambdas) / (np.sum(self.lambdas) + 1e-10)
            self.k = np.argmax(cumulative_var >= 0.9) + 1
            # Ensure the dimension count does not exceed the number of eigenvalues
            self.k = min(self.k, len(self.lambdas))

        # 7. Compute and return reduced-dimensional data (core fix: return projected data, not eigenvectors)
        X_kpca = K_centered @ self.alpha[:, :self.k]

        print(f"KPCA dimensionality reduction complete: original shape {n_samples}x{X.shape[1]} -> reduced shape {n_samples}x{self.k}")
        return X_kpca

    def transform(self, X):
        """Reduce new data with the trained KPCA model (fixed centering logic)."""
        if self.X_train is None or self.alpha is None:
            raise ValueError("Call fit_transform before training the KPCA model.")

        n_train = self.X_train.shape[0]
        n_test = X.shape[0]

        # Compute the kernel matrix between the test set and the training set
        K = self._rbf_kernel(X, self.X_train)

        # Correct centering (core bug fixed)
        one_n_train = np.ones((n_test, n_train)) / n_train
        one_n_test = np.ones((n_train, n_train)) / n_train
        K_centered = K - one_n_train @ self._rbf_kernel(self.X_train,
                                                        self.X_train) - K @ one_n_test + one_n_train @ self._rbf_kernel(
            self.X_train, self.X_train) @ one_n_test

        # Project into the reduced-dimensional space
        X_kpca = K_centered @ self.alpha[:, :self.k]

        # Handle possible NaN values
        X_kpca = np.nan_to_num(X_kpca, nan=0.0, posinf=1e6, neginf=-1e6)

        return X_kpca


# ========================
# Core fix: ELM classifier (fixed pseudo-inverse regularization dimension mismatch)
# ========================
class ELMClassifier:
    def __init__(self, n_hidden=200, activation='sigmoid'):
        self.n_hidden = n_hidden  # Number of hidden-layer nodes (set to 200 according to Reference 2)
        self.activation = activation
        self.input_weights = None  # Input-layer weights
        self.bias = None  # Hidden-layer bias
        self.output_weights = None  # Output-layer weights
        self.input_dim = None  # Recorded input dimension
        self.is_trained = False

    def _sigmoid(self, x):
        """Sigmoid activation function with improved numerical stability."""
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def fit(self, X, y):
        """Train the ELM model (fixed pseudo-inverse regularization dimension mismatch)."""
        n_samples, n_features = X.shape
        self.input_dim = n_features  # Record the input dimension
        n_classes = len(np.unique(y))

        print(f"ELM training started: input shape {n_samples}x{n_features}, hidden-layer nodes {self.n_hidden}")

        # 1. Randomly initialize input weights and biases (uniform distribution in [-1, 1])
        self.input_weights = np.random.uniform(-1, 1, (n_features, self.n_hidden))
        self.bias = np.random.uniform(-1, 1, (1, self.n_hidden))

        # 2. Compute hidden-layer outputs with numerical-stability handling
        H = np.dot(X, self.input_weights) + self.bias
        if self.activation == 'sigmoid':
            H = self._sigmoid(H)

        # 3. One-hot encode labels
        y_onehot = np.eye(n_classes)[y]

        # 4. Solve output weights by pseudo-inverse (core fix: dimension-matched ridge regularization)
        # Replace the original incorrect regularization with the ridge-regression formula: (H.T * H + lambda * I)^-1 * H.T * y
        # Avoid dimension mismatch and improve numerical stability
        lambda_reg = 1e-8  # Regularization coefficient
        H_T = H.T
        # H.T * H has shape (n_hidden, n_hidden), matching the identity matrix
        HTH = np.dot(H_T, H)
        regularized_HTH = HTH + lambda_reg * np.eye(self.n_hidden)

        # Solve the regularized pseudo-inverse
        try:
            HTH_inv = np.linalg.inv(regularized_HTH)
            H_pinv = np.dot(HTH_inv, H_T)
        except np.linalg.LinAlgError:
            # Fallback: use NumPy pseudo-inverse directly (automatically handles singular matrices)
            H_pinv = np.linalg.pinv(H, rcond=1e-8)

        self.output_weights = np.dot(H_pinv, y_onehot)

        self.is_trained = True
        print("ELM training complete")

    def predict(self, X):
        """Predict labels with dimension validation."""
        if not self.is_trained:
            raise ValueError("The model has not been trained.")

        if X.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {X.shape[1]}")

        # Compute hidden-layer outputs
        H = np.dot(X, self.input_weights) + self.bias
        if self.activation == 'sigmoid':
            H = self._sigmoid(H)

        # Predict and return class labels
        y_pred = np.dot(H, self.output_weights)
        return np.argmax(y_pred, axis=1)


# ========================
# Optimized module: CNN classifier (keeps the optimized version)
# ========================
class CNNClassifier:
    def __init__(self, input_shape=(6,), n_classes=4):
        self.input_shape = input_shape  # Input feature dimension (six dimensions: x1-x3 and y1-y3)
        self.n_classes = n_classes  # Four load classes
        self.model = self._build_model()
        self.is_trained = False

    def _build_model(self):
        """Build a 1D-CNN model (fixed input-layer warning)."""
        model = keras.Sequential([
            # Warning fix: use an Input layer as the first layer
            keras.layers.Input(shape=self.input_shape),
            # Dimension expansion: (batch, 6) -> (batch, 6, 1)
            keras.layers.Reshape((*self.input_shape, 1)),
            # Convolution block 1: 64 kernels of size 7x1, ReLU activation, same padding
            keras.layers.Conv1D(64, kernel_size=7, strides=1, padding='same', activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2, padding='same'),
            keras.layers.BatchNormalization(),
            # Convolution block 2: 64 kernels of size 5x1
            keras.layers.Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2, padding='same'),
            keras.layers.BatchNormalization(),
            # Convolution block 3: 64 kernels of size 3x1
            keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2, padding='same'),
            keras.layers.BatchNormalization(),
            # Fully connected layers
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),  # Prevent overfitting
            keras.layers.Dense(self.n_classes, activation='softmax')
        ])
        # Compile the model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model with early stopping to prevent overfitting."""
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        self.model.fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[early_stopping],
                       verbose=1)
        self.is_trained = True

    def predict(self, X):
        """Predict labels."""
        if not self.is_trained:
            raise ValueError("The model has not been trained.")
        return np.argmax(self.model.predict(X, verbose=0), axis=1)


# ========================
# Original module: Lorenz-Rossler model (unchanged)
# ========================
class LorenzRosslerModel:
    def __init__(self, fs=100, T=10.0):
        self.fs = fs
        self.T = T
        self.dt = 1 / fs
        self.N = int(T * fs)
        self.t = np.linspace(0, T, self.N)
        self.min_required_points = 100

        # Model parameters
        self.sigma = 10.0
        self.r = 28.0
        self.b = 8.0 / 3.0
        self.a = 0.2
        self.d = 1.0
        self.c = 5.7
        self.gamma = 0.1

        # Sensor coefficients
        self.k_vib = 1.0
        self.k_torque = 1.0

        # Initial conditions
        self.x1_0 = 0.1
        self.x2_0 = 0.1
        self.x3_0 = 0.1
        self.y1_0 = 0.1
        self.y2_0 = 0.1
        self.y3_0 = 0.1

    def set_parameters(self, sigma=None, r=None, b=None, a=None, d=None, c=None, gamma=None):
        if sigma is not None: self.sigma = sigma
        if r is not None: self.r = r
        if b is not None: self.b = b
        if a is not None: self.a = a
        if d is not None: self.d = d
        if c is not None: self.c = c
        if gamma is not None: self.gamma = gamma

    def coupled_equations(self, t, state):
        x1, x2, x3, y1, y2, y3 = state
        # Lorenz system (vibration subsystem)
        dx1_dt = self.sigma * (x2 - x1) + self.gamma * (y1 + y2)
        dx2_dt = x1 * (self.r - x3) - x2 + self.gamma * (y2 + y3)
        dx3_dt = x1 * x2 - self.b * x3 + self.gamma * (y1 + y3)

        # Rossler system (torque subsystem)
        dy1_dt = -y2 - y3 + self.gamma * (x1 + x2)
        dy2_dt = y1 + self.a * y2 + self.gamma * (x2 + x3)
        dy3_dt = self.d + y1 * y3 - self.c * y3 + self.gamma * (x1 + x3)
        return [dx1_dt, dx2_dt, dx3_dt, dy1_dt, dy2_dt, dy3_dt]

    def solve_ivp_method(self, num_points=None):
        if num_points is None:
            num_points = self.N
        t = np.linspace(0, self.T, num_points)
        y0 = [self.x1_0, self.x2_0, self.x3_0, self.y1_0, self.y2_0, self.y3_0]
        sol = solve_ivp(self.coupled_equations, [0, self.T], y0, t_eval=t, method='RK45')
        return sol.y, t

    def reconstruct_state_from_sensors(self, time_data, vib_signal, torque_signal):
        if len(vib_signal) < self.min_required_points or len(torque_signal) < self.min_required_points:
            raise ValueError(f"Insufficient data points: at least {self.min_required_points} points are required.")

        if len(time_data) > 1:
            actual_fs = 1.0 / np.mean(np.diff(time_data))
            if not (90 <= actual_fs <= 110):
                print(f"Warning: actual sampling frequency ({actual_fs:.2f} Hz) differs greatly from the expected 100 Hz.")
        else:
            actual_fs = self.fs
        print(f"Actual sampling frequency: {actual_fs:.2f} Hz")

        min_len = min(len(time_data), len(vib_signal), len(torque_signal))
        time_data = time_data[:min_len]
        vib_signal = vib_signal[:min_len]
        torque_signal = torque_signal[:min_len]

        # State-variable reconstruction
        x1 = self.k_vib * vib_signal
        y2 = self.k_torque * torque_signal

        if len(x1) < 2:
            x2 = np.zeros_like(x1)
        else:
            x2 = np.gradient(x1) * actual_fs

        window_size = min(50, len(x1) // 20)
        if window_size < 2:
            x3 = x1 - np.mean(x1)
        else:
            x3 = np.convolve(x1, np.ones(window_size) / window_size, mode='same') - np.mean(x1)

        if len(y2) < 2:
            y1 = np.zeros_like(y2)
        else:
            y1 = np.gradient(y2) * actual_fs

        if len(y1) < 2:
            y3 = np.zeros_like(y1)
        else:
            y3 = np.abs(np.gradient(y1)) * actual_fs
        y3 = y3 / np.max(y3) if np.max(y3) > 0 else y3

        return x1, x2, x3, y1, y2, y3, time_data

    def read_group_data(self, file_path, group_info):
        try:
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
            except:
                df = pd.read_excel(file_path, engine='xlrd')

            group_name, time_col, vib_col, torque_col = group_info
            required_cols = [time_col, vib_col, torque_col]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            time_data = df[time_col].values
            vib_signal = df[vib_col].values
            torque_signal = df[torque_col].values

            valid_mask = ~np.isnan(time_data) & ~np.isnan(vib_signal) & ~np.isnan(torque_signal)
            time_data = time_data[valid_mask]
            vib_signal = vib_signal[valid_mask]
            torque_signal = torque_signal[valid_mask]

            print(f"{group_name} - Number of valid data points: {len(time_data)}")
            return time_data, vib_signal, torque_signal
        except Exception as e:
            print(f"{group_name} - Data read error: {str(e)}")
            return None, None, None

    def calculate_jacobian(self, state):
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

    def calculate_max_lyapunov_exponent(self, x1, x2, x3, y1, y2, y3):
        N = len(x1)
        if N < self.min_required_points:
            return np.nan

        step = max(1, N // 500)
        x1 = x1[::step]
        x2 = x2[::step]
        x3 = x3[::step]
        y1 = y1[::step]
        y2 = y2[::step]
        y3 = y3[::step]
        N = len(x1)

        lyapunov_sum = 0
        for k in range(N - 1):
            state = [x1[k], x2[k], x3[k], y1[k], y2[k], y3[k]]
            J = self.calculate_jacobian(state)
            eigenvalues = np.linalg.eigvals(J)
            max_real_part = np.max(np.real(eigenvalues))
            lyapunov_sum += max_real_part * self.dt
        return lyapunov_sum / (N * self.dt)

    def calculate_lyapunov_exponent_time_series(self, x1, x2, x3, y1, y2, y3, window_size=None):
        N = len(x1)
        if window_size is None:
            window_size = min(1000, N // 5)

        if N <= window_size:
            print(f"Warning: data length ({N}) is smaller than the window size ({window_size}); using all data for calculation.")
            window_size = N

        step = max(1, window_size // 100)
        le_time_series = np.full(N, np.nan)

        for i in range(window_size - 1, N, step):
            window_x1 = x1[i - window_size + 1:i + 1]
            window_x2 = x2[i - window_size + 1:i + 1]
            window_x3 = x3[i - window_size + 1:i + 1]
            window_y1 = y1[i - window_size + 1:i + 1]
            window_y2 = y2[i - window_size + 1:i + 1]
            window_y3 = y3[i - window_size + 1:i + 1]

            win_step = max(1, len(window_x1) // 200)
            window_x1 = window_x1[::win_step]
            window_x2 = window_x2[::win_step]
            window_x3 = window_x3[::win_step]
            window_y1 = window_y1[::win_step]
            window_y2 = window_y2[::win_step]
            window_y3 = window_y3[::win_step]

            window_le = self.calculate_max_lyapunov_exponent(
                window_x1, window_x2, window_x3, window_y1, window_y2, window_y3)

            for j in range(i, min(i + step, N)):
                le_time_series[j] = window_le

        return le_time_series

    def objective_function(self, params, target_signals):
        sigma, r, b, a, d, c, gamma = params
        self.set_parameters(sigma, r, b, a, d, c, gamma)

        step = max(1, len(target_signals[0]) // 500)
        downsampled_signals = [sig[::step] for sig in target_signals]

        model_signals, _ = self.solve_ivp_method(len(downsampled_signals[0]))
        rmse = 0
        for model, target in zip(model_signals, downsampled_signals):
            min_len = min(len(model), len(target))
            rmse += np.mean((model[:min_len] - target[:min_len]) ** 2)
        return np.sqrt(rmse / len(model_signals))

    def parameter_estimation_ga(self, target_signals, pop_size=10, generations=15):
        param_ranges = [
            (5.0, 20.0),  # sigma
            (10.0, 40.0),  # r
            (1.0, 5.0),  # b
            (0.1, 1.0),  # a
            (0.5, 5.0),  # d
            (2.0, 10.0),  # c
            (0.01, 0.5)  # gamma
        ]

        population = [[random.uniform(low, high) for low, high in param_ranges]
                      for _ in range(pop_size)]
        best_fitness = float('-inf')
        best_params = None
        generations_rmse = []

        for gen in range(generations):
            fitness = [1 / (self.objective_function(ind, target_signals) + 1e-8)
                       for ind in population]
            current_best_rmse = 1 / max(fitness) - 1e-8
            generations_rmse.append(current_best_rmse)

            total_fitness = sum(fitness)
            if total_fitness == 0:
                probs = [1.0 / pop_size] * pop_size
            else:
                probs = [f / total_fitness for f in fitness]

            new_population = []
            for _ in range(pop_size):
                parent1 = population[np.random.choice(pop_size, p=probs)]
                parent2 = population[np.random.choice(pop_size, p=probs)]
                child = [parent1[j] if random.random() < 0.5 else parent2[j]
                         for j in range(7)]
                for j in range(7):
                    if random.random() < 0.1:
                        child[j] = random.uniform(param_ranges[j][0], param_ranges[j][1])
                new_population.append(child)
            population = new_population

            curr_fitness = max(fitness)
            if curr_fitness > best_fitness:
                best_fitness = curr_fitness
                best_params = population[fitness.index(curr_fitness)]

            if gen % 5 == 0:
                print(f"Iteration {gen}, best RMSE: {current_best_rmse:.6f}")

        return best_params, generations_rmse

    def process_group(self, group_name, time_data, vib_signal, torque_signal):
        start_time = time.time()
        print(f"\n===== Start processing {group_name} data =====")

        if len(time_data) < self.min_required_points:
            print(f"Error: {group_name} has insufficient data points ({len(time_data)}); at least {self.min_required_points} are required.")
            return None

        try:
            x1, x2, x3, y1, y2, y3, time_data = self.reconstruct_state_from_sensors(time_data, vib_signal,
                                                                                    torque_signal)
        except ValueError as e:
            print(f"Error while processing {group_name}: {str(e)}")
            return None

        lambda_max = self.calculate_max_lyapunov_exponent(x1, x2, x3, y1, y2, y3)
        print(f"{group_name} - Before optimization - maximum Lyapunov exponent: {lambda_max:.6f}")

        print(f"{group_name} - Calculating the Lyapunov exponent time series...")
        le_time_series = self.calculate_lyapunov_exponent_time_series(x1, x2, x3, y1, y2, y3)

        print(f"{group_name} - Starting parameter optimization...")
        target_signals = [x1, x2, x3, y1, y2, y3]
        best_params, _ = self.parameter_estimation_ga(target_signals)

        if best_params is not None:
            sigma, r, b, a, d, c, gamma = best_params
            print(f"{group_name} - Optimized parameters: "
                  f"sigma={sigma:.2f}, r={r:.2f}, b={b:.2f}, "
                  f"a={a:.2f}, d={d:.2f}, c={c:.2f}, gamma={gamma:.3f}")

        if best_params is not None:
            self.set_parameters(sigma, r, b, a, d, c, gamma)
        optimized_signals, _ = self.solve_ivp_method(len(target_signals[0]))

        lambda_max_optimized = self.calculate_max_lyapunov_exponent(
            optimized_signals[0], optimized_signals[1], optimized_signals[2],
            optimized_signals[3], optimized_signals[4], optimized_signals[5]
        )
        print(f"{group_name} - Optimized maximum Lyapunov exponent: {lambda_max_optimized:.6f}")

        full_data = pd.DataFrame({
            'Time_s': time_data,
            'Raw_Vibration_Signal': vib_signal[:len(time_data)],
            'Raw_Torque_Signal': torque_signal[:len(time_data)],
            'Reconstructed_x1': x1,
            'Reconstructed_x2': x2,
            'Reconstructed_x3': x3,
            'Reconstructed_y1': y1,
            'Reconstructed_y2': y2,
            'Reconstructed_y3': y3,
            'Optimized_x1': optimized_signals[0][:len(time_data)],
            'Optimized_x2': optimized_signals[1][:len(time_data)],
            'Optimized_x3': optimized_signals[2][:len(time_data)],
            'Optimized_y1': optimized_signals[3][:len(time_data)],
            'Optimized_y2': optimized_signals[4][:len(time_data)],
            'Optimized_y3': optimized_signals[5][:len(time_data)],
            'Lyapunov_Exponent_Time_Series': le_time_series
        })

        processing_time = time.time() - start_time
        print(f"{group_name} processing complete. Elapsed time: {processing_time:.2f} s")

        return {
            'group_name': group_name,
            'time': time_data,
            'lambda_max_before': lambda_max,
            'lambda_max_after': lambda_max_optimized,
            'le_time_series_after': le_time_series,
            'optimized_params': best_params,
            'full_data': full_data
        }


# ========================
# Original module: multidimensional feature preparation (unified data input)
# ========================
def prepare_multidim_features(all_results):
    """Extract reconstructed state variables (x1-x3 and y1-y3) as multidimensional features."""
    X_multidim = []
    y_multidim = []
    class_names = [result['group_name'] for result in all_results]
    label_mapping = {name: i for i, name in enumerate(class_names)}

    for result in all_results:
        label = label_mapping[result['group_name']]
        full_data = result['full_data']
        # Extract six-dimensional features: reconstructed x1-x3 plus reconstructed y1-y3
        features = full_data[['Reconstructed_x1', 'Reconstructed_x2', 'Reconstructed_x3', 'Reconstructed_y1', 'Reconstructed_y2', 'Reconstructed_y3']].values
        # Filter samples containing NaN values
        valid_mask = ~np.any(np.isnan(features), axis=1)
        X_multidim.extend(features[valid_mask])
        y_multidim.extend(np.full(np.sum(valid_mask), label))

    return np.array(X_multidim), np.array(y_multidim), class_names, label_mapping


# ========================
# Original module: model-comparison visualization
# ========================
def plot_model_comparison(model_results):
    """Plot a four-metric comparison chart: accuracy, precision, recall, and training time."""
    results_df = pd.DataFrame(model_results)
    models = results_df['model'].values
    metrics = ['accuracy', 'precision', 'recall', 'train_time']
    titles = ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Training Time (s)']
    y_labels = ['Accuracy', 'Precision', 'Recall', 'Time (s)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Comparison of Three Load-Recognition Models', fontsize=16, fontweight='bold')

    for idx, (metric, title, y_label) in enumerate(zip(metrics, titles, y_labels)):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        # Draw the bar chart
        bars = ax.bar(models, results_df[metric], color=colors, alpha=0.8, edgecolor='black')
        # Set titles and labels
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            fmt = '.4f' if metric != 'train_time' else '.2f'
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    format(height, fmt), ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('model_comparison_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Model-comparison visualization saved as model_comparison_visualization.png")


# ========================
# Main program (integrates the original logic and new model training)
# ========================
if __name__ == "__main__":
    total_start_time = time.time()

    # Part 1: Run the Lorenz-Rossler model (original logic unchanged)
    print("===== Start running the coupled Lorenz-Rossler chaotic-system model =====")
    lorenz_rossler = LorenzRosslerModel(fs=100, T=10.0)

    # Data-group configuration
    groups_config = [
        ('No_Load', 'Time_No_Load', 'Vibration_No_Load', 'Torque_No_Load'),
        ('Light_Load', 'Time_Light_Load', 'Vibration_Light_Load', 'Torque_Light_Load'),
        ('Normal_Load', 'Time_Normal_Load', 'Vibration_Normal_Load', 'Torque_Normal_Load'),
        ('Overload', 'Time_Overload', 'Vibration_Overload', 'Torque_Overload')
    ]
    excel_path = r"C:\Users\20172\Desktop\vibration_torque_data.xlsx"  # Replace with the actual path
    all_results = []

    # Process each data group
    for group_info in groups_config:
        group_name = group_info[0]
        time_data, vib_signal, torque_signal = lorenz_rossler.read_group_data(excel_path, group_info)

        if time_data is None or vib_signal is None or torque_signal is None:
            print(f"Skipping {group_name} data processing.\n")
            continue
        if len(time_data) < lorenz_rossler.min_required_points:
            print(f"{group_name} has insufficient data points and has been skipped.\n")
            continue
        elif not (800 <= len(time_data) <= 1200):
            print(f"{group_name} data-point count differs greatly from the expected range.\n")

        result = lorenz_rossler.process_group(group_name, time_data, vib_signal, torque_signal)
        if result is not None:
            all_results.append(result)
            result['full_data'].to_csv(f'{group_name}_full_data_100hz.csv', index=False, encoding='utf-8-sig')
            print(f"{group_name} data saved.\n")

    # Output the Lorenz-Rossler analysis summary
    print("\n===== Summary of All Group Analysis Results =====")
    print(f"{'Group':<10} | {'Optimized Lyapunov Exponent':<20}")
    print("-" * 40)
    for result in all_results:
        print(f"{result['group_name']:<10} | {result['lambda_max_after']:<20.6f}")

    if all_results:
        results_df = pd.DataFrame(all_results)[['group_name', 'lambda_max_before', 'lambda_max_after']]
        results_df.to_csv('group_analysis_summary_100hz.csv', index=False, encoding='utf-8-sig')
        print("\nAnalysis summary saved.")
    else:
        print("No valid analysis results. Exiting program.")
        exit()

    # ========================
    # Part 2: New model training and comparison
    # ========================
    print("\n===== Start model training and comparison =====")
    # 1. Prepare multidimensional features and the unified dataset
    X_multidim, y_multidim, class_names, label_mapping = prepare_multidim_features(all_results)
    print(f"Multidimensional feature data size: {len(X_multidim)} samples, {len(class_names)} classes")

    # 2. Unified data split (60% training set + 20% validation set + 20% test set)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_multidim, y_multidim, test_size=0.2, random_state=42, stratify=y_multidim
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    print(f"Training set: {len(X_train)} samples, validation set: {len(X_val)} samples, test set: {len(X_test)} samples")

    # 3. Data standardization (for CNN and SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. KPCA dimensionality reduction (for ELM)
    kpca = KPCA(tau=10.0)
    X_train_kpca = kpca.fit_transform(X_train_scaled)
    X_val_kpca = kpca.transform(X_val_scaled)
    X_test_kpca = kpca.transform(X_test_scaled)
    print(f"Feature dimension after KPCA reduction: {X_train_kpca.shape[1]}")

    # 5. Store model results
    model_results = []

    # ------------------------
    # Model 1: SVM (baseline model using multidimensional features)
    # ------------------------
    print("\n===== Train the SVM model =====")
    svm_clf = SVC(kernel='rbf', gamma='scale', random_state=42, probability=True)
    svm_start = time.time()
    svm_clf.fit(X_train_scaled, y_train)
    svm_time = time.time() - svm_start

    y_pred_svm = svm_clf.predict(X_test_scaled)
    svm_metrics = {
        'model': 'SVM',
        'accuracy': accuracy_score(y_test, y_pred_svm),
        'precision': precision_score(y_test, y_pred_svm, average='weighted'),
        'recall': recall_score(y_test, y_pred_svm, average='weighted'),
        'train_time': svm_time
    }
    model_results.append(svm_metrics)
    print(f"SVM training time: {svm_time:.2f} s, accuracy: {svm_metrics['accuracy']:.4f}")

    # ------------------------
    # Model 2: CNN (1D-CNN structure according to Reference 1)
    # ------------------------
    print("\n===== Train the CNN model =====")
    cnn_clf = CNNClassifier(input_shape=(6,), n_classes=4)
    cnn_start = time.time()
    cnn_clf.fit(X_train_scaled, y_train, X_val_scaled, y_val, epochs=50, batch_size=32)
    cnn_time = time.time() - cnn_start

    y_pred_cnn = cnn_clf.predict(X_test_scaled)
    cnn_metrics = {
        'model': 'CNN',
        'accuracy': accuracy_score(y_test, y_pred_cnn),
        'precision': precision_score(y_test, y_pred_cnn, average='weighted'),
        'recall': recall_score(y_test, y_pred_cnn, average='weighted'),
        'train_time': cnn_time
    }
    model_results.append(cnn_metrics)
    print(f"CNN training time: {cnn_time:.2f} s, accuracy: {cnn_metrics['accuracy']:.4f}")

    # ------------------------
    # Model 3: KPCA-ELM (fusion strategy according to Reference 2)
    # ------------------------
    print("\n===== Train the KPCA-ELM model =====")
    elm_clf = ELMClassifier(n_hidden=200, activation='sigmoid')
    elm_start = time.time()
    elm_clf.fit(X_train_kpca, y_train)
    elm_time = time.time() - elm_start

    y_pred_elm = elm_clf.predict(X_test_kpca)
    elm_metrics = {
        'model': 'KPCA-ELM',
        'accuracy': accuracy_score(y_test, y_pred_elm),
        'precision': precision_score(y_test, y_pred_elm, average='weighted'),
        'recall': recall_score(y_test, y_pred_elm, average='weighted'),
        'train_time': elm_time
    }
    model_results.append(elm_metrics)
    print(f"KPCA-ELM training time: {elm_time:.2f} s, accuracy: {elm_metrics['accuracy']:.4f}")

    # 6. Output model-comparison results
    print("\n===== Model Performance Comparison Summary =====")
    comparison_df = pd.DataFrame(model_results)
    print(comparison_df.round(4))
    comparison_df.to_csv('model_comparison_results.csv', index=False, encoding='utf-8-sig')
    print("\nComparison results saved as model_comparison_results.csv")

    # 7. Plot the comparison visualization
    plot_model_comparison(model_results)

    # Total running time
    total_time = time.time() - total_start_time
    print(f"\nTotal program running time: {total_time:.2f} s")
