import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit

# ---------------- Font configuration for English-only output ----------------
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- Original experimental data ----------------
# Independent variables: forward speed A (m/s) and drum speed C (r/min)
A = np.array([0.28, 0.28, 0.56, 0.56, 0.14, 0.70, 0.42, 0.42, 0.42, 0.42, 0.42])
C = np.array([130, 190, 130, 190, 160, 160, 100, 220, 160, 160, 160])

# Dependent variables: cleaning rate Y (%) and damage rate S (%)
Y = np.array([98.80, 98.50, 98.20, 97.90, 99.60, 98.55, 98.95, 99.30, 99.45, 99.40, 99.50])
S = np.array([2.90, 4.80, 2.70, 5.20, 3.45, 3.75, 1.70, 7.80, 2.80, 2.75, 2.85])

# ---------------- Quadratic response surface fitting function ----------------
def response_surface_fun(x, a0, a1, a2, a3, a4, a5):
    """y = a0 + a1*A + a2*C + a3*A**2 + a4*C**2 + a5*A*C"""
    A, C = x
    return a0 + a1 * A + a2 * C + a3 * (A ** 2) + a4 * (C ** 2) + a5 * A * C


# Fit model parameters for cleaning rate Y and damage rate S
x_data = (A, C)
popt_Y, _ = curve_fit(response_surface_fun, x_data, Y)  # Fitted parameters for cleaning rate
popt_S, _ = curve_fit(response_surface_fun, x_data, S)  # Fitted parameters for damage rate

# ---------------- Build mesh data for smooth 3D surfaces ----------------
A_grid = np.linspace(0.14, 0.70, 50)  # Forward speed range covering the experimental values
C_grid = np.linspace(100, 220, 50)  # Drum speed range covering the experimental values
A_mesh, C_mesh = np.meshgrid(A_grid, C_grid)

# Calculate fitted response values
Z_Y = response_surface_fun((A_mesh, C_mesh), *popt_Y)  # 3D surface data for cleaning rate Y
Z_S = response_surface_fun((A_mesh, C_mesh), *popt_S)  # 3D surface data for damage rate S

# ---------------- 1. Plot the 3D response surface for cleaning rate Y ----------------
fig1 = plt.figure(figsize=(8, 6), dpi=600)  # High-resolution output at 600 DPI
ax1 = fig1.add_subplot(111, projection='3d')

# Plot the 3D surface
surf_Y = ax1.plot_surface(
    A_mesh,
    C_mesh,
    Z_Y,
    cmap=cm.coolwarm,
    alpha=0.8,
    linewidth=0,
    antialiased=True
)

# Mark the original experimental points
ax1.scatter(A, C, Y, c='black', s=40, label='Experimental Points', zorder=5)

# Axis settings
ax1.set_xlabel('Forward Speed A (m/s)', fontsize=12, labelpad=10)
ax1.set_ylabel('Drum Speed C (r/min)', fontsize=12, labelpad=10)
ax1.set_zlabel('Cleaning Rate Y (%)', fontsize=12, labelpad=10)
ax1.set_title('3D Response Surface of Cleaning Rate', fontsize=14, fontweight='bold', pad=20)

# Color bar and legend
fig1.colorbar(surf_Y, ax=ax1, shrink=0.6, aspect=10, label='Cleaning Rate Y (%)')
ax1.legend(loc='upper left', fontsize=10)

# Save the cleaning rate 3D plot in TIFF format
plt.tight_layout()
plt.savefig('cleaning_rate_Y_3D_response_surface.tif', dpi=600, bbox_inches='tight')
plt.show()

# ---------------- 2. Plot the 3D response surface for damage rate S ----------------
fig2 = plt.figure(figsize=(8, 6), dpi=600)
ax2 = fig2.add_subplot(111, projection='3d')

# Plot the 3D surface
surf_S = ax2.plot_surface(
    A_mesh,
    C_mesh,
    Z_S,
    cmap=cm.hot,
    alpha=0.8,
    linewidth=0,
    antialiased=True
)

# Mark the original experimental points
ax2.scatter(A, C, S, c='black', s=40, label='Experimental Points', zorder=5)

# Axis settings
ax2.set_xlabel('Forward Speed A (m/s)', fontsize=12, labelpad=10)
ax2.set_ylabel('Drum Speed C (r/min)', fontsize=12, labelpad=10)
ax2.set_zlabel('Damage Rate S (%)', fontsize=12, labelpad=10)
ax2.set_title('3D Response Surface of Damage Rate', fontsize=14, fontweight='bold', pad=20)

# Color bar and legend
fig2.colorbar(surf_S, ax=ax2, shrink=0.6, aspect=10, label='Damage Rate S (%)')
ax2.legend(loc='upper left', fontsize=10)

# Save the damage rate 3D plot in TIFF format
plt.tight_layout()
plt.savefig('damage_rate_S_3D_response_surface.tif', dpi=600, bbox_inches='tight')
plt.show()
