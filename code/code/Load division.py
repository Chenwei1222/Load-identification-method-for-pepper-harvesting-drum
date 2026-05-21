import numpy as np
import matplotlib.pyplot as plt
import os

# Create the output directory
if not os.path.exists('vector_output'):
    os.makedirs('vector_output')

# Set English-compatible fonts
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Data extracted from the chart
W = np.array([0.2, 0.4, 0.57, 0.8, 1.2, 1.6, 1.92, 2.2, 2.5])  # Feeding rate (kg/s)
S = np.array([1.25, 1.88, 3.50, 2.65, 2.12, 2.98, 3.52, 4.86, 6.25])  # Damage rate (%)
Y = np.array([99.52, 99.36, 98.50, 98.85, 99.23, 98.96, 98.48, 97.85, 96.92])  # Clean picking rate (%)

# Create a dual-axis plot
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)

# Left Y-axis: damage rate
color = '#d62728'
ax1.set_xlabel(r'Feeding Rate $\it{W}$ (kg/s)', fontsize=14)
ax1.set_ylabel(r'Damage Rate $\it{S}$ (%)', color=color, fontsize=14)
line1, = ax1.plot(W, S, color=color, marker='o', linestyle='-', linewidth=2, markersize=6, label='Damage Rate')
ax1.tick_params(axis='y', labelcolor=color, labelsize=14)
ax1.set_ylim(0, 7)
ax1.set_yticks(np.arange(0, 8, 1))
ax1.tick_params(axis='x', labelsize=14)

# Right Y-axis: clean picking rate
ax2 = ax1.twinx()
color = '#1f77b4'
ax2.set_ylabel(r'Clean Picking Rate $\it{Y}$ (%)', color=color, fontsize=14)
line2, = ax2.plot(W, Y, color=color, marker='s', linestyle='-', linewidth=2, markersize=6, label='Clean Picking Rate')
ax2.tick_params(axis='y', labelcolor=color, labelsize=14)
ax2.set_ylim(96.5, 100)
ax2.set_yticks(np.arange(96.5, 100.1, 0.5))

# Add vertical condition boundary lines and labels
ax1.axvline(x=0.57, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=1.92, color='gray', linestyle='--', alpha=0.5)
ax1.text(0.25, 0.2, 'Light Load', ha='center', va='top', fontsize=12)
ax1.text(1.25, 0.2, 'Normal Load', ha='center', va='top', fontsize=12)
ax1.text(2.25, 0.2, 'Overload', ha='center', va='top', fontsize=12)

# Legend
fig.legend(
    [line1, line2],
    ['Damage Rate', 'Clean Picking Rate'],
    loc='upper right',
    bbox_to_anchor=(0.9, 0.9),
    fontsize=12
)

# Grid
ax1.grid(True, alpha=0.3)

# Title
plt.title('Effect of Feeding Rate on Clean Picking Rate and Damage Rate', fontsize=16)

# Adjust layout
plt.tight_layout()

# Save in vector and high-resolution raster formats
plt.savefig('vector_output/feeding_rate_curve.svg', format='svg', dpi=300, bbox_inches='tight', transparent=True)
plt.savefig('vector_output/feeding_rate_curve.eps', format='eps', dpi=300, bbox_inches='tight', transparent=True)
plt.savefig('vector_output/feeding_rate_curve.pdf', format='pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.savefig('vector_output/feeding_rate_curve.png', format='png', dpi=300, bbox_inches='tight', transparent=True)

print("Vector graphics have been saved to the vector_output directory.")
print("Included formats: SVG, EPS, PDF, PNG at 300 DPI.")

# Display the chart
plt.show()
