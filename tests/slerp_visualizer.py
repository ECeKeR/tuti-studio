# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

# ── 1. Academic Presentation Styling ─────────────────────────────────────────
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.style.use('dark_background')

# ── 2. Vector Space Setup (3D Projection of High-Dim Space) ─────────────────
# In Qwen3-TTS, embeddings are ~1024 dimensions. 
# We project them onto a 3D unit sphere for geometric visualization.

# Vector A: Ryan's original speaker token
vec_ryan = np.array([0.9, 0.1, 0.4])
vec_ryan = vec_ryan / np.linalg.norm(vec_ryan)

# Vector B: User's reference centroid (extracted via VAD)
vec_ref = np.array([-0.6, 0.7, 0.5])
vec_ref = vec_ref / np.linalg.norm(vec_ref)

# Calculate the angle between them (Theta)
dot_product = np.dot(vec_ryan, vec_ref)
theta = np.arccos(np.clip(dot_product, -1.0, 1.0))

# ── 3. Mathematical Functions: LERP vs SLERP ────────────────────────────────

def lerp(v1, v2, t):
    """Linear Interpolation: Traces a straight chord through the sphere."""
    return (1.0 - t) * v1 + t * v2

def slerp(v1, v2, t, omega=theta):
    """Spherical Linear Interpolation: Traces an arc on the surface of the sphere."""
    if omega < 1e-5:
        return v1
    s0 = np.sin((1.0 - t) * omega) / np.sin(omega)
    s1 = np.sin(t * omega) / np.sin(omega)
    return s0 * v1 + s1 * v2

# Pre-calculate animation paths
t_values = np.linspace(0, 1, 120)
lerp_path = np.array([lerp(vec_ryan, vec_ref, t) for t in t_values])
slerp_path = np.array([slerp(vec_ryan, vec_ref, t) for t in t_values])

# ── 4. Matplotlib Figure Configuration ───────────────────────────────────────
fig = plt.figure(figsize=(12, 8), facecolor='#0a0a0a')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#0a0a0a')

# Remove axis panes for a cleaner "space" look
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#333333')
ax.yaxis.pane.set_edgecolor('#333333')
ax.zaxis.pane.set_edgecolor('#333333')

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
ax.set_xlabel('Dim 1', color='#aaaaaa', fontsize=10, labelpad=10)
ax.set_ylabel('Dim 2', color='#aaaaaa', fontsize=10, labelpad=10)
ax.set_zlabel('Dim 3', color='#aaaaaa', fontsize=10, labelpad=10)
ax.tick_params(axis='both', colors='#666666', labelsize=8)

# ── 5. Drawing the Unit Sphere (The Latent Space Boundary) ──────────────────
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 40)
x_s = np.outer(np.cos(u), np.sin(v))
y_s = np.outer(np.sin(u), np.sin(v))
z_s = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x_s, y_s, z_s, color='#222222', alpha=0.3, linewidth=0.5)

# ── 6. Static Elements (Endpoints and Formulas) ─────────────────────────────
# Origin point
ax.scatter([0], [0], [0], color='#ffffff', s=20, marker='o')

# Ryan Vector
ax.plot([0, vec_ryan[0]], [0, vec_ryan[1]], [0, vec_ryan[2]], color='#00ffff', lw=2, alpha=0.8)
ax.scatter(*vec_ryan, color='#00ffff', s=100, label='Speaker Token (Ryan)', depthshade=False)

# Reference Vector
ax.plot([0, vec_ref[0]], [0, vec_ref[1]], [0, vec_ref[2]], color='#ff00ff', lw=2, alpha=0.8)
ax.scatter(*vec_ref, color='#ff00ff', s=100, label='Reference Centroid (User)', depthshade=False)

# Dynamic Lines and Points (to be updated in animation)
lerp_line, = ax.plot([], [], [], color='#ff3333', lw=2.5, linestyle='--', label='LERP Path (Chord)')
slerp_line, = ax.plot([], [], [], color='#33ff33', lw=3, label='SLERP Path (Arc)')

lerp_dot, = ax.plot([], [], [], 'o', color='#ff3333', markersize=8)
slerp_dot, = ax.plot([], [], [], 'o', color='#33ff33', markersize=8)

# Dynamic Vectors from Origin (to show magnitude shrinking vs staying constant)
lerp_vec_line, = ax.plot([], [], [], color='#ff3333', lw=1.5, alpha=0.5)
slerp_vec_line, = ax.plot([], [], [], color='#33ff33', lw=1.5, alpha=0.5)

# ── 7. Text Overlays (Formulas and Real-time Metrics) ───────────────────────
# Title
ax.set_title("Speaker Embedding Space Interpolation\nPreserving Vector Magnitude in Latent Space", 
             color='white', fontsize=16, fontweight='bold', pad=20)

# Formulas (Top Left)
formula_text = (
    r"$\mathbf{LERP}: \quad \vec{v}_{lerp} = (1-t)\vec{v}_1 + t\vec{v}_2$" + "\n"
    r"$\mathbf{SLERP}: \quad \vec{v}_{slerp} = \frac{\sin((1-t)\theta)}{\sin(\theta)}\vec{v}_1 + \frac{\sin(t\theta)}{\sin(\theta)}\vec{v}_2$" + "\n"
    r"$\theta = \arccos(\vec{v}_1 \cdot \vec{v}_2) \approx " + f"{np.degrees(theta):.1f}^\circ$"
)
ax.text2D(0.02, 0.95, formula_text, transform=ax.transAxes, color='#dddddd', 
          fontsize=12, family='serif', verticalalignment='top',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#111111', edgecolor='#333333', alpha=0.9))

# Metrics Panel (Top Right)
metrics_template = (
    "INTERPOLATION STATE\n"
    "─────────────────────\n"
    r"$t$ (Alpha):        {:.2f}" + "\n"
    "─────────────────────\n"
    r"$\| \vec{v}_{lerp} \|$  :  {:.4f}" + "\n"
    r"$\| \vec{v}_{slerp} \|$:  {:.4f}" + "\n"
    "─────────────────────\n"
    r"$\Delta$ L2 Norm:    {:.4f}"
)
metrics_text = ax.text2D(0.75, 0.95, "", transform=ax.transAxes, color='#dddddd', 
                         fontsize=12, family='monospace', verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='#111111', edgecolor='#333333', alpha=0.9))

# Legend
ax.legend(loc='lower left', facecolor='#111111', edgecolor='#444444', labelcolor='white', fontsize=10)

# Camera angle
ax.view_init(elev=25, azim=45)

# ── 8. Animation Logic ──────────────────────────────────────────────────────
def init():
    lerp_line.set_data([], []); lerp_line.set_3d_properties([])
    slerp_line.set_data([], []); slerp_line.set_3d_properties([])
    lerp_dot.set_data([], []); lerp_dot.set_3d_properties([])
    slerp_dot.set_data([], []); slerp_dot.set_3d_properties([])
    lerp_vec_line.set_data([], []); lerp_vec_line.set_3d_properties([])
    slerp_vec_line.set_data([], []); slerp_vec_line.set_3d_properties([])
    return lerp_line, slerp_line, lerp_dot, slerp_dot, lerp_vec_line, slerp_vec_line, metrics_text

def update(frame):
    t = t_values[frame]
    
    # Update Paths
    lerp_line.set_data(lerp_path[:frame+1, 0], lerp_path[:frame+1, 1])
    lerp_line.set_3d_properties(lerp_path[:frame+1, 2])
    
    slerp_line.set_data(slerp_path[:frame+1, 0], slerp_path[:frame+1, 1])
    slerp_line.set_3d_properties(slerp_path[:frame+1, 2])
    
    # Update Dots
    l_curr, s_curr = lerp_path[frame], slerp_path[frame]
    
    lerp_dot.set_data([l_curr[0]], [l_curr[1]]); lerp_dot.set_3d_properties([l_curr[2]])
    slerp_dot.set_data([s_curr[0]], [s_curr[1]]); slerp_dot.set_3d_properties([s_curr[2]])
    
    # Update Vectors from Origin (Visualizing Magnitude)
    lerp_vec_line.set_data([0, l_curr[0]], [0, l_curr[1]]); lerp_vec_line.set_3d_properties([0, l_curr[2]])
    slerp_vec_line.set_data([0, s_curr[0]], [0, s_curr[1]]); slerp_vec_line.set_3d_properties([0, s_curr[2]])
    
    # Calculate Metrics
    l_norm = np.linalg.norm(l_curr)
    s_norm = np.linalg.norm(s_curr)
    delta_norm = l_norm - s_norm
    
    metrics_text.set_text(metrics_template.format(t, l_norm, s_norm, delta_norm))
    
    # Slowly rotate camera for dynamic presentation
    ax.view_init(elev=25, azim=45 + frame*0.5)
    
    return lerp_line, slerp_line, lerp_dot, slerp_dot, lerp_vec_line, slerp_vec_line, metrics_text

# Create animation (blit=False is required for 3D text/camera updates)
ani = FuncAnimation(fig, update, frames=len(t_values), init_func=init, blit=False, interval=40)

# ── 9. Display or Save ──────────────────────────────────────────────────────
plt.tight_layout()
plt.show()
