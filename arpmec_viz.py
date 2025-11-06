from arpmec_config import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import matplotlib.widgets as mwidgets
from colorama import Fore, Style
import numpy as np
import arpmec_state as state  # usage du module d'état global

def assign_bs_to_ch(clusters, base_stations):
    bs_assign = {}
    for c in clusters:
        distances = [(bs, np.linalg.norm(bs.pos - c.ch.pos)) for bs in base_stations]
        bs_closest = min(distances, key=lambda x: x[1])[0]
        bs_assign[c.ch.id] = bs_closest
    return bs_assign

def assign_mec_to_bs(base_stations, mec_servers):
    mec_assign = {}
    for bs in base_stations:
        distances = [(mec, np.linalg.norm(mec.pos - bs.pos)) for mec in mec_servers]
        mec_closest = min(distances, key=lambda x: x[1])[0]
        mec_assign[bs.id] = mec_closest
    return mec_assign

def draw_clusters(ax, clusters, bs_assign, mec_assign):
    ax.clear()
    ax.set_xlim(0, AREA_SIZE)
    ax.set_ylim(0, AREA_SIZE)
    ax.set_title("ARPMEC+FT: Advanced Clustering with BS & MEC")
    colors = plt.colormaps['tab10'].resampled(max(len(clusters), 1))
    for idx, cluster in enumerate(clusters):
        color = colors(idx)
        points = [cluster.ch.pos] + [m.pos for m in cluster.members]
        centroid = np.mean(points, axis=0)
        max_dist = max(np.linalg.norm(p - centroid) for p in points)
        radius = max(max_dist + 5, 12)
        circle = plt.Circle(centroid, radius, edgecolor=color, linestyle='--', fill=False, linewidth=1.5)
        ax.add_patch(circle)
        for m in cluster.members:
            ax.plot([cluster.ch.pos[0], m.pos[0]], [cluster.ch.pos[1], m.pos[1]], c=color, linewidth=0.7)
            ax.plot(m.pos[0], m.pos[1], 'o', color=color)
            ax.text(m.pos[0] + 0.7, m.pos[1] + 0.7, f"{m.id}", fontsize=8, color=color)
        ch = cluster.ch
        ax.plot(ch.pos[0], ch.pos[1], 'X', color=color, markersize=12, markeredgecolor='k', markeredgewidth=1.5)
        ax.text(ch.pos[0] + 0.7, ch.pos[1] + 0.7, f"CH{ch.id}", fontsize=9, fontweight='bold', color=color)
        ax.text(centroid[0], centroid[1] + radius + 2, f"Cluster {idx}", fontsize=10, fontweight='bold', color=color)
        bs = bs_assign.get(ch.id)
        if bs:
            ax.plot(bs.pos[0], bs.pos[1], 's', color='red', markersize=10)
            mec = mec_assign.get(bs.id)
            if mec:
                ax.plot(mec.pos[0], mec.pos[1], 'D', color='purple', markersize=10)
                ax.plot([bs.pos[0], mec.pos[0]], [bs.pos[1], mec.pos[1]], 'purple', linestyle=':', linewidth=1.2)
            ax.plot([ch.pos[0], bs.pos[0]], [ch.pos[1], bs.pos[1]], 'r-', linewidth=1.2)
    legend_items = [
        mpatches.Patch(color='blue', label='Members'),
        mpatches.Patch(color='orange', label='cluster Head (CH)'),
        mpatches.Patch(color='red', label='Base Stations (BS)'),
        mpatches.Patch(color='purple', label='MEC Serveurs'),
        mpatches.Patch(facecolor='none', label='Bounded Clusters', linestyle='--', edgecolor='gray')
    ]
    ax.legend(handles=legend_items, loc='upper right')

def visualize_simulation():
    paused = [False]
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.25)
    n_rounds = len(state.clusters_by_round)
    bs_assign_all = [assign_bs_to_ch(c, state.base_stations) for c in state.clusters_by_round]
    mec_assign_all = [assign_mec_to_bs(state.base_stations, state.mec_servers) for _ in state.clusters_by_round]

    def update(frame):
        round_num = int(frame)
        ax.set_title(f"ARPMEC+FT Simulation - Round {round_num}")
        if round_num < n_rounds:
            clusters = state.clusters_by_round[round_num]
            bs_assign = bs_assign_all[round_num]
            mec_assign = mec_assign_all[round_num]
            draw_clusters(ax, clusters, bs_assign, mec_assign)
        fig.canvas.draw_idle()

    slider_ax = plt.axes([0.25, 0.1, 0.50, 0.03])
    round_slider = mwidgets.Slider(slider_ax, 'Round', 0, n_rounds - 1, valinit=0, valstep=1)
    round_slider.on_changed(lambda val: update(int(val)))

    def on_pause(event):
        paused[0] = not paused[0]
        btn_pause.label.set_text("Play" if paused[0] else "Pause")

    pause_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    btn_pause = mwidgets.Button(pause_ax, 'Pause')
    btn_pause.on_clicked(on_pause)

    ani = animation.FuncAnimation(fig, update, frames=range(n_rounds), interval=1500, repeat=False)
    plt.show()

