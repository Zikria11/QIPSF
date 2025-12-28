import os
import matplotlib.pyplot as plt
import networkx as nx

class LivePlotter:
    def __init__(self, run_id):
        self.run_id = run_id
        self.fig, self.axes = plt.subplots(1, 2, figsize=(10, 4))
        self.times = []
        self.S_history = []
        plt.ion()

    def update(self, t, S_t, G, redraw=True, snapshot=False):
        self.times.append(t)
        self.S_history.append(S_t)

        if not redraw:
            return

        ax_net, ax_stab = self.axes
        ax_net.clear()
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(G, pos=pos, ax=ax_net,
                         node_size=80, with_labels=False)
        ax_net.set_title(f"Network at t={t:.1f}")

        ax_stab.clear()
        ax_stab.plot(self.times, self.S_history, '-b')
        ax_stab.set_xlabel("Time")
        ax_stab.set_ylabel("Stability index S(t)")
        ax_stab.set_title("Stability over time")

        plt.tight_layout()
        plt.pause(0.01)

        # Save snapshot of network for report
        if snapshot:
            snap_path = f"outputs/snapshots/network_t{int(t):04d}_{self.run_id}.png"
            self.fig.savefig(snap_path, dpi=200)

    def save_final_figures(self):
        # Full stability plot only
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.times, self.S_history, '-b')
        ax.set_xlabel("Time")
        ax.set_ylabel("Stability index S(t)")
        ax.set_title("Stability over time")
        fig.tight_layout()
        fig.savefig(f"outputs/figures/stability_{self.run_id}.png", dpi=300)
        plt.close(fig)

    def finalize(self):
        self.save_final_figures()
        plt.ioff()
        plt.show()