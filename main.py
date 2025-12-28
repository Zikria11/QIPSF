# main.py
import numpy as np
import networkx as nx

from quantum_rules import (
    update_link_probabilities,
    apply_entanglement_correlation,
    stability_index,
)
from network_generator import (
    initialize_prob_matrix,
    sample_graph_from_probabilities,
    apply_backbone_floor,
)
from agent_rl import QLearningAgent
from visualization import LivePlotter
from logging_utils import create_run_id, log_params, init_csv_logger, log_step, ensure_dirs


def run_qipsf_simulation(
    n_nodes=30,
    p0=0.2,
    lam=0.03,
    eta=0.1,
    omega=0.2,
    rho0=0.3,
    mu=0.01,
    T=1000,
    dt=1.0,
    correlated_pairs=None,
    backbone_p_min=0.1,
    max_steps_per_episode=20,
    reward_goal=1.0,
    reward_step=-0.005,
    reward_fail=-0.2,
    seed=42,
    plot_interval=20,
    snapshot_interval=100,
):
    if correlated_pairs is None:
        correlated_pairs = []

    ensure_dirs()
    run_id = create_run_id()

    params = dict(
        n_nodes=n_nodes,
        p0=p0,
        lam=lam,
        eta=eta,
        omega=omega,
        rho0=rho0,
        mu=mu,
        T=T,
        dt=dt,
        correlated_pairs=correlated_pairs,
        backbone_p_min=backbone_p_min,
        max_steps_per_episode=max_steps_per_episode,
        reward_goal=reward_goal,
        reward_step=reward_step,
        reward_fail=reward_fail,
        seed=seed,
    )
    log_params(run_id, params)

    P = initialize_prob_matrix(n_nodes, p0, seed=seed)
    agent = QLearningAgent(n_nodes=n_nodes, seed=seed)

    # Define a simple ring backbone: (0,1), (1,2), ..., (n-1,0)
    backbone_edges = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]

    plotter = LivePlotter(run_id)
    csv_file, csv_writer = init_csv_logger(run_id)

    rng = np.random.default_rng(seed)

    episode_reward = 0.0
    src = rng.integers(0, n_nodes)
    dst = rng.integers(0, n_nodes)
    while dst == src:
        dst = rng.integers(0, n_nodes)
    state = src

    for step in range(T):
        t = step * dt

        # 1) Update link probabilities + correlation + backbone floor
        P = update_link_probabilities(P, lam, eta, omega, dt, t, noise_std=0.02)
        P = apply_entanglement_correlation(P, correlated_pairs, rho0, mu, t)
        P = apply_backbone_floor(P, backbone_edges, p_min=backbone_p_min)

        # 2) Sample G_t
        G = sample_graph_from_probabilities(P, seed=rng.integers(0, 1_000_000))

        # 3) Check structural reachability (for analysis)
        try:
            is_reachable = nx.has_path(G, state, dst)
        except nx.NetworkXError:
            is_reachable = False

        # Reset episode every max_steps_per_episode
        if step % max_steps_per_episode == 0 and step > 0:
            agent.decay_epsilon()
            # new episode
            src = rng.integers(0, n_nodes)
            dst = rng.integers(0, n_nodes)
            while dst == src:
                dst = rng.integers(0, n_nodes)
            state = src
            episode_reward = 0.0

        # 4) RL interaction
        if state == dst:
            reward = reward_goal
            next_state = state
        else:
            neighbors = list(G.neighbors(state))
            if len(neighbors) == 0:
                reward = reward_fail
                next_state = state
                next_neighbors = []
            else:
                action = agent.select_action(state, neighbors)
                if action == dst:
                    reward = reward_goal
                else:
                    reward = reward_step
                next_state = action
                next_neighbors = list(G.neighbors(next_state))

            if len(neighbors) > 0:
                agent.update(state, action, reward, next_state, next_neighbors)

            state = next_state

        episode_reward += reward

        # 5) Stability
        S_t = stability_index(P)

        # 6) Logging and plotting
        redraw = (step % plot_interval == 0) or (step == T - 1)
        snapshot = (step % snapshot_interval == 0) or (step == T - 1)

        log_step(
            csv_writer,
            t,
            S_t,
            G,
            episode_reward=episode_reward,
            epsilon=agent.epsilon,
            is_reachable=is_reachable,
        )
        plotter.update(t, S_t, G, redraw=redraw, snapshot=snapshot)

    csv_file.close()
    plotter.finalize()
    return run_id


if __name__ == "__main__":
    correlated_pairs = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    ]

    run_id = run_qipsf_simulation(
        n_nodes=30,
        p0=0.2,
        lam=0.03,
        eta=0.1,
        omega=0.2,
        rho0=0.3,
        mu=0.01,
        T=500,
        dt=1.0,
        correlated_pairs=correlated_pairs,
        backbone_p_min=0.1,
        max_steps_per_episode=20,
        reward_goal=1.0,
        reward_step=-0.005,
        reward_fail=-0.2,
        seed=42,
        plot_interval=20,
        snapshot_interval=100,
    )
    print("Finished improved QIPSF run:", run_id)
    print("Check outputs/figures, outputs/snapshots, outputs/logs.")
