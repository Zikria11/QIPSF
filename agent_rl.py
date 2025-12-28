# agent_rl.py
import numpy as np
import random

class QLearningAgent:
    def __init__(
        self,
        n_nodes,
        alpha=0.2,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.98,
        seed=42,
    ):
        self.n_nodes = n_nodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_nodes, n_nodes), dtype=float)

    def select_action(self, state, neighbors):
        if len(neighbors) == 0:
            return state

        if self.rng.random() < self.epsilon:
            return random.choice(neighbors)

        q_values = [(a, self.Q[state, a]) for a in neighbors]
        max_q = max(q_values, key=lambda x: x[1])[1]
        best_actions = [a for a, q in q_values if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_neighbors):
        if len(next_neighbors) == 0:
            max_next_q = 0.0
        else:
            max_next_q = np.max(self.Q[next_state, next_neighbors])

        td_target = reward + self.gamma * max_next_q
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
