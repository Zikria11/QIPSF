import numpy as np
import networkx as nx
from quantum_rules import update_link_probabilities, apply_entanglement_correlation

class QIPSFEnv:
    def __init__(self, num_nodes=30, lam=0.03, eta=0.1, rho0=0.3):
        self.num_nodes = num_nodes
        self.lam = lam  # Decay constant (lambda) [cite: 67]
        self.eta = eta  # Noise floor [cite: 70]
        self.rho0 = rho0 # Initial correlation strength
        self.omega = 0.5 # Interference frequency
        self.dt = 1.0    # Time step
        
        # Initialize the 30-node directed graph [cite: 59]
        self.graph = nx.gnm_random_graph(num_nodes, num_nodes * 2, directed=True, seed=42)
        self.start_node = 0
        self.target_node = num_nodes - 1
        
        # Physical parameters [cite: 65, 70]
        self.P = np.zeros((num_nodes, num_nodes))
        self.reset()

    def reset(self):
        """Resets the environment for a new episode[cite: 153]."""
        self.state = self.start_node
        self.time = 0
        # Initialize all existing edges with high probability P0 [cite: 70]
        self.P.fill(0.0)
        for u, v in self.graph.edges():
            self.P[u, v] = 1.0 
        return self.state

    def step(self, action):
        """
        Executes a transition based on the current vanishing topology[cite: 68].
        """
        self.time += self.dt
        
        # 1. Update physics via Euler-Maruyama and Oscillatory interference [cite: 65, 70]
        self.P = update_link_probabilities(
            self.P, self.lam, self.eta, self.omega, self.dt, self.time
        )
        
        # 2. Check if the link exists at this specific moment [cite: 253]
        link_persistence = self.P[self.state, action]
        
        if np.random.random() < link_persistence:
            # Successful transition
            self.state = action
            reached_target = (self.state == self.target_node)
            
            # Reward: Bonus for target, penalty for time 
            reward = 100.0 if reached_target else (link_persistence - 1.0)
            done = reached_target
        else:
            # Link Collapse: Agent stays at current node [cite: 253]
            reward = -5.0 # High penalty for link failure [cite: 253]
            done = False
            
        # Terminal condition for max steps to avoid infinite loops
        if self.time > 100:
            done = True
            
        return self.state, reward, done, {"stability": self.P[self.state, action]}

    def get_valid_actions(self, node):
        """Returns neighbors in the fixed graph[cite: 76]."""
        return list(self.graph.neighbors(node))
