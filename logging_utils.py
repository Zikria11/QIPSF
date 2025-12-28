# logging_utils.py
import csv
import os
from datetime import datetime
import json

def create_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dirs():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/snapshots", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

def log_params(run_id, params: dict):
    ensure_dirs()
    path = f"outputs/logs/params_{run_id}.json"
    with open(path, "w") as f:
        json.dump(params, f, indent=2)

def init_csv_logger(run_id):
    ensure_dirs()
    path = f"outputs/logs/metrics_{run_id}.csv"
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "t", "S_t", "num_edges", "avg_degree",
        "episode_reward", "epsilon", "is_reachable"
    ])
    return f, writer

def log_step(writer, t, S_t, G, episode_reward, epsilon, is_reachable):
    num_edges = G.number_of_edges()
    n = G.number_of_nodes()
    avg_degree = (2 * num_edges / n) if n > 0 else 0.0
    writer.writerow([t, S_t, num_edges, avg_degree, episode_reward, epsilon, int(is_reachable)])
