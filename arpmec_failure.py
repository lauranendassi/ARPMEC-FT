from arpmec_config import *
from arpmec_classes import *
from colorama import Fore, Style
import numpy as np
import random
import arpmec_state as state  # import de l'état global

def generate_failure_plan(seed=42):
    random.seed(seed)
    plan = {}
    for round_num in range(ROUNDS):
        plan[round_num] = []
        if round_num == 3:
            for c in state.clusters_by_round[0] if state.clusters_by_round else []:
                if c.ch.alive:
                    plan[3].append(c.ch.id)
                    break
        forced_random = random.sample(range(NUM_NODES), k=random.randint(0, 2))
        plan[round_num].extend(forced_random)
    return plan

def path_loss(d, R_c=R_c, alpha=alpha):
    return (d / R_c) ** (2 * alpha) if d != 0 else 0

def compute_rssi(P_t, d):
    if d == 0:
        return P_t
    Ld = 10 * np.log10(path_loss(d)) if path_loss(d) > 0 else 0
    X_sigma = np.random.normal(0, sigma)
    RSSI = P_t - Ld + X_sigma
    return RSSI

def compute_pr_d(d):
    if d == 0:
        return 0
    val = 1 - (R_c / d) ** (2 * alpha)
    return max(0, val)

def predict_failures_arpmec(clusters, seuil_rssi, seuil_pr_d):
    at_risk_nodes = []
    for c in clusters:
        ch = c.ch
        if not ch.alive:
            continue
        for node in c.members:
            if not node.alive:
                continue
            d = np.linalg.norm(node.pos - ch.pos)
            pr_d = compute_pr_d(d)
            rssi = compute_rssi(P_t, d)
            if rssi < seuil_rssi or pr_d < seuil_pr_d:
                at_risk_nodes.append((node, c))
                state.logs.append(f"{Fore.YELLOW}[Prédiction] Node {node.id} à risque (RSSI={rssi:.2f}, Pr(d)={pr_d:.3f}) dans cluster CH {ch.id}{Style.RESET_ALL}")
    return at_risk_nodes

def failover_recovery_arpmec(fallen_node, clusters, seuil_rssi, seuil_pr_d):
    for c in clusters:
        if c.ch.id == fallen_node.id:
            candidates = []
            for node in c.members:
                if not node.alive:
                    continue
                d = np.linalg.norm(node.pos - fallen_node.pos)
                pr_d = compute_pr_d(d)
                rssi = compute_rssi(P_t, d)
                if rssi > seuil_rssi and pr_d > seuil_pr_d:
                    candidates.append((node, rssi, pr_d))
            if not candidates:
                state.logs.append(f"{Fore.RED}[Failover] Aucune alternative CH trouvée pour cluster CH {fallen_node.id}{Style.RESET_ALL}")
                return None
            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            new_ch = candidates[0][0]
            state.logs.append(f"{Fore.GREEN}[Failover] Nouveau CH provisoire sélectionné: Node {new_ch.id} pour remplacer CH {fallen_node.id}{Style.RESET_ALL}")
            c.ch = new_ch
            chemin_recup = []
            for node in c.members:
                if node.id == new_ch.id or not node.alive:
                    continue
                d = np.linalg.norm(node.pos - new_ch.pos)
                pr_d = compute_pr_d(d)
                rssi = compute_rssi(P_t, d)
                if rssi > seuil_rssi and pr_d > seuil_pr_d:
                    chemin_recup.append(node)
            if not chemin_recup:
                state.logs.append(f"{Fore.RED}[Failover] Chemin de récupération vide, relancer recherche{Style.RESET_ALL}")
                return None
            return chemin_recup
    return None

def trigger_failures(round_num, failure_plan=None):
    failed_nodes = []
    forced_ids = failure_plan.get(round_num, []) if failure_plan else []
    for nid in forced_ids:
        if state.nodes[nid].alive:
            state.nodes[nid].alive = False
            failed_nodes.append(nid)
            state.logs.append(f"{Fore.RED}[PLAN] Node {nid} panne planifiée au round {round_num}{Style.RESET_ALL}")
    for n in state.nodes:
        if n.alive and n.id not in forced_ids:
            if n.energy <= 20.0:
                n.alive = False
                failed_nodes.append(n.id)
                state.logs.append(f"{Fore.RED}[AUTO] Node {n.id} panne énergie{Style.RESET_ALL}")
            else:
                prob_failure = 0.03 + (1 - n.energy / 100) * 0.05
                if random.random() < prob_failure:
                    n.alive = False
                    failed_nodes.append(n.id)
                    state.logs.append(f"{Fore.RED}[AUTO] Node {n.id} panne aléatoire (E={n.energy:.2f}){Style.RESET_ALL}")
    return failed_nodes

