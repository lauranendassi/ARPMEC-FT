#!/usr/bin/env python3
"""
ARPMEC+FT : Simulation avancée de clustering dynamique avec tolérance aux pannes
Features incluses :
- Clustering intelligent via k-means avec reclustering dynamique
- Réélection automatique des CH en cas de panne
- Dissolution et reclustering forcé si clusters trop petits
- Mobilité dirigée avec rebond dans la zone
- Modèle panne énergétique + panne ciblée
- Visualisation interactive avec slider et boutons Pause/Play
- Visualisation améliorée : clusters, CH, BS, MEC, liens, légende
- Export GIF + MP4
- Logs colorés en console via colorama
- Export métriques détaillées CSV
- Interface utilisateur avec sauvegarde/restauration
- Routage adaptatif simulé avec métriques de succès/échec

Tolérance aux pannes intégrée selon :
- Algorithme 1 : Prédiction des pannes (Pr(d), RSSI)
- Algorithme 2 : Récupération en cas de panne (sélection CH provisoire, chemins de récupération)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import matplotlib.widgets as mwidgets
import numpy as np
import random
import time
import csv
import statistics

from scapy.all import Ether, IP, UDP, Raw, wrpcap

from sklearn.cluster import KMeans
from colorama import init as colorama_init, Fore, Style

colorama_init(autoreset=True)

# === Paramètres globaux ===
NUM_NODES = 50
AREA_SIZE = 150
MIN_CLUSTER_SIZE = 3  # CH + au moins 2 membres
ROUNDS = 20
NODE_SPEED_MAX = 1.0
BS_COUNT = 5
MEC_COUNT = 3

# === Paramètres physiques et seuils tolérance ===
R_c = 30.0          # Portée de transmission
alpha = 2.0         # Coef d’atténuation
P_t = 100.0         # Puissance émission (dBm)
sigma = 2.0         # Ecart-type bruit X_sigma
SEUIL_RSSI = 20.0   # Seuil RSSI pour risque panne
SEUIL_PR_D = 0.4    # Seuil Pr(d) pour risque panne

# === Classes ===

class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.energy = 100.0
        self.alive = True
        self.pos = np.random.rand(2) * AREA_SIZE
        self.direction = np.random.uniform(0, 2*np.pi)
        self.speed = random.uniform(0.1, NODE_SPEED_MAX)

    def move(self):
        if not self.alive:
            return
        dx = np.cos(self.direction) * self.speed
        dy = np.sin(self.direction) * self.speed
        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy
        if new_x < 0 or new_x > AREA_SIZE:
            self.direction = np.pi - self.direction
            new_x = max(0, min(AREA_SIZE, new_x))
        if new_y < 0 or new_y > AREA_SIZE:
            self.direction = -self.direction
            new_y = max(0, min(AREA_SIZE, new_y))
        self.pos = np.array([new_x, new_y])
        energy_consumed = 0.05 * self.speed
        self.energy = max(0.0, self.energy - energy_consumed)
        if self.energy == 0:
            self.alive = False

class Cluster:
    def __init__(self, ch: Node, members: list):
        self.ch = ch
        self.members = members  # List[Node]

class BaseStation:
    def __init__(self, bs_id):
        self.id = bs_id
        self.pos = np.random.rand(2) * AREA_SIZE

class MECServer:
    def __init__(self, mec_id):
        self.id = mec_id
        self.pos = np.random.rand(2) * AREA_SIZE

# === Initialisation globale ===
nodes = [Node(i) for i in range(NUM_NODES)]
base_stations = [BaseStation(i) for i in range(BS_COUNT)]
mec_servers = [MECServer(i) for i in range(MEC_COUNT)]

clusters_by_round = []
logs = []
pcap_packets = []


def generate_failure_plan(seed=42):
    random.seed(seed)
    plan = {}
    for round_num in range(ROUNDS):
        plan[round_num] = []
        if round_num == 3:
            # Forçage CH pour tolérance testée au même moment
            for c in clusters_by_round[0] if clusters_by_round else []:
                if c.ch.alive:
                    plan[3].append(c.ch.id)
                    break
        forced_random = random.sample(range(NUM_NODES), k=random.randint(0, 2))
        plan[round_num].extend(forced_random)
    return plan

simulation_history = []

# === Fonctions clustering ===

def perform_kmeans_clustering(nodes_alive, n_clusters):
    positions = np.array([n.pos for n in nodes_alive])
    if len(nodes_alive) < n_clusters:
        clusters = [Cluster(ch=n, members=[]) for n in nodes_alive]
        return clusters
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(positions)
    clusters = []
    for cluster_id in range(n_clusters):
        members_idx = np.where(labels == cluster_id)[0]
        cluster_nodes = [nodes_alive[i] for i in members_idx]
        center = kmeans.cluster_centers_[cluster_id]
        distances = [np.linalg.norm(n.pos - center) for n in cluster_nodes]
        ch_idx = np.argmin(distances)
        ch = cluster_nodes[ch_idx]
        members = [n for i,n in enumerate(cluster_nodes) if i != ch_idx]
        clusters.append(Cluster(ch=ch, members=members))
    return clusters

def recluster_if_needed(clusters):
    valid_clusters = []
    recluster_needed = False
    for c in clusters:
        alive_members = [m for m in c.members if m.alive]
        if not c.ch.alive or len(alive_members) < MIN_CLUSTER_SIZE -1:
            recluster_needed = True
            logs.append(f"{Fore.YELLOW}[RECLUSTER] Cluster CH {c.ch.id} dissous (CH mort ou membres insuffisants){Style.RESET_ALL}")
        else:
            valid_clusters.append(c)
    if recluster_needed:
        alive_nodes = [n for n in nodes if n.alive]
        nb_clusters = max(1, int(np.sqrt(len(alive_nodes))))
        logs.append(f"{Fore.MAGENTA}[RECLUSTER] Reclustering automatique avec {nb_clusters} clusters{Style.RESET_ALL}")
        new_clusters = perform_kmeans_clustering(alive_nodes, nb_clusters)
        return new_clusters
    else:
        return clusters

def update_clusters(clusters):
    updated_clusters = []
    ch_reelected_count = 0
    for c in clusters:
        if not c.ch.alive:
            alive_members = [m for m in c.members if m.alive]
            if len(alive_members) >= MIN_CLUSTER_SIZE -1:
                new_ch = max(alive_members, key=lambda n: n.energy)
                new_members = [m for m in alive_members if m.id != new_ch.id]
                updated_clusters.append(Cluster(new_ch, new_members))
                ch_reelected_count +=1
                logs.append(f"{Fore.GREEN}[Tolérance] CH {c.ch.id} remplacé par {new_ch.id}{Style.RESET_ALL}")
            else:
                logs.append(f"{Fore.RED}[Tolérance] Cluster dissous (CH {c.ch.id} mort et pas assez membres){Style.RESET_ALL}")
        else:
            valid_members = [m for m in c.members if m.alive]
            if len(valid_members) >= MIN_CLUSTER_SIZE -1:
                updated_clusters.append(Cluster(c.ch, valid_members))
            else:
                logs.append(f"{Fore.RED}[Tolérance] Cluster dissous (trop peu de membres autour de CH {c.ch.id}){Style.RESET_ALL}")
    return updated_clusters, ch_reelected_count

# === Mobilité ===

def move_nodes():
    for n in nodes:
        n.move()

# === Algorithme 1 - Prédiction des pannes ===

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
                logs.append(f"{Fore.YELLOW}[Prédiction] Node {node.id} à risque (RSSI={rssi:.2f}, Pr(d)={pr_d:.3f}) dans cluster CH {ch.id}{Style.RESET_ALL}")
    return at_risk_nodes

# === Algorithme 2 - Récupération en cas de panne ===

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
                logs.append(f"{Fore.RED}[Failover] Aucune alternative CH trouvée pour cluster CH {fallen_node.id}{Style.RESET_ALL}")
                return None
            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            new_ch = candidates[0][0]
            logs.append(f"{Fore.GREEN}[Failover] Nouveau CH provisoire sélectionné: Node {new_ch.id} pour remplacer CH {fallen_node.id}{Style.RESET_ALL}")
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
                logs.append(f"{Fore.RED}[Failover] Chemin de récupération vide, relancer recherche{Style.RESET_ALL}")
                return None
            return chemin_recup
    return None

# === Panne énergétique + aléatoire ===

def trigger_failures(round_num, failure_plan=None):
    failed_nodes = []
    forced_ids = failure_plan.get(round_num, []) if failure_plan else []

    for nid in forced_ids:
        if nodes[nid].alive:
            nodes[nid].alive = False
            failed_nodes.append(nid)
            logs.append(f"{Fore.RED}[PLAN] Node {nid} panne planifiée au round {round_num}{Style.RESET_ALL}")

    for n in nodes:
        if n.alive and n.id not in forced_ids:
            if n.energy <= 20.0:
                n.alive = False
                failed_nodes.append(n.id)
                logs.append(f"{Fore.RED}[AUTO] Node {n.id} panne énergie{Style.RESET_ALL}")
            else:
                prob_failure = 0.03 + (1 - n.energy/100)*0.05
                if random.random() < prob_failure:
                    n.alive = False
                    failed_nodes.append(n.id)
                    logs.append(f"{Fore.RED}[AUTO] Node {n.id} panne aléatoire (E={n.energy:.2f}){Style.RESET_ALL}")
    return failed_nodes


# === ROUTING ADAPTATIF ===
def simulate_routing(clusters, bs_assign, round_num):
    total_packets = 0
    lost_packets = 0

    for cluster in clusters:
        ch = cluster.ch
        if not ch.alive:
            logs.append(f"{Fore.RED}[ROUTING] CH {ch.id} est mort — cluster ignoré (round {round_num}){Style.RESET_ALL}")
            continue

        bs = bs_assign.get(ch.id)
        if not bs:
            logs.append(f"{Fore.YELLOW}[ROUTING] Aucun BS assigné au CH {ch.id}, routage ignoré (round {round_num}){Style.RESET_ALL}")
            continue

        for member in cluster.members:
            if not member.alive:
                continue

            total_packets += 1

            # Étape 1 : de membre vers CH
            dist_m_to_ch = np.linalg.norm(member.pos - ch.pos)
            prob_success_m_to_ch = max(0.9 - dist_m_to_ch / AREA_SIZE, 0.5)

            if random.random() > prob_success_m_to_ch:
                lost_packets += 1
                logs.append(f"{Fore.RED}[ROUTING] Échec Membre {member.id} → CH {ch.id} (d={dist_m_to_ch:.1f}) au round {round_num}{Style.RESET_ALL}")
                continue

            # Étape 2 : de CH vers BS
            dist_ch_to_bs = np.linalg.norm(ch.pos - bs.pos)
            prob_success_ch_to_bs = max(0.95 - dist_ch_to_bs / AREA_SIZE, 0.6)

            if random.random() > prob_success_ch_to_bs:
                lost_packets += 1
                logs.append(f"{Fore.RED}[ROUTING] Échec CH {ch.id} → BS {bs.id} (d={dist_ch_to_bs:.1f}) au round {round_num}{Style.RESET_ALL}")
                continue

            # Succès complet du paquet
            logs.append(f"{Fore.GREEN}[ROUTING] Succès complet : Membre {member.id} → CH {ch.id} → BS {bs.id} (Round {round_num}){Style.RESET_ALL}")
            payload = f"Data from Node {member.id}"
            log_packet_to_pcap(src_id=member.id, dst_id=bs.id, payload=payload, round_num=round_num)

    if total_packets > 0:
        success_rate = (total_packets - lost_packets) / total_packets
        logs.append(f"{Fore.CYAN}[ROUTING] Taux de succès routage au round {round_num}: {success_rate*100:.1f}% "
                    f"({total_packets - lost_packets}/{total_packets}){Style.RESET_ALL}")
    else:
        success_rate = 0.0
        logs.append(f"{Fore.YELLOW}[ROUTING] Aucun paquet transmis au round {round_num}. Succès = 0.0%{Style.RESET_ALL}")

    return success_rate



# === Métriques et logs ===

def log_packet_to_pcap(src_id, dst_id, payload, round_num):
    src_ip = f"10.0.0.{src_id + 1}"
    dst_ip = f"10.0.0.{dst_id + 1}"
    pkt = Ether() / IP(src=src_ip, dst=dst_ip) / UDP(sport=src_id + 1000, dport=dst_id + 1000) / Raw(load=f"{payload} (Round {round_num})")
    pcap_packets.append(pkt)


def log_metrics(round_num, clusters, ch_reelected_count, failed_nodes, round_time, routing_success):
    alive_nodes = [n for n in nodes if n.alive]
    avg_energy = sum(n.energy for n in alive_nodes) / len(alive_nodes) if alive_nodes else 0.0
    num_clusters = len(clusters)
    total_failures = len(failed_nodes)

    avg_energy_clusters = []
    for c in clusters:
        members_energy = [m.energy for m in c.members if m.alive]
        ch_energy = c.ch.energy if c.ch.alive else 0
        cluster_avg = (sum(members_energy) + ch_energy) / (len(members_energy)+1) if members_energy else ch_energy
        avg_energy_clusters.append(cluster_avg)

    print(f"{Fore.CYAN}--- Round {round_num} ---{Style.RESET_ALL} Clusters: {num_clusters} | "
          f"CH réélus: {ch_reelected_count} | Pannes ce round: {total_failures} | Temps: {round_time:.2f}s | "
          f"Energie moyenne: {avg_energy:.2f}J | Routing succès: {routing_success*100:.1f}%")

    simulation_history.append({
        "round": round_num,
        "clusters": num_clusters,
        "ch_reelected": ch_reelected_count,
        "failures": total_failures,
        "avg_energy": avg_energy,
        "avg_energy_clusters": avg_energy_clusters,
        "time": round_time,
        "routing_success": routing_success
    })

def export_metrics_csv(filename="metrics_detailed.csv"):
    if not simulation_history:
        print("[WARN] Pas de données pour exporter CSV.")
        return
    with open(filename, "w", newline='') as csvfile:
        fieldnames = ["round", "clusters", "ch_reelected", "failures", "avg_energy", "time", "routing_success"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in simulation_history:
            writer.writerow({
                "round": entry["round"],
                "clusters": entry["clusters"],
                "ch_reelected": entry["ch_reelected"],
                "failures": entry["failures"],
                "avg_energy": f"{entry['avg_energy']:.2f}",
                "time": f"{entry['time']:.3f}",
                "routing_success": f"{entry['routing_success']:.3f}",
            })
    print(f"[EXPORT] Données métriques exportées vers {filename}")

# === Visualisation interactive ===

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
            ax.text(m.pos[0]+0.7, m.pos[1]+0.7, f"{m.id}", fontsize=8, color=color)

        ch = cluster.ch
        ax.plot(ch.pos[0], ch.pos[1], 'X', color=color, markersize=12, markeredgecolor='k', markeredgewidth=1.5)
        ax.text(ch.pos[0]+0.7, ch.pos[1]+0.7, f"CH{ch.id}", fontsize=9, fontweight='bold', color=color)

        ax.text(centroid[0], centroid[1] + radius + 2, f"Cluster {idx}", fontsize=10, fontweight='bold', color=color)

        # BS et MEC liés
        bs = bs_assign.get(ch.id)
        if bs:
            ax.plot(bs.pos[0], bs.pos[1], 's', color='red', markersize=10)
            mec = mec_assign.get(bs.id)
            if mec:
                ax.plot(mec.pos[0], mec.pos[1], 'D', color='purple', markersize=10)
                ax.plot([bs.pos[0], mec.pos[0]], [bs.pos[1], mec.pos[1]], 'purple', linestyle=':', linewidth=1.2)
            ax.plot([ch.pos[0], bs.pos[0]], [ch.pos[1], bs.pos[1]], 'r-', linewidth=1.2)

    # Légende
    legend_items = [
        mpatches.Patch(color='blue', label='Members'),
        mpatches.Patch(color='orange', label='cluster Head (CH)'),
        mpatches.Patch(color='red', label='Base Stations (BS)'),
        mpatches.Patch(color='purple', label='MEC Serveurs'),
        mpatches.Patch(facecolor='none', label='Bounded Clusters', linestyle='--', edgecolor='gray')

    ]
    ax.legend(handles=legend_items, loc='upper right')

def visualize_simulation():
    paused = [False]  # Liste mutable pour éviter 'nonlocal'

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.25)

    n_rounds = len(clusters_by_round)
    bs_assign_all = [assign_bs_to_ch(c, base_stations) for c in clusters_by_round]
    mec_assign_all = [assign_mec_to_bs(base_stations, mec_servers) for _ in clusters_by_round]

    def update(frame):
        round_num = int(frame)
        ax.set_title(f"ARPMEC+FT Simulation - Round {round_num}")
        if round_num < n_rounds:
            clusters = clusters_by_round[round_num]
            bs_assign = bs_assign_all[round_num]
            mec_assign = mec_assign_all[round_num]
            draw_clusters(ax, clusters, bs_assign, mec_assign)
        fig.canvas.draw_idle()

    slider_ax = plt.axes([0.25, 0.1, 0.50, 0.03])
    round_slider = mwidgets.Slider(slider_ax, 'Round', 0, n_rounds-1, valinit=0, valstep=1)
    round_slider.on_changed(lambda val: update(int(val)))

    def on_pause(event):
        paused[0] = not paused[0]
        btn_pause.label.set_text("Play" if paused[0] else "Pause")

    pause_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    btn_pause = mwidgets.Button(pause_ax, 'Pause')
    btn_pause.on_clicked(on_pause)

    ani = animation.FuncAnimation(fig, update, frames=range(n_rounds), interval=1500, repeat=False)
    plt.show()


# === Simulation principale ===

def simulate_arpmec_ft():
    print("Lancement simulation ARPMEC+FT avec tolérance aux pannes...")

    # Clustering initial
    clusters = perform_kmeans_clustering([n for n in nodes if n.alive], n_clusters=3)

    # Réinitialisation des historiques
    clusters_by_round.clear()
    simulation_history.clear()
    logs.clear()
    pcap_packets.clear()  # Vider la liste pour cette simulation

    for round_num in range(ROUNDS):
        start_time = time.time()

        move_nodes()
        predict_failures_arpmec(clusters, SEUIL_RSSI, SEUIL_PR_D)
        failed_this_round = trigger_failures(round_num, failure_plan={})
        clusters, ch_reelected = update_clusters(clusters)
        clusters = recluster_if_needed(clusters)
        routing_success = simulate_routing(clusters, assign_bs_to_ch(clusters, base_stations), round_num)
        clusters_by_round.append(clusters)
        round_time = time.time() - start_time
        log_metrics(round_num, clusters, ch_reelected, failed_this_round, round_time, routing_success)

    # Export PCAP après simulation
    if pcap_packets:
        wrpcap("simulated_arpmec.pcap", pcap_packets)
        print(f"{Fore.CYAN}[PCAP] Export terminé : 'simulated_arpmec.pcap'{Style.RESET_ALL}")

    # Export CSV final
    export_metrics_csv()



def print_logs():
    if not logs:
        print("[INFO] Aucun log disponible, lancez d'abord une simulation.")
        return
    print("\n".join(logs))

def menu():
    while True:
        print("\n--- MENU ARPMEC+FT ---")
        print("1 - Lancer la simulation")
        print("2 - Afficher les logs")
        print("3 - Exporter métriques CSV")
        print("4 - Visualiser la simulation")
        print("5 - Quitter")
        print("6 - Afficher les courbes comparatives")
        print("7 - Comparer avec version sans tolérance")
        print("8 - Générer la comparaison avec barres d’erreurs")
        print("9 - Générer les courbes individuellement")

        choice = input("Choisissez une option (1-9) : ")

        if choice == '1':
            simulate_arpmec_ft()
        elif choice == '2':
            print_logs()
        elif choice == '3':
            export_metrics_csv()
        elif choice == '4':
            if not clusters_by_round:
                print("[INFO] Pas de simulation disponible, lancez d'abord la simulation.")
            else:
                visualize_simulation()
        elif choice == '5':
            print("Au revoir!")
            break
        elif choice == '6':
            courbes_comparatives()
        elif choice == '7':
            courbes_comparatives_tolerance()
        elif choice == '8':
            courbes_comparatives_moyenne(n_runs=10)
        elif choice == '9':
            export_courbes_separees(n_runs=10)
        else:
            print("[ERREUR] Option invalide. Veuillez choisir un nombre entre 1 et 9.")


            
# === COURBES COMPARATIVES ===

def courbes_comparatives():
    if not simulation_history:
        print("[WARN] Aucune simulation à afficher.")
        return

    rounds = [d["round"] for d in simulation_history]
    routing = [d["routing_success"]*100 for d in simulation_history]
    pannes = [d["failures"] for d in simulation_history]
    chs = [d["ch_reelected"] for d in simulation_history]
    energy = [d["avg_energy"] for d in simulation_history]
    clusters = [d["clusters"] for d in simulation_history]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(rounds, routing, 'g-o')
    plt.title("Succès de routage (%)")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(rounds, pannes, 'r-s')
    plt.title("Nombre de pannes")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(rounds, chs, 'm-d')
    plt.title("CH réélus")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(rounds, energy, 'b-^', label='Énergie')
    plt.plot(rounds, clusters, 'k--', label='Clusters')
    plt.title("Énergie moyenne & Clusters actifs")
    plt.legend()
    plt.grid(True)

    plt.suptitle("Courbes comparatives ARPMEC+FT", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("courbes_comparatives.png", dpi=300)
    print(f"{Fore.GREEN}[EXPORT] Courbes comparatives exportées vers 'courbes_comparatives.png'{Style.RESET_ALL}")
    plt.show()
    
def simulate_arpmec(version_tolérance=True, failure_plan=None):
    clusters = perform_kmeans_clustering([n for n in nodes if n.alive], n_clusters=3)
    sim_hist = []

    for round_num in range(ROUNDS):
        move_nodes()
        start_time = time.time()

        if version_tolérance:
            predict_failures_arpmec(clusters, SEUIL_RSSI, SEUIL_PR_D)

        failed_this_round = trigger_failures(round_num, failure_plan=failure_plan)

        if version_tolérance:
            clusters, ch_reelected = update_clusters(clusters)
            clusters = recluster_if_needed(clusters)
        else:
            clusters = [c for c in clusters if c.ch.alive and all(m.alive for m in c.members)]
            ch_reelected = 0

        routing_success = simulate_routing(clusters, assign_bs_to_ch(clusters, base_stations), round_num)


        round_time = time.time() - start_time
        alive_nodes = [n for n in nodes if n.alive]
        avg_energy = sum(n.energy for n in alive_nodes) / len(alive_nodes) if alive_nodes else 0.0

        sim_hist.append({
            "round": round_num,
            "clusters": len(clusters),
            "ch_reelected": ch_reelected,
            "failures": len(failed_this_round),
            "avg_energy": avg_energy,
            "time": round_time,
            "routing_success": routing_success
        })

    return sim_hist

def courbes_comparatives_tolerance():
    failure_plan = generate_failure_plan(seed=123)

    print("\n[INFO] Simulation ARPMEC+FT (avec tolérance)...")
    for n in nodes:
        n.alive = True
        n.energy = 100.0
        n.pos = np.random.rand(2) * AREA_SIZE
    hist_tolerant = simulate_arpmec(version_tolérance=True, failure_plan=failure_plan)

    print("\n[INFO] Simulation ARPMEC (sans tolérance)...")
    for n in nodes:
        n.alive = True
        n.energy = 100.0
        n.pos = np.random.rand(2) * AREA_SIZE
    hist_notolerant = simulate_arpmec(version_tolérance=False, failure_plan=failure_plan)

    rounds = [d["round"] for d in hist_tolerant]

    def extract(metric, history):
        return [d[metric] for d in history]

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(rounds, extract("routing_success", hist_tolerant), 'g-o', label="Avec tolérance")
    plt.plot(rounds, extract("routing_success", hist_notolerant), 'r--s', label="Sans tolérance")
    plt.title("Succès du routage")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(rounds, extract("failures", hist_tolerant), 'g-o')
    plt.plot(rounds, extract("failures", hist_notolerant), 'r--s')
    plt.title("Nombre de pannes")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(rounds, extract("avg_energy", hist_tolerant), 'g-o')
    plt.plot(rounds, extract("avg_energy", hist_notolerant), 'r--s')
    plt.title("Énergie moyenne")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(rounds, extract("clusters", hist_tolerant), 'g-o')
    plt.plot(rounds, extract("clusters", hist_notolerant), 'r--s')
    plt.title("Clusters actifs")
    plt.grid(True)

    plt.suptitle("Comparaison +FT vs ARPMEC sans tolérance (pannes synchronisées)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("comparaison_tolerance.png", dpi=300)
    print(f"{Fore.GREEN}[EXPORT] Graphique exporté sous 'comparaison_tolerance.png'{Style.RESET_ALL}")
    plt.show()

def export_courbes_separees(n_runs=10):
    print(f"\n[INFO] Moyennage sur {n_runs} simulations...")

    failure_plan = generate_failure_plan(seed=123)

    history_tol = [[] for _ in range(ROUNDS)]
    history_no_tol = [[] for _ in range(ROUNDS)]

    for run in range(n_runs):
        print(f"[RUN {run+1}/{n_runs}] Tolérance activée...")
        for n in nodes:
            n.alive = True
            n.energy = 100.0
            n.pos = np.random.rand(2) * AREA_SIZE
        hist1 = simulate_arpmec(version_tolérance=True, failure_plan=failure_plan)

        print(f"[RUN {run+1}/{n_runs}] Tolérance désactivée...")
        for n in nodes:
            n.alive = True
            n.energy = 100.0
            n.pos = np.random.rand(2) * AREA_SIZE
        hist2 = simulate_arpmec(version_tolérance=False, failure_plan=failure_plan)

        for r in range(ROUNDS):
            history_tol[r].append(hist1[r])
            history_no_tol[r].append(hist2[r])

    def moyenne_std(metric, history_r):
        moy = statistics.mean([entry[metric] for entry in history_r])
        std = statistics.stdev([entry[metric] for entry in history_r])
        return moy, std

    rounds = list(range(ROUNDS))
    def extract_curve(history):
        return {
            "routing": [moyenne_std("routing_success", history[r]) for r in rounds],
            "failures": [moyenne_std("failures", history[r]) for r in rounds],
            "energy": [moyenne_std("avg_energy", history[r]) for r in rounds],
            "clusters": [moyenne_std("clusters", history[r]) for r in rounds],
        }

    tol_data = extract_curve(history_tol)
    no_tol_data = extract_curve(history_no_tol)

    def split_moy_std(data_list):
        moy = [v[0] for v in data_list]
        std = [v[1] for v in data_list]
        return moy, std

    metrics_info = [
        ("routing", "Routing Success Rate (%)", "routing_success_avg.png", 'green'),
        ("failures", "Number of Failures", "failures_avg.png", 'red'),
        ("energy", "Average Energy", "avg_energy_avg.png", 'blue'),
        ("clusters", "Active Clusters", "clusters_avg.png", 'black')
    ]

    for key, title, filename, color in metrics_info:
        plt.figure(figsize=(8,6))
        moy1, std1 = split_moy_std(tol_data[key])
        moy2, std2 = split_moy_std(no_tol_data[key])
        plt.errorbar(rounds, moy1, yerr=std1, fmt='-o', color='green', label='With Fault Tolerance')
        plt.errorbar(rounds, moy2, yerr=std2, fmt='--s', color='red', label='Without Fault Tolerance')
        plt.title(title, fontsize=14)
        plt.xlabel("Rounds")
        plt.ylabel(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"{Fore.GREEN}[EXPORT] '{filename}' exportée.{Style.RESET_ALL}")
        plt.close()

    print(f"{Fore.CYAN}[INFO] Courbes moyennées avec barres d'erreur générées pour {n_runs} simulations.{Style.RESET_ALL}")



def courbes_comparatives_moyenne(n_runs=10):
    print(f"\n[INFO] Moyennage sur {n_runs} simulations...")

    # Génération du plan de pannes fixe
    failure_plan = generate_failure_plan(seed=123)

    # Initialisation
    history_tol = [[] for _ in range(ROUNDS)]
    history_no_tol = [[] for _ in range(ROUNDS)]
    

    for run in range(n_runs):
        print(f"[RUN {run+1}/{n_runs}] Tolérance activée...")
        for n in nodes:
            n.alive = True
            n.energy = 100.0
            n.pos = np.random.rand(2) * AREA_SIZE
        hist1 = simulate_arpmec(version_tolérance=True, failure_plan=failure_plan)

        print(f"[RUN {run+1}/{n_runs}] Tolérance désactivée...")
        for n in nodes:
            n.alive = True
            n.energy = 100.0
            n.pos = np.random.rand(2) * AREA_SIZE
        hist2 = simulate_arpmec(version_tolérance=False, failure_plan=failure_plan)

        for r in range(ROUNDS):
            history_tol[r].append(hist1[r])
            history_no_tol[r].append(hist2[r])

    def moyenne_std(metric, history_r):
        moy = statistics.mean([entry[metric] for entry in history_r])
        std = statistics.stdev([entry[metric] for entry in history_r])
        return moy, std

    # Construction des courbes
    rounds = list(range(ROUNDS))
    def extract_curve(history):
        return {
            "routing": [moyenne_std("routing_success", history[r]) for r in rounds],
            "failures": [moyenne_std("failures", history[r]) for r in rounds],
            "energy": [moyenne_std("avg_energy", history[r]) for r in rounds],
            "clusters": [moyenne_std("clusters", history[r]) for r in rounds],
        }

    tol_data = extract_curve(history_tol)
    no_tol_data = extract_curve(history_no_tol)

    def split_moy_std(data_list):
        moy = [v[0] for v in data_list]
        std = [v[1] for v in data_list]
        return moy, std

    # === PLOT FINAL ===
    plt.figure(figsize=(14, 10))

    for i, (key, title, color) in enumerate([
        ("routing", "Routing Success Rate", 'green'),
        ("failures", "Number of Failures", 'red'),
        ("energy", "Average Energy", 'blue'),
        ("clusters", "Active Clusters", 'black')
    ]):
        plt.subplot(2, 2, i+1)
        moy1, std1 = split_moy_std(tol_data[key])
        moy2, std2 = split_moy_std(no_tol_data[key])
        plt.errorbar(rounds, moy1, yerr=std1, fmt='-o', color='green', label='With Fault Tolerance')
        plt.errorbar(rounds, moy2, yerr=std2, fmt='--s', color='red', label='Without Fault Tolerance')
        plt.title(title)
        plt.grid(True)
        plt.legend()

    plt.suptitle(f"Comparison of ARPMEC+FT vs ARPMEC (average over {n_runs} runs)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("comparaison_tolerance_avg.png", dpi=300)
    print(f"{Fore.GREEN}[EXPORT] Moyenne avec barres d’erreurs exportée sous 'comparaison_tolerance_avg.png'{Style.RESET_ALL}")
    plt.show()





if __name__ == "__main__":
    menu()

