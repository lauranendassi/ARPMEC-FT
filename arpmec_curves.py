import arpmec_state as state
import arpmec_config as config
from arpmec_failure import generate_failure_plan, trigger_failures
from arpmec_failure import predict_failures_arpmec
from arpmec_clustering import perform_kmeans_clustering, update_clusters, recluster_if_needed
from arpmec_mobility import move_nodes
from arpmec_routing import simulate_routing
from arpmec_viz import assign_bs_to_ch
from colorama import Fore, Style
import matplotlib.pyplot as plt
import statistics
import numpy as np
import time

def courbes_comparatives():
    if not state.simulation_history:
        print("[WARN] Aucune simulation à afficher.")
        return

    rounds = [d["round"] for d in state.simulation_history]
    routing = [d["routing_success"] * 100 for d in state.simulation_history]
    pannes = [d["failures"] for d in state.simulation_history]
    chs = [d["ch_reelected"] for d in state.simulation_history]
    energy = [d["avg_energy"] for d in state.simulation_history]
    clusters = [d["clusters"] for d in state.simulation_history]

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
    clusters = perform_kmeans_clustering([n for n in state.nodes if n.alive], n_clusters=3)
    sim_hist = []

    for round_num in range(config.ROUNDS):
        move_nodes()
        start_time = time.time()

        if version_tolérance:
            predict_failures_arpmec(clusters, config.SEUIL_RSSI, config.SEUIL_PR_D)

        failed_this_round = trigger_failures(round_num, failure_plan=failure_plan)

        if version_tolérance:
            clusters, ch_reelected = update_clusters(clusters)
            clusters = recluster_if_needed(clusters)
        else:
            clusters = [c for c in clusters if c.ch.alive and all(m.alive for m in c.members)]
            ch_reelected = 0

        routing_success = simulate_routing(clusters, assign_bs_to_ch(clusters, state.base_stations), round_num)

        round_time = time.time() - start_time
        alive_nodes = [n for n in state.nodes if n.alive]
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
    for n in state.nodes:
        n.alive = True
        n.energy = 100.0
        n.pos = np.random.rand(2) * config.AREA_SIZE
    hist_tolerant = simulate_arpmec(version_tolérance=True, failure_plan=failure_plan)

    print("\n[INFO] Simulation ARPMEC (sans tolérance)...")
    for n in state.nodes:
        n.alive = True
        n.energy = 100.0
        n.pos = np.random.rand(2) * config.AREA_SIZE
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

    history_tol = [[] for _ in range(config.ROUNDS)]
    history_no_tol = [[] for _ in range(config.ROUNDS)]

    for run in range(n_runs):
        print(f"[RUN {run+1}/{n_runs}] Tolérance activée...")
        for n in state.nodes:
            n.alive = True
            n.energy = 100.0
            n.pos = np.random.rand(2) * config.AREA_SIZE
        hist1 = simulate_arpmec(version_tolérance=True, failure_plan=failure_plan)

        print(f"[RUN {run+1}/{n_runs}] Tolérance désactivée...")
        for n in state.nodes:
            n.alive = True
            n.energy = 100.0
            n.pos = np.random.rand(2) * config.AREA_SIZE
        hist2 = simulate_arpmec(version_tolérance=False, failure_plan=failure_plan)

        for r in range(config.ROUNDS):
            history_tol[r].append(hist1[r])
            history_no_tol[r].append(hist2[r])

    def moyenne_std(metric, history_r):
        moy = statistics.mean([entry[metric] for entry in history_r])
        std = statistics.stdev([entry[metric] for entry in history_r])
        return moy, std

    rounds = list(range(config.ROUNDS))
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
    history_tol = [[] for _ in range(config.ROUNDS)]
    history_no_tol = [[] for _ in range(config.ROUNDS)]

    for run in range(n_runs):
        print(f"[RUN {run+1}/{n_runs}] Tolérance activée...")
        for n in state.nodes:
            n.alive = True
            n.energy = 100.0
            n.pos = np.random.rand(2) * config.AREA_SIZE
        hist1 = simulate_arpmec(version_tolérance=True, failure_plan=failure_plan)

        print(f"[RUN {run+1}/{n_runs}] Tolérance désactivée...")
        for n in state.nodes:
            n.alive = True
            n.energy = 100.0
            n.pos = np.random.rand(2) * config.AREA_SIZE
        hist2 = simulate_arpmec(version_tolérance=False, failure_plan=failure_plan)

        for r in range(config.ROUNDS):
            history_tol[r].append(hist1[r])
            history_no_tol[r].append(hist2[r])

    def moyenne_std(metric, history_r):
        moy = statistics.mean([entry[metric] for entry in history_r])
        std = statistics.stdev([entry[metric] for entry in history_r])
        return moy, std

    # Construction des courbes
    rounds = list(range(config.ROUNDS))
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

