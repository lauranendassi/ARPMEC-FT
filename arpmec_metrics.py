from arpmec_config import *
from colorama import Fore, Style
import csv
import arpmec_state as state

def log_metrics(round_num, clusters, ch_reelected_count, failed_nodes, round_time, routing_success):
    alive_nodes = [n for n in state.nodes if n.alive]
    avg_energy = sum(n.energy for n in alive_nodes) / len(alive_nodes) if alive_nodes else 0.0
    num_clusters = len(clusters)
    total_failures = len(failed_nodes)
    avg_energy_clusters = []
    for c in clusters:
        members_energy = [m.energy for m in c.members if m.alive]
        ch_energy = c.ch.energy if c.ch.alive else 0
        cluster_avg = (sum(members_energy) + ch_energy) / (len(members_energy) + 1) if members_energy else ch_energy
        avg_energy_clusters.append(cluster_avg)
    print(f"{Fore.CYAN}--- Round {round_num} ---{Style.RESET_ALL} Clusters: {num_clusters} | CH réélus: {ch_reelected_count} | Pannes ce round: {total_failures} | Temps: {round_time:.2f}s | Energie moyenne: {avg_energy:.2f}J | Routing succès: {routing_success*100:.1f}%")
    state.simulation_history.append({
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
    if not state.simulation_history:
        print("[WARN] Pas de données pour exporter CSV.")
        return
    # Colonnes incluant la liste avg_energy_clusters sous forme chaîne séparée par ;
    fieldnames = ["round", "clusters", "ch_reelected", "failures", "avg_energy", "avg_energy_clusters", "time", "routing_success"]
    with open(filename, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in state.simulation_history:
            avg_energy_clusters_str = ";".join(f"{e:.3f}" for e in entry.get("avg_energy_clusters", []))
            writer.writerow({
                "round": entry["round"],
                "clusters": entry["clusters"],
                "ch_reelected": entry["ch_reelected"],
                "failures": entry["failures"],
                "avg_energy": f"{entry['avg_energy']:.3f}",
                "avg_energy_clusters": avg_energy_clusters_str,
                "time": f"{entry['time']:.3f}",
                "routing_success": f"{entry['routing_success']:.5f}",
            })
    print(f"[EXPORT] Données métriques exportées vers {filename}")

def print_logs():
    if not state.logs:
        print("[INFO] Aucun log disponible, lancez d'abord une simulation.")
        return
    print("\n".join(state.logs))
