import time
from colorama import Fore, Style
from scapy.all import wrpcap

import arpmec_state as state
import arpmec_config as config
from arpmec_classes import global_init
from arpmec_clustering import perform_kmeans_clustering, update_clusters, recluster_if_needed
from arpmec_mobility import move_nodes
from arpmec_failure import predict_failures_arpmec, trigger_failures
from arpmec_routing import simulate_routing
from arpmec_metrics import log_metrics, export_metrics_csv
from arpmec_viz import assign_bs_to_ch, visualize_simulation
from arpmec_curves import courbes_comparatives, courbes_comparatives_tolerance, courbes_comparatives_moyenne, export_courbes_separees


def simulate_arpmec_ft():
    print("Lancement simulation ARPMEC+FT avec tolérance aux pannes...")

    # Initialisation globale des objets
    global_init()

    # Clustering initial
    clusters = perform_kmeans_clustering([n for n in state.nodes if n.alive], n_clusters=3)

    # Réinitialisation des historiques
    state.clusters_by_round.clear()
    state.simulation_history.clear()
    state.logs.clear()
    state.pcap_packets.clear()  # Vider la liste pour cette simulation

    for round_num in range(config.ROUNDS):
        start_time = time.time()

        move_nodes()
        predict_failures_arpmec(clusters, config.SEUIL_RSSI, config.SEUIL_PR_D)
        failed_this_round = trigger_failures(round_num, failure_plan={})
        clusters, ch_reelected = update_clusters(clusters)
        clusters = recluster_if_needed(clusters)
        routing_success = simulate_routing(clusters, assign_bs_to_ch(clusters, state.base_stations), round_num)
        state.clusters_by_round.append(clusters)
        round_time = time.time() - start_time
        log_metrics(round_num, clusters, ch_reelected, failed_this_round, round_time, routing_success)

    # Export PCAP après simulation
    if state.pcap_packets:
        wrpcap("simulated_arpmec.pcap", state.pcap_packets)
        print(f"{Fore.CYAN}[PCAP] Export terminé : 'simulated_arpmec.pcap'{Style.RESET_ALL}")

    # Export CSV final
    export_metrics_csv()


def print_logs():
    if not state.logs:
        print("[INFO] Aucun log disponible, lancez d'abord une simulation.")
        return
    print("\n".join(state.logs))


def menu():
    while True:
        print("\n--- MENU ARPMEC+FT ---")
        print("1 - Lancer la simulation")
        print("2 - Afficher les logs")
        print("3 - Exporter métriques CSV")
        print("4 - Visualiser la simulation")
        print("5 - Quitter")
        #print("6 - Afficher les courbes comparatives")
        #print("7 - Comparer avec version sans tolérance")
        #print("8 - Générer la comparaison avec barres d’erreurs")
        print("6 - Générer les courbes individuellement")

        choice = input("Choisissez une option (1-6) : ")

        if choice == '1':
            simulate_arpmec_ft()
        elif choice == '2':
            print_logs()
        elif choice == '3':
            export_metrics_csv()
        elif choice == '4':
            if not state.clusters_by_round:
                print("[INFO] Pas de simulation disponible, lancez d'abord la simulation.")
            else:
                visualize_simulation()
        elif choice == '5':
            print("Au revoir!")
            break
      #  elif choice == '6':
      #      courbes_comparatives()
      #  elif choice == '7':
       #     courbes_comparatives_tolerance()
        #elif choice == '8':
         #   courbes_comparatives_moyenne(n_runs=10)
        elif choice == '6':
            export_courbes_separees(n_runs=10)
        else:
            print("[ERREUR] Option invalide. Veuillez choisir un nombre entre 1 et 9.")


if __name__ == "__main__":
    menu()

