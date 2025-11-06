import numpy as np
from arpmec_config import *
from arpmec_classes import Cluster
import arpmec_state as state
from colorama import Fore, Style
from sklearn.cluster import KMeans

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
        members = [n for i, n in enumerate(cluster_nodes) if i != ch_idx]
        clusters.append(Cluster(ch=ch, members=members))
    return clusters

def recluster_if_needed(clusters):
    valid_clusters = []
    recluster_needed = False
    for c in clusters:
        alive_members = [m for m in c.members if m.alive]
        if not c.ch.alive or len(alive_members) < MIN_CLUSTER_SIZE - 1:
            recluster_needed = True
            state.logs.append(f"{Fore.YELLOW}[RECLUSTER] Cluster CH {c.ch.id} dissous (CH mort ou membres insuffisants){Style.RESET_ALL}")
        else:
            valid_clusters.append(c)
    if recluster_needed:
        alive_nodes = [n for n in state.nodes if n.alive]
        nb_clusters = max(1, int(np.sqrt(len(alive_nodes))))
        state.logs.append(f"{Fore.MAGENTA}[RECLUSTER] Reclustering automatique avec {nb_clusters} clusters{Style.RESET_ALL}")
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
            if len(alive_members) >= MIN_CLUSTER_SIZE - 1:
                new_ch = max(alive_members, key=lambda n: n.energy)
                new_members = [m for m in alive_members if m.id != new_ch.id]
                updated_clusters.append(Cluster(new_ch, new_members))
                ch_reelected_count += 1
                state.logs.append(f"{Fore.GREEN}[Tolérance] CH {c.ch.id} remplacé par {new_ch.id}{Style.RESET_ALL}")
            else:
                state.logs.append(f"{Fore.RED}[Tolérance] Cluster dissous (CH {c.ch.id} mort et pas assez membres){Style.RESET_ALL}")
        else:
            valid_members = [m for m in c.members if m.alive]
            if len(valid_members) >= MIN_CLUSTER_SIZE - 1:
                updated_clusters.append(Cluster(c.ch, valid_members))
            else:
                state.logs.append(f"{Fore.RED}[Tolérance] Cluster dissous (trop peu de membres autour de CH {c.ch.id}){Style.RESET_ALL}")
    return updated_clusters, ch_reelected_count

