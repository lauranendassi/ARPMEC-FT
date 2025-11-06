from arpmec_config import *
from colorama import Fore, Style
from scapy.all import Ether, IP, UDP, Raw
import numpy as np
import random
import arpmec_state as state  # Import du module d'état partagé

def simulate_routing(clusters, bs_assign, round_num):
    total_packets = 0
    lost_packets = 0
    for cluster in clusters:
        ch = cluster.ch
        if not ch.alive:
            state.logs.append(f"{Fore.RED}[ROUTING] CH {ch.id} est mort — cluster ignoré (round {round_num}){Style.RESET_ALL}")
            continue
        bs = bs_assign.get(ch.id)
        if not bs:
            state.logs.append(f"{Fore.YELLOW}[ROUTING] Aucun BS assigné au CH {ch.id}, routage ignoré (round {round_num}){Style.RESET_ALL}")
            continue
        for member in cluster.members:
            if not member.alive:
                continue
            total_packets += 1
            dist_m_to_ch = np.linalg.norm(member.pos - ch.pos)
            prob_success_m_to_ch = max(0.9 - dist_m_to_ch / AREA_SIZE, 0.5)
            if random.random() > prob_success_m_to_ch:
                lost_packets += 1
                state.logs.append(f"{Fore.RED}[ROUTING] Échec Membre {member.id} → CH {ch.id} (d={dist_m_to_ch:.1f}) au round {round_num}{Style.RESET_ALL}")
                continue
            dist_ch_to_bs = np.linalg.norm(ch.pos - bs.pos)
            prob_success_ch_to_bs = max(0.95 - dist_ch_to_bs / AREA_SIZE, 0.6)
            if random.random() > prob_success_ch_to_bs:
                lost_packets += 1
                state.logs.append(f"{Fore.RED}[ROUTING] Échec CH {ch.id} → BS {bs.id} (d={dist_ch_to_bs:.1f}) au round {round_num}{Style.RESET_ALL}")
                continue
            state.logs.append(f"{Fore.GREEN}[ROUTING] Succès complet : Membre {member.id} → CH {ch.id} → BS {bs.id} (Round {round_num}){Style.RESET_ALL}")
            payload = f"Data from Node {member.id}"
            log_packet_to_pcap(src_id=member.id, dst_id=bs.id, payload=payload, round_num=round_num)
    if total_packets > 0:
        success_rate = (total_packets - lost_packets) / total_packets
        state.logs.append(f"{Fore.CYAN}[ROUTING] Taux de succès routage au round {round_num}: {success_rate*100:.1f}% ({total_packets - lost_packets}/{total_packets}){Style.RESET_ALL}")
    else:
        success_rate = 0.0
        state.logs.append(f"{Fore.YELLOW}[ROUTING] Aucun paquet transmis au round {round_num}. Succès = 0.0%{Style.RESET_ALL}")
    return success_rate

def log_packet_to_pcap(src_id, dst_id, payload, round_num):
    src_ip = f"10.0.0.{src_id + 1}"
    dst_ip = f"10.0.0.{dst_id + 1}"
    pkt = Ether() / IP(src=src_ip, dst=dst_ip) / UDP(sport=src_id + 1000, dport=dst_id + 1000) / Raw(load=f"{payload} (Round {round_num})")
    state.pcap_packets.append(pkt)

