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
MIN_CLUSTER_SIZE = 3
ROUNDS = 20
NODE_SPEED_MAX = 1.0
BS_COUNT = 5
MEC_COUNT = 3

# === Paramètres physiques et seuils tolérance ===
R_c = 30.0
alpha = 2.0
P_t = 100.0
sigma = 2.0
SEUIL_RSSI = 20.0
SEUIL_PR_D = 0.4

