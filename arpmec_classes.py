import numpy as np
import random
from arpmec_config import *
import arpmec_state as state  # Import du module d'état partagé

class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.energy = 100.0
        self.alive = True
        self.pos = np.random.rand(2) * AREA_SIZE
        self.direction = np.random.uniform(0, 2 * np.pi)
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
    def __init__(self, ch, members):
        self.ch = ch
        self.members = members

class BaseStation:
    def __init__(self, bs_id):
        self.id = bs_id
        self.pos = np.random.rand(2) * AREA_SIZE

class MECServer:
    def __init__(self, mec_id):
        self.id = mec_id
        self.pos = np.random.rand(2) * AREA_SIZE

# Initialisation globale des listes dans le module d'état
def global_init():
    state.nodes = [Node(i) for i in range(NUM_NODES)]
    state.base_stations = [BaseStation(i) for i in range(BS_COUNT)]
    state.mec_servers = [MECServer(i) for i in range(MEC_COUNT)]

