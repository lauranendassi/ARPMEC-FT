# Projet ARPMEC+FT - Simulation de clustering dynamique avec tolérance aux pannes

Ce projet contient une simulation avancée ARPMEC+FT, réalisée en Python et intégrée à NS3, avec gestion de la mobilité, tolérance aux pannes, routage adaptatif et visualisation.

## Contenu du dépôt

- Code Python modulaire pour simulation, clustering, prédiction des pannes, routage, visualisation, métriques
- Scripts d’analyse et export CSV
- Fichiers de configuration
- Fichier `requirement.txt` listant toutes les dépendances Python

## Prérequis

- Système Linux (Ubuntu recommandé)
- Python 3.8+
- Environnement virtuel Python (`venv`)
- NS3 (Network Simulator 3) installé et configuré
- Wireshark pour analyse des fichiers PCAP générés
- Bibliothèques Python listées dans `requirement.txt`

## Installation et configuration

1. Cloner le dépôt :
```bash
https://github.com/lauranendassi/ARPMEC-FT.git
cd ARPMEC-FT
```

2. Créer un environnement virtuel Python et l’activer :
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Installer les dépendances Python :
```bash
pip install -r requirement.txt
```

4. Installer Wireshark (nécessaire pour analyser les fichiers PCAP générés par la simulation) :
```bash
sudo apt update
sudo apt install wireshark
```

## Lancement de la simulation

1. Après activation de l’environnement virtuel, lancer la simulation principale :
```bash
python arpmec_main.py
```

2. Utiliser le menu interactif pour exécuter les différentes options (lancer simulation, visualiser, exporter métriques, afficher courbes...)

## Conseils

- Assurez-vous que NS3 est bien installé sur ta machine (suivre la documentation officielle de NS3 : https://www.nsnam.org/docs/).
- Wireshark te permettra d’ouvrir les fichiers `.pcap` générés par la simulation, pour analyser le trafic réseau simulé.
- L’environnement virtuel Python garantit que les bonnes versions des packages sont utilisées sans conflit.
- Commencer toujours par lancer la simulation avant les visualisations ou exports.
