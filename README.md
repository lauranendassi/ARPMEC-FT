project:
  name: "ARPMEC+FT"
  description: >
    ARPMEC+FT est un simulateur Python avancé pour les réseaux sans fil utilisant un clustering dynamique
    avec tolérance aux pannes. Conçu pour être exécuté dans un environnement virtuel Python, garantissant
    l'isolation des dépendances.
    
features:
  clustering:
    description: "Clustering intelligent via k-means"
    details:
      - Chaque cluster possède un Cluster Head (CH) et au moins 2 membres
      - Reclustering automatique en cas de panne de CH ou dissolution du cluster
  fault_tolerance:
    description: "Tolérance aux pannes"
    details:
      - Prédiction des pannes basée sur RSSI et Pr(d)
      - Récupération automatique avec réélection de CH
      - Gestion des pannes énergétiques et aléatoires
  mobility:
    description: "Mobilité des nœuds"
    details:
      - Déplacement dirigé avec rebond
      - Simulation de consommation d'énergie liée au déplacement
  routing:
    description: "Routage adaptatif"
    details:
      - Transmission des données des membres vers le CH puis vers la Base Station (BS)
      - Échecs de transmission simulés selon distance et RSSI
      - Journaux détaillés de routage, export TXT possible
  visualization:
    description: "Visualisation interactive"
    details:
      - Affichage des clusters, CH, membres, BS et serveurs MEC
      - Slider pour naviguer entre les rounds et boutons Pause/Play
      - Export automatique PNG, GIF et MP4
  metrics:
    description: "Métriques et analyses"
    details:
      - Énergie moyenne des nœuds
      - Nombre de clusters actifs et CH réélus
      - Nombre de pannes par round
      - Taux de réussite du routage
      - Export CSV, courbes comparatives avec/sans tolérance
      - Moyennage sur plusieurs simulations avec barres d'erreur

installation:
  steps:
    - step: "Créer un environnement virtuel"
      command: "python -m venv venv_arpmec"
    - step: "Activer l'environnement"
      command_windows: "venv_arpmec\\Scripts\\activate"
      command_linux_mac: "source venv_arpmec/bin/activate"
    - step: "Installer les dépendances"
      command: "pip install -r requirements.txt"
  dependencies:
    - numpy
    - matplotlib
    - scikit-learn
    - scapy
    - colorama

usage:
  launch: "python arpmec_tolerant.py"
  menu_options:
    - 1: "Lancer la simulation"
    - 2: "Afficher les logs détaillés"
    - 3: "Exporter les métriques CSV"
    - 4: "Visualiser la simulation"
    - 5: "Quitter"
    - 6: "Afficher les logs de routage"
    - 7: "Exporter les logs de routage en fichier .txt"
    - 8: "Lancer des simulations moyennées et générer les courbes"

project_structure:
  - arpmec_tolerant.py: "Fichier principal"
  - requirements.txt: "Dépendances Python"
  - README.md: "Documentation"
  - metrics_detailed.csv: "Export métriques détaillées"
  - simulated_arpmec.pcap: "Export des paquets simulés"
  - courbes_comparatives.png: "Courbes comparatives"
  - comparaison_tolerance.png: "Comparaison Tolérance / Non Tolérance"

example_simulation:
  nodes: 50
  rounds: 20
  initial_clusters: 3
  base_stations: 5
  mec_servers: 3
  visualization: "Interactive, animation des clusters, CH, BS et MEC"

contribution:
  description: "Les contributions sont les bienvenues"
  suggestions:
    - "Amélioration des algorithmes de clustering et tolérance"
    - "Modèle de mobilité et consommation d’énergie"
    - "Visualisation et animations interactives"

license:
  type: "MIT"
  file: "LICENSE"

contact:
  author: "Elvie Laurane NGUEDEM NDASSI"
  email: "ndassilaurane@gmail.com"
  notes: "Développé dans le cadre de recherche sur les réseaux distribués et la tolérance aux pannes"

