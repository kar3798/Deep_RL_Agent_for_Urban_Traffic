==================================================
Deep Reinforcement Learning for Urban Traffic Signal Optimization - Project README
==================================================

This repository contains code, data, configuration, and pretrained models for a deep reinforcement learning approach to adaptive urban traffic signal control using SUMO.

----------------------------------------
Directory Structure
----------------------------------------

Traffic_Simulator_RL-iteration-final/
│
├── .venv/                       # Python virtual environment (optional, for local use)
│
├── data/                        # CSV files: intersection and street network data
│   ├── high_injury_network_2020.csv
│   ├── Intersection_Controls.csv
│   ├── Potential_Intersections.csv
│   └── Street_Centerline.csv
│
├── env/                         # SUMO environment config and route files
│   ├── *.sumocfg                # SUMO configuration files for various scenarios
│   ├── *.xml                    # Network, junction, node, and logic files
│   ├── *.rou.xml                # Generated route files for different scenarios
│
├── scripts/                     # All source code and models
│   ├── dqn/                     # Contains model checkpoints, logs, and results
│   │   ├── logs/
│   │   ├── results/
│   │   ├── tensorboard_logs/
│   │   ├── curriculum_final.zip
│   │   ├── curriculum_phase_1.zip
│   │   ├── curriculum_phase_2.zip
│   │   ├── curriculum_phase_3.zip
│   │   ├── curriculum_phase_4.zip
│   │   ├── traffic_env.py/
│   │   ├── train_dqn.py/
│   │   ├── visualization_env.py/
│   │   └── visualize_models.py/
│   ├── generate_routes.py
│   ├── get_best_intersection.py
│   └── scenario_generator.py
│
├── requirements.txt             # Python dependencies
├── readme.txt                   # This file
├── .gitignore                   # Files/folders to be ignored by git

----------------------------------------
Getting Started
----------------------------------------

1. (Optional) Create and Activate Virtual Environment
----------------------------------------------------
Windows:
    python -m venv .venv
    .venv\Scripts\activate

Linux/Mac:
    python3 -m venv .venv
    source .venv/bin/activate

2. Install Required Python Packages
-----------------------------------
    pip install -r requirements.txt

3. Prepare Data and Generate Routes
-----------------------------------
Run the script to generate SUMO routes based on provided intersection data:
    python scripts/generate_routes.py

4. Train the RL Agent (Optional: Use Pretrained Models)
-------------------------------------------------------
To train the agent from scratch with curriculum learning:
    python scripts/train_dqn.py

Model checkpoints will be saved in scripts/dqn/.

To skip training, you can use the provided pretrained models (curriculum_phase_*.zip).

5. Analyze Results
------------------
To compute and summarize evaluation metrics:
    python scripts/analyze_result.py

6. Visualize Results and Models
-------------------------------
To generate plots and visual summaries:
    python scripts/visualize_models.py

7. View TensorBoard Logs (Optional)
-----------------------------------
To visualize training progress and logs:
    tensorboard --logdir=scripts/dqn/tensorboard_logs/

