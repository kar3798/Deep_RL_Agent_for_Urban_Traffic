# Curriculum Learning implementation for Traffic Signal Control
# Goes from simple traffic to complex turning movements

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import traci
from traffic_env import TrafficSignalEnv
import xml.etree.ElementTree as ET
import random
import os

# Set all seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Traffic env for learning, with 16D normalized state and 8 actions
class TrainingEnv(TrafficSignalEnv):

    def __init__(self, config_path="../../env/market_street.sumocfg", max_steps=3000, gui=False, step_delay=0.1):
        super().__init__(config_path, max_steps, gui, step_delay)
        self.observation_space = gym.spaces.Box(low=0, high=1.0, shape=(16,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(8)
        self.action_mapping = {
            0: {"direction": "NS", "duration": 25, "type": "through"},
            1: {"direction": "NS", "duration": 35, "type": "through"},
            2: {"direction": "NS", "duration": 50, "type": "through"},
            3: {"direction": "EW", "duration": 25, "type": "through"},
            4: {"direction": "EW", "duration": 35, "type": "through"},
            5: {"direction": "EW", "duration": 50, "type": "through"},
            6: {"direction": "NS_LEFT", "duration": 20, "type": "protected_left"},
            7: {"direction": "EW_LEFT", "duration": 20, "type": "protected_left"}
        }

        self.turning_movement_counts = {
            'north': {'through': 0, 'right': 0, 'left': 0, 'uturn': 0},
            'south': {'through': 0, 'right': 0, 'left': 0, 'uturn': 0},
            'east': {'through': 0, 'right': 0, 'left': 0, 'uturn': 0},
            'west': {'through': 0, 'right': 0, 'left': 0, 'uturn': 0}
        }

    # Returns normalized vector with turning movements and intersection state.
    def _get_observation(self):
        try:
            turning_counts = self._count_turning_movements()
            max_per_movement = 15
            obs = []
            for direction in ['north', 'south', 'east', 'west']:
                through = min(turning_counts[direction]['through'] / max_per_movement, 1.0)
                right = min(turning_counts[direction]['right'] / max_per_movement, 1.0)
                left = min(turning_counts[direction]['left'] / max_per_movement, 1.0)
                obs.extend([through, right, left])
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            phase_normalized = current_phase / 8.0
            sim_time = traci.simulation.getTime()
            phase_duration = min((sim_time % 120) / 120.0, 1.0)
            conflict_pressure = self._calculate_conflict_pressure(turning_counts)
            hour_of_day = (7 + (sim_time / 3600)) % 24
            time_normalized = hour_of_day / 24.0
            obs.extend([phase_normalized, phase_duration, conflict_pressure, time_normalized])
            return np.array(obs, dtype=np.float32)
        except Exception:
            return np.zeros(16, dtype=np.float32)

    # Counts vehicles by approach and turning movement
    def _count_turning_movements(self):
        counts = {d: {'through': 0, 'right': 0, 'left': 0, 'uturn': 0} for d in ['north', 'south', 'east', 'west']}
        try:
            for edge_id in ["e_north", "e_south", "e_east", "e_west"]:
                vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
                for veh_id in vehicles:
                    try:
                        route_id = traci.vehicle.getRouteID(veh_id)
                        if route_id.startswith('north_'):
                            if 'south' in route_id:
                                counts['north']['through'] += 1
                            elif 'east' in route_id:
                                counts['north']['right'] += 1
                            elif 'west' in route_id:
                                counts['north']['left'] += 1
                            elif 'north' in route_id:
                                counts['north']['uturn'] += 1
                        elif route_id.startswith('south_'):
                            if 'north' in route_id:
                                counts['south']['through'] += 1
                            elif 'west' in route_id:
                                counts['south']['right'] += 1
                            elif 'east' in route_id:
                                counts['south']['left'] += 1
                            elif 'south' in route_id:
                                counts['south']['uturn'] += 1
                        elif route_id.startswith('east_'):
                            if 'west' in route_id:
                                counts['east']['through'] += 1
                            elif 'south' in route_id:
                                counts['east']['right'] += 1
                            elif 'north' in route_id:
                                counts['east']['left'] += 1
                            elif 'east' in route_id:
                                counts['east']['uturn'] += 1
                        elif route_id.startswith('west_'):
                            if 'east' in route_id:
                                counts['west']['through'] += 1
                            elif 'north' in route_id:
                                counts['west']['right'] += 1
                            elif 'south' in route_id:
                                counts['west']['left'] += 1
                            elif 'west' in route_id:
                                counts['west']['uturn'] += 1
                    except traci.TraCIException:
                        continue
        except Exception:
            pass
        return counts

    # Computes left-turn and through movement conflict intensity
    def _calculate_conflict_pressure(self, turning_counts):
        ns_left_pressure = (turning_counts['north']['left'] + turning_counts['south']['left']) * \
                           (turning_counts['south']['through'] + turning_counts['north']['through'])
        ew_left_pressure = (turning_counts['east']['left'] + turning_counts['west']['left']) * \
                           (turning_counts['west']['through'] + turning_counts['east']['through'])
        total_pressure = ns_left_pressure + ew_left_pressure
        return min(total_pressure / 100.0, 1.0)

    # Calculates reward based on waiting time, throughput, turning efficiency, conflict, and phase switching
    def _calculate_reward(self, action):
        try:
            turning_counts = self._count_turning_movements()
            waiting_times = [
                traci.edge.getWaitingTime("e_north"),
                traci.edge.getWaitingTime("e_south"),
                traci.edge.getWaitingTime("e_east"),
                traci.edge.getWaitingTime("e_west")
            ]
            total_waiting = sum(waiting_times)
            waiting_penalty = -min(total_waiting * 0.05, 30.0)
            vehicles_arrived = traci.simulation.getArrivedNumber()
            throughput_reward = vehicles_arrived * 3.0
            action_info = self.action_mapping[action]
            direction = action_info["direction"]
            action_type = action_info["type"]
            turning_efficiency = 0
            if action_type == "protected_left":
                if direction == "NS_LEFT":
                    left_demand = turning_counts['north']['left'] + turning_counts['south']['left']
                    if left_demand > 3:
                        turning_efficiency = 10.0
                elif direction == "EW_LEFT":
                    left_demand = turning_counts['east']['left'] + turning_counts['west']['left']
                    if left_demand > 3:
                        turning_efficiency = 10.0
            elif action_type == "through":
                if direction == "NS":
                    through_demand = turning_counts['north']['through'] + turning_counts['south']['through']
                    if through_demand > turning_counts['north']['left'] + turning_counts['south']['left']:
                        turning_efficiency = 5.0
                elif direction == "EW":
                    through_demand = turning_counts['east']['through'] + turning_counts['west']['through']
                    if through_demand > turning_counts['east']['left'] + turning_counts['west']['left']:
                        turning_efficiency = 5.0
            conflict_pressure = self._calculate_conflict_pressure(turning_counts)
            conflict_bonus = 8.0 if action_type == "protected_left" and conflict_pressure > 0.3 else 0
            total_vehicles = sum(sum(counts.values()) for counts in turning_counts.values())
            clearance_bonus = 15.0 if total_vehicles < 20 else 0
            switch_penalty = 0
            if getattr(self, 'last_action', None) is not None:
                last_type = self.action_mapping[self.last_action]["type"]
                current_type = self.action_mapping[action]["type"]
                if last_type != current_type:
                    switch_penalty = -3.0
            total_reward = (waiting_penalty + throughput_reward + turning_efficiency +
                            conflict_bonus + clearance_bonus + switch_penalty)
            return max(min(total_reward, 50.0), -80.0)
        except Exception:
            return -5.0


# Generates and saves SUMO configs and route files for each transfer training phase.
def create_phases():
    tree = ET.parse("../../env/routes.rou.xml")
    root = tree.getroot()
    phases = [
        {
            "name": "through_only",
            "keep_routes": ["north_south", "south_north", "east_west", "west_east"],
            "max_depart": 120,
            "vehicles": 300
        },
        {
            "name": "through_and_right",
            "keep_routes": ["north_south", "south_north", "east_west", "west_east",
                            "north_east", "south_west", "east_south", "west_north"],
            "max_depart": 180,
            "vehicles": 500
        },
        {
            "name": "all_except_uturn",
            "keep_routes": ["north_south", "south_north", "east_west", "west_east",
                            "north_east", "south_west", "east_south", "west_north",
                            "north_west", "south_east", "east_north", "west_south"],
            "max_depart": 250,
            "vehicles": 800
        },
        {
            "name": "full_complexity",
            "keep_routes": None,
            "max_depart": 400,
            "vehicles": 1200
        }
    ]
    for phase in phases:
        create_phase_config(root, phase)
    return phases


# Writes SUMO XML and config file for a given phase
def create_phase_config(original_root, phase_config):
    new_root = ET.Element("routes")
    for child in original_root:
        if child.tag in ["vType", "route"]:
            if child.tag == "route":
                route_id = child.get("id")
                if phase_config["keep_routes"] is None or route_id in phase_config["keep_routes"]:
                    new_root.append(child)
            else:
                new_root.append(child)
    for vehicle in original_root.findall("vehicle"):
        depart_time = float(vehicle.get("depart", 0))
        route_id = vehicle.get("route")
        if depart_time < phase_config["max_depart"]:
            if phase_config["keep_routes"] is None or route_id in phase_config["keep_routes"]:
                new_root.append(vehicle)
    phase_tree = ET.ElementTree(new_root)
    routes_path = f"../../env/routes_{phase_config['name']}.rou.xml"
    phase_tree.write(routes_path, xml_declaration=True, encoding='UTF-8')
    config_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<sumoConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="market_street_with_tls.net.xml"/>
        <route-files value="routes_{phase_config['name']}.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="800"/>
        <step-length value="0.1"/>
    </time>
    <report>
        <verbose value="true"/>
        <no-step-log value="true"/>
    </report>
</sumoConfiguration>'''
    with open(f"../../env/market_street_{phase_config['name']}.sumocfg", "w") as f:
        f.write(config_content)


#  Trains DQN models sequentially over each phase and saves each checkpoint
def train_dqn_agent():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    phases = create_phases()
    training_configs = [
        {"timesteps": 100000, "max_steps": 2000, "learning_rate": 0.0005},
        {"timesteps": 120000, "max_steps": 2500, "learning_rate": 0.0003},
        {"timesteps": 150000, "max_steps": 3000, "learning_rate": 0.0002},
        {"timesteps": 200000, "max_steps": 4000, "learning_rate": 0.0001}
    ]
    model = None
    for i, (phase, config) in enumerate(zip(phases, training_configs)):
        env = TrainingEnv(
            config_path=f"../../env/market_street_{phase['name']}.sumocfg",
            max_steps=config['max_steps'],
            gui=False
        )
        env = Monitor(env, f"logs/phase_{i + 1}/")
        if model is None:
            model = DQN(
                policy="MlpPolicy",
                env=env,
                learning_rate=config['learning_rate'],
                buffer_size=150000,
                learning_starts=8000,
                batch_size=128,
                tau=0.01,
                gamma=0.99,
                target_update_interval=2500,
                exploration_fraction=0.4,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.02,
                verbose=1,
                device=device,
                tensorboard_log="./tensorboard_logs/",
                policy_kwargs=dict(
                    net_arch=[512, 512, 256, 128],
                    activation_fn=torch.nn.ReLU,
                ),
                seed=42
            )
        else:
            model.set_env(env)
            model.learning_rate = config['learning_rate']
        model.learn(
            total_timesteps=config['timesteps'],
            log_interval=25,
            progress_bar=True
        )
        model.save(f"curriculum_phase_{i + 1}")
        env.close()
    model.save("curriculum_final")
    return model


if __name__ == "__main__":
    torch.set_num_threads(4)
    for i in range(1, 5):
        os.makedirs(f"logs/phase_{i}", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    model = train_dqn_agent()
