import gymnasium as gym
import numpy as np
import traci


# Base traffic signal control environment using SUMO simulation
class TrafficSignalEnv(gym.Env):

    def __init__(self, config_path="../../env/market_street.sumocfg", max_steps=5000, gui=False, step_delay=0.1):
        super().__init__()
        self.config_path = config_path
        self.max_steps = max_steps
        self.gui = gui
        self.step_count = 0
        self.step_delay = step_delay
        self.sumo_binary = "sumo-gui" if self.gui else "sumo"
        self.tls_id = "n_center"

        # Action space: 6 traffic signal timing options
        self.action_space = gym.spaces.Discrete(8)
        self.action_mapping = {
            0: {"direction": "NS", "duration": 20},
            1: {"direction": "NS", "duration": 30},
            2: {"direction": "NS", "duration": 45},
            3: {"direction": "EW", "duration": 20},
            4: {"direction": "EW", "duration": 30},
            5: {"direction": "EW", "duration": 45},
            6: {"direction": "NS_LEFT", "duration": 20, "type": "protected_left"},  # NEW
            7: {"direction": "EW_LEFT", "duration": 20, "type": "protected_left"}  # NEW
        }

        # Observation space: vehicle counts from 6 lanes
        self.observation_space = gym.spaces.Box(low=0, high=200, shape=(6,), dtype=np.float32)

        # State tracking
        self.last_action = None
        self.phase_start_time = 0
        self.total_vehicles_passed = 0
        self.action_history = []

    # Resets the SUMO environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if traci.isLoaded():
            traci.close()

        traci.start([self.sumo_binary, "-c", self.config_path, "--start"])

        self.step_count = 0
        self.last_action = None
        self.phase_start_time = traci.simulation.getTime()
        self.total_vehicles_passed = 0
        self.action_history = []

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        action = int(action) if hasattr(action, 'item') else int(action)
        action_info = self.action_mapping[action]
        direction = action_info["direction"]
        duration = action_info["duration"]
        # Apply traffic light control
        if direction == "NS":
            traci.trafficlight.setPhase(self.tls_id, 0)
        elif direction == "EW":
            traci.trafficlight.setPhase(self.tls_id, 2)
        elif direction == "NS_LEFT":
            traci.trafficlight.setPhase(self.tls_id, 1)
        elif direction == "EW_LEFT":
            traci.trafficlight.setPhase(self.tls_id, 3)
        vehicles_before = traci.simulation.getArrivedNumber()
        # Advance simulation
        steps_to_run = max(10, duration // 2)
        for i in range(steps_to_run):
            traci.simulationStep()
            self.step_count += 1
            if self.step_count % 50 == 0:
                current_phase = traci.trafficlight.getPhase(self.tls_id)
                waiting_vehicles = sum(self._get_lane_vehicle_counts())
                print(f"[Step {self.step_count}] Action: {action}({direction}-{duration}s), "
                      f"Phase: {current_phase}, Waiting: {waiting_vehicles}")
        vehicles_after = traci.simulation.getArrivedNumber()
        self.total_vehicles_passed += (vehicles_after - vehicles_before)
        self.action_history.append(action)
        obs = self._get_observation()
        reward = self._calculate_reward(action)
        terminated = False
        truncated = self.step_count >= self.max_steps
        self.last_action = action
        return obs, reward, terminated, truncated, {
            'total_vehicles_passed': self.total_vehicles_passed,
            'action_duration': duration,
            'direction': direction
        }

    def _get_observation(self):
        lane_counts = self._get_lane_vehicle_counts()
        return np.array(lane_counts, dtype=np.float32)

    # Get vehicle counts from all intersection approaches
    def _get_lane_vehicle_counts(self):
        try:
            north_count = traci.edge.getLastStepVehicleNumber("e_north")
            south_count = traci.edge.getLastStepVehicleNumber("e_south")

            # Multi-lane east-west approaches
            east_lane1_count = traci.lane.getLastStepVehicleNumber("e_east_0")
            east_lane2_count = traci.lane.getLastStepVehicleNumber("e_east_1")
            west_lane1_count = traci.lane.getLastStepVehicleNumber("e_west_0")
            west_lane2_count = traci.lane.getLastStepVehicleNumber("e_west_1")

            return [north_count, south_count, east_lane1_count, east_lane2_count,
                    west_lane1_count, west_lane2_count]

        except Exception as e:
            # Fallback to edge-level counts
            try:
                north_count = traci.edge.getLastStepVehicleNumber("e_north")
                south_count = traci.edge.getLastStepVehicleNumber("e_south")
                east_count = traci.edge.getLastStepVehicleNumber("e_east")
                west_count = traci.edge.getLastStepVehicleNumber("e_west")

                return [north_count, south_count, east_count // 2, east_count // 2,
                        west_count // 2, west_count // 2]
            except:
                return [0, 0, 0, 0, 0, 0]

    def _calculate_reward(self, action):
        try:
            # Waiting time penalty
            north_waiting = traci.edge.getWaitingTime("e_north")
            south_waiting = traci.edge.getWaitingTime("e_south")
            east_waiting = traci.edge.getWaitingTime("e_east")
            west_waiting = traci.edge.getWaitingTime("e_west")
            total_waiting_time = north_waiting + south_waiting + east_waiting + west_waiting
        except Exception:
            total_waiting_time = 0
        # Reward components
        waiting_penalty = -total_waiting_time
        throughput_reward = self.total_vehicles_passed * 0.1
        # Fairness penalty for unbalanced waiting times
        lane_waiting_times = [north_waiting, south_waiting, east_waiting, west_waiting]
        if len(lane_waiting_times) > 1 and max(lane_waiting_times) > 0:
            fairness_penalty = -np.std(lane_waiting_times) * 0.05
        else:
            fairness_penalty = 0
        # Action switching penalty
        if self.last_action is not None and self._actions_conflict(self.last_action, action):
            switch_penalty = -1.0
        else:
            switch_penalty = 0
        # Lane utilization bonus
        lane_counts = self._get_lane_vehicle_counts()
        east_total = lane_counts[2] + lane_counts[3]
        west_total = lane_counts[4] + lane_counts[5]
        if east_total > 0 and west_total > 0:
            east_balance = min(lane_counts[2], lane_counts[3]) / max(lane_counts[2], lane_counts[3], 1)
            west_balance = min(lane_counts[4], lane_counts[5]) / max(lane_counts[4], lane_counts[5], 1)
            lane_balance_bonus = (east_balance + west_balance) * 0.5
        else:
            lane_balance_bonus = 0
        total_reward = (waiting_penalty + throughput_reward + fairness_penalty +
                        switch_penalty + lane_balance_bonus)
        return total_reward

    # Check if two actions control different traffic directions
    def _actions_conflict(self, action1, action2):
        dir1 = self.action_mapping[action1]["direction"]
        dir2 = self.action_mapping[action2]["direction"]
        return dir1 != dir2

    def close(self):
        if traci.isLoaded():
            traci.close()
