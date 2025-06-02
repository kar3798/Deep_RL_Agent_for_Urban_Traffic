import os
import time
import numpy as np
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
from visualization_env import VisualizationEnvironment

PHASES = [f"Phase {i}" for i in range(1, 5)]
MODEL_FILES = {f"Phase {i}": f"curriculum_phase_{i}.zip" for i in range(1, 5)}

CONFIG_PATH = "../../env/market_street_low_traffic.sumocfg"
FALLBACK_CONFIG = "../../env/market_street.sumocfg"
MAX_STEPS = 3000
STEP_DELAY = 0.2


# Finds available RL model files
def find_available_models():
    models = {}
    for phase in PHASES:
        if os.path.exists(MODEL_FILES[phase]):
            models[phase] = MODEL_FILES[phase]
    return models


# Implements the fixed-cycle Philadelphia baseline controller
class PhiladelphiaBaselineVisual:
    def __init__(self):
        self.cycle_start_time = 0

    def get_action(self, simulation_time):
        hour = (7 + (simulation_time / 3600)) % 24
        timing = self._get_timing_plan(hour)
        time_in_cycle = (simulation_time - self.cycle_start_time) % timing["cycle_length"]
        if time_in_cycle < 1.0 and simulation_time > timing["cycle_length"]:
            self.cycle_start_time = simulation_time
            time_in_cycle = 0
        if time_in_cycle < timing["major_green"]:
            action = 1
        elif time_in_cycle < timing["major_green"] + timing["yellow"]:
            action = 1
        elif time_in_cycle < timing["major_green"] + timing["yellow"] + timing["all_red"]:
            action = 1
        elif time_in_cycle < timing["major_green"] + timing["yellow"] + timing["all_red"] + timing["minor_green"]:
            action = 4
        elif time_in_cycle < timing["major_green"] + timing["yellow"] + timing["all_red"] + timing["minor_green"] + \
                timing["yellow"]:
            action = 4
        else:
            action = 4
        return action

    def _get_timing_plan(self, time_of_day):
        if 7 <= time_of_day <= 9 or 17 <= time_of_day <= 19:
            return {"cycle_length": 100, "major_green": 65, "minor_green": 25, "yellow": 4, "all_red": 2}
        elif 12 <= time_of_day <= 14:
            return {"cycle_length": 85, "major_green": 55, "minor_green": 22, "yellow": 4, "all_red": 2}
        else:
            return {"cycle_length": 75, "major_green": 45, "minor_green": 22, "yellow": 4, "all_red": 2}


def get_config_path():
    return CONFIG_PATH if os.path.exists(CONFIG_PATH) else FALLBACK_CONFIG


# Runs and visualizes an RL model, returns total reward
def visualize_model(model_path, model_name, gui=True, step_delay=STEP_DELAY, max_steps=MAX_STEPS):
    config_path = get_config_path()
    try:
        model = DQN.load(model_path)
    except Exception as e:
        print(f"Could not load model {model_name}: {e}")
        return
    env = VisualizationEnvironment(config_path=config_path, max_steps=max_steps, gui=gui, step_delay=step_delay)
    obs, info = env.reset()
    episode_reward = 0
    step_count = 0
    print(f"Visualizing {model_name}...")
    while step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        if hasattr(action, 'item'):
            action = action.item()
        elif isinstance(action, np.ndarray):
            action = int(action[0])
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        if gui:
            if step_count % 25 == 0:
                print(f"  Step {step_count}  |  Reward: {episode_reward:.1f}")
            time.sleep(step_delay)
        if terminated or truncated:
            break
    env.close()
    print(f"{model_name} complete. Total Reward: {episode_reward:.1f}, Steps: {step_count}\n")
    return episode_reward


# Runs and visualizes the baseline, returns total reward
def visualize_baseline(gui=True, step_delay=STEP_DELAY, max_steps=MAX_STEPS):
    config_path = get_config_path()
    env = VisualizationEnvironment(config_path=config_path, max_steps=max_steps, gui=gui, step_delay=step_delay)
    baseline = PhiladelphiaBaselineVisual()
    obs, info = env.reset()
    episode_reward = 0
    step_count = 0
    simulation_time = 0
    print("Visualizing Philadelphia Baseline...")
    while step_count < max_steps:
        action = baseline.get_action(simulation_time)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        simulation_time += 1.2
        if gui:
            if step_count % 25 == 0:
                print(f"  Step {step_count}  |  Reward: {episode_reward:.1f}")
            time.sleep(step_delay)
        if terminated or truncated:
            break
    env.close()
    print(f"Philadelphia Baseline complete. Total Reward: {episode_reward:.1f}, Steps: {step_count}\n")
    return episode_reward

# Saves the bar chart of rewards for all phases and baseline
def plot_rewards(rewards):
    plt.figure(figsize=(7, 5))
    phases_order = PHASES + ["Philadelphia Baseline"]
    y = [rewards[p] for p in phases_order]
    plt.bar(phases_order, y, color="skyblue")
    plt.ylabel("Total Reward")
    plt.title("Rewards by Model Phase")
    plt.xticks(rotation=20)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/visualized_phase_rewards.png")
    plt.show()
    print("Reward plot saved")


def main():
    models = find_available_models()
    rewards = {}
    for phase in PHASES:
        if phase in models:
            # Store reward for each phase
            reward = visualize_model(models[phase], phase)
            rewards[phase] = reward if reward is not None else 0
    # Baseline
    baseline_reward = visualize_baseline()
    rewards["Philadelphia Baseline"] = baseline_reward if baseline_reward is not None else 0
    plot_rewards(rewards)
    print("All visualizations finished.")


if __name__ == "__main__":
    main()
