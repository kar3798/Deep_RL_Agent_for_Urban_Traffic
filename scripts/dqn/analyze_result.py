import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
from stable_baselines3 import DQN
from visualization_env import VisualizationEnvironment

TRAFFIC_SCENARIOS = {
    "low": {
        "config": "../../env/market_street_low_traffic.sumocfg",
        "label": "Low Traffic",
        "route file": "../../env/routes_low_traffic.rou.xml"
    },
    "medium": {
        "config": "../../env/market_street_medium_traffic.sumocfg",
        "label": "Medium Traffic",
        "route file": "../../env/routes_medium_traffic.rou.xml"
    },
    "high": {
        "config": "../../env/market_street_high_traffic.sumocfg",
        "label": "High Traffic",
        "route file": "../../env/routes_high_traffic.rou.xml"
    }
}

MODEL_ORDER = ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Philadelphia Baseline"]

SCENARIO_COLORS = {
    "low": "green",
    "medium": "orange",
    "high": "red"
}


# Finds available RL model files for phases 1-4
def find_available_models():
    models = {}
    for phase in range(1, 5):
        model_file = f"curriculum_phase_{phase}.zip"
        if os.path.exists(model_file):
            models[f"Phase {phase}"] = model_file
    return models


# Counts the number of vehicles in a scenario
def count_vehicles_in_routefile(routefile_path):
    try:
        tree = ET.parse(routefile_path)
        root = tree.getroot()
        return sum(1 for v in root.iter('vehicle'))
    except Exception as e:
        print(f"Error counting vehicles in {routefile_path}: {e}")
        return None


# Runs model and collects metrics
def run_model(model_path, env_config, max_steps=3000):
    """Evaluate a trained RL model on the given environment config."""
    env = VisualizationEnvironment(config_path=env_config, max_steps=max_steps, gui=False, step_delay=0)
    obs, info = env.reset()
    episode_reward = 0
    step_count = 0
    model = DQN.load(model_path)
    while step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        if hasattr(action, 'item'):
            action = action.item()
        elif isinstance(action, np.ndarray):
            action = int(action[0])
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        if terminated or truncated:
            break
    env.close()
    return {
        'reward': episode_reward,
        'steps': step_count,
        'total_waiting_time': info.get('total_waiting_time', 0),
        'phase_switches': info.get('phase_switches', 0),
    }


# Implements the Philadelphia-style baseline controller for traffic signal timing
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
        return action, None, timing

    def _get_timing_plan(self, time_of_day):
        if 7 <= time_of_day <= 9 or 17 <= time_of_day <= 19:
            return {"cycle_length": 100, "major_green": 65, "minor_green": 25, "yellow": 4, "all_red": 2}
        elif 12 <= time_of_day <= 14:
            return {"cycle_length": 85, "major_green": 55, "minor_green": 22, "yellow": 4, "all_red": 2}
        else:
            return {"cycle_length": 75, "major_green": 45, "minor_green": 22, "yellow": 4, "all_red": 2}


#  Runs baseline and collects metrics
def run_baseline(env_config, max_steps=3000):
    env = VisualizationEnvironment(config_path=env_config, max_steps=max_steps, gui=False, step_delay=0)
    baseline = PhiladelphiaBaselineVisual()
    obs, info = env.reset()
    episode_reward = 0
    step_count = 0
    simulation_time = 0
    while step_count < max_steps:
        action, _, _ = baseline.get_action(simulation_time)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        simulation_time += 1.2
        if terminated or truncated:
            break
    env.close()
    return {
        'reward': episode_reward,
        'steps': step_count,
        'total_waiting_time': info.get('total_waiting_time', 0),
        'phase_switches': info.get('phase_switches', 0),
    }


# Evaluates all models and baseline for given scenario and saves results
def analyze_scenario(scenario_key, max_steps=3000):
    scenario = TRAFFIC_SCENARIOS[scenario_key]
    print(f"\nScenario: {scenario['label']}")
    models = find_available_models()
    results = {}
    no_of_vehicles = count_vehicles_in_routefile(scenario["route file"])
    for model_name in MODEL_ORDER:
        if model_name == "Philadelphia Baseline":
            print(f"Evaluating: {model_name}")
            metrics = run_baseline(scenario["config"], max_steps)
        elif model_name in models:
            print(f"Evaluating: {model_name}")
            model_path = models[model_name]
            metrics = run_model(model_path, scenario["config"], max_steps)
        else:
            continue
        metrics['vehicles_defined'] = no_of_vehicles
        results[model_name] = metrics
    df = pd.DataFrame(results).T
    df = df.reset_index().rename(columns={'index': 'Model'})

    # USED FOR DEBUGGING
    # Print results table
    # print("\nResults:")
    # print(df[["Model", "reward", "steps", "total_waiting_time", "phase_switches"]].round(3))
    #
    # # Print summary of rewards
    # print("\nReward summary:")
    # for idx, row in df.iterrows():
    #     print(f"{row['Model']}: {row['reward']:.2f}")
    # print(f"Max reward: {df['reward'].max():.2f}")
    # print(f"Min reward: {df['reward'].min():.2f}")
    # print(f"Mean reward: {df['reward'].mean():.2f}")

    csv_filename = f"results/{scenario_key}_results.csv"
    df.to_csv(csv_filename, index=False, float_format="%.3f")
    return df


# Generates and saves comparative plots
def plot_results(dfs, scenario_keys):
    os.makedirs("results", exist_ok=True)
    metrics = ['reward', 'steps', 'total_waiting_time']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for df, key in zip(dfs, scenario_keys):
            plot_df = df.set_index("Model").reindex(MODEL_ORDER)
            plt.plot(plot_df.index, plot_df[metric], marker='o', label=TRAFFIC_SCENARIOS[key]['label'],
                     color=SCENARIO_COLORS.get(key, "blue"))
        plt.title(f"{metric.replace('_', ' ').title()} by Model/Phase (All Traffic Scenarios)")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xlabel("Model / Phase")
        plt.xticks(rotation=20)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        filename = f"results/{metric}_all_scenarios_comparison.png"
        plt.savefig(filename)
        plt.close()


def main():
    scenario_keys = list(TRAFFIC_SCENARIOS.keys())
    all_dfs = []
    for key in scenario_keys:
        df = analyze_scenario(key)
        all_dfs.append(df)
    plot_results(all_dfs, scenario_keys)
    print("Analysis complete")


if __name__ == "__main__":
    main()
