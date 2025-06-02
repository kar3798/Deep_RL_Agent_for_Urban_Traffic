import traci
from train_dqn import TrainingEnv


# Environment for visual demo
class VisualizationEnvironment(TrainingEnv):

    def __init__(self, config_path, max_steps=3000, gui=True, step_delay=0.1):
        super().__init__(config_path, max_steps, gui, step_delay)
        self.visualization_metrics = {
            'vehicles_completed': 0,
            'total_waiting_time': 0,
            'phase_switches': 0,
            'last_phase': None,
            'efficiency_score': 0
        }

    # Performs a step and records visualization metrics
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        try:
            self.visualization_metrics['vehicles_completed'] = traci.simulation.getArrivedNumber()
            waiting_time = sum([
                traci.edge.getWaitingTime("e_north"),
                traci.edge.getWaitingTime("e_south"),
                traci.edge.getWaitingTime("e_east"),
                traci.edge.getWaitingTime("e_west")
            ])
            self.visualization_metrics['total_waiting_time'] += waiting_time
            current_phase = traci.trafficlight.getPhase(self.tls_id)
            if (self.visualization_metrics['last_phase'] is not None and
                    current_phase != self.visualization_metrics['last_phase']):
                self.visualization_metrics['phase_switches'] += 1
            self.visualization_metrics['last_phase'] = current_phase
            vehicles = self.visualization_metrics['vehicles_completed']
            switches = self.visualization_metrics['phase_switches']
            if switches > 0:
                self.visualization_metrics['efficiency_score'] = vehicles / switches
            info.update(self.visualization_metrics)
        except Exception:
            pass
        return obs, reward, terminated, truncated, info
