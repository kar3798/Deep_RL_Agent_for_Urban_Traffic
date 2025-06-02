import xml.etree.ElementTree as ET
import numpy as np
import os
from pathlib import Path

# Scenario Movement
SCENARIO_MOVEMENT_PROBABILITIES = {
    "low": {
        "north_south": 0.16, "south_north": 0.16,
        "east_west": 0.18, "west_east": 0.18,
        "north_east": 0.06, "south_west": 0.06,
        "east_south": 0.04, "west_north": 0.04,
        "north_west": 0.03, "south_east": 0.03,
        "east_north": 0.02, "west_south": 0.02,
        "north_north": 0.005, "south_south": 0.005,
        "east_east": 0.005, "west_west": 0.005
    },
    "medium": {
        "north_south": 0.14, "south_north": 0.14,
        "east_west": 0.16, "west_east": 0.16,
        "north_east": 0.07, "south_west": 0.07,
        "east_south": 0.05, "west_north": 0.05,
        "north_west": 0.04, "south_east": 0.04,
        "east_north": 0.03, "west_south": 0.03,
        "north_north": 0.008, "south_south": 0.008,
        "east_east": 0.004, "west_west": 0.004
    },
    "high": {
        "north_south": 0.12, "south_north": 0.12,
        "east_west": 0.14, "west_east": 0.14,
        "north_east": 0.08, "south_west": 0.08,
        "east_south": 0.06, "west_north": 0.06,
        "north_west": 0.05, "south_east": 0.05,
        "east_north": 0.04, "west_south": 0.04,
        "north_north": 0.01, "south_south": 0.01,
        "east_east": 0.005, "west_west": 0.005
    }
}

SCENARIO_VEHICLE_TYPE_PROBABILITIES = {
    "low": {"car": 0.90, "bus": 0.03, "truck": 0.07},
    "medium": {"car": 0.85, "bus": 0.05, "truck": 0.10},
    "high": {"car": 0.90, "bus": 0.04, "truck": 0.06}
}


def generate_departure_times_with_total(total_vehicles, simulation_duration, get_traffic_intensity, scenario_name):
    # Generates N random intervals with intensity effect
    intervals = []
    current_time = 0
    for i in range(total_vehicles):
        base_interval = simulation_duration / total_vehicles
        intensity = get_traffic_intensity(current_time, scenario_name)
        if scenario_name == "low":
            interval = np.random.exponential(base_interval * 1.2 / intensity)
        elif scenario_name == "medium":
            interval = np.random.exponential(base_interval / intensity)
        else:  # high
            interval = np.random.exponential(base_interval * 0.8 / intensity)
        intervals.append(interval)
        current_time += interval
    # Cumulative sum to get times
    departure_times = np.cumsum(intervals)
    # If the last depart_time > simulation_duration, rescale all times
    if departure_times[-1] > simulation_duration:
        departure_times = departure_times * (simulation_duration / departure_times[-1])
    return departure_times


def generate_traffic_scenario(scenario_name, vehicles_per_hour, simulation_duration=2400, output_dir="../env"):
    # Calculates total vehicles for the simulation duration
    total_vehicles = int((vehicles_per_hour * simulation_duration) / 3600)

    # Create root element
    root = ET.Element("routes")

    # Vehicle type definitions
    car_type = ET.SubElement(root, "vType")
    car_type.set("id", "car")
    car_type.set("accel", "1.5")
    car_type.set("decel", "4.5")
    car_type.set("sigma", "0.3")
    car_type.set("length", "5")
    car_type.set("minGap", "2.5")
    car_type.set("maxSpeed", "13.89")
    car_type.set("guiShape", "passenger")

    bus_type = ET.SubElement(root, "vType")
    bus_type.set("id", "bus")
    bus_type.set("accel", "1.0")
    bus_type.set("decel", "4.0")
    bus_type.set("sigma", "0.2")
    bus_type.set("length", "15")
    bus_type.set("minGap", "3.0")
    bus_type.set("maxSpeed", "11.00")
    bus_type.set("guiShape", "bus")

    truck_type = ET.SubElement(root, "vType")
    truck_type.set("id", "truck")
    truck_type.set("accel", "0.8")
    truck_type.set("decel", "4.0")
    truck_type.set("sigma", "0.2")
    truck_type.set("length", "12")
    truck_type.set("minGap", "3.5")
    truck_type.set("maxSpeed", "9.50")
    truck_type.set("guiShape", "truck")

    # Route definitions
    route_definitions = [
        ("north_south", "e_north e_center_south"),
        ("north_east", "e_north e_center_east"),
        ("north_west", "e_north e_center_west"),
        ("north_north", "e_north e_center_north"),
        ("south_north", "e_south e_center_north"),
        ("south_west", "e_south e_center_west"),
        ("south_east", "e_south e_center_east"),
        ("south_south", "e_south e_center_south"),
        ("east_west", "e_east e_center_west"),
        ("east_south", "e_east e_center_south"),
        ("east_north", "e_east e_center_north"),
        ("east_east", "e_east e_center_east"),
        ("west_east", "e_west e_center_east"),
        ("west_north", "e_west e_center_north"),
        ("west_south", "e_west e_center_south"),
        ("west_west", "e_west e_center_west")
    ]

    for route_id, edges in route_definitions:
        route_elem = ET.SubElement(root, "route")
        route_elem.set("id", route_id)
        route_elem.set("edges", edges)

    # Probabilities per scenario
    movement_probabilities = SCENARIO_MOVEMENT_PROBABILITIES[scenario_name]
    # Normalize
    total_prob = sum(movement_probabilities.values())
    movement_probabilities = {k: v / total_prob for k, v in movement_probabilities.items()}

    # Probabilities per vehicles
    vehicle_type_probabilities = SCENARIO_VEHICLE_TYPE_PROBABILITIES[scenario_name]
    # Normalize
    total_vehicle_type_prob = sum(vehicle_type_probabilities.values())
    vehicle_type_probabilities = {k: v / total_vehicle_type_prob for k, v in vehicle_type_probabilities.items()}

    # Time intensity profile
    def get_traffic_intensity(time_seconds, scenario):
        hour = (time_seconds / 3600) % 24
        if scenario == "low":
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                return 1.2
            elif 12 <= hour <= 13:
                return 1.1
            elif 22 <= hour or hour <= 6:
                return 0.7
            else:
                return 1.0
        elif scenario == "medium":
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                return 1.4
            elif 12 <= hour <= 13:
                return 1.2
            elif 22 <= hour or hour <= 6:
                return 0.4
            else:
                return 1.0
        else:  # high
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                return 1.6
            elif 12 <= hour <= 13:
                return 1.3
            elif 22 <= hour or hour <= 6:
                return 0.2
            else:
                return 1.0

    departure_times = generate_departure_times_with_total(
        total_vehicles, simulation_duration, get_traffic_intensity, scenario_name
    )

    route_names = list(movement_probabilities.keys())
    route_probs = list(movement_probabilities.values())
    vehicle_types = list(vehicle_type_probabilities.keys())
    type_probs = list(vehicle_type_probabilities.values())
    type_counts = {"car": 0, "bus": 0, "truck": 0}
    for i, depart_time in enumerate(departure_times):
        vehicle_type = np.random.choice(vehicle_types, p=type_probs)
        route_id = np.random.choice(route_names, p=route_probs)
        vehicle = ET.SubElement(root, "vehicle")
        vehicle.set("id", f"{vehicle_type}_{i + 1}")
        vehicle.set("type", vehicle_type)
        vehicle.set("route", route_id)
        vehicle.set("depart", f"{depart_time:.2f}")
        if route_id.startswith(('east_', 'west_')):
            vehicle.set("departLane", str(np.random.randint(0, 2)))
    print(f"{scenario_name.title()} vehicle type counts: {type_counts}")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    routes_file = f"routes_{scenario_name}_traffic.rou.xml"
    routes_path = os.path.join(output_dir, routes_file)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(routes_path, 'wb') as f:
        tree.write(f, xml_declaration=True, encoding='UTF-8')
    return routes_file


def write_sumo_config(
        config_path,
        net_file="market_street_with_tls.net.xml",
        route_file="routes_high_traffic.rou.xml",
        begin=0,
        end=800,
        step_length=0.1,
        verbose=True,
        no_step_log=True
):
    root = ET.Element(
        "sumoConfiguration",
        attrib={
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/sumoConfiguration.xsd"
        }
    )

    input_elem = ET.SubElement(root, "input")
    ET.SubElement(input_elem, "net-file", value=net_file)
    ET.SubElement(input_elem, "route-files", value=route_file)
    time_elem = ET.SubElement(root, "time")
    ET.SubElement(time_elem, "begin", value=str(begin))
    ET.SubElement(time_elem, "end", value=str(end))
    ET.SubElement(time_elem, "step-length", value=str(step_length))
    report_elem = ET.SubElement(root, "report")
    ET.SubElement(report_elem, "verbose", value=str(verbose).lower())
    ET.SubElement(report_elem, "no-step-log", value=str(no_step_log).lower())
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(config_path, encoding="UTF-8", xml_declaration=True)


def create_sumo_configs(simulation_duration, output_dir):
    output_dir = Path(output_dir)
    config_paths = []
    for scenario in ["low", "medium", "high"]:
        config_path = output_dir / f"market_street_{scenario}_traffic.sumocfg"
        route_file = output_dir / f"routes_{scenario}_traffic.rou.xml"
        write_sumo_config(
            config_path=config_path,
            net_file="market_street_with_tls.net.xml",
            route_file=route_file.name,
            begin=0,
            end=simulation_duration,
            step_length=0.1,
            verbose=True,
            no_step_log=True
        )
        config_paths.append(str(config_path))
    return config_paths
