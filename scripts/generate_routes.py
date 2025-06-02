from scenario_generator import generate_traffic_scenario, create_sumo_configs


# Generates SUMO route and config files for several traffic scenarios
def main():
    output_dir = "../env"
    scenarios = [
        ("low", 300),
        ("medium", 400),
        ("high", 500)
    ]
    sim_duration = 2400

    for sname, vph in scenarios:
        routes_file = generate_traffic_scenario(sname, vph, sim_duration, output_dir)
        print(f"{sname.title()} traffic: {routes_file}")

    config_paths = create_sumo_configs(sim_duration, output_dir)
    for config in config_paths:
        print(f"Config created: {config}")


if __name__ == "__main__":
    main()
