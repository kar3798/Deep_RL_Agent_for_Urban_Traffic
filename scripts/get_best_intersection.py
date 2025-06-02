import pandas as pd

# Load datasets
traffic_controls = pd.read_csv("../data/Intersection_Controls.csv")
street_centerline = pd.read_csv("../data/Street_Centerline.csv")
high_injury_network = pd.read_csv("../data/high_injury_network_2020.csv")

# Normalize text fields
street_centerline['ST_NAME'] = street_centerline['ST_NAME'].str.upper()
high_injury_network['street_name'] = high_injury_network['street_name'].str.upper()

# Filter relevant intersection types
relevant_controls = traffic_controls[
    traffic_controls['stoptype'].isin(['All Way', 'Signalized', 'Conventional'])
]

# Tag high-injury segments
high_injury_streets = set(high_injury_network['street_name'].unique())
street_centerline['is_high_injury'] = street_centerline['ST_NAME'].isin(high_injury_streets)

# Match intersection node_id to FNODE_ in street data
intersection_matches = pd.merge(
    relevant_controls,
    street_centerline,
    left_on='node_id',
    right_on='FNODE_',
    how='inner'
)

# Filter only those that are on high-injury segments
intersection_matches_high_injury = intersection_matches[intersection_matches['is_high_injury']]

# Select key columns
candidate_intersections = intersection_matches_high_injury[[
    'node_id', 'stoptype', 'ST_NAME', 'X', 'Y', 'SEG_ID'
]].drop_duplicates()

# Save to CSV
candidate_intersections.to_csv("Potential_Intersections.csv", index=False)