import csv

# Define actions and mappings
directions = ["north", "south", "east", "west"]
actions = ["look", "talk", "attack"]
items = ["sword", "axe", "dagger", "bow", "shield", "staff", "mace", "spear", "hammer", "pickaxe"]

# Directional mappings for forward/backward/left/right
direction_mappings = {
    "forward": "north",
    "backward": "south",
    "left": "west",
    "right": "east"
}

# Create synthetic data
data = []

# Generate variations of movement commands
for direction in directions:
    variations = [
        f"move {direction}",
        f"go {direction}",
        f"walk {direction}",
        f"head {direction}",
        f"travel {direction}",
        f"move to the {direction}",
        f"head towards {direction}",
    ]
    for variation in variations:
        data.append([variation, f"move {direction}"])

# Add variations for "go" and "travel" to only return "move" for directions like "forward", "backward", "left", "right"
for key, value in direction_mappings.items():
    data.append([f"go {key}", f"move {value}"])
    data.append([f"travel {key}", f"move {value}"])

# Additional "go" and "travel" cases for each direction
for direction in directions:
    data.append([f"go {direction}", f"move {direction}"])
    data.append([f"travel {direction}", f"move {direction}"])

# Generate variations of "look" action
look_variations = [
    "look around",
    "look at the room",
    "examine the room",
    "inspect the room",
    "check the surroundings",
    "look here",
    "observe",
]
for variation in look_variations:
    data.append([variation, "look"])

# Generate variations of "take" action
for item in items:
    take_variations = [
        f"take the {item}",
        f"grab the {item}",
        f"pick up the {item}",
        f"get the {item}",
        f"collect the {item}",
        f"pick the {item}",
    ]
    for variation in take_variations:
        data.append([variation, f"take {item}"])

# Generate variations of "talk" action
talk_variations = [
    "talk",
    "speak",
    "chat",
    "converse",
    "speak to someone",
    "have a conversation",
]
for variation in talk_variations:
    data.append([variation, "talk"])

# Generate variations of "attack" action
attack_variations = [
    "attack",
    "fight",
    "battle",
    "hit",
    "strike",
    "engage in combat",
    "take action against",
]
for variation in attack_variations:
    data.append([variation, "attack"])
    
data.append(["move", "move"])

    
# Write to CSV
csv_file_path = 'data.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Input", "Output"])  # header
    writer.writerows(data)
    

print(csv_file_path)

