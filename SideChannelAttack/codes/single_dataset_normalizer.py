import json

INPUT_FILE = "./2005070/2005070/dataset.json"
OUTPUT_FILE = "./2005070/2005070/normalized_dataset.json"
INPUT_SIZE = 1000  # Ensure consistency with training code

def normalize_trace(trace):
    """Normalize a single trace using min-max scaling."""
    if not trace:
        return [0.0] * INPUT_SIZE
    
    min_val = min(trace)
    max_val = max(trace)

    # Avoid division by zero if all values are equal
    if max_val == min_val:
        return [0.0] * INPUT_SIZE

    normalized = [(x - min_val) / (max_val - min_val) for x in trace]

    # Ensure padding/truncation to INPUT_SIZE
    if len(normalized) < INPUT_SIZE:
        return normalized + [0.0] * (INPUT_SIZE - len(normalized))
    return normalized[:INPUT_SIZE]

def normalize_dataset(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    for entry in data:
        entry["trace_data"] = normalize_trace(entry["trace_data"])

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[✓] Normalized dataset saved to: {output_path}")

if __name__ == "__main__":
    normalize_dataset(INPUT_FILE, OUTPUT_FILE)
