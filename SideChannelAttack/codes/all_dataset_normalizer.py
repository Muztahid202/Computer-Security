import os
import json

INPUT_SIZE = 1000  # Must match your model config

def normalize_trace(trace):
    """Normalize a single trace using min-max scaling."""
    if not trace:
        return [0.0] * INPUT_SIZE

    min_val = min(trace)
    max_val = max(trace)

    if max_val == min_val:
        return [0.0] * INPUT_SIZE

    normalized = [(x - min_val) / (max_val - min_val) for x in trace]
    return normalized[:INPUT_SIZE] + [0.0] * (INPUT_SIZE - len(normalized)) if len(normalized) < INPUT_SIZE else normalized[:INPUT_SIZE]

def normalize_all_datasets(base_path="."):
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder, folder)  # Expecting structure: ./2005XXX/2005XXX/
        dataset_path = os.path.join(folder_path, "dataset.json")

        if os.path.isfile(dataset_path):
            try:
                with open(dataset_path, "r") as f:
                    data = json.load(f)

                for entry in data:
                    entry["trace_data"] = normalize_trace(entry["trace_data"])

                normalized_path = os.path.join(folder_path, "normalized_dataset.json")
                with open(normalized_path, "w") as f:
                    json.dump(data, f, indent=2)

                print(f"[✓] Normalized: {normalized_path}")
            except Exception as e:
                print(f"[✗] Failed to normalize {dataset_path}: {e}")

if __name__ == "__main__":
    normalize_all_datasets(".")
