import os
import json
import re

def is_valid_student_dir(path):
    """
    Check if the path matches ./2005XXX/2005XXX/ pattern.
    For example: ./2005111/2005111/normalized_dataset.json
    """
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 3:
        return False
    # Get the last two directory names
    parent, student = parts[-3], parts[-2]
    # Check if both match the 2005XXX pattern
    return re.fullmatch(r'2005\d{3}', parent) and parent == student

def find_all_normalized_files(base_path="."):
    """Find all valid normalized_dataset.json files inside ./2005XXX/2005XXX/ directories only."""
    normalized_files = []
    for root, dirs, files in os.walk(base_path):
        if "normalized_dataset.json" in files:
            full_path = os.path.join(root, "normalized_dataset.json")
            if is_valid_student_dir(full_path):
                normalized_files.append(full_path)
    return normalized_files

def merge_normalized_datasets(base_path=".", output_file="merged_dataset.json"):
    all_data = []

    normalized_files = find_all_normalized_files(base_path)
    if not normalized_files:
        print("[!] No valid normalized_dataset.json files found.")
        return

    print(f"[+] Found {len(normalized_files)} valid normalized dataset files")

    for file_path in normalized_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"  - Loaded {len(data)} samples from: {file_path}")
        except Exception as e:
            print(f"[✗] Failed to load {file_path}: {e}")

    # Save merged dataset
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"[✓] Merged dataset saved to: {output_file}")
    print(f"[✓] Total samples: {len(all_data)}")

if __name__ == "__main__":
    merge_normalized_datasets()
