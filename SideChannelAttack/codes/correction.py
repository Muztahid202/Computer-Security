# import json

# # Path to the merged dataset
# DATASET_PATH = "merged_dataset.json"

# # Load the dataset
# with open(DATASET_PATH, 'r') as f:
#     data = json.load(f)

# # Extract unique website URLs
# unique_websites = sorted(set(entry["website"] for entry in data))

# # Print the results
# print("Unique websites found in the dataset:")
# for idx, site in enumerate(unique_websites):
#     print(f"{idx}: {site}")




import json

# Path to the target file
path = "./2005070/2005070/dataset.json"

# Correct mapping
url_to_index = {
    "https://cse.buet.ac.bd/moodle/": 0,
    "https://google.com": 1,
    "https://prothomalo.com": 2,
}

# Load the dataset
with open(path, "r") as f:
    data = json.load(f)

# Fix indices
for entry in data:
    website = entry.get("website")
    if website in url_to_index:
        entry["website_index"] = url_to_index[website]
    else:
        print(f"⚠️ Unknown website URL: {website}")

# Save corrected data (overwrite)
with open(path, "w") as f:
    json.dump(data, f, indent=4)

print("✅ Fixed website_index values and saved to", path)
