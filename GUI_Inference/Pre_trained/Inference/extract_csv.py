import csv
import json

def csv_to_json(csv_path, json_path):
    """
    Converts a CSV file (id, label) to a JSON file {id: label}.
    
    Args:
        csv_path (str): Path to the input CSV file.
        json_path (str): Path to save the output JSON file.
    """
    label_dict = {}
    
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            label_dict[row['id']] = row['name']
    
    with open(json_path, 'w') as json_file:
        json.dump(label_dict, json_file, indent=4)
    
    print(f"Successfully converted {csv_path} to {json_path}")

if __name__ == '__main__':
    # Replace these paths with your actual file paths
    csv_path = 'labels.csv'    # Input CSV file (format: id,label)
    json_path = 'kinetics_labels.json'  # Output JSON file
    
    csv_to_json(csv_path, json_path)