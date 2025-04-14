import json
import os

def extract_labels_from_txt(txt_path, output_json_path):
    
    label_dict = {}
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                video_path = parts[0]  
                class_idx = parts[1]   
                
                class_name = video_path.split('/')[0]
                
                label_dict[class_idx] = class_name
    
    # Save to JSON
    with open(output_json_path, 'w') as f:
        json.dump(label_dict, f, indent=4)
    
    print(f"Saved labels to {output_json_path}")

if __name__ == '__main__':
    txt_path = 'temporal-shift-module/tools/kinetics_label_map.txt'  
    output_json_path = 'kinetics_labels.json'         
    
    extract_labels_from_txt(txt_path, output_json_path)