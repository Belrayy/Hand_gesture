import json

gesture_list = [
    "Swiping Left",
    "Swiping Right",
    "Swiping Down",
    "Swiping Up",
    "Pushing Hand Away",
    "Pulling Hand In",
    "Sliding Two Fingers Left",
    "Sliding Two Fingers Right",
    "Sliding Two Fingers Down",
    "Sliding Two Fingers Up",
    "Pushing Two Fingers Away",
    "Pulling Two Fingers In",
    "Rolling Hand Forward",
    "Rolling Hand Backward",
    "Turning Hand Clockwise",
    "Turning Hand Counterclockwise",
    "Zooming In With Full Hand",
    "Zooming Out With Full Hand",
    "Zooming In With Two Fingers",
    "Zooming Out With Two Fingers",
    "Thumb Up",
    "Thumb Down",
    "Shaking Hand",
    "Stop Sign",
    "Drumming Fingers",
    "No gesture",
    "Doing other things"
]

# Create a dictionary with numerical keys (as strings)
gesture_dict = {}
for index, gesture in enumerate(gesture_list):
    gesture_dict[str(index)] = gesture

# Specify the filename for the JSON file
filename = "gestures.json"

# Open the file in write mode ('w')
with open(filename, 'w') as f:
    # Use json.dump() to write the dictionary to the file
    json.dump(gesture_dict, f, indent=4)

print(f"The list has been successfully transferred to '{filename}' in the desired JSON format.")