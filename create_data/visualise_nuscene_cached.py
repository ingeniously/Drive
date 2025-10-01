import pickle
import os

file_path = '/media/seonho/34B6BDA8B6BD6ACE/AAAI/project/Drive/create_data/cached_nuscenes_info.pkl'

# Check if the file exists before trying to load it
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    try:
        # Open the file in binary read mode ('rb')
        with open(file_path, 'rb') as f:
            # Load the data from the file
            data = pickle.load(f)

        print(f"Successfully loaded data from: {file_path}")
        print("---")
        print(f"Data Type: {type(data)}")
        
        # Print a concise summary based on the assumed data type
        if isinstance(data, dict):
            print(f"Number of keys (items): {len(data)}")
            # Print the first few keys to understand the structure
            print(f"First 5 keys: {list(data.keys())[:5]}")
            # If it's a dictionary and you want a closer look at an item:
            if data:
                first_key = list(data.keys())[0]
                print(f"Structure of the first item (key: {first_key}):")
                print(f"  Type: {type(data[first_key])}")
                if isinstance(data[first_key], dict):
                    print(f"  Item keys: {list(data[first_key].keys())}")
        
        elif isinstance(data, list):
            print(f"Number of elements: {len(data)}")
            if data:
                print(f"Type of first element: {type(data[0])}")
        
        else:
            # For other types (e.g., set, str, int)
            print(f"Value (first 200 characters if string/bytes): {str(data)[:200]}")


    except pickle.UnpicklingError as e:
        print(f"Error loading pickle file (corrupted or incompatible format): {e}")
    except EOFError:
        print("Error: EOFError. The pickle file might be empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")