import pickle

file_path = "./data/ebird_occurance_habitat_10_noyear.csv"
try:
    with open(file_path, "r") as f:
        spe_name = f.readline().strip().split(",")[36:]
    assert len(spe_name) == 332
    
    # Create bird_dict: index → species name
    bird_dict = {i: name for i, name in enumerate(spe_name)}

    # Optionally save to esrd_trans.pkl
    with open("./data/esrd_trans.pkl", "wb") as f:
        pickle.dump(bird_dict, f)
    print("bird_dict saved to ./data/esrd_trans.pkl")

except FileNotFoundError:
    print(f"Error: {file_path} not found.")
except AssertionError:
    print(f"Error: The species list doesn't have 404 species.")
except Exception as e:
    print(f"Unexpected error: {e}")