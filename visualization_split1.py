import os
import pickle
import subprocess

# Run the script to get model numbers
subprocess.run(["python3", "get_model_number.py"])

for cnt in range(1, 13):
    # Load the model number list
    model_num_file = f"./data/small_map_data_{cnt}/model_number_{cnt}.pkl"
    with open(model_num_file, "rb") as f:
        model_num = pickle.load(f)

    corrupt = 0

    for i, num in enumerate(model_num):
        print(num)
        if num == -1 or num == 0:
            corrupt += 1
            continue

        j = cnt
        print(f"Processing model {i}, j = {j}")

        command = [
            "python3",
            "visual_embedding_traits.py",
            "--visual_dir", f"./vis_{i}_{j}",
            "--mon", str(j),
            "--case", str(i)
        ]
        print("Running command:", " ".join(command))

        subprocess.run(command)

    print("Corrupt entries:", corrupt)
