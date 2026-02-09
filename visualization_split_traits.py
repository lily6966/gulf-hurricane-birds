import os
import pickle
import subprocess
import numpy as np

# Run the script to get model numbers
subprocess.run(["python3", "get_model_number.py"])

for cnt in range(1, 13):
    # model_num_file = f"./data/small_map_data_{cnt}/model_number_{cnt}.pkl"
    # with open(model_num_file, "rb") as f:
    #     model_num = pickle.load(f)

    # corrupt = 0

    # for i, num in enumerate(model_num):
    #     if num == -1 or num == 0:
    #         corrupt += 1
    #         continue

    j = cnt
    visual_dir = f"./results/vis_mean"
    print(f"Processing month_{j}")

    # Step 1: Run original visualization script
    command = [
        "python3",
        "visual_PCA_continuousgroup_filtered.py", # change to visual_embedding_continuousgroup_filtered_.py for t-sne analysis, change to visual_cor_matrix_filtered.py for correlation matrix
        #"visual_embedding_continuousgroup_filtered.py",
        "--visual_dir", visual_dir,
        "--mon", str(j),
        #"--case", str(i)
    ]
    print("Running command:", " ".join(command))
    subprocess.run(command)

        
    #print("Corrupt entries:", corrupt)

