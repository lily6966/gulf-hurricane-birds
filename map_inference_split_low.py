import os
import pickle

os.system("python3 get_model_number.py")

for cnt in range(1, 13):
    with open("./data/small_map_data_%d/model_number_%d.pkl" % (cnt, cnt), "rb") as f:
        model_num = pickle.load(f)
        print(f"Loaded {len(model_num)} model numbers: {model_num[:10]}")
    #assert (len(model_num) == 482)

    cur = -1
    corrupt = 0

    for i in range(len(model_num)):
        print(model_num[i])
        if model_num[i] in (-1, 0):  # Check for corrupt data
            corrupt += 1
            cur += 1  # Move index only for valid models
            continue
        

        for j in range(1):
            j = cnt 
            cur += 1
            print (cur)
            j_name = "test_bird_realworld_%d_%d" % (i, j)
            command = "python3 inference_low.py --checkpoint_path ./data/small_model_mon%d/model_%d_%d/model-%d.weights.h5 --summary_dir ./summary_%d_%d --visual_dir ./vis_%d_%d --data_dir ./data/small_esrd_low/small_esrd_%d_%d.npy --test_idx ./data/small_region/small_case%d_bird_test_idx_%d.npy --r_dim 332 --r_max_dim 332 --mon %d --case %d | tee ./data/small_map_data_%d/%s" % (cnt, i, j, model_num[cur], i, j, i, j, i, j, i, j, j, i, cnt, j_name)
            print(f"Executing command: {command}")

            print (command)
            os.system(command)

    #print ("corrupt: ", corrupt)
