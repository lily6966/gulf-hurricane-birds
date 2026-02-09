import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import datetime
import get_data 
import pandas as pd
import config 
import os
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math
import urllib
from pyheatmap.heatmap import HeatMap
import seaborn as sns
from absl import flags, app
import os
import model_2
FLAGS = flags.FLAGS



def calculate_mdt_threshold(species, indiv_prob, input_label, n_thresholds=100, printNow=False):
    thresholds = np.linspace(0, 1, n_thresholds)
    best_thresh = 0
    min_diff = float('inf')
    # Ensure both classes are present
    if len(set(input_label.flatten())) == 1:

        print("Only one class present in input_label. Skipping ROC AUC calculation.")
        return None  # or handle it differently if you want
    for t in thresholds:
                pred_label = (indiv_prob >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(input_label, pred_label).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                diff = abs(sensitivity - specificity)

                if diff < min_diff:
                    min_diff = diff
                    best_thresh = t

    
    print(f"Best threshold for species #{species}: {best_thresh:.4f} with min diff {min_diff:.4f}")
    # Vectorized computation of TP, FP, TN, FN
    pred_label = (indiv_prob > best_thresh).astype(int)

    TP = np.sum((pred_label == 1) & (input_label == 1))
    FP = np.sum((pred_label == 1) & (input_label == 0))
    TN = np.sum((pred_label == 0) & (input_label == 0))
    FN = np.sum((pred_label == 0) & (input_label == 1))

    

    

       
            
    # Metrics calculation
    total = TP + TN + FP + FN
    eps = 1e-6
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    accuracy = (TP + TN) / total
    F1 = 2 * precision * recall / (precision + recall + eps)
    auc = roc_auc_score(input_label, indiv_prob)

    # Reshape for AUC and average precision score calculation
    indiv_prob = np.reshape(indiv_prob, (-1))
    input_label = np.reshape(input_label, (-1))

    new_auc = roc_auc_score(input_label, indiv_prob)
    ap = average_precision_score(input_label, indiv_prob)

    if printNow:
        print(f"\nAnalysis of species #{species}:")
        print(f"Occurrence rate: {np.mean(input_label)}")
        print(f"Overall AUC: {auc:.6f}, New AUC: {new_auc:.6f}, AP: {ap:.6f}")
        print(f"F1: {F1}, Accuracy: {accuracy}")
        print(f"Precision: {precision}, Recall: {recall}")
        print(f"TP={TP/total:.6f}, TN={TN/total:.6f}, FN={FN/total:.6f}, FP={FP/total:.6f}")
    
    return {
    "species_index": species,
    "threshold": best_thresh,
    "occurrence_rate": np.mean(input_label),
    "AUC": auc,
    "new_AUC": new_auc,
    "AP": ap,
    "F1": F1,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "TP_rate": TP / total,
    "TN_rate": TN / total,
    "FP_rate": FP / total,
    "FN_rate": FN / total
    }


def test_step(classifier, data, test_idx):
    print('Testing...')
    all_nll_loss, all_l2_loss, all_total_loss = 0, 0, 0
    all_indiv_prob, all_label = [], []
    prob_res, loc_res = [], []

    real_batch_size = min(FLAGS.testing_size, len(test_idx))
    N_test_batch = (len(test_idx) - 1) // real_batch_size + 1

    for i in range(N_test_batch):
        print(f"{(i * 100.0 / N_test_batch):.1f}% completed")

        start, end = real_batch_size * i, min(real_batch_size * (i + 1), len(test_idx))

        # Get data for the current batch
        input_loc = get_data.get_loc(data, test_idx[start:end])
        input_nlcd = get_data.get_nlcd(data, test_idx[start:end])
        input_label = get_data.get_label(data, test_idx[start:end])

        # Run model inference
        indiv_prob, prob_res_sample, nll_loss, marginal_loss, l2_loss, total_loss, covariance = classifier([input_nlcd, input_label], training=False)

        # Save individual predictions and locations
        for ii in range(indiv_prob.shape[0]):
            prob_res.append(indiv_prob[ii].numpy())  # Convert Tensor to NumPy
            loc_res.append(input_loc[ii])

        last_prob = prob_res[-1]

        # Convert Tensor to numpy
        last_sample = prob_res_sample.numpy() if hasattr(prob_res_sample, 'numpy') else prob_res_sample

        # If it's scalar (0D tensor), broadcast it to match shape
        if np.isscalar(last_sample) or np.ndim(last_sample) == 0:
            # Broadcast scalar to same shape as last_prob
            last_sample = np.full_like(last_prob, float(last_sample))

        # Now you can safely compute MSE
        ttt = [(float(a) - float(b)) ** 2 for a, b in zip(last_prob, last_sample)]
        print(np.mean(ttt))
        

        # Accumulate losses
        all_nll_loss += nll_loss * (end - start)
        all_l2_loss += l2_loss * (end - start)
        all_total_loss += total_loss * (end - start)

        # Accumulate probabilities and labels
        all_indiv_prob.append(indiv_prob.numpy())
        all_label.append(input_label)

    # Average losses
    nll_loss = all_nll_loss / len(test_idx)
    l2_loss = all_l2_loss / len(test_idx)
    total_loss = all_total_loss / len(test_idx)

    print(f"Performance on test set: nll_loss={nll_loss:.6f}, l2_loss={l2_loss:.6f}, total_loss={total_loss:.6f}")
    print(datetime.datetime.now().isoformat())

    os.makedirs(FLAGS.visual_dir, exist_ok=True)
    np.save(os.path.join(FLAGS.visual_dir, "cov.npy"), covariance)

    return np.concatenate(all_indiv_prob), np.concatenate(all_label), prob_res, loc_res



def main(_):
    print('Reading npy...')
    data = np.load(FLAGS.data_dir, allow_pickle=True)
    test_idx = np.load(FLAGS.test_idx) if "esrd" not in FLAGS.data_dir else list(range(data.shape[0]))
    print('Reading completed')

    # Build model using Keras
    classifier = model_2.MODEL(is_training=False)
    # ---------------Dummy inputs to build the model(Make sure the dimension is aligned with data)---------------
    dummy_input_nlcd = tf.zeros((1, 34))  # Replace with actual input shape
    dummy_input_label = tf.zeros((1, 332))   # Replace with actual label shape
    _ = classifier((dummy_input_nlcd, dummy_input_label), is_training=False)

    # Restore model weights (must be .weights.h5 or compatible)
    classifier.load_weights(FLAGS.checkpoint_path)
    print(f'Restored weights from: {FLAGS.checkpoint_path}')

    # Extract feature embedding from layer 'r_mu'
    feature_embedding = classifier.get_layer("r_mu").get_weights()[0]
    feature_embedding = np.transpose(feature_embedding)

    # Save the feature embedding
    os.makedirs("./results/thresholds/", exist_ok=True)
    #np.save(os.path.join(FLAGS.visual_dir, f"feature_emb_{FLAGS.mon}"), feature_embedding)

    # Read species names
    file_path = "./data/ebird_occurance_habitat_10_withyear.csv"
    try:
        with open(file_path, "r") as f:
            spe_name = f.readline().strip().split(",")[36:]
        assert len(spe_name) == 332
        print(spe_name[:10])  # Check first 10 species for sanity
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return
    except AssertionError:
        print(f"Error: The species list doesn't have 332 species.")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

    # Perform testing and evaluation
    all_indiv_prob, all_label, prob_res, loc_res = test_step(classifier, data, test_idx)

    summary_metrics = []
    thresholds = [None] * FLAGS.r_dim
    for i in range(FLAGS.r_dim):
        res = calculate_mdt_threshold(
            species=i,
            indiv_prob=all_indiv_prob[:, i],
            input_label=all_label[:, i],
            printNow=True
        )
        if res is not None:
            thresholds[i] = res["threshold"]
            res["species_name"] = spe_name[i]
            summary_metrics.append(res)
        else:
        # Still include the species name and index with NaNs
            res = {
                "species_index": i,
                "species_name": spe_name[i],
                "threshold": None,
                "occurrence_rate": None,
                "AUC": None,
                "new_AUC": None,
                "AP": None,
                "F1": None,
                "Accuracy": None,
                "Precision": None,
                "Recall": None,
                "TP_rate": None,
                "TN_rate": None,
                "FP_rate": None,
                "FN_rate": None
            }
            summary_metrics.append(res)
    
    # Step 1: Create the tidy DataFrame
    summary_df = pd.DataFrame(summary_metrics)

    # Step 2: Set species_name as columns and metric names as rows
    # First, drop species_index and set species_name as column names
    summary_df = summary_df.drop(columns=["species_index"])
    summary_df = summary_df.set_index("species_name").transpose()

    # Step 3: Save this transposed version
    summary_df.to_csv(f"./results/thresholds/mdt_thresholds_transposed_case_{FLAGS.case}_month_{FLAGS.mon}.csv")
    # # Save the prediction results to CSV
    # with open("./results/thresholds/mdt_thresholds_case_%d_month_%d.csv" % (FLAGS.case, FLAGS.mon), "w") as f:
    #     f.write(",".join(spe_name) + "\n")
    #     f.write(",".join([f"{t:.6f}" if t is not None else "" for t in thresholds]) + "\n")



    # summary_df = pd.DataFrame(summary_metrics)
    # summary_df = summary_df[["species_index", "species_name", "threshold", "occurrence_rate", "AUC", 
    #                         "new_AUC", "AP", "F1", "Accuracy", "Precision", "Recall", 
    #                         "TP_rate", "TN_rate", "FP_rate", "FN_rate"]]

    # summary_df.to_csv(f"./results/thresholds/mdt_summary_case_{FLAGS.case}_month_{FLAGS.mon}.csv", index=False)

    
    

if __name__ == '__main__':
    app.run(main)