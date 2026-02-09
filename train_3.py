import tensorflow as tf
import numpy as np
import pandas as pd
import time
import datetime
import model_2
import get_data
import config
from sklearn.metrics import average_precision_score, roc_auc_score
from absl import flags, app
import os, sys
from scipy import stats
FLAGS = flags.FLAGS



def make_summary(name, value, writer, step):
    """Logs a scalar summary using TensorFlow 2.x summary API."""
    step = tf.convert_to_tensor(step, dtype=tf.int64)  # Ensure step is an int64 tensor
    with writer.as_default():
        tf.summary.scalar(name, value, step=step)

# def MakeSummary(name, value):
#     """Creates a tf.Summary proto with the given name and value."""
#     summary = tf.Summary()
#     val = summary.value.add()
#     val.tag = str(name)
#     val.simple_value = float(value)
#     return summary

def Species_acc(pred, Y): #the smaller the better
    return np.mean(np.abs(pred - Y))

def Species_Dis(pred, Y): #the larger the better
    res = []
    for i in range(pred.shape[1]):
        try:
            auc = roc_auc_score(Y[:, i] ,pred[:, i])
            res.append(auc)
        except:
            res.append(1.0) #print("AUC nan", i, np.mean(Y[:, i]), np.mean(pred[:, i]))
        
    return np.mean(res)

def Species_Cali(pred, Y):
    res = []
    for j in range(pred.shape[1]):
        p = pred[:, j]
        y = Y[:, j]

        bin1 = np.zeros(10)
        bin2 = np.zeros(10)
        th = np.zeros(10)

        for k in range(10):
            th[k] = np.percentile(p, (k+1)*10)

        for i in range(p.shape[0]):
            for k in range(10):
                if (p[i] <= th[k]):
                    bin1[k] += p[i]
                    bin2[k] += y[i]
                    break

        diff = np.sum(np.abs(bin1 - bin2))
        #print(bin1)
        #print(bin2)
        res.append(diff)
    return np.mean(res)

def Species_Prec(pred, Y): #the smaller the better
    return np.mean(np.sqrt(pred * (1 - pred)))

def Richness_Acc(pred, Y): #the smaller the better
    return np.sqrt(np.mean((np.sum(pred, axis = 1)-np.sum(Y, axis = 1)) ** 2))

def Richness_Dis(pred, Y): #the larger the better
    return stats.spearmanr(np.sum(pred, axis = 1), np.sum(Y, axis = 1))[0]

def Richness_Cali(pred, Y): #the smaller the better
    samples = [] #100, n, sp
    for i in range(100):
        samples.append(np.random.binomial(1, pred))
    richness = np.sum(samples, axis = 2) #100, n
    gt_richness = np.sum(Y, axis = 1)

    res = []

    for i in range(pred.shape[0]):
        if (gt_richness[i] <= np.percentile(richness[:, i], 75) and gt_richness[i] >= np.percentile(richness[:, i], 25)):
            res.append(1)
        else:
            res.append(0)
    p = np.mean(res)
    return np.abs(p - 0.5)

def Richness_Prec(pred, Y): #the smaller the better
    samples = [] #100, n, sp
    for i in range(100):
        samples.append(np.random.binomial(1, pred))

    return np.mean(np.std(np.sum(samples, axis = 2), axis = 0))

def Beta_SOR(x, y):
    if (np.sum(x * y) == 0 and np.sum(x + y) == 0):
        return 0

    return 1 - 2 * np.sum(x * y)/np.maximum(np.sum(x + y), 1e-9)

def Beta_SIM(x, y):
    if (np.sum(x * y) == 0 and np.minimum(np.sum(x), np.sum(y)) == 0):
        return 0
    return 1 - np.sum(x * y)/np.maximum(np.minimum(np.sum(x), np.sum(y)), 1e-9)

def Beta_NES(x, y):
    return Beta_SOR(x, y) - Beta_SIM(x, y)

def get_dissim(pred, Y):
    samples = [] #100, n, sp
    for i in range(100):
        samples.append(np.random.binomial(1, pred))

    pairs = []
    N = 300
    for i in range(N):
        x = np.random.randint(pred.shape[0])
        y = np.random.randint(pred.shape[0])
        pairs.append([x, y])


    SOR = np.zeros((N, 100))
    SIM = np.zeros((N, 100))
    NES = np.zeros((N, 100))

    gt_SOR = []
    gt_SIM = []
    gt_NES = []
    for i in range(N):
        x, y = pairs[i]
        for j in range(100):
            SOR[i][j] = Beta_SOR(samples[j][x], samples[j][y])
            SIM[i][j] = Beta_SIM(samples[j][x], samples[j][y])
            NES[i][j] = Beta_NES(samples[j][x], samples[j][y])

        gt_SOR.append(Beta_SOR(Y[x], Y[y]))
        gt_SIM.append(Beta_SIM(Y[x], Y[y]))
        gt_NES.append(Beta_NES(Y[x], Y[y]))
    return SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES

def Community_Acc(pred, Y): #the smaller the better
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    return np.sqrt(np.mean((np.mean(SOR, axis = 1) - gt_SOR)**2)),\
    np.sqrt(np.mean((np.mean(SIM, axis = 1) - gt_SIM)**2)),\
    np.sqrt(np.mean((np.mean(NES, axis = 1) - gt_NES)**2))

def Community_Dis(pred, Y): #the larger the better
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    return stats.spearmanr(np.mean(SOR, axis = 1), gt_SOR)[0],\
    stats.spearmanr(np.mean(SIM, axis = 1), gt_SIM)[0],\
    stats.spearmanr(np.mean(NES, axis = 1), gt_NES)[0]

def Community_Cali(pred, Y):
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    tmp1 = np.abs(np.mean(np.logical_and(np.less_equal(np.percentile(SOR, 25, axis = 1), gt_SOR),\
     np.greater_equal(np.percentile(SOR, 75, axis = 1),gt_SOR)).astype("float")) - 0.5)

    tmp2 = np.abs(np.mean(np.logical_and(np.less_equal(np.percentile(SIM, 25, axis = 1), gt_SIM),\
     np.greater_equal(np.percentile(SIM, 75, axis = 1),gt_SIM)).astype("float")) - 0.5)

    tmp3 = np.abs(np.mean(np.logical_and(np.less_equal(np.percentile(NES, 25, axis = 1), gt_NES),\
     np.greater_equal(np.percentile(NES, 75, axis = 1),gt_NES)).astype("float")) - 0.5)

    return tmp1, tmp2, tmp3

def Community_Prec(pred, Y): #the smaller the better
    SOR, SIM, NES, gt_SOR, gt_SIM, gt_NES = get_dissim(pred, Y)

    return np.mean(np.std(SOR, axis = 1)), \
    np.mean(np.std(SIM, axis = 1)), \
    np.mean(np.std(NES, axis = 1))

def train_step(hg, input_nlcd, input_label, optimizer, step, writer):
    """Performs a training step, computes gradients, and logs metrics."""

    with tf.GradientTape() as tape:
        # Forward pass with inputs
        indiv_prob, eprob, nll_loss, marginal_loss, l2_loss, total_loss, covariance = hg(
            [input_nlcd, input_label], is_training=True
        )

    # Compute gradients
    gradients = tape.gradient(total_loss, hg.trainable_variables)

    # Apply gradients
    optimizer.apply_gradients(zip(gradients, hg.trainable_variables))

    # Convert losses to scalars for easy debugging
    indiv_prob, nll_loss, marginal_loss, l2_loss, total_loss = indiv_prob.numpy(), nll_loss.numpy(), marginal_loss.numpy(), l2_loss.numpy(), total_loss.numpy()

    # Log metrics
    with writer.as_default():
        tf.summary.scalar("train/nll_loss", nll_loss, step=step)
        tf.summary.scalar("train/marginal_loss", nll_loss, step=step)
        tf.summary.scalar("train/l2_loss", l2_loss, step=step)
        tf.summary.scalar("train/total_loss", total_loss, step=step)

    return indiv_prob, nll_loss, marginal_loss, l2_loss, total_loss



def validation_step(hg, data, valid_idx, writer, step, metrics, metric_names):
    """Performs one validation step and logs the results."""
    print("Validating...")
 
    all_nll_loss, all_marginal_loss, all_l2_loss, all_total_loss = 0, 0, 0, 0
    all_preds, all_Ys = [], []

    for i in range(0, len(valid_idx)-(len(valid_idx) % FLAGS.batch_size), FLAGS.batch_size):
        batch_indices = valid_idx[i:i + FLAGS.batch_size]
    valid_log = get_data.Log()

    
    
    batch_size = FLAGS.batch_size

    
    
    for i in range(0, len(valid_idx)-(len(valid_idx) % FLAGS.batch_size), FLAGS.batch_size):
        batch_indices = valid_idx[i:i + FLAGS.batch_size]
        input_nlcd = get_data.get_nlcd(data, batch_indices)
        input_label = get_data.get_label(data, batch_indices)

        feed_dict={}
        feed_dict[hg.input_nlcd]=input_nlcd
        feed_dict[hg.input_label]=input_label
        
        #feed_dict[hg.keep_prob]=1.0
        # Forward pass (no gradient calculation during validation)
         # Forward pass (no gradient calculation during validation)
        preds, eprob, nll_loss, marginal_loss, l2_loss, total_loss, covariance = hg([input_nlcd, input_label], is_training=False)

        
        preds = np.array(preds)
        
        # Aggregate results
        all_nll_loss += nll_loss * len(batch_indices)
        all_l2_loss += l2_loss * len(batch_indices)
        all_total_loss += total_loss * len(batch_indices)
        all_marginal_loss += marginal_loss * len(batch_indices)   
        
        #all_preds = np.concatenate((all_preds, preds), axis=0)
        #all_Ys = np.concatenate((all_Ys, preds), axis=0)
        
        #all_preds = np.array(all_preds)
        
        #all_Ys = np.array(all_Ys) 
        # Compute average metrics
        mean_nll_loss = all_nll_loss / len(valid_idx)
        mean_l2_loss = all_l2_loss / len(valid_idx)
        mean_total_loss = all_total_loss / len(valid_idx)
        mean_marginal_loss = all_marginal_loss / len(valid_idx)

        # Compute average precision and AUC

        all_Ys = np.concatenate(input_label).flatten()
        all_preds = np.concatenate(preds).flatten()
        print(all_preds)
       
        ap = average_precision_score(all_Ys, all_preds)
        try:
            auc = roc_auc_score(all_Ys, all_preds)
        except ValueError:
            auc = 0.0

    # Log results to TensorBoard
    with writer.as_default():
        tf.summary.scalar("validation/ap", ap, step = step)
        tf.summary.scalar("validation/auc", auc, step = step)
        tf.summary.scalar("validation/nll_loss", mean_nll_loss, step = step)
        tf.summary.scalar("validation/marginal_loss", mean_marginal_loss, step = step)
        tf.summary.scalar("validation/l2_loss", mean_l2_loss, step = step)
        tf.summary.scalar("validation/total_loss", mean_total_loss, step = step)

    return mean_nll_loss, [preds, input_label]
    
    
    # all_nll_loss, all_marginal_loss, all_l2_loss, all_total_loss = 0, 0, 0, 0
    # all_indiv_prob, all_label = [], []

    
    # for i in range(0, len(valid_idx)-(len(valid_idx) % FLAGS.batch_size), FLAGS.batch_size):
    #     batch_indices = valid_idx[i:i + FLAGS.batch_size]

    #     input_nlcd = get_data.get_nlcd(data, batch_indices)
    #     input_label = get_data.get_label(data, batch_indices)

    #     # Forward pass (no gradient calculation during validation)
    #     indiv_prob, eprob, nll_loss, marginal_loss, l2_loss, total_loss, covariance = hg([input_nlcd, input_label], is_training=False)

    #     # Aggregate results
    #     all_nll_loss += nll_loss * len(batch_indices)
    #     all_l2_loss += l2_loss * len(batch_indices)
    #     all_total_loss += total_loss * len(batch_indices)
    #     all_marginal_loss += marginal_loss * len(batch_indices)   
    #     for ii in indiv_prob:
    #         all_indiv_prob.append(ii)
    #     for ii in input_label:
    #         all_label.append(ii)
   
    # all_indiv_prob = np.array(all_indiv_prob)
    # all_label = np.array(all_label)         
    
    # # Compute average metrics
    # mean_nll_loss = all_nll_loss / len(valid_idx)
    # mean_l2_loss = all_l2_loss / len(valid_idx)
    # mean_total_loss = all_total_loss / len(valid_idx)
    # mean_marginal_loss = all_marginal_loss / len(valid_idx)

    # # Compute average precision and AUC

    # all_indiv_prob = np.concatenate(all_indiv_prob).flatten()
    # all_label = np.concatenate(all_label).flatten()

    # ap = average_precision_score(all_label, all_indiv_prob)
    
                
    # try:
    #     auc = roc_auc_score(all_label, all_indiv_prob)
    # except ValueError:
    #     auc = 0.0

    # # Log results to TensorBoard
    # with writer.as_default():
    #     tf.summary.scalar("validation/auc", auc, step=step)
    #     tf.summary.scalar("validation/ap", ap, step=step)
    #     tf.summary.scalar("validation/nll_loss", mean_nll_loss, step=step)
    #     tf.summary.scalar("validation/marginal_loss", mean_marginal_loss, step=step)
    #     tf.summary.scalar("validation/l2_loss", mean_l2_loss, step=step)
    #     tf.summary.scalar("validation/total_loss", mean_total_loss, step=step)

    # return mean_nll_loss

def main(_):
    st_time = time.time()
    print('Reading npy...')
    np.random.seed(19950420)

    data = np.load(FLAGS.data_dir, allow_pickle=True)
    train_idx = np.load(FLAGS.train_idx)
    valid_idx = np.load(FLAGS.valid_idx)

    labels = get_data.get_label(data, train_idx)
    print("Label distribution: ", np.mean(labels))

    one_epoch_iter = len(train_idx) // FLAGS.batch_size
    print('Reading completed')
    metrics = [Species_acc, Species_Dis, Species_Cali, Species_Prec, \
                Richness_Acc, Richness_Dis, Richness_Cali, Richness_Prec, 
                Community_Acc, Community_Dis, Community_Cali, Community_Prec]

    metric_names = ["Species_acc", "Species_Dis", "Species_Cali", "Species_Prec", \
                 "Richness_Acc", "Richness_Dis", "Richness_Cali", "Richness_Prec", \
                 "Community_Acc", "Community_Dis", "Community_Cali", "Community_Prec"]
    # GPU memory configuration
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Building the model
    print('Building model...')
    hg = model_2.MODEL(is_training=True)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Learning rate schedule
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True
    )

    # Optimizer setup
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    global_step = optimizer.iterations

    # Ensure checkpoint directory exists
    checkpoint_dir = FLAGS.model_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=hg)
    
    print ('building finished')
    # TensorBoard summary writer
    summary_writer = tf.summary.create_file_writer(FLAGS.summary_dir)

    best_loss = float('inf')
    current_step = global_step.numpy()  # Convert to integer
    best_loss = float('inf')
    best_epoch = 0
    drop_cnt = 0
    max_epoch = FLAGS.max_epoch
    step = 0
    for epoch in range(FLAGS.max_epoch):
        print(f'Epoch {epoch + 1} starts!')

        np.random.shuffle(train_idx)

        smooth_nll_loss = 0.0
        smooth_marginal_loss = 0.0
        smooth_l2_loss = 0.0
        smooth_total_loss = 0.0
        temp_label = []
        temp_indiv_prob = []

        for i in range(one_epoch_iter):
            start = i * FLAGS.batch_size
            end = start + FLAGS.batch_size

            input_nlcd = get_data.get_nlcd(data, train_idx[start:end])
            input_label = get_data.get_label(data, train_idx[start:end])
            
            # Perform training step
            indiv_prob, nll_loss, marginal_loss, l2_loss, total_loss = train_step(hg, input_nlcd, input_label, optimizer, current_step, summary_writer)

            # Update smooth losses
            smooth_nll_loss += nll_loss
            smooth_marginal_loss += marginal_loss
            smooth_l2_loss += l2_loss
            smooth_total_loss += total_loss

            # Store labels and predictions for AP & AUC
            temp_label.append(input_label)
            temp_indiv_prob.append(indiv_prob)
            
            
            # Log progress every `check_freq` iterations
            if (i + 1) % FLAGS.check_freq == 0:
                mean_nll_loss = smooth_nll_loss / FLAGS.check_freq
                mean_marginal_loss = smooth_marginal_loss/FLAGS.check_freq
                mean_l2_loss = smooth_l2_loss / FLAGS.check_freq
                mean_total_loss = smooth_total_loss / FLAGS.check_freq
                
                # Flatten and reshape predictions and labels
                temp_indiv_prob = np.reshape(np.array(temp_indiv_prob), (-1))
                temp_label = np.reshape(np.array(temp_label), (-1))

                # Compute AP & AUC
                ap = average_precision_score(temp_label, temp_indiv_prob.reshape(-1, 1))

                try:
                    auc = roc_auc_score(temp_label, temp_indiv_prob)
                except ValueError:
                    print('Warning: AUC computation failed due to label mismatch.')
                    auc = None 

                
                # Write to TensorBoard
                with summary_writer.as_default():
                    tf.summary.scalar('train/ap', ap, step=current_step)
                    if auc is not None:
                        tf.summary.scalar('train/auc', auc, step=current_step)
                    tf.summary.scalar('train/nll_loss', mean_nll_loss, step=current_step)
                    tf.summary.scalar('train/marginal_loss', mean_marginal_loss, step=current_step)
                    tf.summary.scalar('train/l2_loss', mean_l2_loss, step=current_step)
                    tf.summary.scalar('train/total_loss', mean_total_loss, step=current_step)
                
                time_str = datetime.datetime.now().isoformat()

                print ("train step: %s\tap=%.6f\tnll_loss=%.6f\tmarginal_loss=%.6f\tl2_loss=%.6f\ttotal_loss=%.6f" % (time_str, ap, nll_loss, marginal_loss, l2_loss, total_loss))
                #print ("validation results: ap=%.6f\tnll_loss=%.6f\tmarginal_loss=%.6f\tl2_loss=%.6f\ttotal_loss=%.6f" % (ap, nll_loss, mean_marginal_loss, l2_loss, total_loss))

                # Reset accumulators
                temp_indiv_prob = []
                temp_label = []
                smooth_nll_loss = 0.0
                smooth_marginal_loss = 0.0
                smooth_l2_loss = 0.0
                smooth_total_loss = 0.0

        save_epoch = 1
        # Validation
        if (epoch + 1) % save_epoch == 0:
            # Evaluate on test set
            Res = []
            test_loss, (preds, Ys) = validation_step(hg, data, valid_idx, summary_writer, epoch, metrics, metric_names)
            
            for i in range(len(metrics)):
                f = metrics[i]
                name = metric_names[i]
                res = (name, f(preds, Ys))
                print(res)
                if (isinstance(res[1], tuple)):
                    for x in res[1]:
                        Res.append(x)
                else:
                    Res.append(res[1])
            
            
            
            print(f"Epoch {epoch+1} validation loss: {test_loss}")
            
            # Save if improved
            if test_loss < best_loss:
                print(f"New best loss: {test_loss} (previous: {best_loss})")
                best_loss = test_loss
                best_iter = epoch
                best_res = [preds, Ys]
                checkpoint.save(file_prefix=checkpoint_prefix)

                drop_count = 0
            else:
                drop_count += 1
            
            # Early stopping
            if drop_count > 10:
                print("Early stopping triggered")
                break

    print('Training completed!')
    print(f'Best validation loss: {best_loss}')
    print ('the best checkpoint is '+str(best_iter))
    ed_time = time.time()
    print ("running time: ", ed_time - st_time)   
    (preds, Ys) = best_res
    Res = []
    for i in range(len(metrics)):
        f = metrics[i]
        name = metric_names[i]
        res = (name, f(preds, Ys))
        print(res)
        if (isinstance(res[1], tuple)):
            for x in res[1]:
                Res.append(x)
        else:
            Res.append(res[1])

    np.save("results/%s_%s"%(sys.argv[1], sys.argv[2]), Res)
    

if __name__ == '__main__':
    app.run(main)
