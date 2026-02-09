import numpy as np
import pickle
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.manifold import TSNE
import os
import config
from absl import flags, app
import pandas as pd
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import zscore


FLAGS = flags.FLAGS

eps = 1e-9

def tSNE(X):
    import numpy as Math
    import pylab as Plot
    
    def Hbeta(D = Math.array([]), beta = 1.0):
        """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""
    
        # Compute P-row and corresponding perplexity
        #print (np.max(-D.copy() * beta), np.min(-D.copy() * beta))
        P = Math.exp(-D.copy() * beta);
        sumP = sum(P + eps);
        H = Math.log(sumP) + beta * Math.sum(D * P) / sumP;
        P = P / sumP;
        return H, P;
    
    
    def x2p(X = Math.array([]), tol = 1e-5, perplexity = 20.0):
        """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""
    
        # Initialize some variables
        print("Computing pairwise distances...")
        (n, d) = X.shape;
        sum_X = Math.sum(Math.square(X), 1);

        #D =  -Math.dot(X, X.T) #   negative inner product as the distance 
        D = Math.add(Math.add(-2 * Math.dot(X, X.T), sum_X).T, sum_X) # L2 distance


        P = Math.zeros((n, n));
        beta = Math.ones((n, 1));
        logU = Math.log(perplexity);
    
        # Loop over all datapoints
        for i in range(n):
    
            # Print progress
            if i % 500 == 0:
                print("Computing P-values for point ", i, " of ", n, "...")
    
            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -Math.inf;
            betamax =  Math.inf;
            Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))];
            (H, thisP) = Hbeta(Di, beta[i]);
    
            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU;
            tries = 0;
            while Math.abs(Hdiff) > tol and tries < 50:
    
                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].copy();
                    if betamax == Math.inf or betamax == -Math.inf:
                        beta[i] = beta[i] * 2;
                    else:
                        beta[i] = (beta[i] + betamax) / 2;
                else:
                    betamax = beta[i].copy();
                    if betamin == Math.inf or betamin == -Math.inf:
                        beta[i] = beta[i] / 2;
                    else:
                        beta[i] = (beta[i] + betamin) / 2;
    
                # Recompute the values
                (H, thisP) = Hbeta(Di, beta[i]);
                Hdiff = H - logU;
                tries = tries + 1;
    
            # Set the final row of P
            P[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))] = thisP;
    
        # Return final P-matrix
        print("Mean value of sigma: ", Math.mean(Math.sqrt(1 / beta)));
        return P;
    
    
    def pca(X = Math.array([]), no_dims = 50):
        """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""
    
        print("Preprocessing the data using PCA...")
        (n, d) = X.shape;
        X = X - Math.tile(Math.mean(X, 0), (n, 1));
        (l, M) = Math.linalg.eig(Math.dot(X.T, X));
        print("singular values:\n", l)
        Y = Math.dot(X, M[:,0:no_dims]);
        return Y;
    
    
    def tsne(X = Math.array([]), no_dims = 2, initial_dims = 50, perplexity = 40.0, max_iter=1000):
        """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
        The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""
    
        # Check inputs
        if isinstance(no_dims, float):
            print("Error: array X should have type float.")
            return -1;
        if round(no_dims) != no_dims:
            print("Error: number of dimensions should be an integer.")
            return -1;
    
        # Initialize variables
        X = pca(X, initial_dims).real;
        (n, d) = X.shape;
        initial_momentum = 0.5;
        final_momentum = 0.8;
        eta = 50;
        min_gain = 0.01;
        Y = Math.random.randn(n, no_dims);
        dY = Math.zeros((n, no_dims));
        iY = Math.zeros((n, no_dims));
        gains = Math.ones((n, no_dims));
    
        # Compute P-values
        P = x2p(X, 1e-5, perplexity);
        P = P + Math.transpose(P);
        P = P / Math.sum(P);
        P = P * 4;                                    # early exaggeration
        P = Math.maximum(P, 1e-12);
    
        # Run iterations
        for iter in range(max_iter):
    
            # Compute pairwise affinities
            sum_Y = Math.sum(Math.square(Y), 1);
            num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y));
            num[range(n), range(n)] = 0;
            Q = num / Math.sum(num);
            Q = Math.maximum(Q, 1e-12);
    
            # Compute gradient
            PQ = P - Q;
            for i in range(n):
                dY[i,:] = Math.sum(Math.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);
    
            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
            gains[gains < min_gain] = min_gain;
            iY = momentum * iY - eta * (gains * dY);
            Y = Y + iY;
            Y = Y - Math.tile(Math.mean(Y, 0), (n, 1));
    
            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                C = Math.sum(P * Math.log(P / Q));
                print("Iteration ", (iter + 1), ": error is ", C)
    
            # Stop lying about P-values
            if iter == 100:
                P = P / 4;
    
        # Return solution
        return Y;

    return tsne(X, 2, 80, 30, 2000) 

def correlation(cov, show=False):
    indices = []
    sigma = np.zeros((89,89))
    for i in range(89):
        for j in range(89):
            sigma[i][j]=cov[i][j]/np.sqrt(cov[i][i]*cov[j][j])
            indices.append((i,j))
    plt.imshow(sigma, cmap='jet', interpolation='nearest')
    if (show):
        plt.show()
    return sigma, indices

def compute_global_cmap_range(L_all_months):
    """Given a list of L matrices (one per month), compute global symmetric vmin and vmax."""
    all_off_diag_vals = []
    for L in L_all_months:
        # Normalize only off-diagonal elements if needed
        mask = ~np.eye(L.shape[0], dtype=bool)
        vals = L[mask]
        all_off_diag_vals.extend(vals.flatten())
    
    # Use symmetric color scale centered at 0
    min_val = np.min(all_off_diag_vals)
    max_val = np.max(all_off_diag_vals)
    abs_max = max(abs(min_val), abs(max_val))
    return -abs_max, abs_max



def normalize_off_diagonal(L):
    L = L.copy()  # avoid modifying original
    diag = np.diag(L)  # save diagonal

    # Create a mask for off-diagonal elements
    off_diag_mask = ~np.eye(L.shape[0], dtype=bool)

    # Extract off-diagonal values and z-score them
    off_diag_vals = L[off_diag_mask]
    norm_vals = np.sign(off_diag_vals) * np.abs(zscore(off_diag_vals))

    # Put normalized values back into L
    L[off_diag_mask] = norm_vals
    np.fill_diagonal(L, diag)  # restore original diagonal
    return L

def show_embed(L, ele_dict):
    plt.clf()
    #X_embedded = TSNE(n_components=2, init='pca', perplexity = 8.0).fit_transform(L) 

    X_embedded = tSNE(L)
    x_min, x_max = X_embedded.min(0), X_embedded.max(0)
    X_embedded = (X_embedded - x_min) / (x_max - x_min)
    #ele_color = np.load("ele_color.npy").item()
 

    for i in range(X_embedded.shape[0]):
        x = X_embedded[i][0]
        y = X_embedded[i][1]
        #print(x, y, ele_dict[i])
        plt.text(x, y, ele_dict[i], fontsize=8, ha='center', va='center')
        key = str(ele_dict[i])
        #print(key)
        #plt.plot(x, y, 'o', ms=20, c=ele_color[key])
        plt.plot(x, y, 'o', ms=10)
    plt.savefig(os.path.join(FLAGS.visual_dir, "small_feature_emb.jpg"))
    #plt.savefig( "./cor_emb.jpg")
    #plt.show()

def show_embed_correlation(L, ele_dict, mon, trait_file="./results/AVONET2_eBird.xlsx"):
    trait = "Trofic.Level"
    plt.clf()

    # Step 1: t-SNE embedding
    X_embedded = TSNE(n_components=2, perplexity=8, max_iter=1000, init='pca', random_state=42).fit_transform(L)
    #X_embedded = tSNE(L)
    x_min, x_max = X_embedded.min(0), X_embedded.max(0)
    X_embedded = (X_embedded - x_min) / (x_max - x_min)

    # Step 2: Load trait data
    traits_df = pd.read_excel(trait_file, engine='openpyxl')[['Species2', trait]]
    traits_df = traits_df.dropna(subset=['Species2', trait])

    # Step 3: Map species scientific name to common name
    taxonomy = pd.read_csv("./results/ebird-taxonomy.csv")
    sci_to_common = dict(zip(taxonomy['scientific_name'], taxonomy['common_name']))
    traits_df['Species2'] = traits_df['Species2'].map(sci_to_common)

    # 

    # Create mapping: species name → category
    species_to_group = dict(zip(traits_df['Species2'], traits_df[trait]))
    
    # Identify unique categories and assign colors
    unique_groups = sorted(traits_df[trait].unique())
    cmap = plt.get_cmap('tab20', len(unique_groups))  # Up to 20 distinct colors
    group_to_color = {group: cmap(i) for i, group in enumerate(unique_groups)}
    
    # Step 4: Bin Wing.Length trait into intervals
    # num_bins = 8
    # traits_df['TraitBin'] = pd.cut(traits_df[trait], bins=num_bins)
    # # Step 4: Bin trait values
    # num_bins = 8
    # traits_df['TraitBin'] = pd.cut(traits_df[trait], bins=num_bins)

    # # Create species to bin label mapping
    # species_to_group = dict(zip(traits_df['Species2'], traits_df['TraitBin']))

    # # Step 5: Assign colors to bins
    # unique_groups = sorted(traits_df['TraitBin'].dropna().unique())
    # cmap = plt.get_cmap('tab10', len(unique_groups))
    # group_to_color = {group: cmap(i) for i, group in enumerate(unique_groups)}

    # Step 6: Plot each point
    for i in range(X_embedded.shape[0]):
        x, y = X_embedded[i]
        species = ele_dict[i]
        
        key = str(ele_dict[i])
        #print(key)
        group = species_to_group.get(species, "Unknown")
        color = group_to_color.get(group, 'gray')
        plt.plot(x, y, 'o', ms=8, color=color)
        plt.text(x, y, ele_dict[i], fontsize=8, ha='center', va='center')
    # Step 7: Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=grp,
                   markerfacecolor=group_to_color[grp], markersize=8)
        for grp in unique_groups
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title=trait)

    # Step 8: Save the figure
    os.makedirs(FLAGS.visual_dir, exist_ok=True)
    out_path = os.path.join(FLAGS.visual_dir, f"TrophicL_cor_emb_month_{mon}.pdf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format='pdf')
    print(f"Saved t-SNE plot to {out_path}")


def show_embed_feature(L, ele_dict, mon, trait_file="./results/AVONET2_eBird.xlsx"):
    trait = "Habitat"
    plt.clf()

    # Step 1: Embed
    X_embedded = TSNE(n_components=2, perplexity=8, max_iter=1000, init='pca', random_state=42).fit_transform(L)
    #X_embedded = tSNE(L)
    x_min, x_max = X_embedded.min(0), X_embedded.max(0)
    X_embedded = (X_embedded - x_min) / (x_max - x_min)

    # Step 2: Load trait data
    traits_df = pd.read_excel(trait_file, engine='openpyxl')[['Species2', trait]]
    traits_df = traits_df.dropna(subset=['Species2', trait])

    # Step 3: Map to common names
    taxonomy = pd.read_csv("./results/ebird-taxonomy.csv")
    sci_to_common = dict(zip(taxonomy['scientific_name'], taxonomy['common_name']))
    traits_df['Species2'] = traits_df['Species2'].map(sci_to_common)

    # Create mapping: species name → category
    species_to_group = dict(zip(traits_df['Species2'], traits_df[trait]))
    
    # Identify unique categories and assign colors
    unique_groups = sorted(traits_df[trait].unique())
    cmap = plt.get_cmap('tab20', len(unique_groups))  # Up to 20 distinct colors
    group_to_color = {group: cmap(i) for i, group in enumerate(unique_groups)}

    # # Step 4: Bin trait values
    # num_bins = 8
    # traits_df['TraitBin'] = pd.cut(traits_df[trait], bins=num_bins)

    # # Create species to bin label mapping
    # species_to_group = dict(zip(traits_df['Species2'], traits_df['TraitBin']))

    # # Step 5: Assign colors to bins
    # unique_groups = sorted(traits_df['TraitBin'].dropna().unique())
    # cmap = plt.get_cmap('tab10', len(unique_groups))
    # group_to_color = {group: cmap(i) for i, group in enumerate(unique_groups)}

    # Step 6: Plot each point
    for i in range(X_embedded.shape[0]):
        x, y = X_embedded[i]
        species = ele_dict[i]
        
        key = str(ele_dict[i])
        #print(key)
        group = species_to_group.get(species, "Unknown")
        color = group_to_color.get(group, 'gray')
        plt.plot(x, y, 'o', ms=8, color=color)
        plt.text(x, y, ele_dict[i], fontsize=4, ha='center', va='top')
    # Step 7: Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=grp,
                   markerfacecolor=group_to_color[grp], markersize=8)
        for grp in unique_groups
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title=trait)

    # Step 8: Save figure
    os.makedirs(FLAGS.visual_dir, exist_ok=True)
    out_path = os.path.join(FLAGS.visual_dir, f"Habitat_feature_emb_month_{mon}.pdf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format='pdf')
    print(f"Saved t-SNE plot to {out_path}")



def show_embed_pca(L, ele_dict, mon, trait_file="./results/AVONET2_eBird.xlsx", trait="Habitat"):
    plt.clf()

    # Step 1: PCA embedding
    pca = PCA(n_components=2)
    X_embedded = pca.fit_transform(L)
    x_min, x_max = X_embedded.min(0), X_embedded.max(0)
    X_embedded = (X_embedded - x_min) / (x_max - x_min)

    # Step 2: Load trait data
    traits_df = pd.read_excel(trait_file, engine='openpyxl')[['Species2', trait]]
    traits_df = traits_df.dropna(subset=['Species2', trait])

    # Step 3: Map scientific name to common name
    taxonomy = pd.read_csv("./results/ebird-taxonomy.csv")
    sci_to_common = dict(zip(taxonomy['scientific_name'], taxonomy['common_name']))
    traits_df['Species2'] = traits_df['Species2'].map(sci_to_common)

    # Step 4: Map species to trait
    species_to_group = dict(zip(traits_df['Species2'], traits_df[trait]))
    unique_groups = sorted(traits_df[trait].unique())
    cmap = plt.get_cmap('tab20', len(unique_groups))
    group_to_color = {group: cmap(i) for i, group in enumerate(unique_groups)}

    # Step 5: Plot points
    for i in range(X_embedded.shape[0]):
        x, y = X_embedded[i]
        species = ele_dict[i]
        group = species_to_group.get(species, "Unknown")
        color = group_to_color.get(group, 'gray')
        plt.plot(x, y, 'o', ms=8, color=color)
        plt.text(x, y, species, fontsize=4, ha='center', va='top')

    # Step 6: Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=grp,
                   markerfacecolor=group_to_color[grp], markersize=8)
        for grp in unique_groups
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title=trait)

    # Step 7: Save figure
    os.makedirs(FLAGS.visual_dir, exist_ok=True)
    out_path = os.path.join(FLAGS.visual_dir, f"{trait}_pca_emb_month_{mon}.pdf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format='pdf')
    print(f"Saved PCA plot to {out_path}")


def main(_):
    visual_dir = FLAGS.visual_dir
    
    mon = FLAGS.mon

    # Load correlation matrix and features
    cov = np.load(os.path.join(visual_dir, f"cov_mean_{mon}.npy"))
    correlation_emb = np.linalg.cholesky(cov)
    feature_emb = np.load(os.path.join(visual_dir, f"feature_emb_mean_{mon}.npy"))
    print("Feature embedding shape:", feature_emb.shape)
    print("Correlation embedding shape:", correlation_emb.shape)

    # Load full bird_dict and trim to match embedding shape
    with open("./data/esrd_trans.pkl", "rb") as f:
        bird_dict_full = pickle.load(f)
    bird_dict = {i: bird_dict_full[i] for i in range(correlation_emb.shape[0])}

    # Step 2: Load threshold file (column names = species to keep)
    threshold_path = f"./results/thresholds/filtered_mdt_threshold_month_{mon}.csv"
    threshold_df = pd.read_csv(threshold_path)
    birds_to_keep = set(threshold_df.columns)

    # Step 3: Filter by species names (preserve index mapping)
    filtered_indices = [i for i, bird in bird_dict.items() if bird in birds_to_keep]

    # Filter embeddings
    correlation_emb = correlation_emb[np.ix_(filtered_indices, filtered_indices)]
    feature_emb = feature_emb[filtered_indices]

    # Create reindexed dict: 0-based index for visualization
    filtered_bird_dict = {new_i: bird_dict[old_i] for new_i, old_i in enumerate(filtered_indices)}

    print("Generating visualizations...")
    #show_embed_correlation(correlation_emb / 1e5, filtered_bird_dict, mon)
    show_embed_pca(feature_emb, filtered_bird_dict, mon)
    

if __name__ == '__main__':
    app.run(main)
