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
    trait = "Beak.Length_Nares"
    plt.clf()

    # Step 1: t-SNE embedding
    X_embedded = TSNE(n_components=2, perplexity=8, max_iter=1000, init='pca', random_state=42).fit_transform(L)
    x_min, x_max = X_embedded.min(0), X_embedded.max(0)
    X_embedded = (X_embedded - x_min) / (x_max - x_min)

    # Step 2: Load trait data
    traits_df = pd.read_excel(trait_file, engine='openpyxl')[['Species2', trait]]
    traits_df = traits_df.dropna(subset=['Species2', trait])

    # Step 3: Map species scientific name to common name
    taxonomy = pd.read_csv("./results/ebird-taxonomy.csv")
    sci_to_common = dict(zip(taxonomy['scientific_name'], taxonomy['common_name']))
    traits_df['Species2'] = traits_df['Species2'].map(sci_to_common)

    # Step 4: Bin Wing.Length trait into intervals
    num_bins = 8
    traits_df['TraitBin'] = pd.cut(traits_df[trait], bins=num_bins)

    # Mapping: species → binned group
    species_to_bin = dict(zip(traits_df['Species2'], traits_df['TraitBin']))

    # Step 5: Assign colors per bin
    unique_bins = sorted(traits_df['TraitBin'].dropna().unique())
    cmap = plt.get_cmap('tab10', len(unique_bins))
    bin_to_color = {bin_lbl: cmap(i) for i, bin_lbl in enumerate(unique_bins)}

    # Step 6: Plot each embedded species
    for i in range(X_embedded.shape[0]):
        x, y = X_embedded[i]
        species = ele_dict[i]
        bin_label = species_to_bin.get(species, "Unknown")
        color = bin_to_color.get(bin_label, 'gray')
        plt.plot(x, y, 'o', ms=8, color=color)

    # Step 7: Add legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=str(bin_lbl),
                   markerfacecolor=bin_to_color[bin_lbl], markersize=8)
        for bin_lbl in unique_bins
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title=f'{trait} bins')

    # Step 8: Save the figure
    os.makedirs(FLAGS.visual_dir, exist_ok=True)
    out_path = os.path.join(FLAGS.visual_dir, f"beakL_cor_emb_month_{mon}.pdf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format='pdf')
    print(f"Saved t-SNE plot to {out_path}")


def show_embed_feature(L, ele_dict, mon, trait_file="./results/AVONET2_eBird.xlsx"):
    trait = "Beak.Length_Nares"
    plt.clf()

    # Step 1: Embed
    X_embedded = TSNE(n_components=2, perplexity=8, max_iter=1000, init='pca', random_state=42).fit_transform(L)
    x_min, x_max = X_embedded.min(0), X_embedded.max(0)
    X_embedded = (X_embedded - x_min) / (x_max - x_min)

    # Step 2: Load trait data
    traits_df = pd.read_excel(trait_file, engine='openpyxl')[['Species2', trait]]
    traits_df = traits_df.dropna(subset=['Species2', trait])

    # Step 3: Map to common names
    taxonomy = pd.read_csv("./results/ebird-taxonomy.csv")
    sci_to_common = dict(zip(taxonomy['scientific_name'], taxonomy['common_name']))
    traits_df['Species2'] = traits_df['Species2'].map(sci_to_common)

    # Step 4: Bin trait values
    num_bins = 8
    traits_df['TraitBin'] = pd.cut(traits_df[trait], bins=num_bins)

    # Create species to bin label mapping
    species_to_bin = dict(zip(traits_df['Species2'], traits_df['TraitBin']))

    # Step 5: Assign colors to bins
    unique_bins = sorted(traits_df['TraitBin'].dropna().unique())
    cmap = plt.get_cmap('tab10', len(unique_bins))
    bin_to_color = {bin_lbl: cmap(i) for i, bin_lbl in enumerate(unique_bins)}

    # Step 6: Plot each point
    for i in range(X_embedded.shape[0]):
        x, y = X_embedded[i]
        species = ele_dict[i]
        bin_label = species_to_bin.get(species, "Unknown")
        color = bin_to_color.get(bin_label, 'gray')
        plt.plot(x, y, 'o', ms=8, color=color)

    # Step 7: Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=str(bin_lbl),
                   markerfacecolor=bin_to_color[bin_lbl], markersize=8)
        for bin_lbl in unique_bins
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title=f'{trait} bins')

    # Step 8: Save figure
    os.makedirs(FLAGS.visual_dir, exist_ok=True)
    out_path = os.path.join(FLAGS.visual_dir, f"beakL_feature_emb_month_{mon}.pdf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, format='pdf')
    print(f"Saved t-SNE plot to {out_path}")


# def show_correlation_matrix(L, ele_dict, mon, species_names=None):
#     """
#     Visualize the pairwise correlation matrix of species feature embeddings using both
#     a standard heatmap and a clustered heatmap.

#     Args:
#         L (ndarray): Correlation matrix of shape [n_species, n_species].
#         ele_dict (dict or list): Index-to-species mapping (e.g., {0: "Species A", 1: "Species B", ...}).
#         mon (int): Month index (used in output file name).
#         species_names (list or None): Optional custom list of species names.
#         out_dir (str): Output directory for saving the figure.
#     """
#     # Step 1: Prepare species labels
#     if species_names is None:
#         if isinstance(ele_dict, dict):
#             species_names = [ele_dict[i] for i in range(len(L))]
#         else:
#             species_names = list(ele_dict)

#     # Step 2: Convert matrix to DataFrame
#     df_corr = pd.DataFrame(L, index=species_names, columns=species_names)

#     # === Plot 1: Regular heatmap ===
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(df_corr, cmap="coolwarm", vmin=0, vmax=1, square=True,
#                 xticklabels=True, yticklabels=True, cbar_kws={"label": "Correlation"})
#     plt.title(f"Species Correlation Matrix - Month {mon}")
#     plt.xticks(rotation=90)
#     plt.yticks(rotation=0)
#     plt.tight_layout()

#     # Save regular heatmap
   
#     heatmap_path = os.path.join(FLAGS.visual_dir, f"species_corr_matrix_month_{mon}.pdf")
#     plt.savefig(heatmap_path, dpi=300, format='pdf')
#     print(f"Saved correlation matrix heatmap to {heatmap_path}")
#     plt.close()

#     # === Plot 2: Clustered heatmap ===
#     g = sns.clustermap(df_corr, cmap="coolwarm", vmin=0, vmax=1, 
#                        linewidths=0.5, figsize=(12, 12), 
#                        cbar_kws={"label": "Correlation"})
#     g.fig.suptitle(f"Clustered Species Correlation Matrix - Month {mon}", y=1.02)

#     # Save clustered heatmap
#     cluster_path = os.path.join(FLAGS.visual_dir, f"species_corr_clustermap_month_{mon}.pdf")
#     g.savefig(cluster_path, dpi=300, format='pdf')
#     print(f"Saved clustered correlation matrix to {cluster_path}")


def show_correlation_matrix(L, ele_dict, mon, species_names=None):
    """
    Visualize the pairwise correlation matrix of species feature embeddings using both
    a standard heatmap and a clustered heatmap. The color scale adapts to min/max correlation.

    Args:
        L (ndarray): Correlation matrix of shape [n_species, n_species].
        ele_dict (dict or list): Index-to-species mapping (e.g., {0: "Species A", 1: "Species B", ...}).
        mon (int): Month index (used in output file name).
        species_names (list or None): Optional custom list of species names.
        out_dir (str): Output directory for saving the figure.
    """
    # Step 1: Prepare species labels
    if species_names is None:
        if isinstance(ele_dict, dict):
            species_names = [ele_dict[i] for i in range(len(L))]
        else:
            species_names = list(ele_dict)

    # Step 2: Convert matrix to DataFrame
    df_corr = pd.DataFrame(L, index=species_names, columns=species_names)

    # Step 3: Determine dynamic vmin and vmax for color scale
        # Step 3: Determine dynamic vmin and vmax for color scale
    # Exclude diagonal to focus on meaningful pairwise correlations
    mask = ~np.eye(L.shape[0], dtype=bool)
    off_diag_values = L[mask]

    min_val = np.min(off_diag_values)
    max_val = np.max(off_diag_values)

    # Optional: add a small buffer to enhance visual contrast
    margin = 0.01 * (max_val - min_val)
    vmin = max(0.0, min_val - margin)
    vmax = min(1.0, max_val + margin)

   


    # === Plot 1: Regular heatmap ===
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, cmap="coolwarm", vmin=vmin, vmax=vmax, square=True,
                xticklabels=True, yticklabels=True, cbar_kws={"label": "Correlation"})
    plt.title(f"Species Correlation Matrix - Month {mon}")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save regular heatmap
    
    heatmap_path = os.path.join(FLAGS.visual_dir, f"species_corr_matrix_month_{mon}.pdf")
    plt.savefig(heatmap_path, dpi=300, format='pdf')
    print(f"Saved correlation matrix heatmap to {heatmap_path}")
    plt.close()

    # === Plot 2: Clustered heatmap ===
    g = sns.clustermap(df_corr, cmap="coolwarm", vmin=min_val, vmax=max_val,
                       linewidths=0.5, figsize=(12, 12),
                       cbar_kws={"label": "Correlation"})
    g.fig.suptitle(f"Clustered Species Correlation Matrix - Month {mon}", y=1.02)

    # Save clustered heatmap
    cluster_path = os.path.join(FLAGS.visual_dir, f"species_corr_clustermap_month_{mon}.pdf")
    g.savefig(cluster_path, dpi=300, format='pdf')
    print(f"Saved clustered correlation matrix to {cluster_path}")


def main(_):
    
    

   
    visual_dir = FLAGS.visual_dir
    mon = FLAGS.mon

    # Load correlation matrix and features
    cov = np.load(os.path.join(visual_dir, f"cov_mean_{mon}.npy"))
    correlation_emb = np.linalg.cholesky(cov)
    feature_emb = np.load(os.path.join(visual_dir, f"feature_emb_mean_{mon}.npy"))
    print("Feature embedding shape:", feature_emb.shape)
    print("Correlation embedding shape:", correlation_emb.shape)

    # Load bird dict
    with open("./data/esrd_trans.pkl", "rb") as f:
        bird_dict = pickle.load(f)

    # Step 2: Load threshold file (column names = species to keep)
    threshold_path = f"./results/thresholds/filtered_mdt_threshold_month_{mon}.csv"
    threshold_df = pd.read_csv(threshold_path)

    # Extract bird names to keep
    birds_to_keep = set(threshold_df.columns)

    # Step 3: Filter bird_dict
    filtered_bird_dict = {
        i: bird for i, bird in bird_dict.items() if bird in birds_to_keep
        }

    print("Generating visualizations...")
    show_embed_correlation(correlation_emb / 1e5, filtered_bird_dict, mon)
    show_embed_feature(feature_emb / 1e5, filtered_bird_dict, mon)
    show_correlation_matrix(correlation_emb / 1e5, filtered_bird_dict, mon, species_names=None)

if __name__ == '__main__':
    app.run(main)