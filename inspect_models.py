import pickle
import pandas as pd
import anndata as ad
import numpy as np
import os # Added to check file paths

print("--- Starting Model Inspection ---")

# Define the path to your Model directory
# This assumes 'Model' is a direct subdirectory of where this script is run
MODEL_DIR = 'Model'

# --- 1. Inspect model_genes.pkl ---
genes_path = os.path.join(MODEL_DIR, 'model_genes.pkl')
try:
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    print(f"\n--- Content of {genes_path} ---")
    print(f"Type: {type(genes)}")
    print(f"Number of genes: {len(genes)}")
    # Print first few and last few genes to get an idea
    if len(genes) > 10:
        print(f"First 5 genes: {genes[:5]}")
        print(f"Last 5 genes: {genes[-5:]}")
    else:
        print(f"All genes: {genes}")
    # Basic check for expected type (list of strings)
    if all(isinstance(g, str) for g in genes[:10]):
        print("Genes appear to be strings (as expected).")
    else:
        print("Warning: Genes do not appear to be strings.")
except FileNotFoundError:
    print(f"Error: {genes_path} not found. Make sure it's in the 'Model' folder.")
except Exception as e:
    print(f"Error loading or inspecting {genes_path}: {e}")

# --- 2. Inspect model_yng.pkl ---
yng_path = os.path.join(MODEL_DIR, 'model_yng.pkl')
try:
    with open(yng_path, 'rb') as f:
        yng_data_sparse = pickle.load(f)
    print(f"\n--- Content of {yng_path} ---")
    print(f"Type of loaded object: {type(yng_data_sparse)}")

    # Attempt to convert to dense DataFrame for easier viewing
    # This assumes 'genes' was successfully loaded from model_genes.pkl
    if 'genes' in locals() and isinstance(genes, (list, np.ndarray)):
        # Your original code uses .todense(), implying scipy sparse matrix
        # If it's a sparse matrix, todense() is correct.
        # If it's already a numpy array, todense() will cause an AttributeError,
        # so we check if it has the todense method.
        if hasattr(yng_data_sparse, 'todense'):
            yng_df = pd.DataFrame(yng_data_sparse.todense(), columns=genes)
        else: # Assume it's already a numpy array or similar
            yng_df = pd.DataFrame(yng_data_sparse, columns=genes)

        print(f"Shape (rows, columns): {yng_df.shape}")
        print("First 5 rows and 5 columns of data (as DataFrame):")
        print(yng_df.iloc[:5, :5])
        print(f"Mean value (first 50 elements): {np.mean(yng_df.iloc[:, :50].values):.4f}")
        print(f"Min value: {yng_df.min().min():.4f}, Max value: {yng_df.max().max():.4f}")
        print(f"Contains NaNs: {yng_df.isnull().any().any()}")
    else:
        print("Cannot convert to DataFrame without valid 'genes' list.")
        print(f"Raw data (first few elements): {yng_data_sparse}")

except FileNotFoundError:
    print(f"Error: {yng_path} not found. Make sure it's in the 'Model' folder.")
except Exception as e:
    print(f"Error loading or inspecting {yng_path}: {e}")

# --- 3. Inspect model_old.pkl ---
old_path = os.path.join(MODEL_DIR, 'model_old.pkl')
try:
    with open(old_path, 'rb') as f:
        old_data_sparse = pickle.load(f)
    print(f"\n--- Content of {old_path} ---")
    print(f"Type of loaded object: {type(old_data_sparse)}")

    if 'genes' in locals() and isinstance(genes, (list, np.ndarray)):
        if hasattr(old_data_sparse, 'todense'):
            old_df = pd.DataFrame(old_data_sparse.todense(), columns=genes)
        else:
            old_df = pd.DataFrame(old_data_sparse, columns=genes)

        print(f"Shape (rows, columns): {old_df.shape}")
        print("First 5 rows and 5 columns of data (as DataFrame):")
        print(old_df.iloc[:5, :5])
        print(f"Mean value (first 50 elements): {np.mean(old_df.iloc[:, :50].values):.4f}")
        print(f"Min value: {old_df.min().min():.4f}, Max value: {old_df.max().max():.4f}")
        print(f"Contains NaNs: {old_df.isnull().any().any()}")
    else:
        print("Cannot convert to DataFrame without valid 'genes' list.")
        print(f"Raw data (first few elements): {old_data_sparse}")

except FileNotFoundError:
    print(f"Error: {old_path} not found. Make sure it's in the 'Model' folder.")
except Exception as e:
    print(f"Error loading or inspecting {old_path}: {e}")

print("\n--- Model Inspection Complete ---")