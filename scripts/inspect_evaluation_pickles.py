import pickle
import pandas as pd
import numpy as np
import anndata as ad # Potentially useful for single-cell data
import os

print("--- Starting Evaluation Pickle Inspection ---")

# Define the path to your data directory relative to where this script is run
# This assumes your .pkl files are in 'forecasts_package/data/'
# and this script is in 'forecasts_package/scripts/'
# So, to go from 'scripts' to 'data', we go up one level (..) then into 'data'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


files_to_inspect = [
    'feature_importance.pkl',
    'randomgridsearch.pkl',
    'sc_data_obs.pkl',
    'sc_obs.pkl'
]

for filename in files_to_inspect:
    filepath = os.path.join(DATA_DIR, filename)
    print(f"\n--- Inspecting {filepath} ---")
    try:
        with open(filepath, 'rb') as f:
            content = pickle.load(f)

        print(f"Type of loaded object: {type(content)}")

        if isinstance(content, pd.DataFrame):
            print(f"Shape: {content.shape}")
            print("Head (first 5 rows):")
            print(content.head())
            if 'score' in content.columns and 'moran' in content.columns:
                print(f"Sample 'score' values: {content['score'].head().tolist()}")
                print(f"Sample 'moran' values: {content['moran'].head().tolist()}")
        elif isinstance(content, pd.Series):
            print(f"Length: {len(content)}")
            print("Head (first 5 elements):")
            print(content.head())
        elif isinstance(content, ad.AnnData):
            print(f"AnnData object: n_obs={content.n_obs}, n_vars={content.n_vars}")
            print(f"Keys in .obs: {list(content.obs.keys())}")
            print(f"Keys in .var: {list(content.var.keys())}")
            print("First 5 obs values:")
            print(content.obs.head())
        elif isinstance(content, (list, np.ndarray)):
            print(f"Length/Shape: {len(content) if isinstance(content, list) else content.shape}")
            print(f"First 10 elements: {str(content[:10])}")
        else:
            print("Object is not a common data structure (DataFrame, Series, AnnData, list, array).")
            print(f"Representation: {str(content)[:200]}...") # Print first 200 chars of string representation

    except FileNotFoundError:
        print(f"Error: {filepath} not found. Make sure it's in the '{DATA_DIR}' folder.")
    except Exception as e:
        print(f"Error loading or inspecting {filepath}: {e}")

print("\n--- Evaluation Pickle Inspection Complete ---")