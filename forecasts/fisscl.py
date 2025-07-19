import pandas as pd
import numpy as np
import scanpy
import qnorm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from anndata import AnnData
import squidpy as sq
import time as t
import pickle
import os # <-- ADDED THIS IMPORT


class fisscl:
    def __init__(self):
        # Define a base path relative to the current file to access the Model directory
        # This makes the path robust regardless of where the script is executed from
        # os.path.abspath(__file__) gives the full path to fisscl.py
        # os.path.dirname(...) gets the directory it's in (e.g., forecasts_package/forecasts/)
        # os.path.dirname(os.path.dirname(...)) goes up one more level (e.g., forecasts_package/)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'Model')

        # Load single-cell training data and log normalize
        # Paths now use os.path.join for cross-OS compatibility
        with open(os.path.join(model_dir, 'model_genes.pkl'), 'rb') as f:
            genes = pickle.load(f)
            
        with open(os.path.join(model_dir, 'model_yng.pkl'), 'rb') as f:
            yng = pd.DataFrame(
                pickle.load(f).todense(),
                index = [0] * 400,
                columns = genes
            )
            
        with open(os.path.join(model_dir, 'model_old.pkl'), 'rb') as f:
            old = pd.DataFrame(
                pickle.load(f).todense(),
                index = [1] * 400,
                columns = genes
            )
            
        yng = AnnData(yng)
        old = AnnData(old)
        scanpy.pp.normalize_total(yng, exclude_highly_expressed = True)
        scanpy.pp.normalize_total(old, exclude_highly_expressed = True)
        scanpy.pp.log1p(yng)
        scanpy.pp.log1p(old)
        yng = yng.to_df().astype(float)
        old = old.to_df().astype(float)
        
        # Split to train & test and store
        yng_trn, yng_tst = train_test_split(yng, test_size = 0.1, random_state = 42)
        old_trn, old_tst = train_test_split(old, test_size = 0.1, random_state = 42)
        
        self._trn_X = pd.concat([yng_trn, old_trn])
        self._trn_y = list(self._trn_X.index)
        self._tst_X = pd.concat([yng_tst, old_tst])
        self._tst_y = list(self._tst_X.index)

        
    def _normalize(self, st_df, logarithmized, scale):
        """
        Perform quantile normalization across both the input dataset and the
        tang et al. scRNA-seq dataset.
        
        Returns the quantile normalized scRNA-seq and input datasets. Each 
        spot/cell/barcode in both datasets will have identical means. Ties 
        in the original datasets (i.e. zero counts) are resolved by
        setting tied values to their means; as such each spot will similar
        but not necessarily identical standard deviations.

        Parameters
        ----------
        st_raw : pandas.DataFrame
        
        A transcriptomic dataset in pandas.DataFrame type with rows 
        corresponding to spots/cells/barcodes and columns 
        corresponding to genes.
        
        Returns
        -------
        list [pandas.DataFrame, pandas.DataFrame]
        A list containing the normalized scRNA-seq dataset at index 0 and
        the normalized input dataset at index 1.

        """
        # Logarithmize
        if not logarithmized:
            st_df = pd.DataFrame(
                data = scanpy.pp.log1p(np.asarray(st_df)), 
                index = st_df.index, 
                columns = st_df.columns
            )
        
        # Subset to genes common among the datasets
        common_genes = self._trn_X.columns.intersection(st_df.columns)
        trn_X = self._trn_X[common_genes]
        tst_X = self._tst_X[common_genes]
        st_df = st_df[common_genes]
        
        # Quantile normalize
        all_samples = pd.concat([trn_X, tst_X, st_df], axis = 0)
        quantile_normalized = qnorm.quantile_normalize(all_samples, axis = 0)
        
        # Split into original dataframes
        trn_X = quantile_normalized[ : self._trn_X.shape[0]]
        tst_X = quantile_normalized[self._trn_X.shape[0] : self._trn_X.shape[0] + self._tst_X.shape[0]]
        st_df = quantile_normalized[self._trn_X.shape[0] + self._tst_X.shape[0] : ]
        
        print(trn_X.shape) # This print statement might be verbose for a publication, consider removing or making conditional on 'verbose'
        
        # Scale data
        if scale:
            scaler = StandardScaler()
            trn_X = scaler.fit_transform(trn_X)
            tst_X = scaler.fit_transform(tst_X)
            st_df = scaler.fit_transform(st_df)
        
        return [trn_X, tst_X, st_df]

    
    def classify(self, data, n_jobs = 1, logarithmized = True, verbose = True, layer = None, scale = False):
        if type(data) != pd.DataFrame and type(data) != AnnData:
            raise TypeError("Input datatype must be pandas.DataFrame or AnnData.")
        if type(n_jobs) != int:
            raise TypeError("n_jobs must be of type int.")
        
        if verbose:
            start = t.time()
            print("Pre-processing...")
        
        # Convert to pandas for maths
        if type(data) == AnnData:
            st_df = data.to_df(layer = layer)
        else:
            st_df = data
    
        trn_X, tst_X, st_df = self._normalize(st_df, logarithmized = logarithmized, scale = scale)
        
        if verbose:
            print(f"Pre-processing complete in {t.time() - start:.2f} seconds.")
            print(f"{st_df.shape[1]} genes in model")
            print("Modelling...")
            start = t.time()
        
        # Classify and test output accuracy
        clf = RandomForestClassifier(
            criterion = 'gini',
            n_estimators = 500,
            max_depth = 11,
            max_features = 148,
            min_samples_split = 2,
            min_samples_leaf = 1,
            max_samples = 1.0,
            random_state = 42,
            n_jobs = n_jobs,
            ccp_alpha = 0.0
        ).fit(trn_X, self._trn_y)
        
        if verbose:
            preds = clf.predict(tst_X)
            accuracy = sum(np.equal(preds, self._tst_y)) / len(preds)
            print(f"Model accuracy on hold-out scRNA-seq dataset of {accuracy*100}%.")
            
        probabilities = [x[1] for x in clf.predict_proba(st_df)]
        
        if verbose:
            print(f"Modelling complete in {t.time() - start:.2f} seconds.")
        
        return probabilities
    
    
if __name__ == '__main__':
    # This block will only run when fisscl.py is executed directly
    # Adjust this path if your Ganier data is not directly in forecasts_package/Ganier/
    print("Running fisscl.py as a standalone script (for testing purposes).")
    try:
        data = sq.read.visium(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Ganier/WSSKNKCLsp10446617"), counts_file = 'filtered_feature_bc_matrix.h5')
        data.var_names_make_unique()
        clf = fisscl()
        preds = clf.classify(data)
        print("Classification successful for example data.")
    except Exception as e:
        print(f"Error when running example usage: {e}")
        print("Please ensure 'Ganier/WSSKNKCLsp10446617' path is correct relative to forecasts_package root.")