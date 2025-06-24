import pandas as pd
import numpy as np
import scanpy
import qnorm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from anndata import AnnData
import squidpy as sq # Keeping for now as requested
import time as t
import pickle
import anndata as ad
import logging
from typing import Union, Optional # Added for type hinting

# Configure logger for your module
# This basic configuration will print INFO and higher messages to the console.
# You can customize format, level, and output destination (e.g., to a file).
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FORECASTS:
    """
    A Python class for classifying cellular states (e.g., 'young' vs. 'old')
    in transcriptomics data, particularly suited for spatial transcriptomics
    and single-cell RNA sequencing data.

    The class initializes with pre-trained single-cell reference datasets
    and employs normalization, quantile normalization, and a Random Forest
    classifier to predict the probability of a given cellular state in new data.
    """
    def __init__(self, model_genes_path: str, model_yng_path: str, model_old_path: str):
        """
        Initializes the FORECASTS classifier by loading and pre-processing
        the single-cell training data.

        Parameters
        ----------
        model_genes_path : str
            Path to the .pkl file containing the list of model genes.
        model_yng_path : str
            Path to the .pkl file containing the single-cell expression data
            for 'young' cells (sparse matrix).
        model_old_path : str
            Path to the .pkl file containing the single-cell expression data
            for 'old' cells (sparse matrix).
        """
        try:
            # Load single-cell training data
            with open(model_genes_path, 'rb') as f:
                genes = pickle.load(f)

            with open(model_yng_path, 'rb') as f:
                yng_data = pickle.load(f).todense()
                yng = pd.DataFrame(
                    yng_data,
                    index=[0] * yng_data.shape[0], # Make index flexible based on data size
                    columns=genes
                )

            with open(model_old_path, 'rb') as f:
                old_data = pickle.load(f).todense()
                old = pd.DataFrame(
                    old_data,
                    index=[1] * old_data.shape[0], # Make index flexible based on data size
                    columns=genes
                )
        except FileNotFoundError as e:
            logger.error(f"Required model file not found: {e.filename}. Please check paths.")
            raise FileNotFoundError(f"Required model file not found: {e.filename}. Please check paths.")
        except Exception as e:
            logger.error(f"Error loading model data: {e}")
            raise RuntimeError(f"Error loading model data: {e}")

        # Ensure genes are unique and match column count (basic check)
        if len(genes) != yng.shape[1] or len(genes) != old.shape[1]:
            logger.error("Number of genes loaded from model_genes.pkl does not match data dimensions in model_yng.pkl or model_old.pkl.")
            raise ValueError("Number of genes loaded from model_genes.pkl does not match data dimensions in model_yng.pkl or model_old.pkl.")

        # Convert to AnnData and apply normalization/log1p for reference data
        # Note: These operations are applied to the internal reference data during initialization
        yng_adata = AnnData(yng)
        old_adata = AnnData(old)
        
        scanpy.pp.normalize_total(yng_adata, exclude_highly_expressed=True)
        scanpy.pp.normalize_total(old_adata, exclude_highly_expressed=True)
        scanpy.pp.log1p(yng_adata)
        scanpy.pp.log1p(old_adata)
        
        self.yng = yng_adata.to_df().astype(float)
        self.old = old_adata.to_df().astype(float)

        # Store genes used in the model
        self.model_genes = genes

        # Split to train & test and store
        yng_trn, yng_tst = train_test_split(self.yng, test_size=0.1, random_state=42)
        old_trn, old_tst = train_test_split(self.old, test_size=0.1, random_state=42)

        self._trn_X = pd.concat([yng_trn, old_trn])
        self._trn_y = list(self._trn_X.index)
        self._tst_X = pd.concat([yng_tst, old_tst])
        self._tst_y = list(self._tst_X.index)

    def _normalize(self, st_df: pd.DataFrame, normalized: bool, logarithmized: bool) -> list[np.ndarray]:
        """
        Perform normalization, logarithmization, quantile normalization, and scaling
        across the input dataset and the internal single-cell reference data.

        This method ensures comparable gene expression distributions between the
        user-provided data and the pre-trained reference.

        Parameters
        ----------
        st_df : pandas.DataFrame
            A transcriptomic dataset with rows corresponding to spots/cells/barcodes
            and columns corresponding to genes.
        normalized : bool
            If False, performs total-count normalization on `st_df` using scanpy.pp.normalize_total.
            If True, assumes `st_df` is already total-count normalized.
        logarithmized : bool
            If False, applies log1p transformation to `st_df` using scanpy.pp.log1p.
            If True, assumes `st_df` is already logarithmized.

        Returns
        -------
        list [numpy.ndarray, numpy.ndarray, numpy.ndarray]
            A list containing three NumPy arrays in the following order:
            [0] : Scaled training data (NumPy array).
            [1] : Scaled testing data (NumPy array).
            [2] : Scaled input spatial data (NumPy array).

        Raises
        ------
        ValueError
            If the input DataFrame does not contain any genes common with the model genes.
        """
        # Make a copy to avoid modifying the original DataFrame in place
        st_df_copy = st_df.copy()

        if not normalized:
            # Create AnnData object properly with obs and var to preserve indices/columns
            adata = ad.AnnData(X = st_df_copy.values, obs = st_df_copy.index.to_frame(), var = st_df_copy.columns.to_frame())
            scanpy.pp.normalize_total(adata, exclude_highly_expressed = True)
            st_df_copy = adata.to_df()

        # Logarithmize
        if not logarithmized:
            st_df_copy = pd.DataFrame(
                data = scanpy.pp.log1p(st_df_copy.values), # Apply log1p to the underlying numpy array
                index = st_df_copy.index,
                columns = st_df_copy.columns
            )

        # Subset to genes common among the datasets
        common_genes = self.model_genes.intersection(st_df_copy.columns)
        if common_genes.empty:
            raise ValueError("No common genes found between input data and model genes. Please ensure gene names match.")

        trn_X = self._trn_X[common_genes]
        tst_X = self._tst_X[common_genes]
        st_df_subsetted = st_df_copy[common_genes]

        # Handle duplicate columns if any (though `intersection` should already handle uniqueness)
        st_df_subsetted = st_df_subsetted.loc[:, ~st_df_subsetted.columns.duplicated(keep = 'first')]

        # Quantile normalize
        all_samples = pd.concat([trn_X, tst_X, st_df_subsetted], axis = 0)
        quantile_normalized_values = qnorm.quantile_normalize(all_samples.values, axis = 0) # Apply to values for qnorm

        # Split into original dataframes (now numpy arrays)
        trn_X_normalized = quantile_normalized_values[ : self._trn_X.shape[0]]
        tst_X_normalized = quantile_normalized_values[self._trn_X.shape[0] : self._trn_X.shape[0] + self._tst_X.shape[0]]
        st_df_normalized = quantile_normalized_values[self._trn_X.shape[0] + self._tst_X.shape[0] : ]

        # Scale data
        scaler = StandardScaler()
        # Fit scaler on training data, then transform all other sets
        trn_X_scaled = scaler.fit_transform(trn_X_normalized)
        tst_X_scaled = scaler.transform(tst_X_normalized) # Use transform
        st_df_scaled = scaler.transform(st_df_normalized) # Use transform

        return [trn_X_scaled, tst_X_scaled, st_df_scaled]

    def classify(self, data: Union[pd.DataFrame, AnnData], n_jobs: int = 1,
                 normalized: bool = True, logarithmized: bool = True,
                 verbose: bool = True, layer: Optional[str] = None) -> list[float]:
        """
        Classifies an input transcriptomic dataset (spatial or single-cell)
        into 'young' or 'old' states based on the pre-trained model.

        Parameters
        ----------
        data : pandas.DataFrame or anndata.AnnData
            The input transcriptomic dataset to classify. Rows should be cells/spots
            and columns should be genes.
        n_jobs : int, optional
            The number of jobs to run in parallel for the Random Forest classifier.
            -1 means using all available processors. Default is 1.
        normalized : bool, optional
            If False, performs total-count normalization on `data` before classification.
            If True, assumes `data` is already total-count normalized. Default is True.
        logarithmized : bool, optional
            If False, applies log1p transformation to `data` before classification.
            If True, assumes `data` is already logarithmized. Default is True.
        verbose : bool, optional
            If True, prints progress messages and model accuracy on the hold-out test set.
            Default is True.
        layer : str, optional
            If `data` is an `AnnData` object, specifies the layer to use for expression
            data. If None, uses `data.X`. Default is None.

        Returns
        -------
        list of float
            A list of probabilities (between 0 and 1), where each probability
            corresponds to the likelihood of a given cell/spot in the input `data`
            being classified as 'old'. The order of probabilities matches the
            order of rows in the input `data`.

        Raises
        ------
        TypeError
            If input `data` is not a pandas.DataFrame or AnnData object,
            or if `n_jobs` is not an integer.
        ValueError
            If an underlying normalization or data handling step fails (e.g., no common genes).
        """
        # Type validation
        if not isinstance(data, (pd.DataFrame, AnnData)):
            raise TypeError("Input datatype 'data' must be pandas.DataFrame or AnnData.")
        if not isinstance(n_jobs, int):
            raise TypeError("n_jobs must be of type int.")

        if verbose:
            start = t.time()
            logger.info("Pre-processing...")

        # Convert to pandas for maths if AnnData
        if isinstance(data, AnnData):
            st_df = data.to_df(layer = layer)
        else:
            st_df = data

        try:
            trn_X_final, tst_X_final, st_df_processed = self._normalize(st_df, normalized = normalized, logarithmized = logarithmized)
        except ValueError as e:
            logger.error(f"Normalization or data subsetting failed during pre-processing: {e}")
            raise # Re-raise the error after logging

        if verbose:
            logger.info(f"Pre-processing complete in {t.time() - start:.2f} seconds.")
            logger.info(f"{st_df_processed.shape[1]} genes in model after subsetting.")
            logger.info("Modelling...")
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
        ).fit(trn_X_final, self._trn_y) # Use trn_X_final here

        if verbose:
            preds = clf.predict(tst_X_final) # Use tst_X_final here
            accuracy = sum(np.equal(preds, self._tst_y)) / len(preds)
            logger.info(f"Model accuracy on hold-out scRNA-seq dataset of {accuracy*100:.2f}%.") # Formatted to 2 decimal places

        probabilities = [x[1] for x in clf.predict_proba(st_df_processed)] # Use st_df_processed here

        if verbose:
            logger.info(f"Modelling complete in {t.time() - start:.2f} seconds.")

        return probabilities