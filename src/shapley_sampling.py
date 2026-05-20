import numpy as np
import pandas as pd

def _finalize(result, runs, shape):
    """
    Normalizes the accumulated Shapley scores by the total number of sampling runs
    and reshapes the output matrix to match the target structure.

    Args:
        result (np.ndarray): The accumulated prediction deltas from the sampling loop.
        runs (int): The number of completed permutation runs used for averaging.
        shape (list): The target shape dimensions for the final output array.

    Returns:
        np.ndarray: The finalized, averaged, and reshaped Shapley values.
    """
    # Calculate the average marginal contribution per feature across all runs
    shapley = result.copy() / runs
    
    # Enforce the requested shape format before returning the results
    return shapley.reshape(shape)


def run_shapley_sampling(model, xs, runs=128, class_flag=False, callback=None):
    """
    Computes Shapley feature importance values using a polynomial sampling approach.
    DataFrame-compliant version that preserves feature names to prevent Scikit-Learn UserWarnings.

    Args:
        model: Trained estimator (e.g., Scikit-Learn Regressor/Classifier or GridSearchCV).
        xs (pd.DataFrame): Input features dataframe of shape (n_samples, n_features).
        runs (int): Number of permutation sampling iterations. Default is 128.
        class_flag (bool): Set to True for classification (uses predict_proba), 
                           False for regression (uses predict). Default is False.
        callback (callable): Optional callback function for tracking progress.

    Returns:
        np.ndarray: Reshaped matrix containing the calculated Shapley values.
    """
    # Ensure input data structure is a pandas DataFrame to maintain feature names consistency
    if not isinstance(xs, pd.DataFrame):
        raise ValueError("xs must be a pandas DataFrame to preserve feature names.")
        
    n_samples = xs.shape[0]
    n_features = xs.shape[1]
    
    print("Data shape for Shapley: ", xs.shape)
    print("Runs: ", runs)
    
    # Initialize the results matrix and callback tracking parameters
    result = np.zeros((n_samples, n_features))
    reconstruction_shape = [n_samples, n_features]
    next_callback = 1

    for r in range(runs):
        # Generate a random permutation of feature indices for the current run
        p = np.random.permutation(n_features)
        
        # Create an isolated deep copy of the dataframe to prevent mutating the original data
        x = xs.copy(deep=True)  
        y = None
        
        for i in p:
            # Establish the base prediction for the current feature set combination
            if y is None:
                y = model.predict_proba(x) if class_flag else model.predict(x)

            # Mask/baseline the current feature using .iloc index positioning
            x.iloc[:, i] = -1
            
            # Get the new prediction after masking the feature
            y0 = model.predict_proba(x) if class_flag else model.predict(x)
            assert y0.shape == y.shape

            # Calculate the absolute change in model predictions
            prediction_delta = np.abs(y0 - y)
            
            # If multi-class outputs exist, aggregate the delta across the class axis (axis=1)
            # while preserving individual sample identity (axis=0)
            if prediction_delta.ndim > 1:
                prediction_delta = np.sum(prediction_delta, axis=1)
                
            # Accumulate the impact score for the respective feature
            result[:, i] += prediction_delta
            y = y0

        # Trigger callback at exponential steps (1, 2, 4, 8, ...) if provided
        if (r+1) == next_callback and callback is not None:
            next_callback = next_callback * 2
            callback((r+1, _finalize(result, r+1, reconstruction_shape)))

    return _finalize(result, runs, reconstruction_shape)