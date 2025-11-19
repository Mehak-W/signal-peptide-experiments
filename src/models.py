"""
Model Building Utilities
Neural network and machine learning models for signal peptide prediction
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict, Optional


class FocalLoss(keras.losses.Loss):
    """
    Focal loss for addressing class imbalance in multi-class classification.

    Parameters
    ----------
    gamma : float
        Focusing parameter (default: 2.0)
    alpha : array-like, optional
        Class weights
    """

    def __init__(self, gamma=2.0, alpha=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1. - epsilon)

        ce = -y_true * keras.backend.log(y_pred)
        p_t = keras.backend.sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_term = keras.backend.pow(1 - p_t, self.gamma)

        if self.alpha is not None:
            alpha_t = y_true * self.alpha
            focal_term = focal_term * alpha_t

        loss = focal_term * ce
        return keras.backend.mean(keras.backend.sum(loss, axis=-1))


class ProteinClassifierNN:
    """
    Neural network classifier for protein embeddings.
    Converts continuous WA values to probability distributions over bins.

    Parameters
    ----------
    n_bins : int
        Number of bins for discretization
    bin_strategy : str
        Binning strategy ('quantile' or 'uniform')
    """

    def __init__(self, n_bins: int = 10, bin_strategy: str = 'quantile'):
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.bin_edges = None
        self.bin_centers = None
        self.model = None

    def create_bins(self, y_train: np.ndarray) -> None:
        """Create bins from training data."""
        if self.bin_strategy == 'quantile':
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            self.bin_edges = np.percentile(y_train, percentiles)
            self.bin_edges = np.unique(self.bin_edges)
        else:
            self.bin_edges = np.linspace(y_train.min(), y_train.max(), self.n_bins + 1)

        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def encode_labels(self, y: np.ndarray) -> np.ndarray:
        """Convert continuous values to one-hot encoded bins."""
        bin_indices = np.digitize(y, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.bin_centers) - 1)

        y_encoded = np.zeros((len(y), len(self.bin_centers)))
        y_encoded[np.arange(len(y)), bin_indices] = 1
        return y_encoded

    def encode_labels_soft(self, y: np.ndarray, temperature: float = 0.5) -> np.ndarray:
        """
        Soft label encoding using Gaussian weighting.
        Spreads probability mass across neighboring bins.
        """
        y_encoded = np.zeros((len(y), len(self.bin_centers)))

        for i, value in enumerate(y):
            distances = np.abs(self.bin_centers - value)
            similarities = np.exp(-distances**2 / (2 * temperature**2))
            y_encoded[i] = similarities / np.sum(similarities)

        return y_encoded

    def decode_predictions(self, y_pred_probs: np.ndarray) -> np.ndarray:
        """Convert probability distributions back to continuous values."""
        return np.sum(y_pred_probs * self.bin_centers, axis=1)

    def create_model(
        self,
        input_dim: int,
        architecture: str = 'single_layer',
        units: List[int] = [256],
        dropout_rates: List[float] = [0.3],
        l2_reg: float = 0.001
    ) -> keras.Model:
        """
        Build neural network model.

        Parameters
        ----------
        input_dim : int
            Input feature dimension
        architecture : str
            'single_layer' or 'two_layer'
        units : list
            Number of units in hidden layers
        dropout_rates : list
            Dropout rates for each layer
        l2_reg : float
            L2 regularization strength

        Returns
        -------
        model : keras.Model
            Compiled Keras model
        """
        model = keras.Sequential()

        # First layer
        model.add(layers.Dense(
            units[0],
            activation='relu',
            input_shape=(input_dim,),
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rates[0]))

        # Second layer for two_layer architecture
        if architecture == 'two_layer' and len(units) > 1:
            model.add(layers.Dense(
                units[1],
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rates[1] if len(dropout_rates) > 1 else dropout_rates[0]))

        # Output layer
        model.add(layers.Dense(len(self.bin_centers), activation='softmax'))

        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        architecture: str = 'single_layer',
        units: List[int] = [256],
        dropout_rates: List[float] = [0.3],
        l2_reg: float = 0.001,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        verbose: int = 1
    ) -> Dict:
        """
        Train the neural network classifier.

        Returns
        -------
        history : dict
            Training history
        """
        # Create bins and encode labels
        self.create_bins(y_train)
        y_train_encoded = self.encode_labels_soft(y_train)

        # Create model
        self.model = self.create_model(
            input_dim=X_train.shape[1],
            architecture=architecture,
            units=units,
            dropout_rates=dropout_rates,
            l2_reg=l2_reg
        )

        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            y_val_encoded = self.encode_labels_soft(y_val)
            validation_data = (X_val, y_val_encoded)

        # Train
        history = self.model.fit(
            X_train, y_train_encoded,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=15,
                    restore_best_weights=True
                )
            ]
        )

        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict continuous WA values.

        Parameters
        ----------
        X : np.ndarray
            Input features

        Returns
        -------
        predictions : np.ndarray
            Predicted WA values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        y_pred_probs = self.model.predict(X, verbose=0)
        return self.decode_predictions(y_pred_probs)


def build_random_forest(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42
) -> RandomForestRegressor:
    """
    Build Random Forest regressor with standard hyperparameters.

    Parameters
    ----------
    n_estimators : int
        Number of trees
    max_depth : int, optional
        Maximum tree depth
    min_samples_split : int
        Minimum samples to split a node
    min_samples_leaf : int
        Minimum samples in a leaf
    random_state : int
        Random seed

    Returns
    -------
    model : RandomForestRegressor
        Configured Random Forest model
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )

    return model
