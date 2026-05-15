import argparse
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_absolute_error
from scipy.stats import rankdata, pearsonr, spearmanr
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from tqdm import tqdm
from typing import Any, Dict, List


class PLSRegressionCustom(PLSRegression):
    def __init__(
            self, n_components=2, *, copy=True
    ):
        super().__init__(
            n_components=n_components,
            copy=copy,
        )
    def fit_transform(self, X, y=None):
        X_t, _ = self.fit(X, y).transform(X, y)
        return X_t


def load_data(path: str, taget_var_name='CGIExit') -> tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Loads data into a Pandas dataframe and separate into target and features numpy arrays. Also extract feature names

    Drops 'ID' column, and replaces ' ' with nans

    :param path: Path to the csv file containing the data
    :param taget_var_name: Name of the target variable (Default: 'CGIExit')
    :return: (Data in a Pandas dataframe, features, target, feature_names)
    """
    data = pd.read_csv(path)
    if 'ID' in data.columns:
        data = data.drop(columns=["ID"])
    data = data.replace(' ', np.nan)

    # Separate features and target
    X = data.drop(columns=[taget_var_name]).to_numpy().astype(np.float64)
    y = data[taget_var_name].to_numpy()

    # Extract feature names
    feature_names = list(data.drop(columns=[taget_var_name]).columns)

    return data, X, y, feature_names


def get_defaults() -> tuple[callable, Pipeline, Dict[str, Any]]:
    """
    Created the default scorer (MAE), pipeline and parameter grid (for search)

    :return: (scorer, pipeline, parameter_grid)
    """
    # Define scorer for mean absolute error
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Define the default pipeline
    pipeline = Pipeline([
        ("preprocess", Pipeline([
            ("impute", KNNImputer(n_neighbors=7, metric="nan_euclidean")),
            ("scale", MinMaxScaler())
        ])),
        ("pls", PLSRegressionCustom()),
        ("svr", SVR(kernel="linear"))
    ])

    # Define parameter grid for tuning PLS and SVR
    param_grid = {
        "pls__n_components": [1, 2, 5, 10, 15],
        "svr__C": np.logspace(-3, 3, 100),
    }

    return scorer, pipeline, param_grid


def calculate_model_stats(y_true:np.ndarray, y_pred:np.ndarray) -> Dict[str, Any]:
    """
    Calculates model statistics
        - Pearson correlation
            - r value
            - p-value
        - Spearman correlation
            - rho value
            - p-value
        - Mean absolute error
            - Model's MAE
            - Baseline MAE (using the mean of target as predictor)

    :param y_true: True targets
    :param y_pred: Predicted targets
    :return: Calculated model stats in a dict
    """

    model_stats = {}

    # Get Pearson correlation
    model_stats['pearson_r_value'], model_stats['pearson_p_value'] = pearsonr(y_true, y_pred)

    # Spearman correlation
    model_stats['spearman_rho_value'], model_stats['spearman_p_value'] = spearmanr(y_true, y_pred)

    # Mean absolute error of the model and using just the mean value
    model_stats['mae'] = mean_absolute_error(y_true, y_pred)
    model_stats['baseline_mae'] = mean_absolute_error(y_true, y_true * 0 + np.mean(y_true))

    return model_stats


def _plot_predictions(
        y_true:np.ndarray,
        y_pred:np.ndarray,
        model_stats:Dict[str, Any],
        fig: plt.figure,
        ax: plt.axis,
        fig_letter: str
) -> None:
    """
    Plot for predictions

    :param y_true: True targets
    :param y_pred: Predicted targets
    :param model_stats: Model statistics
    :param fig: Figure
    :param ax: Axis to plot on
    :param fig_letter: Figure letter shown top left
    :return:
    """
    sns.regplot(
        x=y_pred, y=y_true,
        ax=ax,
        scatter_kws={'s': 100},
        line_kws={'linewidth': 5},
        color=sns.color_palette('Greys')[4]
    )

    # Labeling
    ax.set_xlabel(f'Predicted CGIExit', fontsize=16)
    ax.set_ylabel(f'Observed CGIExit', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Put figure letter above the y label
    fig.canvas.draw()  # Ensure the figure is rendered
    ylab_box = ax.yaxis.label.get_window_extent()
    ylab_pos_axes = ax.transAxes.inverted().transform(ylab_box.p0)
    ax.text(ylab_pos_axes[0]-0.025, 1.06, fig_letter, ha='left', va='top', fontsize=20, fontweight='bold', transform=ax.transAxes)

    # PLot stats
    stats_text = (
        f"r={model_stats['pearson_r_value']:.2f} (p={model_stats['pearson_p_value']:.2g})\n"
        f"ρ={model_stats['spearman_rho_value']:.2f} (p={model_stats['spearman_p_value']:.2g})\n"
        f"MAE = {model_stats['mae']:.2f}\n"
        f"Baseline MAE = {model_stats['baseline_mae']:.2f}"
    )
    ax.text(
        0.025, 0.975,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="black")
    )


def _plot_feature_importance(
        feature_names:List[str],
        shap_values: np.ndarray,
        fig: plt.figure,
        ax: plt.axis,
        fig_letter: str,
        max_features: int = 15
) -> None:
    """
    Plot for SHAP (feature importances)

    :param feature_names: List of feature names
    :param shap_values: Shap values
    :param fig: Figure
    :param ax: Axis to plot on
    :param fig_letter: Figure letter shown top left
    :param max_features: Maximum number of feature displayed (+1 will be displayed with the sum of the rest)
    :return: None
    """
    # Take the mean of the absolute shap values for each feature
    shap_values_abs_mean = np.mean(np.abs(shap_values), axis=0)

    # Order it so that more important features are in front
    sorted_indices = np.argsort(shap_values_abs_mean)[::-1]
    shap_values_abs_mean_sorted = shap_values_abs_mean[sorted_indices]
    feature_names_sorted = [feature_names[i] for i in sorted_indices]

    # Get the best max_features
    values = shap_values_abs_mean_sorted[:max_features]
    names = [feature_names_sorted[i] for i in range(max_features)]


    # Plot shap values
    sns.barplot(
        x=values, y=names,
        orient='h',
        color=sns.color_palette('Greys')[3],
        ax=ax
    )

    # Labels, ticks and layout adjustments
    ax.set_xlabel("mean(|SHAP value|)", fontsize=16)
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.3g}'))

    # Put figure letter above the y label
    fig.canvas.draw()  # Ensure the figure is rendered
    ylab_box = ax.yaxis.label.get_window_extent()
    ylab_pos_axes = ax.transAxes.inverted().transform(ylab_box.p0)
    ax.text(ylab_pos_axes[0], 1.06, fig_letter, ha='left', va='top', fontsize=20, fontweight='bold', transform=ax.transAxes)


def plot(
        y_true:np.ndarray,
        y_pred:np.ndarray,
        model_stats:Dict[str, Any],
        feature_names:List[str],
        shap_values:np.ndarray) -> None:
    """
    Main plot. Plots predictions with stats and feature importance (SHAP).

    Saves it as a vector graphics image named: "model_results.pdf"

    :param y_true: True targets
    :param y_pred: Predicted targets
    :param model_stats: Model statistics
    :param feature_names: List of feature names
    :param shap_values: Shap values
    :return:
    """
    # Create figure
    figsize = (1000, 500)
    def_dpi = plt.rcParams['figure.dpi']
    figsize = ((figsize[0]) / float(def_dpi), figsize[1] / float(def_dpi))
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 2)
    axs = [
        fig.add_subplot(gs[0, 0]),  # Predictions
        fig.add_subplot(gs[0, 1]),  # Feature importance
    ]
    fig_letters = ['A', 'B']

    # Plot predictions
    _plot_predictions(y_true, y_pred, model_stats, fig, axs[0], fig_letters[0])

    # Plot feature importance
    _plot_feature_importance(feature_names, shap_values, fig, axs[1], fig_letters[1])

    # Save figure
    fig.savefig('ml_results.pdf', format='pdf')


def run_model(
        data_path: str,
        num_outer_folds:int = 10,
        num_inner_folds:int = 10,
        rand_state:int = 42
) -> None:
    """
    Runs the model with default settings

    :param data_path: Path to the csv file
    :param num_outer_folds: Number of outer folds. (Default: 10)
    :param num_inner_folds: Number of inner folds. (Default: 10)
    :param rand_state: Random state for reproducibility (Default: 42)
    :return: None
    """
    # Load data
    data, X, y, feature_names = load_data(data_path)

    # Scale the target
    y_scaler = MinMaxScaler()
    y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Get default scorer, pipeline and parameter grid
    scorer, pipeline, param_grid = get_defaults()

    # Define outer and inner cvs
    outer_cv = KFold(n_splits=num_outer_folds, shuffle=True, random_state=rand_state)
    inner_cv = KFold(n_splits=num_inner_folds, shuffle=True, random_state=rand_state)

    # Placeholder for predictions
    y_pred = np.zeros_like(y)
    shap_values = []

    for train_idx, test_idx in tqdm(outer_cv.split(X), desc="Outer CV", total=outer_cv.get_n_splits()):
        # Split into train and test datasets
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner grid search
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scorer,
            cv=inner_cv,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Get best model and predict
        best_model = grid_search.best_estimator_
        y_pred[test_idx] = best_model.predict(X_test)

        # Compute feature importance with SHAP values on outer test set
        explainer = shap.KernelExplainer(best_model.predict, shap.sample(X_train, 100))
        shap_values.append(explainer.shap_values(X_test))

    # Concatenate the folds
    shap_values = np.concatenate(shap_values, axis=0)

    # Rescale predictions back to original scale
    y_true = y_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    y_pred_rescaled = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Quantile Mapping to align predicted distribution with true values
    y_true_sorted = np.sort(y_true)
    y_pred_sorted = np.sort(y_pred_rescaled)
    y_pred_rescaled_qm = np.interp(rankdata(y_pred_rescaled), rankdata(y_pred_sorted), y_true_sorted)

    # Calculate model statistics
    model_stats = calculate_model_stats(y_true, y_pred_rescaled)
    model_stats_qm = calculate_model_stats(y_true, y_pred_rescaled_qm)

    # Create plots
    plot(y_true, y_pred_rescaled, model_stats, feature_names, shap_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='Dataset/dummy_dataset.csv')
    parser.add_argument('--num_outer_folds', type=int, default=10)
    parser.add_argument('--num_inner_folds', type=int, default=10)
    parser.add_argument('--rand_state', type=int, default=42)

    config = vars(parser.parse_args())

    run_model(**config)
