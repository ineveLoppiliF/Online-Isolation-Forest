from matplotlib import rcParams, pyplot as plt
from numpy import array_split, asarray, empty, floor, hstack, linspace, meshgrid, ndarray, ones, vstack, zeros
from OnlineIForest import OnlineIForest
from scipy.stats import multivariate_normal, uniform
from sklearn.metrics import auc, roc_curve
from sklearn.utils import shuffle

#%% Define execution parameters
oiforest_params: dict = {'num_trees': 128,
                         'max_leaf_samples': 8,
                         'window_size': 1024}

n_points: int = 1000
fraction_outliers: float = 0.1
n_gauss: int = 3
mean_range: list = [[0, 1], [0, 1]]
variance: float = 0.005

batch_size: int = 1
n_plots: int = 11
n_evaluation_windows: int = 31

#%% Generate and plot synthetic dataset
n_inliers: int = int(floor((1-fraction_outliers)*n_points))
n_outliers: int = int(floor(fraction_outliers*n_points))

# Generate inliers
inliers: list = []
for i in range(n_gauss):
    inliers.extend(multivariate_normal.rvs(mean=[uniform.rvs(mean_range[0][0], mean_range[0][1] - mean_range[0][0]),
                                                 uniform.rvs(mean_range[1][0], mean_range[1][1] - mean_range[1][0])],
                                           cov=variance,
                                           size=int(floor(n_inliers/n_gauss))))
inliers: ndarray = asarray(inliers)

# Generate outliers
outliers: ndarray = vstack([uniform.rvs(mean_range[0][0], mean_range[0][1] - mean_range[0][0], (n_outliers,)),
                            uniform.rvs(mean_range[1][0], mean_range[1][1] - mean_range[1][0], (n_outliers,))]).T

# Compose dataset
data: ndarray = vstack([inliers, outliers])
inlier_mask: ndarray = hstack([ones((inliers.shape[0],), dtype=bool), zeros((outliers.shape[0],), dtype=bool)])

# Shuffle dataset
data, inlier_mask = shuffle(data, inlier_mask)

# Plot dataset
plt.rcParams["figure.figsize"] = (6, 6)
plt.figure()
plt.scatter(x=data[inlier_mask, 0], y=data[inlier_mask, 1], s=4*rcParams['lines.markersize'], c='#78D695')
plt.scatter(x=data[~inlier_mask, 0], y=data[~inlier_mask, 1], s=4*rcParams['lines.markersize'], c='#FF6969', marker='x')
plt.xlim([data[:, 0].min(), data[:, 0].max()])
plt.ylim([data[:, 1].min(), data[:, 1].max()])
x0, x1 = plt.xlim()
y0, y1 = plt.ylim()
plt.gca().set_aspect(abs(x1-x0)/abs(y1-y0))
plt.xticks([], [])
plt.yticks([], [])
plt.tight_layout()
plt.show()

#%% Execute Online-iForest and plot heatmaps of Anomaly Scores
data_indices_so_far: ndarray = empty(shape=(0,), dtype=int)
data_indices_current_window: ndarray = empty(shape=(0,), dtype=int)
scores: ndarray = empty(shape=(0,))
roc_aucs_cumulative: ndarray = empty(shape=(0,))
roc_aucs_windows: ndarray = empty(shape=(0,))

batch_splits: list = array_split(range(data.shape[0]), floor(data.shape[0]/batch_size)+1)
plots_times: ndarray = linspace(0, len(batch_splits)-1, n_plots, dtype=int)
evaluation_windows_times: ndarray = linspace(0, len(batch_splits)-1, n_evaluation_windows, dtype=int)

# Initialize Online Isolation Forest
oiforest: OnlineIForest = OnlineIForest.create(**oiforest_params)

# Execute Online Isolation Forest for each batch
for t, batch_indices in enumerate(batch_splits):
    print('t = ' + str(t))
    data_indices_so_far: ndarray = hstack([data_indices_so_far, batch_indices])
    data_indices_current_window: ndarray = hstack([data_indices_current_window, batch_indices])

    # Update Online Isolation Forest
    oiforest.learn_batch(data[batch_indices])

    # Score samples
    scores: ndarray = hstack([scores, oiforest.score_batch(data[batch_indices])])

    # Plot heatmap
    if t in plots_times:
        x: ndarray = linspace(data[:, 0].min(), data[:, 0].max(), 100)
        y: ndarray = linspace(data[:, 1].min(), data[:, 1].max(), 100)
        X, Y = meshgrid(x, y)
        coords: ndarray = vstack([X.ravel(), Y.ravel()]).T
        scores_heat: ndarray = oiforest.score_batch(coords)

        plt.rcParams["figure.figsize"] = (6, 6)
        plt.figure()
        plt.scatter(coords[:, 0], coords[:, 1], c=scores_heat, cmap='rainbow', alpha=0.8, marker='.', s=120,
                    edgecolors='none')
        plt.xlim([data[:, 0].min(), data[:, 0].max()])
        plt.ylim([data[:, 1].min(), data[:, 1].max()])
        plt.colorbar()
        x0, x1 = plt.xlim()
        y0, y1 = plt.ylim()
        plt.gca().set_aspect(abs(x1 - x0) / abs(y1 - y0))
        plt.xticks([], [])
        plt.yticks([], [])
        plt.tight_layout()
        plt.show()

    # Compute cumulative FPR, TPR and AUC
    fpr, tpr, _ = roc_curve(~inlier_mask[data_indices_so_far], scores[data_indices_so_far])
    roc_aucs_cumulative: ndarray = hstack([roc_aucs_cumulative, auc(fpr, tpr)])
    # Compute windowed cumulative FPR, TPR and AUC
    if t in evaluation_windows_times:
        fpr, tpr, _ = roc_curve(~inlier_mask[data_indices_current_window], scores[data_indices_current_window])
        roc_aucs_windows: ndarray = hstack([roc_aucs_windows, auc(fpr, tpr)])
        data_indices_current_window: ndarray = empty(shape=(0,), dtype=int)

#%% Plot cumulative and windowed AUC for each time instant
plt.figure()
plt.plot(range(int(floor(data.shape[0]/batch_size))+1), roc_aucs_cumulative, label='AUC cumulative (' + str(roc_aucs_cumulative[-1].round(decimals=2)) + ')')
plt.plot(evaluation_windows_times, roc_aucs_windows, label='AUC windows')
plt.ylim([0.0, 1.05])
plt.xlabel('batch #')
plt.ylabel('AUC')
plt.legend(loc="best")
plt.tight_layout()
plt.show()
