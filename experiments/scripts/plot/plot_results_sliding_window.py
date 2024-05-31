from matplotlib import pyplot as plt
from numpy import asarray, genfromtxt, nanmedian, ndarray
from pickle import load
from sklearn.metrics import roc_auc_score


# region Define parameters
dataset_algorithm_name: str = 'kqt'
dataset_type: str = 'non-stationary'
dataset_name: str = 'insects_ae-aegypti-female_cx-quinq-male_ae-albopictus-male'
change_points: list = [51255,  74381, 100944, 142291, 152207, 212514]
dataset_path: str = '../../../datasets/' + dataset_type + '/' + dataset_algorithm_name + '/' + dataset_name + '.csv'
results_path: str = '../../results/' + dataset_type + '/' + dataset_algorithm_name + '/' + dataset_name + '/'
n_batches: int = None
auc_quantiles: list = [0.5, 0.5]
time_quantiles: list = [0.5, 0.5]
auc_conv_num: int = 5000
auc_conv_step: int = 10

# Colorblind-friendly palette
colors = {
    'blue':    '#377eb8',
    'orange':  '#ff7f00',
    'green':   '#4daf4a',
    'pink':    '#f781bf',
    'brown':   '#a65628',
    'purple':  '#984ea3',
    'gray':    '#999999',
    'red':     '#e41a1c',
    'yellow':  '#dede00'
}
# endregion

# region Load and disaggregate results structures
with open(results_path + 'results.pkl', 'rb') as handle:
    results: dict = load(handle)

ifor_results: dict = results['ifor']
boif_results: dict = results['boif']
ifasd_results: dict = results['ifasd']
hst_results: dict = results['hst']
rrcf_results: dict = results['rrcf']
loda_results: dict = results['loda']
oif_results: dict = results['oif']
# endregion

# region Load information for plots
with open(results_path + 'info.pkl', 'rb') as handle:
    info: dict = load(handle)

batch_size: int = info['batch_size']
n_batches: int = n_batches if n_batches else info['n_batches']
evaluation_windows_times: ndarray = info['evaluation_windows_times']
evaluation_holdout_times: ndarray = info['evaluation_holdout_times']
# endregion

# region Load dataset and collect inlier mask
dataset: ndarray = genfromtxt(dataset_path, delimiter=',')
dataset: ndarray = dataset[1:, 1:]
inlier_mask: ndarray = ~dataset[:, -1].astype(dtype=bool)
# endregion

# region Plot results

# region Plot smoothed convoluted AUC, change points and ground truth for each time instant
plt.figure()

# asdIFOR
ifasd_scores: ndarray = nanmedian(ifasd_results['scores'], axis=0)
ifasd_aucs_conv: list = []
for i in asarray(range(ifasd_scores.shape[0]-auc_conv_num)):
    if (i % auc_conv_step) == 0:
        try:
            ifasd_aucs_conv.append(roc_auc_score(~inlier_mask[i:i+auc_conv_num], ifasd_scores[i:i+auc_conv_num]))
        except ValueError:
            ifasd_aucs_conv.append(None)
ifasd_aucs_conv: ndarray = asarray(ifasd_aucs_conv)
plt.plot((asarray(range(ifasd_aucs_conv.shape[0]))/ifasd_aucs_conv.shape[0])*ifasd_scores.shape[0], ifasd_aucs_conv, color=colors['orange'], linewidth=0.75, alpha=0.75)
# HST
hst_scores: ndarray = nanmedian(hst_results['scores'], axis=0)
hst_aucs_conv: list = []
for i in asarray(range(hst_scores.shape[0]-auc_conv_num)):
    if (i % auc_conv_step) == 0:
        try:
            hst_aucs_conv.append(roc_auc_score(~inlier_mask[i:i+auc_conv_num], hst_scores[i:i+auc_conv_num]))
        except ValueError:
            hst_aucs_conv.append(None)
hst_aucs_conv: ndarray = asarray(hst_aucs_conv)
plt.plot((asarray(range(hst_aucs_conv.shape[0]))/hst_aucs_conv.shape[0])*hst_scores.shape[0], hst_aucs_conv, color=colors['purple'], linewidth=0.75, alpha=0.75)
# RRCF
rrcf_scores: ndarray = nanmedian(rrcf_results['scores'], axis=0)
rrcf_aucs_conv: list = []
for i in asarray(range(rrcf_scores.shape[0]-auc_conv_num)):
    if (i % auc_conv_step) == 0:
        try:
            rrcf_aucs_conv.append(roc_auc_score(~inlier_mask[i:i+auc_conv_num], rrcf_scores[i:i+auc_conv_num]))
        except ValueError:
            rrcf_aucs_conv.append(None)
rrcf_aucs_conv: ndarray = asarray(rrcf_aucs_conv)
plt.plot((asarray(range(rrcf_aucs_conv.shape[0]))/rrcf_aucs_conv.shape[0])*rrcf_scores.shape[0], rrcf_aucs_conv, color=colors['brown'], linewidth=0.75, alpha=0.75)
# LODA
loda_scores: ndarray = nanmedian(loda_results['scores'], axis=0)
loda_aucs_conv: list = []
for i in asarray(range(loda_scores.shape[0]-auc_conv_num)):
    if (i % auc_conv_step) == 0:
        try:
            loda_aucs_conv.append(roc_auc_score(~inlier_mask[i:i+auc_conv_num], loda_scores[i:i+auc_conv_num]))
        except ValueError:
            loda_aucs_conv.append(None)
loda_aucs_conv: ndarray = asarray(loda_aucs_conv)
plt.plot((asarray(range(loda_aucs_conv.shape[0]))/loda_aucs_conv.shape[0])*loda_scores.shape[0], loda_aucs_conv, color=colors['pink'], linewidth=0.75, alpha=0.75)
# oIFOR
oif_scores: ndarray = nanmedian(oif_results['scores'], axis=0)
oif_aucs_conv: list = []
for i in asarray(range(oif_scores.shape[0]-auc_conv_num)):
    if (i % auc_conv_step) == 0:
        try:
            oif_aucs_conv.append(roc_auc_score(~inlier_mask[i:i+auc_conv_num], oif_scores[i:i+auc_conv_num]))
        except ValueError:
            oif_aucs_conv.append(None)
oif_aucs_conv: ndarray = asarray(oif_aucs_conv)
plt.plot((asarray(range(oif_aucs_conv.shape[0]))/oif_aucs_conv.shape[0])*oif_scores.shape[0], oif_aucs_conv, color=colors['green'], linewidth=0.75, alpha=0.75)
# Change points
for p in change_points[:-1]:
    plt.axvline(p, color=colors['red'], linestyle='dotted', linewidth=1)

plt.ylim([0.0, 1.05])
plt.xlabel('# samples', fontsize=16)
plt.ylabel('AUC', fontsize=16)
plt.tight_layout()
plt.savefig(results_path + 'convoluted_aucs.pdf', format='pdf')
plt.close()
#plt.show()

# endregion

# endregion
