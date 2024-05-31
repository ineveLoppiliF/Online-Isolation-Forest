from autorank import autorank, plot_stats
from collections import defaultdict
from matplotlib import pyplot as plt
from numpy import asarray, cumsum, hstack, nanmedian, nanquantile, ndarray, vstack
from pandas import DataFrame
from pickle import load


# region Define parameters
datasets_algorithms_name: list = ['ifor', 'hst', 'rrcf']
datasets_type: str = 'stationary'
datasets_name: list = [['Http', 'ForestCover', 'Mulcross', 'Smtp', 'Shuttle', 'Mammography', 'Annthyroid', 'Satellite'],
                       ['donors', 'fraud'],
                       ['nyc_taxi_shingle']]
datasets_cardinalities: list = [[567497, 286048, 262144, 95156, 49097, 11183, 6832, 6435],
                                [619326, 284807],
                                [10273]]
datasets_prop_anomalies: list = [[0.4, 0.9, 10, 0.03, 7, 2, 7, 32],
                                 [5.9, 0.17],
                                 [5.2]]
competitors_name: list = ['ifasd', 'hst', 'rrcf', 'loda', 'oif']
n_batches: int = 11
auc_quantiles: list = [0.5, 0.5]
time_quantiles: list = [0.5, 0.5]

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

# region Load and rearrange results structures
results: dict = defaultdict(dict)
for i, datasets_algorithm_name in enumerate(datasets_algorithms_name):
    for j, dataset_name in enumerate(datasets_name[i]):
        results_path: str = '../../results/' + datasets_type + '/' + datasets_algorithms_name[i] + '/' + datasets_name[i][j] + '/'
        with open(results_path + 'results.pkl', 'rb') as handle:
            results[datasets_algorithm_name][dataset_name]: dict = load(handle)
# endregion

# region Merge results structures among all methods and datasets
results_merged: dict = defaultdict(dict)
for competitor_name in competitors_name:
    results_merged_temp: list = []
    for i, datasets_algorithm_name in enumerate(datasets_algorithms_name):
        for j, dataset_name in enumerate(datasets_name[i]):
            results_merged_temp.append(results[datasets_algorithm_name][dataset_name][competitor_name])

    results_merged_temp_new: dict = defaultdict(dict)
    for key in results_merged_temp[0].keys():
        results_merged_temp_new[key]: list = vstack([[result[:n_batches] for result in result_merged_temp[key]]
                                                     for result_merged_temp in results_merged_temp])

    results_merged[competitor_name]: dict = results_merged_temp_new

ifasd_results: dict = results_merged['ifasd']
hst_results: dict = results_merged['hst']
rrcf_results: dict = results_merged['rrcf']
loda_results: dict = results_merged['loda']
oif_results: dict = results_merged['oif']
# endregion

# region Load and rearrange information structures
info: dict = defaultdict(dict)
for i, datasets_algorithm_name in enumerate(datasets_algorithms_name):
    for j, dataset_name in enumerate(datasets_name[i]):
        results_path: str = '../../results/' + datasets_type + '/' + datasets_algorithms_name[i] + '/' + datasets_name[i][j] + '/'
        with open(results_path + 'info.pkl', 'rb') as handle:
            info[datasets_algorithm_name][dataset_name]: dict = load(handle)

batch_size: int = info[datasets_algorithms_name[0]][datasets_name[0][0]]['batch_size']
n_batches: int = n_batches if n_batches else info[datasets_algorithms_name[0]][datasets_name[0][0]]['n_batches']
# endregion

# region Plot results

# region Plot AUC

# region Plot final cumulative AUC for each method for each dataset
plt.figure()

prop_anomalies: ndarray = hstack(datasets_prop_anomalies)
labels: ndarray = hstack(datasets_name)
sort_idx: ndarray = prop_anomalies.argsort()
# asdIFOR
roc_aucs: ndarray = asarray([nanmedian(results[datasets_algorithm_name][dataset_name]['ifasd']['roc_aucs_cumulative'], axis=0)[-1]
                          for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                          for j, dataset_name in enumerate(datasets_name[i])])
plt.plot(prop_anomalies[sort_idx], roc_aucs[sort_idx], c=colors['orange'], label='asdIFOR', marker='s', alpha=0.75, markersize=8, markeredgewidth=0.0)
# HST
roc_aucs: ndarray = asarray([nanmedian(results[datasets_algorithm_name][dataset_name]['hst']['roc_aucs_cumulative'], axis=0)[-1]
                             for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                             for j, dataset_name in enumerate(datasets_name[i])])
plt.plot(prop_anomalies[sort_idx], roc_aucs[sort_idx], c=colors['purple'], label='HST', marker='D', alpha=0.75, markersize=8, markeredgewidth=0.0)
# RRCF
roc_aucs: ndarray = asarray([nanmedian(results[datasets_algorithm_name][dataset_name]['rrcf']['roc_aucs_cumulative'], axis=0)[-1]
                             for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                             for j, dataset_name in enumerate(datasets_name[i])])
plt.plot(prop_anomalies[sort_idx], roc_aucs[sort_idx], c=colors['brown'], label='RRCF', marker='^', alpha=0.75, markersize=8, markeredgewidth=0.0)
# LODA
roc_aucs: ndarray = asarray([nanmedian(results[datasets_algorithm_name][dataset_name]['loda']['roc_aucs_cumulative'], axis=0)[-1]
                             for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                             for j, dataset_name in enumerate(datasets_name[i])])
plt.plot(prop_anomalies[sort_idx], roc_aucs[sort_idx], c=colors['pink'], label='LODA', marker='v', alpha=0.75, markersize=8, markeredgewidth=0.0)
# oIFOR
roc_aucs: ndarray = asarray([nanmedian(results[datasets_algorithm_name][dataset_name]['oif']['roc_aucs_cumulative'], axis=0)[-1]
                             for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                             for j, dataset_name in enumerate(datasets_name[i])])
plt.plot(prop_anomalies[sort_idx], roc_aucs[sort_idx], c=colors['green'], label='oIFOR', marker='o', alpha=0.75, markersize=8, markeredgewidth=0.0)

axc = plt.gca()
axc.set_xscale('log')
axc.set_xlabel('% anomalies', fontsize=16)
ax = axc.twiny()
ax.set_xscale('log')
ax.set_xticks(ticks=prop_anomalies, labels=labels, rotation=90)
ax.set_xlim(axc.get_xlim())
ax.set_xlabel('Dataset', fontsize=16)
axc.set_ylim([0.0, 1.05])
axc.set_ylabel('AUC', fontsize=16)
axc.legend(loc='best')
plt.tight_layout()
plt.savefig('../../results/' + datasets_type + '/' + 'roc_aucs.pdf', format='pdf')
plt.close()
#plt.show()

# endregion

# region Plot cumulative AUC for each method for each time instant in the initial window
plt.figure()

# asdIFOR
plt.plot(asarray(range(n_batches))*batch_size, nanmedian(ifasd_results['roc_aucs_cumulative'], axis=0)[:n_batches], label='asdIFOR (' + str(round(nanmedian(ifasd_results['roc_aucs_cumulative'], axis=0)[:n_batches][-1], ndigits=2)) + ')', color=colors['orange'], alpha=0.75)
plt.fill_between(asarray(range(n_batches))*batch_size, nanquantile(ifasd_results['roc_aucs_cumulative'], auc_quantiles, axis=0)[0, :][:n_batches], nanquantile(ifasd_results['roc_aucs_cumulative'], auc_quantiles, axis=0)[1, :][:n_batches], facecolor=colors['orange'], alpha=0.15)
# HST
plt.plot(asarray(range(n_batches))*batch_size, nanmedian(hst_results['roc_aucs_cumulative'], axis=0)[:n_batches], label='HST (' + str(round(nanmedian(hst_results['roc_aucs_cumulative'], axis=0)[:n_batches][-1], ndigits=2)) + ')', color=colors['purple'], alpha=0.75)
plt.fill_between(asarray(range(n_batches))*batch_size, nanquantile(hst_results['roc_aucs_cumulative'], auc_quantiles, axis=0)[0, :][:n_batches], nanquantile(hst_results['roc_aucs_cumulative'], auc_quantiles, axis=0)[1, :][:n_batches], facecolor=colors['purple'], alpha=0.15)
# RRCF
plt.plot(asarray(range(n_batches))*batch_size, nanmedian(rrcf_results['roc_aucs_cumulative'], axis=0)[:n_batches], label='RRCF (' + str(round(nanmedian(rrcf_results['roc_aucs_cumulative'], axis=0)[:n_batches][-1], ndigits=2)) + ')', color=colors['brown'], alpha=0.75)
plt.fill_between(asarray(range(n_batches))*batch_size, nanquantile(rrcf_results['roc_aucs_cumulative'], auc_quantiles, axis=0)[0, :][:n_batches], nanquantile(rrcf_results['roc_aucs_cumulative'], auc_quantiles, axis=0)[1, :][:n_batches], facecolor=colors['brown'], alpha=0.15)
# LODA
plt.plot(asarray(range(n_batches))*batch_size, nanmedian(loda_results['roc_aucs_cumulative'], axis=0)[:n_batches], label='LODA (' + str(round(nanmedian(loda_results['roc_aucs_cumulative'], axis=0)[:n_batches][-1], ndigits=2)) + ')', color=colors['pink'], alpha=0.75)
plt.fill_between(asarray(range(n_batches))*batch_size, nanquantile(loda_results['roc_aucs_cumulative'], auc_quantiles, axis=0)[0, :][:n_batches], nanquantile(loda_results['roc_aucs_cumulative'], auc_quantiles, axis=0)[1, :][:n_batches], facecolor=colors['pink'], alpha=0.15)
# oIFOR
plt.plot(asarray(range(n_batches))*batch_size, nanmedian(oif_results['roc_aucs_cumulative'], axis=0)[:n_batches], label='oIFOR (' + str(round(nanmedian(oif_results['roc_aucs_cumulative'], axis=0)[:n_batches][-1], ndigits=2)) + ')', color=colors['green'], alpha=0.75)
plt.fill_between(asarray(range(n_batches))*batch_size, nanquantile(oif_results['roc_aucs_cumulative'], auc_quantiles, axis=0)[0, :][:n_batches], nanquantile(oif_results['roc_aucs_cumulative'], auc_quantiles, axis=0)[1, :][:n_batches], facecolor=colors['green'], alpha=0.15)

plt.ylim([0.45, 1.05])
plt.xlabel('# samples', fontsize=16)
plt.ylabel('AUC', fontsize=16)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('../../results/' + datasets_type + '/' + 'roc_aucs_initial.pdf', format='pdf')
plt.close()
#plt.show()
# endregion

# region Plot Critical Difference chart for AUC
auc_flattened_table: dict = defaultdict(dict)

# asdIFOR
auc_flattened_table['asdIFOR']: ndarray = hstack([[roc_aucs_cumulative[-1] for roc_aucs_cumulative in results[datasets_algorithm_name][dataset_name]['ifasd']['roc_aucs_cumulative']]
                                                for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                                for j, dataset_name in enumerate(datasets_name[i])])
# HST
auc_flattened_table['HST']: ndarray = hstack([[roc_aucs_cumulative[-1] for roc_aucs_cumulative in results[datasets_algorithm_name][dataset_name]['hst']['roc_aucs_cumulative']]
                                              for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                              for j, dataset_name in enumerate(datasets_name[i])])
# RRCF
auc_flattened_table['RRCF']: ndarray = hstack([[roc_aucs_cumulative[-1] for roc_aucs_cumulative in results[datasets_algorithm_name][dataset_name]['rrcf']['roc_aucs_cumulative']]
                                               for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                               for j, dataset_name in enumerate(datasets_name[i])])
# LODA
auc_flattened_table['LODA']: ndarray = hstack([[roc_aucs_cumulative[-1] for roc_aucs_cumulative in results[datasets_algorithm_name][dataset_name]['loda']['roc_aucs_cumulative']]
                                               for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                               for j, dataset_name in enumerate(datasets_name[i])])
# oIFOR
auc_flattened_table['oIFOR']: ndarray = hstack([[roc_aucs_cumulative[-1] for roc_aucs_cumulative in results[datasets_algorithm_name][dataset_name]['oif']['roc_aucs_cumulative']]
                                              for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                              for j, dataset_name in enumerate(datasets_name[i])])

auc_flattened_dataframe: DataFrame = DataFrame.from_dict(data=auc_flattened_table)
autorank_result = autorank(auc_flattened_dataframe, alpha=0.05, verbose=False)

plt.figure()
plot_stats(autorank_result)
plt.tight_layout()
plt.savefig('../../results/' + datasets_type + '/' + 'cd_roc_aucs.pdf', format='pdf')
#plt.show()
# endregion

# endregion

# region Plot time

# region Plot total time for each method for each dataset
plt.figure()

cardinalities: ndarray = hstack(datasets_cardinalities)
labels: ndarray = hstack(datasets_name)
sort_idx: ndarray = cardinalities.argsort()
# asdIFOR
total_times: ndarray = asarray([(nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['ifasd']['train_times'], axis=1), axis=0)+nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['ifasd']['test_times'], axis=1), axis=0))[-1]
                                for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                for j, dataset_name in enumerate(datasets_name[i])])
plt.plot(cardinalities[sort_idx], total_times[sort_idx], c=colors['orange'], label='asdIFOR', marker='s', alpha=0.75, markersize=8, markeredgewidth=0.0)
# HST
total_times: ndarray = asarray([(nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['hst']['train_times'], axis=1), axis=0)+nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['hst']['test_times'], axis=1), axis=0))[-1]
                                for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                for j, dataset_name in enumerate(datasets_name[i])])
plt.plot(cardinalities[sort_idx], total_times[sort_idx], c=colors['purple'], label='HST', marker='D', alpha=0.75, markersize=8, markeredgewidth=0.0)
# RRCF
total_times: ndarray = asarray([(nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['rrcf']['train_times'], axis=1), axis=0)+nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['rrcf']['test_times'], axis=1), axis=0))[-1]
                                for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                for j, dataset_name in enumerate(datasets_name[i])])
plt.plot(cardinalities[sort_idx], total_times[sort_idx], c=colors['brown'], label='RRCF', marker='^', alpha=0.75, markersize=8, markeredgewidth=0.0)
# LODA
total_times: ndarray = asarray([(nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['loda']['train_times'], axis=1), axis=0)+nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['loda']['test_times'], axis=1), axis=0))[-1]
                                for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                for j, dataset_name in enumerate(datasets_name[i])])
plt.plot(cardinalities[sort_idx], total_times[sort_idx], c=colors['pink'], label='LODA', marker='v', alpha=0.75, markersize=8, markeredgewidth=0.0)
# oIFOR
total_times: ndarray = asarray([(nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['oif']['train_times'], axis=1), axis=0)+nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['oif']['test_times'], axis=1), axis=0))[-1]
                                for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                for j, dataset_name in enumerate(datasets_name[i])])
plt.plot(cardinalities[sort_idx], total_times[sort_idx], c=colors['green'], label='oIFOR', marker='o', alpha=0.75, markersize=8, markeredgewidth=0.0)

axc = plt.gca()
axc.set_xscale('log')
axc.set_xlabel('# samples', fontsize=16)
ax = axc.twiny()
ax.set_xscale('log')
ax.set_xticks(ticks=cardinalities, labels=labels, rotation=90)
ax.set_xlim(axc.get_xlim())
ax.set_xlabel('Dataset', fontsize=16)
axc.set_yscale('log')
axc.set_ylabel('time (s)', fontsize=16)
axc.legend(loc='best')
plt.tight_layout()
plt.savefig('../../results/' + datasets_type + '/' + 'times.pdf', format='pdf')
plt.close()
#plt.show()

# endregion

# region Plot total time for each method for each time instant in the initial window
plt.figure()

# asdIFOR
plt.plot(asarray(range(n_batches))*batch_size, nanmedian(cumsum(ifasd_results['train_times'], axis=1), axis=0)[:n_batches]+nanmedian(cumsum(ifasd_results['test_times'], axis=1), axis=0)[:n_batches], label='asdIFOR (' + str((nanmedian(cumsum(ifasd_results['train_times'], axis=1), axis=0)[:n_batches]+nanmedian(cumsum(ifasd_results['test_times'], axis=1), axis=0)[:n_batches])[-1].round(decimals=1)) + 's)', color=colors['orange'], alpha=0.75)
plt.fill_between(asarray(range(n_batches))*batch_size, nanquantile(cumsum(ifasd_results['train_times'], axis=1), time_quantiles, axis=0)[0, :][:n_batches]+nanquantile(cumsum(ifasd_results['test_times'], axis=1), time_quantiles, axis=0)[0, :][:n_batches], nanquantile(cumsum(ifasd_results['train_times'], axis=1), time_quantiles, axis=0)[1, :][:n_batches]+nanquantile(cumsum(ifasd_results['test_times'], axis=1), time_quantiles, axis=0)[1, :][:n_batches], facecolor=colors['orange'], alpha=0.15)
# HST
plt.plot(asarray(range(n_batches))*batch_size, nanmedian(cumsum(hst_results['train_times'], axis=1), axis=0)[:n_batches]+nanmedian(cumsum(hst_results['test_times'], axis=1), axis=0)[:n_batches], label='HST (' + str((nanmedian(cumsum(hst_results['train_times'], axis=1), axis=0)[:n_batches]+nanmedian(cumsum(hst_results['test_times'], axis=1), axis=0)[:n_batches])[-1].round(decimals=1)) + 's)', color=colors['purple'], alpha=0.75)
plt.fill_between(asarray(range(n_batches))*batch_size, nanquantile(cumsum(hst_results['train_times'], axis=1), time_quantiles, axis=0)[0, :][:n_batches]+nanquantile(cumsum(hst_results['test_times'], axis=1), time_quantiles, axis=0)[0, :][:n_batches], nanquantile(cumsum(hst_results['train_times'], axis=1), time_quantiles, axis=0)[1, :][:n_batches]+nanquantile(cumsum(hst_results['test_times'], axis=1), time_quantiles, axis=0)[1, :][:n_batches], facecolor=colors['purple'], alpha=0.15)
# RRCF
plt.plot(asarray(range(n_batches))*batch_size, nanmedian(cumsum(rrcf_results['train_times'], axis=1), axis=0)[:n_batches]+nanmedian(cumsum(rrcf_results['test_times'], axis=1), axis=0)[:n_batches], label='RRCF (' + str((nanmedian(cumsum(rrcf_results['train_times'], axis=1), axis=0)[:n_batches]+nanmedian(cumsum(rrcf_results['test_times'], axis=1), axis=0)[:n_batches])[-1].round(decimals=1)) + 's)', color=colors['brown'], alpha=0.75)
plt.fill_between(asarray(range(n_batches))*batch_size, nanquantile(cumsum(rrcf_results['train_times'], axis=1), time_quantiles, axis=0)[0, :][:n_batches]+nanquantile(cumsum(rrcf_results['test_times'], axis=1), time_quantiles, axis=0)[0, :][:n_batches], nanquantile(cumsum(rrcf_results['train_times'], axis=1), time_quantiles, axis=0)[1, :][:n_batches]+nanquantile(cumsum(rrcf_results['test_times'], axis=1), time_quantiles, axis=0)[1, :][:n_batches], facecolor=colors['brown'], alpha=0.15)
# LODA
plt.plot(asarray(range(n_batches))*batch_size, nanmedian(cumsum(loda_results['train_times'], axis=1), axis=0)[:n_batches]+nanmedian(cumsum(loda_results['test_times'], axis=1), axis=0)[:n_batches], label='LODA (' + str((nanmedian(cumsum(loda_results['train_times'], axis=1), axis=0)[:n_batches]+nanmedian(cumsum(loda_results['test_times'], axis=1), axis=0)[:n_batches])[-1].round(decimals=1)) + 's)', color=colors['pink'], alpha=0.75)
plt.fill_between(asarray(range(n_batches))*batch_size, nanquantile(cumsum(loda_results['train_times'], axis=1), time_quantiles, axis=0)[0, :][:n_batches]+nanquantile(cumsum(loda_results['test_times'], axis=1), time_quantiles, axis=0)[0, :][:n_batches], nanquantile(cumsum(loda_results['train_times'], axis=1), time_quantiles, axis=0)[1, :][:n_batches]+nanquantile(cumsum(loda_results['test_times'], axis=1), time_quantiles, axis=0)[1, :][:n_batches], facecolor=colors['pink'], alpha=0.15)
# oIFOR
plt.plot(asarray(range(n_batches))*batch_size, nanmedian(cumsum(oif_results['train_times'], axis=1), axis=0)[:n_batches]+nanmedian(cumsum(oif_results['test_times'], axis=1), axis=0)[:n_batches], label='oIFOR (' + str((nanmedian(cumsum(oif_results['train_times'], axis=1), axis=0)[:n_batches]+nanmedian(cumsum(oif_results['test_times'], axis=1), axis=0)[:n_batches])[-1].round(decimals=1)) + 's)', color=colors['green'], alpha=0.75)
plt.fill_between(asarray(range(n_batches))*batch_size, nanquantile(cumsum(oif_results['train_times'], axis=1), time_quantiles, axis=0)[0, :][:n_batches]+nanquantile(cumsum(oif_results['test_times'], axis=1), time_quantiles, axis=0)[0, :][:n_batches], nanquantile(cumsum(oif_results['train_times'], axis=1), time_quantiles, axis=0)[1, :][:n_batches]+nanquantile(cumsum(oif_results['test_times'], axis=1), time_quantiles, axis=0)[1, :][:n_batches], facecolor=colors['green'], alpha=0.15)

axc = plt.gca()
axc.set_yscale('log')
plt.xlabel('# samples', fontsize=16)
plt.ylabel('time (s)', fontsize=16)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('../../results/' + datasets_type + '/' + 'times_initial.pdf', format='pdf')
plt.close()
#plt.show()
# endregion

# region Plot Critical Difference chart for total time
time_flattened_table: dict = defaultdict(dict)

# asdIFOR
time_flattened_table['asdIFOR']: ndarray = -hstack([[train_time[-1]+test_time[-1] for train_time, test_time in zip(cumsum(results[datasets_algorithm_name][dataset_name]['ifasd']['train_times'], axis=1),
                                                                                                                cumsum(results[datasets_algorithm_name][dataset_name]['ifasd']['test_times'], axis=1))]
                                                 for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                                 for j, dataset_name in enumerate(datasets_name[i])])
# HST
time_flattened_table['HST']: ndarray = -hstack([[train_time[-1]+test_time[-1] for train_time, test_time in zip(cumsum(results[datasets_algorithm_name][dataset_name]['hst']['train_times'], axis=1),
                                                                                                              cumsum(results[datasets_algorithm_name][dataset_name]['hst']['test_times'], axis=1))]
                                               for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                               for j, dataset_name in enumerate(datasets_name[i])])
# RRCF
time_flattened_table['RRCF']: ndarray = -hstack([[train_time[-1]+test_time[-1] for train_time, test_time in zip(cumsum(results[datasets_algorithm_name][dataset_name]['rrcf']['train_times'], axis=1),
                                                                                                               cumsum(results[datasets_algorithm_name][dataset_name]['rrcf']['test_times'], axis=1))]
                                                for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                                for j, dataset_name in enumerate(datasets_name[i])])
# LODA
time_flattened_table['LODA']: ndarray = -hstack([[train_time[-1]+test_time[-1] for train_time, test_time in zip(cumsum(results[datasets_algorithm_name][dataset_name]['loda']['train_times'], axis=1),
                                                                                                               cumsum(results[datasets_algorithm_name][dataset_name]['loda']['test_times'], axis=1))]
                                                for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                                for j, dataset_name in enumerate(datasets_name[i])])
# oIFOR
time_flattened_table['oIFOR']: ndarray = -hstack([[train_time[-1]+test_time[-1] for train_time, test_time in zip(cumsum(results[datasets_algorithm_name][dataset_name]['oif']['train_times'], axis=1),
                                                                                                              cumsum(results[datasets_algorithm_name][dataset_name]['oif']['test_times'], axis=1))]
                                               for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                               for j, dataset_name in enumerate(datasets_name[i])])

time_flattened_dataframe: DataFrame = DataFrame.from_dict(data=time_flattened_table)
autorank_result = autorank(time_flattened_dataframe, alpha=0.05, verbose=False)

plt.figure()
plot_stats(autorank_result)
plt.tight_layout()
plt.savefig('../../results/' + datasets_type + '/' + 'cd_times.pdf', format='pdf')
#plt.show()
# endregion

# endregion

# endregion

# region Save tables
labels: ndarray = hstack(datasets_name)

# region Save final cumulative AUC for each method for each dataset
auc_table: dict = defaultdict(dict)

# asdIFOR
auc_table['ifasd']: list = ([round(nanmedian(results[datasets_algorithm_name][dataset_name]['ifasd']['roc_aucs_cumulative'], axis=0)[-1], ndigits=3)
                            for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                            for j, dataset_name in enumerate(datasets_name[i])] +
                            [round(nanmedian(hstack([[roc_aucs_cumulative[-1] for roc_aucs_cumulative in results[datasets_algorithm_name][dataset_name]['ifasd']['roc_aucs_cumulative']]
                                                     for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                                     for j, dataset_name in enumerate(datasets_name[i])]), axis=0), ndigits=3)])
# HST
auc_table['hst']: list = ([round(nanmedian(results[datasets_algorithm_name][dataset_name]['hst']['roc_aucs_cumulative'], axis=0)[-1], ndigits=3)
                          for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                          for j, dataset_name in enumerate(datasets_name[i])] +
                          [round(nanmedian(hstack([[roc_aucs_cumulative[-1] for roc_aucs_cumulative in results[datasets_algorithm_name][dataset_name]['hst']['roc_aucs_cumulative']]
                                                   for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                                   for j, dataset_name in enumerate(datasets_name[i])]), axis=0), ndigits=3)])
# RRCF
auc_table['rrcf']: list = ([round(nanmedian(results[datasets_algorithm_name][dataset_name]['rrcf']['roc_aucs_cumulative'], axis=0)[-1], ndigits=3)
                           for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                           for j, dataset_name in enumerate(datasets_name[i])] +
                           [round(nanmedian(hstack([[roc_aucs_cumulative[-1] for roc_aucs_cumulative in results[datasets_algorithm_name][dataset_name]['rrcf']['roc_aucs_cumulative']]
                                                    for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                                    for j, dataset_name in enumerate(datasets_name[i])]), axis=0), ndigits=3)])
# LODA
auc_table['loda']: list = ([round(nanmedian(results[datasets_algorithm_name][dataset_name]['loda']['roc_aucs_cumulative'], axis=0)[-1], ndigits=3)
                           for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                           for j, dataset_name in enumerate(datasets_name[i])] +
                           [round(nanmedian(hstack([[roc_aucs_cumulative[-1] for roc_aucs_cumulative in results[datasets_algorithm_name][dataset_name]['loda']['roc_aucs_cumulative']]
                                                    for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                                    for j, dataset_name in enumerate(datasets_name[i])]), axis=0), ndigits=3)])
# oIFOR
auc_table['oif']: list = ([round(nanmedian(results[datasets_algorithm_name][dataset_name]['oif']['roc_aucs_cumulative'], axis=0)[-1], ndigits=3)
                           for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                           for j, dataset_name in enumerate(datasets_name[i])] +
                          [round(nanmedian(hstack([[roc_aucs_cumulative[-1] for roc_aucs_cumulative in results[datasets_algorithm_name][dataset_name]['oif']['roc_aucs_cumulative']]
                                                   for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                                   for j, dataset_name in enumerate(datasets_name[i])]), axis=0), ndigits=3)])

auc_table['cardinality']: list = hstack(datasets_cardinalities)
auc_dataframe: DataFrame = DataFrame.from_dict(data=auc_table, orient='index', columns=hstack([labels, ['median']])).transpose()
auc_dataframe.sort_values(by='cardinality', axis=0, ascending=False, inplace=True)
auc_dataframe.drop(labels='cardinality', axis=1, inplace=True)
auc_dataframe.loc['median_rank'] = auc_dataframe.iloc[:-1].rank(axis=1, ascending=False).median(axis=0)
auc_dataframe.loc['mean_rank'] = auc_dataframe.iloc[:-1].rank(axis=1, ascending=False).mean(axis=0)
auc_dataframe.to_csv('../../results/' + datasets_type + '/' + 'roc_aucs.csv')
# endregion

# region Save total time for each method for each dataset
time_table: dict = defaultdict(dict)

# asdIFOR
time_table['ifasd']: list = ([round((nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['ifasd']['train_times'], axis=1), axis=0)+nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['ifasd']['test_times'], axis=1), axis=0))[-1], ndigits=2)
                             for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                             for j, dataset_name in enumerate(datasets_name[i])] +
                             [round(nanmedian(hstack([[train_time[-1]+test_time[-1] for train_time, test_time in zip(cumsum(results[datasets_algorithm_name][dataset_name]['ifasd']['train_times'], axis=1), cumsum(results[datasets_algorithm_name][dataset_name]['ifasd']['test_times'], axis=1))]
                                                      for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                                      for j, dataset_name in enumerate(datasets_name[i])]), axis=0), ndigits=3)])
# HST
time_table['hst']: list = ([round((nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['hst']['train_times'], axis=1), axis=0)+nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['hst']['test_times'], axis=1), axis=0))[-1], ndigits=2)
                           for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                           for j, dataset_name in enumerate(datasets_name[i])] +
                           [round(nanmedian(hstack([[train_time[-1]+test_time[-1] for train_time, test_time in zip(cumsum(results[datasets_algorithm_name][dataset_name]['hst']['train_times'], axis=1), cumsum(results[datasets_algorithm_name][dataset_name]['hst']['test_times'], axis=1))]
                                                    for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                                    for j, dataset_name in enumerate(datasets_name[i])]), axis=0), ndigits=3)])
# RRCF
time_table['rrcf']: list = ([round((nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['rrcf']['train_times'], axis=1), axis=0)+nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['rrcf']['test_times'], axis=1), axis=0))[-1], ndigits=2)
                            for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                            for j, dataset_name in enumerate(datasets_name[i])] +
                            [round(nanmedian(hstack([[train_time[-1]+test_time[-1] for train_time, test_time in zip(cumsum(results[datasets_algorithm_name][dataset_name]['rrcf']['train_times'], axis=1), cumsum(results[datasets_algorithm_name][dataset_name]['rrcf']['test_times'], axis=1))]
                                                     for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                                     for j, dataset_name in enumerate(datasets_name[i])]), axis=0), ndigits=3)])
# LODA
time_table['loda']: list = ([round((nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['loda']['train_times'], axis=1), axis=0)+nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['loda']['test_times'], axis=1), axis=0))[-1], ndigits=2)
                            for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                            for j, dataset_name in enumerate(datasets_name[i])] +
                            [round(nanmedian(hstack([[train_time[-1]+test_time[-1] for train_time, test_time in zip(cumsum(results[datasets_algorithm_name][dataset_name]['loda']['train_times'], axis=1), cumsum(results[datasets_algorithm_name][dataset_name]['loda']['test_times'], axis=1))]
                                                     for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                                     for j, dataset_name in enumerate(datasets_name[i])]), axis=0), ndigits=3)])
# oIFOR
time_table['oif']: list = ([round((nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['oif']['train_times'], axis=1), axis=0)+nanmedian(cumsum(results[datasets_algorithm_name][dataset_name]['oif']['test_times'], axis=1), axis=0))[-1], ndigits=2)
                           for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                           for j, dataset_name in enumerate(datasets_name[i])] +
                           [round(nanmedian(hstack([[train_time[-1]+test_time[-1] for train_time, test_time in zip(cumsum(results[datasets_algorithm_name][dataset_name]['oif']['train_times'], axis=1), cumsum(results[datasets_algorithm_name][dataset_name]['oif']['test_times'], axis=1))]
                                                    for i, datasets_algorithm_name in enumerate(datasets_algorithms_name)
                                                    for j, dataset_name in enumerate(datasets_name[i])]), axis=0), ndigits=3)])

time_table['cardinality']: list = hstack(datasets_cardinalities)
time_dataframe: DataFrame = DataFrame.from_dict(data=time_table, orient='index', columns=hstack([labels, ['median']])).transpose()
time_dataframe.sort_values(by='cardinality', axis=0, ascending=False, inplace=True)
time_dataframe.drop(labels='cardinality', axis=1, inplace=True)
time_dataframe.loc['median_rank'] = time_dataframe.iloc[:-1].rank(axis=1, ascending=True).median(axis=0)
time_dataframe.loc['mean_rank'] = time_dataframe.iloc[:-1].rank(axis=1, ascending=True).mean(axis=0)
time_dataframe.to_csv('../../results/' + datasets_type + '/' + 'times.csv')
# endregion

# endregion
