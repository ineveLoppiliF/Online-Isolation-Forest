from numpy import argwhere, array_split, ceil, empty, floor, genfromtxt, hstack, linspace, ndarray
from numpy.random import choice
from OnlineIForest import OnlineIForest
from os import makedirs
from pickle import dump, HIGHEST_PROTOCOL
from pysad.models import HalfSpaceTrees, IForestASD, LODA, RobustRandomCutForest
from sklearn.metrics import auc, roc_curve
from sklearn.utils import shuffle
from time import time


# region Initialization
# region Define parameters
datasets_algorithm_name: str = 'rrcf'
datasets_type: str = 'stationary'
datasets_name: list = [
                       'nyc_taxi_shingle'  # nyc_taxi_shingle: N = 10273, d = 48, prop_anomalies = 5.2%
                       ]

# Set Isolation Forest ASD parameters
iforestasd_params: dict = {'window_size': 2048,
                           'n_estimators': 32,
                           'max_samples': 256,
                           'max_features': 1.0,
                           'bootstrap': False,
                           'n_jobs': 1}
# Set Half Space Trees parameters
hst_params: dict = {'window_size': 250,
                    'num_trees': 32,
                    'max_depth': 15}
# Set Robust Random Cut Forest parameters
rrcforest_params: dict = {'num_trees': 32,
                          'shingle_size': 1,
                          'tree_size': 256}
# Set LODA parameters
loda_params: dict = {'num_bins': 100,
                     'num_random_cuts': 32}
# Set Online Isolation Forest parameters
oiforest_params: dict = {'iforest_type': 'boundedrandomprojectiononlineiforest',
                         'branching_factor': 2,
                         'metric': 'axisparallel',
                         'num_trees': 32,
                         'max_leaf_samples': 32,
                         'window_size': 2048,
                         'type': 'adaptive',
                         'subsample': 1.0,
                         'n_jobs': 1}

n_executions: int = 30
batch_size: int = 100
n_evaluation_windows: int = 31
n_evaluation_holdout: int = n_evaluation_windows
prop_holdout: float = 0.1
# endregion
# endregion

# region Datasets loop
for j, dataset_name in enumerate(datasets_name[::-1]):
    print('Dataset: ' + dataset_name)
    start_dataset_time: float = time()

    # Define dataset paths
    dataset_path: str = '../../../datasets/' + datasets_type + '/' + datasets_algorithm_name + '/' + dataset_name + '.csv'
    results_path: str = '../../results/' + datasets_type + '/' + datasets_algorithm_name + '/' + dataset_name + '/'

    # Create output folder
    makedirs(results_path, exist_ok=True)

    # region Executions Loop
    # region Define results structures
    ifasd_results: dict = {'roc_aucs_cumulative': [],
                           'roc_aucs_windows': [],
                           'roc_aucs_holdout': [],
                           'train_times': [],
                           'test_times': []}
    hst_results: dict = {'roc_aucs_cumulative': [],
                         'roc_aucs_windows': [],
                         'roc_aucs_holdout': [],
                         'train_times': [],
                         'test_times': []}
    rrcf_results: dict = {'roc_aucs_cumulative': [],
                          'roc_aucs_windows': [],
                          'roc_aucs_holdout': [],
                          'train_times': [],
                          'test_times': []}
    loda_results: dict = {'roc_aucs_cumulative': [],
                          'roc_aucs_windows': [],
                          'roc_aucs_holdout': [],
                          'train_times': [],
                          'test_times': []}
    oif_results: dict = {'roc_aucs_cumulative': [],
                         'roc_aucs_windows': [],
                         'roc_aucs_holdout': [],
                         'train_times': [],
                         'test_times': []}
    # endregion

    for i in range(n_executions):
        print('     Execution #: ' + str(i))

        # region Prepare data
        dataset: ndarray = genfromtxt(dataset_path, delimiter=',')

        # Remove heading and id column
        dataset: ndarray = dataset[1:, 1:]

        # Compose dataset
        data: ndarray = dataset[:, :-1]
        inlier_mask: ndarray = ~dataset[:, -1].astype(dtype=bool)

        # Shuffle dataset
        data, inlier_mask = shuffle(data, inlier_mask)

        # Compute holdout indices
        prop_inlier: float = inlier_mask.sum()/inlier_mask.shape[0]
        holdout_indices: ndarray = hstack([choice(argwhere(inlier_mask).flatten(), int(ceil(prop_inlier*prop_holdout*data.shape[0])), replace=False),
                                           choice(argwhere(~inlier_mask).flatten(), int(ceil((1-prop_inlier)*prop_holdout*data.shape[0])), replace=False)])
        # endregion

        # region Anomaly Detection

        # region Compute splits, windows and holdout times
        n_batches: int = int(floor(data.shape[0]/batch_size))+1
        batch_splits: list = array_split(range(data.shape[0]), n_batches)
        evaluation_windows_times: ndarray = linspace(0, len(batch_splits)-1, n_evaluation_windows, dtype=int)
        evaluation_holdout_times: ndarray = linspace(0, len(batch_splits)-1, n_evaluation_holdout, dtype=int)
        # endregion

        # region Online Anomaly Detection
        print('          Online Anomaly Detection')

        # region Isolation Forest ASD
        print('               Isolation Forest ASD')
        ifasd_scores: ndarray = empty(shape=(0,))
        ifasd_roc_aucs_cumulative: ndarray = empty(shape=(0,))
        ifasd_roc_aucs_windows: ndarray = empty(shape=(0,))
        ifasd_roc_aucs_holdout: ndarray = empty(shape=(0,))
        ifasd_train_times: ndarray = empty(shape=(0,))
        ifasd_test_times: ndarray = empty(shape=(0,))

        # Initialize Isolation Forest ASD
        ifasd: IForestASD = IForestASD(**iforestasd_params)

        data_indices_so_far: ndarray = empty(shape=(0,), dtype=int)
        data_indices_current_window: ndarray = empty(shape=(0,), dtype=int)
        for t, batch_indices in enumerate(batch_splits):
            print('               t = ' + str(t))
            data_indices_so_far: ndarray = hstack([data_indices_so_far, batch_indices])
            data_indices_current_window: ndarray = hstack([data_indices_current_window, batch_indices])

            # Update IFORASD
            start_train_time: float = time()
            ifasd.fit(data[batch_indices])
            end_train_time: float = time()

            # Score samples
            start_test_time: float = time()
            ifasd_scores: ndarray = hstack([ifasd_scores, ifasd.score(data[batch_indices])])
            end_test_time: float = time()

            # Compute cumulative FPR, TPR and AUC
            fpr, tpr, _ = roc_curve(~inlier_mask[data_indices_so_far], ifasd_scores[data_indices_so_far])
            ifasd_roc_aucs_cumulative: ndarray = hstack([ifasd_roc_aucs_cumulative, auc(fpr, tpr)])
            # Compute windowed FPR, TPR and AUC
            if t in evaluation_windows_times:
                fpr, tpr, _ = roc_curve(~inlier_mask[data_indices_current_window], ifasd_scores[data_indices_current_window])
                ifasd_roc_aucs_windows: ndarray = hstack([ifasd_roc_aucs_windows, auc(fpr, tpr)])
                data_indices_current_window: ndarray = empty(shape=(0,), dtype=int)
            # Compute holdout FPR, TPR and AUC
            if t in evaluation_holdout_times:
                holdout_scores: ndarray = ifasd.score(data[holdout_indices])
                fpr, tpr, _ = roc_curve(~inlier_mask[holdout_indices], holdout_scores)
                ifasd_roc_aucs_holdout: ndarray = hstack([ifasd_roc_aucs_holdout, auc(fpr, tpr)])

            # Store times
            ifasd_train_times: ndarray = hstack([ifasd_train_times, end_train_time - start_train_time])
            ifasd_test_times: ndarray = hstack([ifasd_test_times, end_test_time - start_test_time])
        # endregion

        # region Half Space Trees
        print('               Half Space Trees')
        hst_scores: ndarray = empty(shape=(0,))
        hst_roc_aucs_cumulative: ndarray = empty(shape=(0,))
        hst_roc_aucs_windows: ndarray = empty(shape=(0,))
        hst_roc_aucs_holdout: ndarray = empty(shape=(0,))
        hst_train_times: ndarray = empty(shape=(0,))
        hst_test_times: ndarray = empty(shape=(0,))

        # Initialize Half Space Trees
        hst_params['feature_mins']: ndarray = data.min(0)
        hst_params['feature_maxes']: ndarray = data.max(0)
        hst: HalfSpaceTrees = HalfSpaceTrees(**hst_params)

        data_indices_so_far: ndarray = empty(shape=(0,), dtype=int)
        data_indices_current_window: ndarray = empty(shape=(0,), dtype=int)
        for t, batch_indices in enumerate(batch_splits):
            print('               t = ' + str(t))
            data_indices_so_far: ndarray = hstack([data_indices_so_far, batch_indices])
            data_indices_current_window: ndarray = hstack([data_indices_current_window, batch_indices])

            # Update HST
            start_train_time: float = time()
            hst.fit(data[batch_indices])
            end_train_time: float = time()

            # Score samples
            start_test_time: float = time()
            hst_scores: ndarray = hstack([hst_scores, hst.score(data[batch_indices])])
            end_test_time: float = time()

            # Compute cumulative FPR, TPR and AUC
            fpr, tpr, _ = roc_curve(~inlier_mask[data_indices_so_far], hst_scores[data_indices_so_far])
            hst_roc_aucs_cumulative: ndarray = hstack([hst_roc_aucs_cumulative, auc(fpr, tpr)])
            # Compute windowed FPR, TPR and AUC
            if t in evaluation_windows_times:
                fpr, tpr, _ = roc_curve(~inlier_mask[data_indices_current_window], hst_scores[data_indices_current_window])
                hst_roc_aucs_windows: ndarray = hstack([hst_roc_aucs_windows, auc(fpr, tpr)])
                data_indices_current_window: ndarray = empty(shape=(0,), dtype=int)
            # Compute holdout FPR, TPR and AUC
            if t in evaluation_holdout_times:
                holdout_scores: ndarray = hst.score(data[holdout_indices])
                fpr, tpr, _ = roc_curve(~inlier_mask[holdout_indices], holdout_scores)
                hst_roc_aucs_holdout: ndarray = hstack([hst_roc_aucs_holdout, auc(fpr, tpr)])

            # Store times
            hst_train_times: ndarray = hstack([hst_train_times, end_train_time - start_train_time])
            hst_test_times: ndarray = hstack([hst_test_times, end_test_time - start_test_time])
        # endregion

        # region Robust Random Cut Forest
        print('               Robust Random Cut Forest')
        rrcf_scores: ndarray = empty(shape=(0,))
        rrcf_roc_aucs_cumulative: ndarray = empty(shape=(0,))
        rrcf_roc_aucs_windows: ndarray = empty(shape=(0,))
        rrcf_roc_aucs_holdout: ndarray = empty(shape=(0,))
        rrcf_train_times: ndarray = empty(shape=(0,))
        rrcf_test_times: ndarray = empty(shape=(0,))

        # Initialize Robust Random Cut Forest
        rrcf: RobustRandomCutForest = RobustRandomCutForest(**rrcforest_params)

        data_indices_so_far: ndarray = empty(shape=(0,), dtype=int)
        data_indices_current_window: ndarray = empty(shape=(0,), dtype=int)
        for t, batch_indices in enumerate(batch_splits):
            print('               t = ' + str(t))
            data_indices_so_far: ndarray = hstack([data_indices_so_far, batch_indices])
            data_indices_current_window: ndarray = hstack([data_indices_current_window, batch_indices])

            # Update RRCF
            start_train_time: float = time()
            rrcf.fit(data[batch_indices])
            end_train_time: float = time()

            # Score samples
            start_test_time: float = time()
            rrcf_scores: ndarray = hstack([rrcf_scores, rrcf.score(data[batch_indices])])
            end_test_time: float = time()

            # Compute cumulative FPR, TPR and AUC
            fpr, tpr, _ = roc_curve(~inlier_mask[data_indices_so_far], rrcf_scores[data_indices_so_far])
            rrcf_roc_aucs_cumulative: ndarray = hstack([rrcf_roc_aucs_cumulative, auc(fpr, tpr)])
            # Compute windowed FPR, TPR and AUC
            if t in evaluation_windows_times:
                fpr, tpr, _ = roc_curve(~inlier_mask[data_indices_current_window], rrcf_scores[data_indices_current_window])
                rrcf_roc_aucs_windows: ndarray = hstack([rrcf_roc_aucs_windows, auc(fpr, tpr)])
                data_indices_current_window: ndarray = empty(shape=(0,), dtype=int)
            # Compute holdout FPR, TPR and AUC
            if t in evaluation_holdout_times:
                holdout_scores: ndarray = rrcf.score(data[holdout_indices])
                fpr, tpr, _ = roc_curve(~inlier_mask[holdout_indices], holdout_scores)
                rrcf_roc_aucs_holdout: ndarray = hstack([rrcf_roc_aucs_holdout, auc(fpr, tpr)])

            # Store times
            rrcf_train_times: ndarray = hstack([rrcf_train_times, end_train_time - start_train_time])
            rrcf_test_times: ndarray = hstack([rrcf_test_times, end_test_time - start_test_time])
        # endregion

        # region LODA
        print('               LODA')
        loda_scores: ndarray = empty(shape=(0,))
        loda_roc_aucs_cumulative: ndarray = empty(shape=(0,))
        loda_roc_aucs_windows: ndarray = empty(shape=(0,))
        loda_roc_aucs_holdout: ndarray = empty(shape=(0,))
        loda_train_times: ndarray = empty(shape=(0,))
        loda_test_times: ndarray = empty(shape=(0,))

        # Initialize LODA
        loda: LODA = LODA(**loda_params)

        data_indices_so_far: ndarray = empty(shape=(0,), dtype=int)
        data_indices_current_window: ndarray = empty(shape=(0,), dtype=int)
        for t, batch_indices in enumerate(batch_splits):
            print('               t = ' + str(t))
            data_indices_so_far: ndarray = hstack([data_indices_so_far, batch_indices])
            data_indices_current_window: ndarray = hstack([data_indices_current_window, batch_indices])

            # Update LODA
            start_train_time: float = time()
            loda.fit(data[batch_indices])
            end_train_time: float = time()

            # Score samples
            start_test_time: float = time()
            loda_scores: ndarray = hstack([loda_scores, loda.score(data[batch_indices])])
            end_test_time: float = time()

            # Compute cumulative FPR, TPR and AUC
            fpr, tpr, _ = roc_curve(~inlier_mask[data_indices_so_far], loda_scores[data_indices_so_far])
            loda_roc_aucs_cumulative: ndarray = hstack([loda_roc_aucs_cumulative, auc(fpr, tpr)])
            # Compute windowed FPR, TPR and AUC
            if t in evaluation_windows_times:
                fpr, tpr, _ = roc_curve(~inlier_mask[data_indices_current_window], loda_scores[data_indices_current_window])
                loda_roc_aucs_windows: ndarray = hstack([loda_roc_aucs_windows, auc(fpr, tpr)])
                data_indices_current_window: ndarray = empty(shape=(0,), dtype=int)
            # Compute holdout FPR, TPR and AUC
            if t in evaluation_holdout_times:
                holdout_scores: ndarray = loda.score(data[holdout_indices])
                fpr, tpr, _ = roc_curve(~inlier_mask[holdout_indices], holdout_scores)
                loda_roc_aucs_holdout: ndarray = hstack([loda_roc_aucs_holdout, auc(fpr, tpr)])

            # Store times
            loda_train_times: ndarray = hstack([loda_train_times, end_train_time - start_train_time])
            loda_test_times: ndarray = hstack([loda_test_times, end_test_time - start_test_time])
        # endregion

        # region Online Isolation Forest
        print('               Online Isolation Forest')
        oif_scores: ndarray = empty(shape=(0,))
        oif_roc_aucs_cumulative: ndarray = empty(shape=(0,))
        oif_roc_aucs_windows: ndarray = empty(shape=(0,))
        oif_roc_aucs_holdout: ndarray = empty(shape=(0,))
        oif_train_times: ndarray = empty(shape=(0,))
        oif_test_times: ndarray = empty(shape=(0,))

        # Initialize Online Isolation Forest
        oif: OnlineIForest = OnlineIForest.create(**oiforest_params)

        data_indices_so_far: ndarray = empty(shape=(0,), dtype=int)
        data_indices_current_window: ndarray = empty(shape=(0,), dtype=int)
        for t, batch_indices in enumerate(batch_splits):
            print('               t = ' + str(t))
            data_indices_so_far: ndarray = hstack([data_indices_so_far, batch_indices])
            data_indices_current_window: ndarray = hstack([data_indices_current_window, batch_indices])

            # Update OIF
            start_train_time: float = time()
            oif.learn_batch(data[batch_indices])
            end_train_time: float = time()

            # Score samples
            start_test_time: float = time()
            oif_scores: ndarray = hstack([oif_scores, oif.score_batch(data[batch_indices])])
            end_test_time: float = time()

            # Compute cumulative FPR, TPR and AUC
            fpr, tpr, _ = roc_curve(~inlier_mask[data_indices_so_far], oif_scores[data_indices_so_far])
            oif_roc_aucs_cumulative: ndarray = hstack([oif_roc_aucs_cumulative, auc(fpr, tpr)])
            # Compute windowed FPR, TPR and AUC
            if t in evaluation_windows_times:
                fpr, tpr, _ = roc_curve(~inlier_mask[data_indices_current_window], oif_scores[data_indices_current_window])
                oif_roc_aucs_windows: ndarray = hstack([oif_roc_aucs_windows, auc(fpr, tpr)])
                data_indices_current_window: ndarray = empty(shape=(0,), dtype=int)
            # Compute holdout FPR, TPR and AUC
            if t in evaluation_holdout_times:
                holdout_scores: ndarray = oif.score_batch(data[holdout_indices])
                fpr, tpr, _ = roc_curve(~inlier_mask[holdout_indices], holdout_scores)
                oif_roc_aucs_holdout: ndarray = hstack([oif_roc_aucs_holdout, auc(fpr, tpr)])

            # Store times
            oif_train_times: ndarray = hstack([oif_train_times, end_train_time - start_train_time])
            oif_test_times: ndarray = hstack([oif_test_times, end_test_time - start_test_time])
        # endregion

        # endregion

        # endregion

        # region Update results structures
        # Update Isolation Forest ASD result structure
        ifasd_results['roc_aucs_cumulative'].append(ifasd_roc_aucs_cumulative)
        ifasd_results['roc_aucs_windows'].append(ifasd_roc_aucs_windows)
        ifasd_results['roc_aucs_holdout'].append(ifasd_roc_aucs_holdout)
        ifasd_results['train_times'].append(ifasd_train_times)
        ifasd_results['test_times'].append(ifasd_test_times)
        # Update Half Space Trees result structure
        hst_results['roc_aucs_cumulative'].append(hst_roc_aucs_cumulative)
        hst_results['roc_aucs_windows'].append(hst_roc_aucs_windows)
        hst_results['roc_aucs_holdout'].append(hst_roc_aucs_holdout)
        hst_results['train_times'].append(hst_train_times)
        hst_results['test_times'].append(hst_test_times)
        # Update Robust Random Cut Forest result structure
        rrcf_results['roc_aucs_cumulative'].append(rrcf_roc_aucs_cumulative)
        rrcf_results['roc_aucs_windows'].append(rrcf_roc_aucs_windows)
        rrcf_results['roc_aucs_holdout'].append(rrcf_roc_aucs_holdout)
        rrcf_results['train_times'].append(rrcf_train_times)
        rrcf_results['test_times'].append(rrcf_test_times)
        # Update LODA result structure
        loda_results['roc_aucs_cumulative'].append(loda_roc_aucs_cumulative)
        loda_results['roc_aucs_windows'].append(loda_roc_aucs_windows)
        loda_results['roc_aucs_holdout'].append(loda_roc_aucs_holdout)
        loda_results['train_times'].append(loda_train_times)
        loda_results['test_times'].append(loda_test_times)
        # Update Online Isolation Forest result structure
        oif_results['roc_aucs_cumulative'].append(oif_roc_aucs_cumulative)
        oif_results['roc_aucs_windows'].append(oif_roc_aucs_windows)
        oif_results['roc_aucs_holdout'].append(oif_roc_aucs_holdout)
        oif_results['train_times'].append(oif_train_times)
        oif_results['test_times'].append(oif_test_times)
        # endregion

        # region Aggregate and save results structures
        results: dict = {'ifasd': ifasd_results,
                         'hst': hst_results,
                         'rrcf': rrcf_results,
                         'loda': loda_results,
                         'oif': oif_results}
        with open(results_path + 'results.pkl', 'wb') as handle:
            dump(results, handle, protocol=HIGHEST_PROTOCOL)
        # endregion
    # endregion
    end_dataset_time: float = time()

    # region Save parameters information
    params: dict = {'ifasd': iforestasd_params,
                    'hst': hst_params,
                    'rrcf': rrcforest_params,
                    'loda': loda_params,
                    'oif': oiforest_params}
    with open(results_path + 'params.pkl', 'wb') as handle:
        dump(params, handle, protocol=HIGHEST_PROTOCOL)
    # endregion

    # region Save information for plots
    info: dict = {'batch_size': batch_size,
                  'n_batches': n_batches,
                  'evaluation_windows_times': evaluation_windows_times,
                  'evaluation_holdout_times': evaluation_holdout_times,
                  'prop_holdout': prop_holdout,
                  'total_execution_times': end_dataset_time - start_dataset_time}
    with open(results_path + 'info.pkl', 'wb') as handle:
        dump(info, handle, protocol=HIGHEST_PROTOCOL)
    # endregion
# endregion
