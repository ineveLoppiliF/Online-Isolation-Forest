from ..onlineiforest import OnlineIForest
from ..onlineitree import OnlineITree
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from numpy import asarray, finfo, ndarray
from sys import stdout
from tqdm import tqdm


class BoundedRandomProjectionOnlineIForest(OnlineIForest):
    def __init__(self, num_trees: int = 100, max_leaf_samples: int = 32, type: str = 'adaptive', subsample: float = 1.0,
                 window_size: int = 2048, branching_factor: int = 2, metric: str = 'axisparallel', n_jobs: int = 1):
        super().__init__(num_trees, window_size, branching_factor, max_leaf_samples, type, subsample, n_jobs)
        self.metric: str = metric
        self.trees: list[OnlineITree] = [OnlineITree.create('boundedrandomprojectiononlineitree',
                                                            max_leaf_samples=max_leaf_samples,
                                                            type=type,
                                                            subsample=self.subsample,
                                                            branching_factor=self.branching_factor,
                                                            data_size=self.data_size,
                                                            metric=self.metric) for _ in range(self.num_trees)]

    def learn_batch(self, data: ndarray) -> 'BoundedRandomProjectionOnlineIForest':
        # Update the counter of data seen so far
        self.data_size += data.shape[0]
        # Compute the normalization factor
        self.normalization_factor: float = OnlineITree.get_random_path_length(self.branching_factor,
                                                                              self.max_leaf_samples,
                                                                              self.data_size * self.subsample)
        # Instantiate a list of ITrees' learn functions
        learn_funcs: list['function'] = [tree.learn for tree in self.trees]
        # BoundedRandomProjection OnlineITrees learn new data
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            self.trees: list[OnlineITree] = list(tqdm(executor.map(lambda f, x: f(x), learn_funcs,
                                                                   repeat(data, self.num_trees)),
                                                      total=self.num_trees,
                                                      desc='     BoundedRandomProjection Online Isolation Forest -> Learn Batch',
                                                      file=stdout,
                                                      disable=True))
        # If the window size is not None, add new data to the window and eventually remove old ones
        if self.window_size:
            # Update the window of data seen so far
            self.data_window += list(data)
            # If the window size is smaller than the number of data seen so far, unlearn old data
            if self.data_size > self.window_size:
                # Extract old data and update the window of data seen so far
                data, self.data_window = asarray(self.data_window[:self.data_size-self.window_size]), self.data_window[self.data_size-self.window_size:]
                # Update the counter of data seen so far
                self.data_size -= self.data_size-self.window_size
                # Compute the normalization factor
                self.normalization_factor: float = OnlineITree.get_random_path_length(self.branching_factor,
                                                                                      self.max_leaf_samples,
                                                                                      self.data_size * self.subsample)
                # Instantiate a list of ITrees' unlearn functions
                unlearn_funcs: list['function'] = [tree.unlearn for tree in self.trees]
                # BoundedRandomProjection OnlineITrees unlearn new data
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    self.trees: list[OnlineITree] = list(tqdm(executor.map(lambda f, x: f(x), unlearn_funcs,
                                                                           repeat(data, self.num_trees)),
                                                              total=self.num_trees,
                                                              desc='     BoundedRandomProjection Online Isolation Forest -> Unlearn Batch',
                                                              file=stdout,
                                                              disable=True))
        return self

    def score_batch(self, data: ndarray) -> ndarray[float]:
        # Collect ITrees' predict functions
        predict_funcs: list['function'] = [tree.predict for tree in self.trees]
        # Compute the depths of all samples in each tree
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            depths: ndarray[float] = asarray(list(tqdm(executor.map(lambda f, x: f(x), predict_funcs,
                                                                    repeat(data, self.num_trees)),
                                                       total=self.num_trees,
                                                       desc='     BoundedRandomProjection Online Isolation Forest -> Score Batch',
                                                       file=stdout,
                                                       disable=True))).T
        # Compute the mean depth of each sample along all trees
        mean_depths: ndarray[float] = depths.mean(axis=1)
        # Compute normalized mean depths
        normalized_mean_depths: ndarray[float] = 2 ** (-mean_depths / (self.normalization_factor + finfo(float).eps))
        return normalized_mean_depths

    def predict_batch(self, data: ndarray) -> ndarray[float]:
        # Collect ITrees' predict functions
        predict_funcs: list['function'] = [tree.predict for tree in self.trees]
        # Compute the depths of all samples in each tree
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            depths: ndarray[float] = asarray(list(tqdm(executor.map(lambda f, x: f(x), predict_funcs,
                                                                    repeat(data, self.num_trees)),
                                                       total=self.num_trees,
                                                       desc='     BoundedRandomProjection Online Isolation Forest -> Score Batch',
                                                       file=stdout,
                                                       disable=True))).T
        # Compute the mean depth of each sample along all trees
        mean_depths: ndarray[float] = depths.mean(axis=1)
        return mean_depths
