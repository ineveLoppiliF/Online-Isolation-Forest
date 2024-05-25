from .onlineitree import OnlineITree
from abc import ABC, abstractmethod
from multiprocessing import cpu_count
from numpy import ndarray


class OnlineIForest(ABC):
    @staticmethod
    def create(iforest_type: str = 'boundedrandomprojectiononlineiforest', **kwargs) -> 'OnlineIForest':
        # TODO: Find an alternative solution to overcome circular imports
        from .BoundedRandomProjectionOnlineIForest import BoundedRandomProjectionOnlineIForest
        # Map iforest type to an iforest class
        iforest_type_to_iforest_map: dict = {'boundedrandomprojectiononlineiforest': BoundedRandomProjectionOnlineIForest}
        if iforest_type not in iforest_type_to_iforest_map:
            raise ValueError('Bad iforest type {}'.format(iforest_type))
        return iforest_type_to_iforest_map[iforest_type](**kwargs)

    def __init__(self, num_trees: int, window_size: int, branching_factor: int, max_leaf_samples: int, type: str,
                 subsample: float, n_jobs: int):
        self.num_trees: int = num_trees
        self.window_size: int = window_size
        self.branching_factor: int = branching_factor
        self.max_leaf_samples: int = max_leaf_samples
        self.type: str = type
        self.subsample: float = subsample
        self.trees: list[OnlineITree] = []
        self.data_window: list[ndarray] = []
        self.data_size: int = 0
        self.normalization_factor: float = None
        self.n_jobs: int = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())

    @abstractmethod
    def learn_batch(self, data: ndarray):
        pass

    @abstractmethod
    def score_batch(self, data: ndarray):
        pass

    @abstractmethod
    def predict_batch(self, data: ndarray):
        pass
