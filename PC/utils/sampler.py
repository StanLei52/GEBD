import math
import random
import numpy as np

import torch
from torch.utils.data import Sampler
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """
    
    def __init__(self, dataset, n_classes, n_samples):
        if dataset.train:
            self.labels = dataset.train_labels
        else:
            self.labels = dataset.test_labels

        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield from indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset)


class OrderedDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistBalancedBatchSampler(BatchSampler):
    '''
    dataset: dataset to be sampled
    num_classes : number of classes in the dataset
    n_sample_classes : the number of classes to be sampled in one batch
    n_samples: the number of samples to be sampled for each class in *n_sample_classes*
    seed: use the same seed for each replica
    num_replicas: 
    rank:     
    '''
    def __init__(self, dataset, num_classes, n_sample_classes, n_samples, seed=666, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.seed = seed
        self.rank = rank
        # set up batch balancing
        if dataset.train:
            self.labels = dataset.train_labels
        else:
            self.labels = dataset.test_labels
        
        

        self.labels_set = list(np.arange(num_classes))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            # use the same seed for each replica
            np.random.seed(self.seed)
            np.random.shuffle(self.label_to_indices[l])

        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_sample_classes = n_sample_classes
        self.n_samples = n_samples
        
        # batch_size refers to bs per replica
        self.batch_size = self.n_samples * self.n_sample_classes
        
        # for the whole data set, each replica should sample `total_samples_per_replica` samples.
        self.total_samples_per_replica = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))

    def __iter__(self):
        # self.count for each replica
        self.count = 0
        while self.count + self.batch_size < self.total_samples_per_replica:
            classes = np.random.choice(self.labels_set, self.n_sample_classes, replace=False)
            indices = []
            for class_ in classes:
                start = self.used_label_indices_count[class_] + (self.rank % self.num_replicas)
                end = self.used_label_indices_count[class_] + self.n_samples * self.num_replicas
                step = self.num_replicas
                indices.extend(self.label_to_indices[class_][start:end:step])
                self.used_label_indices_count[class_] += self.n_samples * self.num_replicas
                if self.used_label_indices_count[class_] + self.n_samples * self.num_replicas > len(self.label_to_indices[class_]):
                    np.random.seed(self.seed)
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            #  print(f'{self.rank} indices:{indices}.')
            yield from indices
            self.count += self.n_sample_classes * self.n_samples 


    def __len__(self):
        return self.total_samples_per_replica

