from abc import ABC, abstractmethod
import numpy as np

# import mxnet as mx
import warnings
import torch
import copy

# from mxnet import autograd
from utils.model_utils import batch_data
from baseline_constants import INPUT_SIZE


class Model(ABC):
    def __init__(self, seed, lr):
        self.seed = seed
        self.lr = lr

        np.random.seed(123 + self.seed)
        np.random.seed(self.seed)

        self.prev_weights = None
        self.weights = None

        self.model, self.losses, self.optimizer = self.create_model()

    @abstractmethod
    def create_model(self):
        """Creates the model for the task.
        Returns:
            A 3-tuple consisting of:
                model: A neural network workflow.
                loss: An operation that, when run with the features and the
                    labels, computes the loss value.
                train_op: An operation that, when grads are computed, trains
                    the model.
        """
        return None, None, None
    
    def update_model(self, weights):
        self.model.load_state_dict(weights)
        self.weights = copy.deepcopy(self.model.state_dict())

    def delta_weights(self, a, b):
        """Return delta weights for pytorch model weights."""
        if a is None or b is None:
            return None
        return {x: a[x] - b[y] for (x, y) in zip(a, b)}
    
    def weights_to_device(self, weights, dtype):
        """Send model weights to device type dtype."""
        return {name: weights[name].to(dtype) for name in weights}

        return None

    def train(self, data, num_epochs=1, batch_size=10, device="cpu"):
        """
        Trains the client model.
        Args:
            data: Dict of the form {'x': NDArray, 'y': NDArray}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
        """
        self.model.train()
        for _ in range(num_epochs):
            print("---Epoch %d" % (_))
            self.run_epochs(data, batch_size, device)
        
        self.prev_weights = self.weights
        self.weights = self.model.state_dict()

        delta_weights = self.delta_weights(self.weights, self.prev_weights)
        delta_weights = self.weights_to_device(delta_weights, "cpu")
        return delta_weights

    def run_epochs(self, data, batch_size, device):
        # TODO: fix batching, to simply use torch dataset?
        for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
            feat = self.preprocess_x(batched_x).to(device)
            target = self.preprocess_y(batched_y).type(torch.LongTensor).to(device)

            self.optimizer.zero_grad()
            y_hats = self.model(feat)
            lss = self.losses(y_hats, target)
            lss.backward()
            self.optimizer.step()

    @abstractmethod
    def test(self, data):
        """
        Tests the current model on the given data.
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        return None

    def __num_elems(self, shape):
        """Returns the number of elements in the given shape
        Args:
            shape: Parameter shape

        Return:
            tot_elems: int
        """
        tot_elems = 1
        for s in shape:
            tot_elems *= int(s)
        return tot_elems

    @abstractmethod
    def preprocess_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        return None

    @abstractmethod
    def preprocess_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        return None

    # def size(self):
    #     """Returns the size of the network in bytes
    #     The size of the network is calculated by summing up the sizes of each
    #     trainable variable. The sizes of variables are calculated by multiplying
    #     the number of bytes in their dtype with their number of elements, captured
    #     in their shape attribute
    #     Return:
    #         integer representing size of graph (in bytes)
    #     """
    #     params = self.get_params().detach().numpy()
    #     tot_size = 0
    #     for p in params:
    #         tot_elems = self.__num_elems(p.shape)
    #         dtype_size = np.dtype(p.dtype).itemsize
    #         var_size = tot_elems * dtype_size
    #         tot_size += var_size
    #     return tot_size
