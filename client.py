import warnings
import torch


class Client:
    def __init__(self, client_id, group, train_data, test_data, model, device="cpu"):
        self._model = model
        self._model.model.to(device)
        self.id = client_id
        self.group = group

        self.train_data = None
        self.test_data = None

        if train_data != None:
            self.train_data = {
                "x": self.preprocess_data_x(train_data["x"]),
                "y": self.preprocess_data_y(train_data["y"]),
            }
        if test_data != None:
            self.test_data = {
                "x": self.preprocess_data_x(test_data["x"]),
                "y": self.preprocess_data_y(test_data["y"]),
            }

        self.device = device

    def train(self, num_epoch=1, batch_size=10):
        return self.num_train_samples, self.model.train(
            self.train_data, num_epoch, batch_size, self.device
        )

    def test(self, set_to_use="test"):
        assert set_to_use in ["train", "test", "val"]
        if set_to_use == "train":
            data = self.train_data
        elif set_to_use == "test" or set_to_use == "val":
            data = self.test_data
        return self.model.test(data, self.device)

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn(
            "The current implementation shares the model among all clients."
            "Setting it on one client will effectively modify all clients."
        )
        self._model = model

    @property
    def num_train_samples(self):
        if self.train_data is None:
            return 0
        return len(self.train_data["y"])

    @property
    def num_test_samples(self):
        """Number of test samples for this client.
        Return:
            int: Number of test samples for this client
        """
        if self.test_data is None:
            return 0
        return len(self.test_data["y"])

    @property
    def num_samples(self):
        """Number of samples for this client (train + test).
        Return:
            int: Number of samples for this client
        """
        return self.num_train_samples + self.num_test_samples

    def preprocess_data_x(self, data):
        return torch.tensor(data, requires_grad=False)

    def preprocess_data_y(self, data):
        data_y = []
        for i in data:
            data_float = float(i)
            data_y.append(data_float)
        return torch.tensor(data_y, requires_grad=False)
