import os
import numpy as np
import torch
from utils.model_utils import build_model
from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY


class Server:
    def __init__(self, client_model, dataset, model_name, num_classes):
        self.dataset = dataset
        self.model_name = model_name
        self.selected_clients = []
        # build and synchronize the global model
        self.model = build_model(dataset, model_name, num_classes)
        self.model.load_state_dict(client_model.model.state_dict())

        self.updates = []
        self.total = 0

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).
        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:selected_client num_train_samples num_test_samples
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, clients=None):
        """Trains self.model on given clients.

        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.
        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        """
        if clients is None:
            clients = self.selected_clients

        index = 0
        self.total = 0
        self.base_weights = self.model.state_dict()
        for c in clients:
            index += 1
            print("%d th clients start training" % index)
            c.model.update_model(self.base_weights)
            num_samples, update = c.train(num_epochs, batch_size)
            self.total += num_samples
            self.updates.append((num_samples, update))

    def update_model(self):
        print("start global updating")
        for update_item in self.updates:
            num_samples, weights = update_item
            rate = num_samples / self.total
            for k, v in weights.items():
                tmp = v * rate
                tmp = tmp.to(dtype=v.dtype) if tmp.dtype != v.dtype else tmp

                self.base_weights[k] += tmp
        
        self.model.load_state_dict(self.base_weights)
        self.updates = []

    def test_model(self, clients_to_test, set_to_use="test"):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for c in clients_to_test:
            if c.test_data is not None:
                c.model.update_model(self.model.state_dict())
                c_metrics = c.test(set_to_use)
                metrics[c.id] = c_metrics

        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def save_model(self, log_dir):
        """Saves the server model on:
        logs/{self.dataset}/{self.model}.params
        """
        torch.save(self.model.state_dict(), os.path.join(log_dir, self.model_name + ".params"))
