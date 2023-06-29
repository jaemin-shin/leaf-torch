import os
import numpy as np
import torch
import math
from utils.model_utils import build_model
from timesim import Timesim
from baseline_constants import ACCURACY_KEY, LOSS_KEY


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

        self.curr_model_version = 0

    def select_clients(self, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).
        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        """
        num_clients = min(num_clients, len(possible_clients))
        self.selected_clients = np.random.choice(
            possible_clients, num_clients, replace=False
        )

    def initialize_round(self):
        self.total = 0
        self.base_weights = self.model.state_dict()
        self.agg_goal_weights = {}

    def train_model(self, num_epochs=1, batch_size=10, client=None, model_version=0):
        """Trains self.model on given clients.

        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.
        Args:
            client: Client object.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        """
        if client is None:
            assert False
        # print(f"client {client.id} training")
        client.model.update_model(self.base_weights)
        num_samples, update = client.train(num_epochs, batch_size)
        self.total += num_samples
        self.updates.append((num_samples, model_version, update))

    def fedavg_update_model(self):
        print("start global updating")
        for update_item in self.updates:
            num_samples, _, weights = update_item
            rate = num_samples / self.total
            for k, v in weights.items():
                tmp = v * rate
                tmp = tmp.to(dtype=v.dtype) if tmp.dtype != v.dtype else tmp

                self.base_weights[k] += tmp

        self.model.load_state_dict(self.base_weights)
        self.updates = []

    def fedbuff_update_model(self, agg_goal):
        print("start global updating")
        for update_item in self.updates:
            _, model_version, weights = update_item
            rate = 1 / math.sqrt(1 + self.curr_model_version - model_version)
            for k, v in weights.items():
                tmp = v * rate
                tmp = tmp.to(dtype=v.dtype) if tmp.dtype != v.dtype else tmp

                if k not in self.agg_goal_weights.keys():
                    self.agg_goal_weights[k] = tmp
                else:
                    self.agg_goal_weights[k] += tmp

        for k in self.base_weights.keys():
            self.base_weights[k] += self.agg_goal_weights[k] / agg_goal

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

    def average_metrics(self, metrics):
        avg_metrics = {ACCURACY_KEY: [], LOSS_KEY: []}
        for key in metrics:
            avg_metrics[ACCURACY_KEY].append(metrics[key][ACCURACY_KEY])
            avg_metrics[LOSS_KEY].append(metrics[key][LOSS_KEY])

        avg_metrics[ACCURACY_KEY] = np.mean(avg_metrics[ACCURACY_KEY])
        avg_metrics[LOSS_KEY] = np.mean(avg_metrics[LOSS_KEY])
        return avg_metrics

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
        torch.save(
            self.model.state_dict(), os.path.join(log_dir, self.model_name + ".params")
        )

    def syncFL(self, train_clients, test_clients, args, log_fp):
        per_round_time_range = (10, 100)
        timesim = Timesim(per_round_time_range)
        timesim.setup_clients_per_round_time(train_clients)
        timesim.set_time(0)

        for r in range(args.num_rounds):
            print(
                "TIME %.3f --- Round %d of %d: Training %d clients ---"
                % (timesim.get_time(), r, args.num_rounds - 1, args.clients_per_round)
            )
            print(
                "TIME %.3f --- Round %d of %d: Training %d clients ---"
                % (timesim.get_time(), r, args.num_rounds - 1, args.clients_per_round),
                file=log_fp,
                flush=True,
            )

            self.initialize_round()
            self.select_clients(train_clients, args.clients_per_round)

            base_params = {"num_epochs": args.num_epochs, "batch_size": args.batch_size}
            timesim.register_train_events(
                self.selected_clients, self.train_model, base_params
            )
            timesim.pop_events(args.clients_per_round)

            self.fedavg_update_model()

            if (r + 1) % args.eval_every == 0 or (r + 1) == args.num_rounds:
                metrics = self.test_model(test_clients, set_to_use="test")
                avg_metrics = self.average_metrics(metrics)
                print(
                    "TIME %.3f --- Round %d of %d: Testing %d clients --- %s"
                    % (
                        timesim.get_time(),
                        r,
                        args.num_rounds - 1,
                        len(test_clients),
                        str(avg_metrics),
                    ),
                )
                print(
                    "TIME %.3f --- Round %d of %d: Testing %d clients --- %s"
                    % (
                        timesim.get_time(),
                        r,
                        args.num_rounds - 1,
                        len(test_clients),
                        str(avg_metrics),
                    ),
                    file=log_fp,
                    flush=True,
                )

        log_fp.close()

    def asyncFL(self, train_clients, test_clients, args, log_fp):
        per_round_time_range = (10, 100)
        timesim = Timesim(per_round_time_range)
        timesim.setup_clients_per_round_time(train_clients)
        timesim.set_time(0)

        c_training_version = {}
        non_training_clients = []
        current_training_count = 0

        for c in train_clients:
            non_training_clients.append(c)
            c_training_version[c.id] = -1

        self.initialize_round()

        self.select_clients(non_training_clients, args.concurrency)
        base_params = {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "model_version": self.curr_model_version,
        }
        timesim.register_train_events(
            self.selected_clients, self.train_model, base_params
        )
        for c in self.selected_clients:
            non_training_clients.remove(c)

        log_str = "TIME %.3f --- Model version %d: Training %d / %d clients --- " % (
            timesim.get_time(),
            self.curr_model_version,
            len(self.selected_clients),
            args.concurrency,
        )
        print(log_str)
        print(log_str, file=log_fp, flush=True)

        while timesim.get_time() < args.total_time:
            popped_count, popped_clients = timesim.pop_events(1)
            if popped_count == 1:
                non_training_clients.append(popped_clients[0])
                log_str = (
                    "TIME %.3f --- Model version %d: Client %s finished training --- "
                    % (
                        timesim.get_time(),
                        self.curr_model_version,
                        popped_clients[0].id,
                    )
                )
                print(log_str)
                print(log_str, file=log_fp, flush=True)

            if len(self.updates) == args.agg_goal:
                self.fedbuff_update_model(args.agg_goal)
                self.initialize_round()
                self.curr_model_version += 1

                if self.curr_model_version % args.eval_every == 0:
                    metrics = self.test_model(test_clients, set_to_use="test")
                    avg_metrics = self.average_metrics(metrics)
                    log_str = (
                        "TIME %.3f --- Model version %d: Testing %d clients --- %s"
                        % (
                            timesim.get_time(),
                            self.curr_model_version,
                            len(test_clients),
                            str(avg_metrics),
                        )
                    )
                    print(log_str)
                    print(log_str, file=log_fp, flush=True)

            if current_training_count < args.concurrency:
                self.select_clients(
                    non_training_clients, args.concurrency - current_training_count
                )
                base_params = {
                    "num_epochs": args.num_epochs,
                    "batch_size": args.batch_size,
                    "model_version": self.curr_model_version,
                }
                timesim.register_train_events(
                    self.selected_clients, self.train_model, base_params
                )
                for c in self.selected_clients:
                    non_training_clients.remove(c)

                log_str = (
                    "TIME %.3f --- Model version %d: Training %d / %d clients --- "
                    % (
                        timesim.get_time(),
                        self.curr_model_version,
                        len(self.selected_clients),
                        args.concurrency,
                    )
                )
                print(log_str)
                print(log_str, file=log_fp, flush=True)
        log_fp.close()
