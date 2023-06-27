import importlib
import numpy as np
import os
import random
import torch

from client import Client
from server import Server
from baseline_constants import MODEL_PARAMS, ACCURACY_KEY, LOSS_KEY
from utils.args import parse_args
from utils.model_utils import read_data


def main():
    args = parse_args()

    log_dir = os.path.join(args.log_dir, args.dataset, str(args.log_rank))
    os.makedirs(log_dir, exist_ok=True)
    log_fn = "output.%i" % args.log_rank
    log_file = os.path.join(log_dir, log_fn)
    log_fp = open(log_file, "w+")

    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)

    # open&load model from args
    client_path = "%s/client_model.py" % args.dataset
    if not os.path.exists(client_path):
        print("Please specify a valid dataset.", file=log_fp, flush=True)
    client_path = "%s.client_model" % args.dataset
    mod = importlib.import_module(client_path)
    ClientModel = getattr(mod, "ClientModel")

    num_rounds = args.num_rounds
    eval_every = args.eval_every
    clients_per_round = args.clients_per_round

    # get model params
    param_key = "%s.%s" % (args.dataset, args.model)
    model_params = MODEL_PARAMS[param_key]  ###

    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)
    num_classes = model_params[1]

    client_model = ClientModel(args.seed, args.dataset, args.model, *model_params)
    server = Server(client_model, args.dataset, args.model, num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    train_clients, test_clients = setup_clients(args.dataset, client_model, device)
    print("Total number of train clients: %d" % len(train_clients), file=log_fp, flush=True)
    print("Total number of test clients: %d" % len(test_clients), file=log_fp, flush=True)

    print("--- Random Initialization ---", file=log_fp, flush=True)
    print("---start training")
    for r in range(num_rounds):
        print("--- Round %d of %d: Training %d clients ---" % (r, num_rounds - 1, clients_per_round))
        print(
            "--- Round %d of %d: Training %d clients ---" % (r, num_rounds - 1, clients_per_round),
            file=log_fp,
            flush=True,
        )
        server.select_clients(r, train_clients, clients_per_round)

        server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size)
        server.update_model()

        if (r + 1) % eval_every == 0 or (r + 1) == num_rounds:
            metrics = server.test_model(test_clients, set_to_use="test")
            avg_metrics = average_metrics(metrics)
            print(
            "--- Round %d of %d: Testing %d clients --- %s" % (r, num_rounds - 1, len(test_clients), str(avg_metrics)),
            )
            print(
            "--- Round %d of %d: Testing %d clients --- %s" % (r, num_rounds - 1, len(test_clients), str(avg_metrics)),
            file=log_fp,
            flush=True,
            )

    server.save_model(log_dir)
    log_fp.close()

def setup_clients(dataset, model=None, device="cpu"):
    eval_set = "test"
    train_data_dir = os.path.join("data", dataset, "data", "train")
    test_date_dir = os.path.join("data", dataset, "data", eval_set)

    data = read_data(train_data_dir, test_date_dir)
    train_users, train_groups, train_data, test_users, test_groups, test_data = data
    if len(train_groups) == 0:
        train_groups = [[] for _ in train_users]
    if len(test_groups) == 0:
        test_groups = [[] for _ in test_users]
    print("train_user count:", len(train_users), "test_user count:", len(test_users))
    train_clients = [Client(u, g, train_data[u], None, model, device) for u, g in zip(train_users, train_groups)]
    test_clients = [Client(u, g, None, test_data[u], model, device) for u, g in zip(test_users, test_groups)]
    return train_clients, test_clients

def average_metrics(metrics):
    avg_metrics = {ACCURACY_KEY: [], LOSS_KEY: []}
    for key in metrics:
        avg_metrics[ACCURACY_KEY].append(metrics[key][ACCURACY_KEY])
        avg_metrics[LOSS_KEY].append(metrics[key][LOSS_KEY])
    
    avg_metrics[ACCURACY_KEY] = np.mean(avg_metrics[ACCURACY_KEY])
    avg_metrics[LOSS_KEY] = np.mean(avg_metrics[LOSS_KEY])
    return avg_metrics

if __name__ == "__main__":
    main()
