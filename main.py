import importlib
import numpy as np
import os
import random
import torch

from client import Client
from server import Server
from baseline_constants import MODEL_PARAMS
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
    print(f"{device=}", file=log_fp, flush=True)
    train_clients, test_clients = setup_clients(args.dataset, client_model, device)
    print(
        "Total number of train clients: %d" % len(train_clients),
        file=log_fp,
        flush=True,
    )
    print(
        "Total number of test clients: %d" % len(test_clients), file=log_fp, flush=True
    )

    print("--- Random Initialization ---", file=log_fp, flush=True)
    print("---start training")

    if args.mode == "sync":
        server.syncFL(train_clients, test_clients, args, log_fp)
    elif args.mode == "async":
        server.asyncFL(train_clients, test_clients, args, log_fp)


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
    train_clients = [
        Client(u, g, train_data[u], None, model, device)
        for u, g in zip(train_users, train_groups)
    ]
    test_clients = [
        Client(u, g, None, test_data[u], model, device)
        for u, g in zip(test_users, test_groups)
    ]

    return train_clients, test_clients


if __name__ == "__main__":
    main()
