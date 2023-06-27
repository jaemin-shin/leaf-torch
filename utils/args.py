import argparse

DATASETS = ["sent140", "femnist", "shakespeare", "celeba", "synthetic", "reddit"]
SIM_TIMES = ["small", "medium", "large"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset",
                        help="name of dataset;",
                        type=str,
                        choices=DATASETS,
                        default="femnist")
    parser.add_argument("-model",
                        help="name of model;",
                        type=str,
                        default="cnn")
    parser.add_argument("--num-rounds",
                        help="number of rounds to simulate;",
                        type=int,
                        default=1000)
    parser.add_argument("--eval-every",
                        help="evaluate every _ rounds;",
                        type=int,
                        default=1)
    parser.add_argument("--clients-per-round",
                        help="number of clients trained per round;",
                        type=int,
                        default=10)
    parser.add_argument("--batch-size",
                        help="batch size when clients train on data;",
                        type=int,
                        default=32)
    parser.add_argument("--num-epochs",
                        help="number of epochs when clients train on data;",
                        type=int,
                        default=5)
    parser.add_argument("-lr",
                        help="learning rate for local optimizers;",
                        type=float,
                        default=0.001)
    parser.add_argument("--seed",
                        help="seed for random client sampling and batch splitting;",
                        type=int,
                        default=0)
    parser.add_argument("--log-dir",
                        help="dir for log file;",
                        type=str,
                        default="logs")
    parser.add_argument("--log-rank",
                        help="suffix identifier for log file;",
                        type=int,
                        default=0)

    return parser.parse_args()
