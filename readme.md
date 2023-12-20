# LEAF-TORCH: A Pytorch Benchmark for Federated Learning
LEAF-Torch is a torch implementation of LEAF, which is originally implemented by TensorFlow.

More details about LEAF can be found at:

-   Github:  [https://github.com/TalwalkarLab/leaf](https://github.com/TalwalkarLab/leaf)
-   Documentation:  [https://leaf.cmu.edu/](https://leaf.cmu.edu/)
-   Paper: "[LEAF: A Benchmark for Federated Settings](https://arxiv.org/abs/1812.01097)"

## Note

- Go to directory of specific dataset to find the instruction of generating data
- Linux environment is required to run this demo(If your training&test dataset is already prepared,feel free to run this demo in windows)
## Environment & Installation
- Install Python 3 & Conda
- Run `conda env create -f environment.yml`

## Instruction
- Run 'python main.py' with below arguments. Example: `python main.py -mode sync --num-rounds 10`
  
Default values for hyper-parameters are set in `utils/args.py`, including:
| Variable Name | Default Value|Optional Values|Description|
|--|--|--|--|
| -mode|"async"|"sync" or "async"|Federated learning modes|
| -dataset|"femnist"|"femnist"|Dataset used for federated learning|
|-model|"cnn"|"cnn"|Neural network used for federated training.|
|--num-rounds|2|integer|Number of rounds to simulate|
|--eval-every|20|integer|Evaluate the federated model every few rounds.
|--clients-per-round|10|integer|Number of clients participating in each round.
|--batch-size|5|integer|Number of training samples in each batch.
|--num-epochs|2|integer|Number of local epochs in each round.|
|-lr|0.01|float|Learning rate for local optimizers.
|--seed|0|integer|Seed for random client sampling and batch splitting
|--log-dir|"logs"|string|Directory for log files.
|--log-rank|0|integer|Identity for current training process (i.e., `CONTAINER_RANK`). Log files will be written to `logs/{DATASET}/{CONTAINER_RANK}/` (e.g., `logs/femnist/0/`)

## Othernotes
- You can find the output of this demo in logs folder. e.g`logs/femnist/0/output`
- You can also find the parameters of CNN in logs folder after finishing the simulation.e.g`logs/femnist/0/cnn.params`
- The dataset is not uploaded,you should download it by yourself and follow the instruction to generate data as described above.
- Before running the demo,you can remove the `logs` folder since it will be generated when the demo is running. 
