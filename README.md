# Poison_Airport

This repository contains code for conducting experiments on the Airports datasets (USA, Brazil, Europe) using Graph Convolutional Networks (GCNs) with edge perturbations. The experiments include hyperparameter tuning and evaluation of the impact of evenly and concentrated edge perturbations on the model performance.

## Code Structure

- `main.py`: Main script to run the model training, perturbations, and evaluation.
- `model.py`: Contains the definition of the GCN model.
- `utils.py`: Utility functions for data loading, splitting, perturbations, and plotting.
- `README.md`: This file, containing information about the project and instructions to run the code.


## Installation

Install the necessary dependencies:

  ```bash
  pip install torch torch_geometric numpy pandas matplotlib
  ```

## Usage
To run the experiments, execute the following command:

  ```bash
  python main.py
  ```

This will load the datasets, train the models, apply perturbations, and save the results to perturbation_results.csv.


## Results
The results of the experiments, including plots of the test accuracy against the perturbation percentage, will be saved in the current directory.

## Files
The results of the experiments, including plots of the test accuracy against the perturbation percentage, will be saved in the current directory. The CSV file perturbation_results_airports.csv contains the results of the experiments.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
