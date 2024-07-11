import torch
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_airports_data, split_indices, print_dataset_statistics, evenly_perturb_edges, concentrated_perturb_edges
from model import GCNNet, train_model

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(20)

# Hyperparameter grid search
hidden_channels_list = [64, 128, 256, 512]
learning_rates = [0.01, 0.05, 0.005, 0.001, 0.0005]
weight_decays = [1e-4, 1e-5]

# List of countries to process
countries = ['USA', 'Brazil', 'Europe']

# Initialize results DataFrame
results_df = pd.DataFrame(columns=['Country', 'Hidden_Channels', 'Learning_Rate', 'Weight_Decay', 'Accuracy', 'Perturbation_Type', 'Perturbation_Percentage'])

# Process each country and print accuracies
for country in countries:
    print(f'Processing country: {country}')
    data = load_airports_data(country)

    # Print dataset statistics
    print_dataset_statistics(data, country)

    # Prepare the masks
    train_idx, val_idx, test_idx = split_indices(data.num_nodes)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    best_acc = 0
    best_params = None

    # Grid search for best hyperparameters
    for hidden_channels in hidden_channels_list:
        for lr in learning_rates:
            for weight_decay in weight_decays:
                model = GCNNet(data.num_node_features, hidden_channels, data.y.max().item() + 1)
                acc = train_model(model, data, lr, weight_decay)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (hidden_channels, lr, weight_decay)

    print(f"Best accuracy for {country}: {best_acc:.4f} with params {best_params}")

    # Add best model accuracy to the results DataFrame
    new_row = {
        'Country': country,
        'Hidden_Channels': best_params[0],
        'Learning_Rate': best_params[1],
        'Weight_Decay': best_params[2],
        'Accuracy': best_acc,
        'Perturbation_Type': 'None',
        'Perturbation_Percentage': 0
    }
    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    # Perturbation experiments
    perturbation_percentages = [0.05, 0.1, 0.15, 0.2, 0.25]
    perturbation_results = {"Even": [], "Concentrated": []}

    for perturbation_percentage in perturbation_percentages:
        # Even perturbation
        perturbed_data = evenly_perturb_edges(data.clone(), perturbation_percentage)
        model = GCNNet(data.num_node_features, best_params[0], data.y.max().item() + 1)
        even_acc = train_model(model, perturbed_data, best_params[1], best_params[2])
        perturbation_results["Even"].append(even_acc)

        new_row = {
            'Country': country,
            'Hidden_Channels': best_params[0],
            'Learning_Rate': best_params[1],
            'Weight_Decay': best_params[2],
            'Accuracy': even_acc,
            'Perturbation_Type': 'Even',
            'Perturbation_Percentage': perturbation_percentage
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

        # Concentrated perturbation
        perturbed_data, top_k_nodes = concentrated_perturb_edges(data.clone(), perturbation_percentage)
        model = GCNNet(data.num_node_features, best_params[0], data.y.max().item() + 1)
        concentrated_acc = train_model(model, perturbed_data, best_params[1], best_params[2])
        perturbation_results["Concentrated"].append(concentrated_acc)

        new_row = {
            'Country': country,
            'Hidden_Channels': best_params[0],
            'Learning_Rate': best_params[1],
            'Weight_Decay': best_params[2],
            'Accuracy': concentrated_acc,
            'Perturbation_Type': 'Concentrated',
            'Perturbation_Percentage': perturbation_percentage
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot([0] + perturbation_percentages, [best_acc] + perturbation_results["Even"], marker='o', label='Even Perturbation')
    plt.plot([0] + perturbation_percentages, [best_acc] + perturbation_results["Concentrated"], marker='o', label='Concentrated Perturbation')
    plt.xlabel('Perturbation Percentage')
    plt.ylabel('Test Accuracy')
    plt.title(f'Effect of Edge Perturbations on {country} Dataset')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{country}_perturbation_plot.png")
    plt.show()

# Save results to a CSV file
results_df.to_csv('perturbation_results_airports.csv', index=False)
