import csv
import os
import numpy as np
import pandas as pd
from pathlib import Path
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

datasets = ['alarm_interventional_bic', 'barley_interventional_bic', 
            'child_interventional_bic', 'insurance_interventional_bic', 
            'mildew_interventional_bic', 'water_interventional_bic']

for dataset in datasets:
    folder_path = dataset
    models_dir = os.path.join(folder_path, "models")
    os.makedirs(models_dir, exist_ok=True)
    data_file = os.path.join(folder_path, "data.csv") 
    posterior_file = os.path.join(folder_path, "posterior.npy")

    print(f"\n🔧 Processing dataset: {dataset}")
    with open(data_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        node_names = [col.strip() for col in header if col.strip()]

    data = pd.read_csv(data_file)
    posterior = np.load(posterior_file)
    num_models = 5
    
    for k in range(num_models):
        adj = posterior[k]
        edges = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] == 1:
                    edges.append((node_names[i], node_names[j]))
        model = BayesianNetwork(edges)
        
        estimator = MaximumLikelihoodEstimator(model, data)
        cpds = []
        for node in model.nodes():
            cpd = estimator.estimate_cpd(node)
            cpds.append(cpd)
        model.add_cpds(*cpds)

        model_path = os.path.join(models_dir, f'model_{k}.pkl')
        model.save(model_path)
    
    print(f"✅ Completed {dataset} with {num_models} models.\n")

print("🎉 All models generated successfully!")