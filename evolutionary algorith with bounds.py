#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 21:23:37 2025

@author: charliemurray
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 23:19:24 2025

@author
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv("/Users/charliemurray/Documents/all_cohesionless_data/merged_data.csv")

# Extract available inputs and corresponding outputs
input_columns = ["gt_c", "hausner_ratio", "dynamic_angle_of_repose", "rpm"]  
inputs = data.loc[:, input_columns].values
output_columns = ["size", "restitution", "sliding_friction", "rolling_friction"]  
outputs = data.loc[:, output_columns].values

# Normalize data for stable optimization
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

inputs = input_scaler.fit_transform(inputs)
outputs = output_scaler.fit_transform(outputs)

# Define bounds for output parameters
output_bounds = {
    "size": (0, 2),  #
    "restitution": (0, 1),
    "sliding_friction": (0, 1),  
    "rolling_friction": (0, 1) 
}

# Ensure valid min/max values for random initialization
def random_valid_value(param_name):
    return random.uniform(output_bounds[param_name][0], output_bounds[param_name][1])

#  Ensure correct FitnessMulti definition
if "FitnessMulti" in creator.__dict__:
    del creator.FitnessMulti
if "Individual" in creator.__dict__:
    del creator.Individual

creator.create("FitnessMulti", base.Fitness, weights=(-1.0,))  # Single-objective minimization
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Define Genetic Algorithm Operators
toolbox = base.Toolbox()
toolbox.register("attr_float", lambda: [random_valid_value(param) for param in output_bounds])
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Objective Function: Minimize Error Between Predicted and Targeted Bulk Properties
def evaluate(individual):
    param_set = np.array(individual)
    
    # Ensure parameter bounds are respected
    for i, key in enumerate(output_bounds.keys()):
        param_set[i] = max(min(param_set[i], output_bounds[key][1]), output_bounds[key][0])
    
    closest_idx = np.argmin(np.linalg.norm(outputs - param_set, axis=1))
    predicted_inputs = inputs[closest_idx]  
    
    mae = np.mean(np.abs(predicted_inputs - inputs[closest_idx]))
    
    return (mae, predicted_inputs.tolist())

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

def main():
    print("DEBUG: Starting Evolutionary Algorithm for Inverse Mapping...")  
    population = toolbox.population(n=50)  
    hof = tools.ParetoFront()  
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    logbook = tools.Logbook()
    logbook.header = ["gen", "min", "avg"]
    
    input_mappings = []
    
    for gen in range(50):  
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
        fits_outputs = list(map(toolbox.evaluate, offspring))
        
        for ind, (fit, predicted_inputs) in zip(offspring, fits_outputs):
            ind.fitness.values = (fit,)
            input_mappings.append(predicted_inputs + ind)
        
        population = toolbox.select(offspring, k=len(population))
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
    
    # Save results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"inverse_mapping_{timestamp}.csv")
    inverse_mapped_df = pd.DataFrame(input_mappings, columns=input_columns + output_columns)
    inverse_mapped_df.to_csv(results_file, index=False)
    print(f"Inverse mapped parameters saved to {results_file}")
    
    gen = logbook.select("gen")
    min_vals = np.array(logbook.select("min"))
    avg_vals = np.array(logbook.select("avg"))
    
    plt.figure(figsize=(8, 5))
    plt.plot(gen, min_vals, label="Min MAE", marker="o")  
    plt.plot(gen, avg_vals, label="Avg MAE", linestyle="--", marker="s")  
    plt.xlabel("Generation")
    plt.ylabel("MAE")
    plt.title("Inverse Mapping Optimization Progress")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
