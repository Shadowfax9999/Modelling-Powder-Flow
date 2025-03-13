import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms

# Load dataset (replace with actual file path)
data = pd.read_csv("/Users/charliemurray/Documents/all_cohesionless_data/merged_data.csv")

# Extract available inputs and corresponding outputs
input_columns = ["gt_c", "hausner_ratio", "dynamic_angle_of_repose", "rpm"]  # Replace with actual column names
inputs = data.loc[:, input_columns].values
output_columns = ["size", "restitution", "sliding_friction", "rolling_friction"]  # Replace with actual column names
outputs = data.loc[:, output_columns].values


# Replace missing inputs with a marker (e.g., NaN or -1)
inputs = np.where(np.isnan(inputs), -1, inputs)

# Define NSGA-II optimization
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))  # Minimize error
creator.create("Individual", list, fitness=creator.FitnessMulti)

def evaluate(individual):
    """
    Evaluate function: Computes error based on partial input relationships.
    Each individual represents a function that maps partial inputs to outputs.
    """
    weights = np.array(individual).reshape(4, 4)  # Reshape to a 4x4 matrix
    
    # Only use available (non-missing) inputs for predictions
    predictions = []
    for i, row in enumerate(inputs):
        known_indices = row != -1  # Identify known inputs
        if any(known_indices):  # If at least one input is available
            sub_weights = weights[known_indices, :]  # Use only available input rows
            sub_input = row[known_indices].reshape(-1, 1)  # Column vector
            pred = np.sum(sub_weights * sub_input, axis=0)  # Linear combination
        else:
            pred = np.zeros(4)  # Default to zero if no inputs are available
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Compute mean absolute error per output
    error = np.mean(np.abs(predictions - outputs), axis=0)
    return tuple(error)

# Define genetic operators
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)  # Weight range
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 16)  # 4x4 matrix as a flat list
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# Run NSGA-II
def main():
    population = toolbox.population(n=100)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)
    
    algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.2,
                              ngen=50, stats=stats, halloffame=hof, verbose=True)
    return hof

if __name__ == "__main__":
    best_solutions = main()
    print("Best solutions:", best_solutions)
