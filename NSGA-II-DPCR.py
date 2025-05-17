
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair


import time  
start_time = time.time() 


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(13,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3)  
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error')
    return model


model = build_model()
model.load_weights(r'Weights.weights.h5', skip_mismatch=True)


#
file_paths = [
    r'data1.xlsx',
    r'data2.xlsx',
    r'data3.xlsx'
]
selected_columns = ['Features']
datasets = [pd.read_excel(file) for file in file_paths]


normalization_factors = {col: max([df[col].max() for df in datasets]) for col in selected_columns}
for df in datasets:
    for col in selected_columns:
        df[col] /= normalization_factors[col]

epsilon = 1e-6


fixed_values = {
    '###fixed value#'
}

def create_full_input(x):
    design_cols = ['Features']
    design_norm = [x[i] / normalization_factors[col] for i, col in enumerate(design_cols)]
    fixed = [fixed_values[col] for col in fixed_values.keys()]
    return np.concatenate([np.array(fixed), np.array(design_norm)])

def inverse_normalize_output(preds):
    return np.clip(preds[0] * normalization_factors['UTS'], epsilon, None), \
           np.clip(preds[1] * normalization_factors['Kt'], epsilon, None), \
           np.clip(preds[2], epsilon, None)



def load_known_pareto(pareto_path):
    df = pd.read_excel(pareto_path)
    return {
        'X': df[['Configurations', 'Diameter', 'Volume Fraction', 'Varience']].values,
        'UTS': df['UTS'].values,
        'Kt': df['Kt'].values
    }

known_pareto = load_known_pareto(r'known_pareto.xlsx')


class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=num1,
            n_obj=2,
            n_constr=num2,
            xl,
            xu,
            elementwise_evaluation=True
        )
        self.constraint1
        self.constraint2
        self.constraint3
        self.known_pareto 

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.array(x)
        
        full_input = create_full_input(x).reshape(1, -1)
        preds = model.predict(full_input, verbose=0)[0]
        uts, kt, _ = inverse_normalize_output(preds)
        
        out["F"] = [-uts, -kt]
        out["G"] = [
            constraint1,
           constraint2,
           constraint3
        ]


class PartitionMonitor:
    def __init__(self):
        self.uts_bins = [0, a1, a2, a3, a4, float('inf')]
        self.kt_bins = [0, b1, b2, b3, b4, float('inf')]
        self.history = [] 
        self.coverage_grid = np.zeros((5, 5), dtype=float)  

    def update(self, uts_vals, kt_vals):
        uts_digit = np.digitize(uts_vals, self.uts_bins) - 1
        kt_digit = np.digitize(kt_vals, self.kt_bins) - 1
        
        current_coverage = np.zeros((5, 5))
        for u, k in zip(uts_digit, kt_digit):
            if 0 <= u < 5 and 0 <= k < 5:
                current_coverage[u, k] += 1
        
        self.history.append(current_coverage)
        self.coverage_grid += current_coverage

    def check_restart(self):
        total_coverage = np.sum(self.history, axis=0)
        uncovered = np.where(total_coverage == 0)
        return len(uncovered[0]) > 0

    def get_coverage_ratio(self):
        total_coverage = np.sum(self.history, axis=0)
        return np.sum(total_coverage > 0) / 25.0


def chaotic_perturbation(x, scale=0.05):

    x0 = np.random.rand(*x.shape)
    for _ in range(5):  
        x0 = 3.9 * x0 * (1 - x0) 
    perturbation = (x0 - 0.5) * scale
    x_new = x + perturbation
    x_new = np.clip(x_new, problem.xl, problem.xu)

    return x_new



problem = MyProblem()
monitor = PartitionMonitor()


algorithm = NSGA2(
    pop_size=50,
    sampling=FloatRandomSampling(), 
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True,
    repair=RoundingRepair()
)


no_improve_count = 0
best_avg_objective = None
elite_memory = []
trajectory_history = [] 


def initialize_population():
    sampler = FloatRandomSampling()
    pop = sampler.do(problem, 50)

    n_injected = int(0.2 * len(pop))
    indices = np.random.choice(len(known_pareto['X']), n_injected, replace=False)
    for i in range(n_injected):
        idx = indices[i]
        x = known_pareto['X'][idx]

        x_norm = [
            x[0] / normalization_factors['Configurations'],
            x[1] / normalization_factors['Diameter'],
            x[2] / normalization_factors['Volume Fraction'],
            x[3] / normalization_factors['Varience']
        ]
        pop[i].X = np.array(x_norm)

        full_input = create_full_input(pop[i].X).reshape(1, -1)
        preds = model.predict(full_input, verbose=0)[0]
        uts, kt, _ = inverse_normalize_output(preds)
        pop[i].F = [-uts, -kt]
    return pop

pop_init = initialize_population()
algorithm.pop = pop_init  



def callback(algorithm):
    global no_improve_count, best_avg_objective, elite_memory, trajectory_history

    F = algorithm.pop.get("F")
    X = algorithm.pop.get("X")
    uts_vals = -F[:, 0]
    kt_vals = -F[:, 1]
    

    monitor.update(uts_vals, kt_vals)
    

    restart_flag = False
    if algorithm.n_gen % 20 == 0:
        if monitor.check_restart():
            print(f"Generation {algorithm.n_gen}:restart！")
            restart_flag = True


    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    current_front = F[nds]
    

    current_avg = np.mean(np.sum(current_front, axis=1))
    if best_avg_objective is None or current_avg < best_avg_objective:
        best_avg_objective = current_avg
        no_improve_count = 0
    else:
        no_improve_count += 1
    

    if restart_flag or no_improve_count >= 20:
        print(f"Generation {algorithm.n_gen}: restart...")
        num_replace = max(1, int(0.05 * len(X)))
        if len(elite_memory) > 0:
            memory_X = np.array([item['X'] for item in elite_memory])
            indices = np.random.choice(len(memory_X), num_replace)
            new_candidates = [chaotic_perturbation(memory_X[i]) for i in indices]
        else:
            new_candidates = [chaotic_perturbation(X[i]) for i in np.random.choice(len(X), num_replace)]
        
        obj_sum = np.sum(F, axis=1)
        worst_idx = np.argsort(obj_sum)[-num_replace:]
        for i, idx in enumerate(worst_idx):
            X[idx] = new_candidates[i]
            full_input = create_full_input(X[idx]).reshape(1, -1)
            preds = model.predict(full_input, verbose=0)[0]
            uts, kt, _ = inverse_normalize_output(preds)
            F[idx] = [-uts, -kt]
        
        algorithm.pop.set("X", X)
        algorithm.pop.set("F", F)
        no_improve_count = 0
    

    elite_memory.extend([{'X': X[i], 'F': F[i]} for i in nds])
    if len(elite_memory) > 100:
        elite_memory = elite_memory[-100:]
    

    uts_list = []
    kt_list = []
    third_list = []
    for xi in X:
        full_input = create_full_input(xi).reshape(1, -1)
        preds = model.predict(full_input, verbose=0)[0]
        uts, kt, third = inverse_normalize_output(preds)
        uts_list.append(uts)
        kt_list.append(kt)
        third_list.append(third)
    
    trajectory_history.append({
        'X': X.copy(),
        'uts': np.array(uts_list),
        'kt': np.array(kt_list),
        'third': np.array(third_list)
    })


res = minimize(problem,
               algorithm,
               ('n_gen', 50),
               callback=callback,
               seed=42,
               verbose=True,
               save_history=True)



end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")


