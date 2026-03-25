# Quantum Extreme Learning Machine (QELM) for Lottery Prediction
# Lottery prediction generated using a fixed quantum reservoir and a trainable linear readout.
# Quantum Regression Model with Qiskit

# v2: df.copy(); StandardScaler na kvantnim feature-ima; RidgeCV; clip po poziciji; reps/train_window blago povećani.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

from qiskit_machine_learning.utils import algorithm_globals
import random

# ================= SEED PARAMETERS =================
SEED = 39
random.seed(SEED)
np.random.seed(SEED)
algorithm_globals.random_seed = SEED
# ==================================================


# Use the existing dataframe
df_raw = pd.read_csv('/data/loto7hh_4586_k24.csv')
# 4548 historical draws of Lotto 7/39 (Serbia)

_MIN_POS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
_MAX_POS = np.array([33, 34, 35, 36, 37, 38, 39], dtype=int)


def quantum_extreme_learning_predict(df):
    df = df.copy()
    cols = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'Num6', 'Num7']
    predictions = {}

    # Model Hyperparameters
    num_lags = 3
    num_qubits = 3
    train_window = 36  # Balanced for computational efficiency in the kernel

    # Define a fixed (non-trainable) Quantum Feature Map
    # In QELM, the quantum layer acts as a fixed, high-dimensional non-linear reservoir.
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=3, entanglement='linear')

    for idx, col in enumerate(cols):
        # 1. Feature Engineering: 3 Lags to capture more temporal context
        df_col = pd.DataFrame(df[col])
        for i in range(1, num_lags + 1):
            df_col[f'lag_{i}'] = df_col[col].shift(i)

        df_col = df_col.dropna().tail(train_window + 1)

        X_raw = df_col[[f'lag_{i}' for i in range(1, num_lags + 1)]].values
        y_raw = df_col[col].values.astype(np.float64)

        # 2. Scaling to [0, 2*pi] for the quantum feature map encoding
        scaler_x = MinMaxScaler(feature_range=(0, 2 * np.pi))
        X_scaled = scaler_x.fit_transform(X_raw)

        # 3. Quantum Feature Mapping (The "Hidden" Layer)
        # Project classical lags into the 2^n dimensional Hilbert space (8 dimensions for 3 qubits)
        q_features = []
        for x_vec in X_scaled:
            qc_bound = feature_map.assign_parameters(x_vec)
            sv = Statevector.from_instruction(qc_bound)
            # Use the probability distribution of the state as features (non-linear projection)
            q_features.append(np.abs(sv.data) ** 2)

        q_features = np.array(q_features)

        # 4. ELM Training: Linear Readout
        # Only the output weights are learned, making this extremely fast.
        X_train = q_features[:-1]
        y_train = y_raw[:-1]
        X_next = q_features[-1:]

        # v2: skaliranje amplitude pre Ridge-a + automatski alpha
        alphas = np.logspace(-3, 3, 25)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=alphas, cv=min(5, len(y_train)))),
            ]
        )
        model.fit(X_train, y_train)

        # 5. Prediction for the next draw
        y_pred = model.predict(X_next)
        lo, hi = int(_MIN_POS[idx]), int(_MAX_POS[idx])
        predictions[col] = int(round(np.clip(y_pred[0], lo, hi)))

    return predictions

print()
print("Computing predictions using Quantum Extreme Learning Machine (QELM) ...")
print()
q_elm_results = quantum_extreme_learning_predict(df_raw)

# Format for display
q_elm_df = pd.DataFrame([q_elm_results])
# q_elm_df.index = ['Quantum Extreme Learning Machine (QELM) Prediction']

print()
print("Lottery prediction generated using a fixed quantum reservoir and a trainable linear readout.")
print()

print()
print("Quantum Extreme Learning Machine (QELM) Results:")
print(q_elm_df.to_string(index=True))
print()
"""
Computing predictions using Quantum Extreme Learning Machine (QELM) ...

feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=3, entanglement='linear')

Lottery prediction generated using a fixed quantum reservoir and a trainable linear readout.


Quantum Extreme Learning Machine (QELM) Results:
   Num1  Num2  Num3  Num4  Num5  Num6  Num7
0     5    11    x    y    22    z    35
"""



"""
Quantum Extreme Learning Machine (QELM).

v2: df.copy(); num_qubits = 3 i num_lags = 3 ostaju usklađeni; ZZFeatureMap reps 2→3; train_window 30→36; kvantni feature-i idu kroz StandardScaler, readout je RidgeCV (alphas logspace, cv=min(5, len(y_train))); predikcija se clip-uje po poziciji (1..33 … 7..39). 

(v2: readout = StandardScaler + RidgeCV; ZZFeatureMap reps=3; train_window=36; num_qubits=3 kao ulazna dimenzija lag-ova.)
"""
