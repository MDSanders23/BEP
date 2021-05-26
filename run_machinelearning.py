#!/usr/bin/env python
import argparse
import logging
from pathlib import Path

import numpy as np
import json
import sklearn.gaussian_process.kernels as gp
from scipy import stats

from swan.dataset import FingerprintsData
from swan.modeller import SKModeller
from swan.utils.log_config import configure_logger

configure_logger(Path("."))

# Starting logger
LOGGER = logging.getLogger(__name__)


# Scikit learn model hyperparameters
dict_parameters = {
    "decision_tree": {'criterion': 'friedman_mse', 'max_features': 'auto', 'splitter': 'random'},
    "svm": {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'},
    "gaussian_process": {"kernel": gp.ConstantKernel(1.0, (1e-4, 1e4)) * gp.RBF(10.0, (1e-4, 1e4))}
}

# Training variables
properties = [
"Mw",
"#electrons",
"Dipole moment",
"Qxx",
"Qyy",
"Qzz",
"Polarisability",
"IP(vertical)",
"EA(vertical",
"IP(adiabatic)",
"E_trans_S_S_1",
"E_trans_S_S_2",
"E_trans_S_S_3",
"E_trans_S_S_4",
"E_trans_S_S_5",
"E_trans_S_S_6",
"E_trans_S_S_7",
"E_trans_S_S_8",
"E_trans_S_S_9",
"E_trans_S_S_10",
"F_oscil_S_S_1",
"F_oscil_S_S_2",
"F_oscil_S_S_3",
"F_oscil_S_S_4",
"F_oscil_S_S_5",
"F_oscil_S_S_6",
"F_oscil_S_S_7",
"F_oscil_S_S_8",
"F_oscil_S_S_9",
"F_oscil_S_S_10",
"E_trans_S_T_1",
"E_trans_S_T_2",
"E_trans_S_T_3",
"E_trans_S_T_4",
"E_trans_S_T_5",
"E_trans_S_T_6",
"E_trans_S_T_7",
"E_trans_S_T_8",
"E_trans_S_T_9",
"E_trans_S_T_10",
"E_trans_plus_S_S_1",
"E_trans_plus_S_S_2",
"E_trans_plus_S_S_3",
"E_trans_plus_S_S_4",
"E_trans_plus_S_S_5",
"E_trans_plus_S_S_6",
"E_trans_plus_S_S_7",
"E_trans_plus_S_S_8",
"E_trans_plus_S_S_9",
"E_trans_plus_S_S_10",
"F_oscil_plus_S_S_1",
"F_oscil_plus_S_S_2",
"F_oscil_plus_S_S_3",
"F_oscil_plus_S_S_4",
"F_oscil_plus_S_S_5",
"F_oscil_plus_S_S_6",
"F_oscil_plus_S_S_7",
"F_oscil_plus_S_S_8",
"F_oscil_plus_S_S_9",
"F_oscil_plus_S_S_10",
"E_trans_min_S_S_1",
"E_trans_min_S_S_2",
"E_trans_min_S_S_3",
"E_trans_min_S_S_4",
"E_trans_min_S_S_5",
"E_trans_min_S_S_6",
"E_trans_min_S_S_7",
"E_trans_min_S_S_8",
"E_trans_min_S_S_9",
"E_trans_min_S_S_10",
"F_oscil_min_S_S_1",
"F_oscil_min_S_S_2",
"F_oscil_min_S_S_3",
"F_oscil_min_S_S_4",
"F_oscil_min_S_S_5",
"F_oscil_min_S_S_6",
"F_oscil_min_S_S_7",
"F_oscil_min_S_S_8",
"F_oscil_min_S_S_9"
]


def run_all(path_data: str, output_file: str):
    nruns = 3
    models = ["gaussian_process"]  + ["decision_tree", "svm"]
    rvalues = {}
    for p in properties:
        rvalues[p] = {}
        data = FingerprintsData(
            path_data, properties=[p], sanitize=False)
        data.scale_labels()
        for m in models:
            mean = np.mean([run_scikit_model(m, data) for i in range(nruns)])
            print(f"model: {m} property: {p} mean: {mean}")
            rvalues[p][m] = mean

    with open(f"{output_file}.json", 'w') as handler:
        json.dump(rvalues, handler)


def run_scikit_model(name_model: str, data: FingerprintsData):
    parameters = dict_parameters[name_model]
    modeller = SKModeller(data, name_model, **parameters)
    modeller.train_model()
    predicted, expected = modeller.validate_model()
    reg = stats.linregress(predicted, expected.flatten())
    return reg.rvalue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File with the properties", required=True)
    parser.add_argument("-o", "--output", help="output file", required=True)
    args = parser.parse_args()
    run_all(args.file, args.output)


if __name__ == "__main__":
    main()