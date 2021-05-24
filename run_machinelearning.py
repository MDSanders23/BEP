import logging
from pathlib import Path
import torch
from swan.dataset import TorchGeometricGraphData, FingerprintsData, DGLGraphData
from swan.modeller import Modeller
from swan.modeller.models import FingerprintFullyConnected, MPNN, InvariantPolynomial
from swan.modeller.models.se3_transformer import TFN, SE3Transformer
from swan.utils.log_config import configure_logger
from swan.utils.plot import create_scatter_plot
from swan.modeller import scikit_modeller

path_data = "/home/max/miniconda3/swan/tests/files/nnData.csv"

all_properties = [
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
for property in all_properties:
    properties = [property]
    data = FingerprintsData(
            path_data, properties=properties, sanitize=False)
    gaussian = scikit_modeller.SKModeller(data, "gaussianprocess")
    trained_data = gaussian.train_model()
    predicted, expected = gaussian.validate_model()
    create_scatter_plot(predicted, expected, properties, f"{property} GaussianProcess")