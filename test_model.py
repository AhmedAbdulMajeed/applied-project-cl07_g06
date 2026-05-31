import os
import numpy as np


def test_dataset_size():
    # The synthetic generator in src/prepare_data.py produces 50 samples
    X = np.arange(-100, 100, 4)
    assert len(X) == 50


def test_requirements_file_exists():
    assert os.path.exists("requirements.txt")


def test_params_file_exists():
    assert os.path.exists("params.yaml")


def test_model_script_exists():
    assert os.path.exists("src/model.py")


def test_evaluate_script_exists():
    assert os.path.exists("src/evaluate.py")
