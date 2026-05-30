import os
import numpy as np


def test_dataset_size():
    X = np.arange(-100, 100, 4)
    assert len(X) == 50


def test_requirements_file_exists():
    assert os.path.exists("requirements.txt")


def test_model_file_exists():
    assert os.path.exists("model.py")