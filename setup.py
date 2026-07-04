# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="metabci_3class",
    version="3.0.0",
    description="EEG cognitive impairment detection based on MetaBCI",
    author="MetaBCI Team",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy", "scipy", "scikit-learn", "pandas",
        "mne", "joblib", "matplotlib", "seaborn",
    ],
    extras_require={
        "tabpfn": ["tabpfn", "torch"],
        "stim": ["psychopy"],
        "lsl": ["pylsl"],
    },
)
