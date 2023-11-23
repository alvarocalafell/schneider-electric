# Data Business Challenge
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://gitlab.code.hfactory.io/maria-susanne.stoelben/data-business-challenge/blob/main/.pre-commit-config.yaml)

Authors: Alvaro Calafell, Elizaveta Barysheva, João Melo, Madhura Nirale, Maria Stoelben

## Description

In today's business landscape, accurately predicting the costs associated with plastic materials is crucial for Schneider Electric (SE). However, it can be challenging to foresee how plastic costs will evolve in the future due to various factors influencing the market. To address this issue, leveraging data and AI technologies can provide valuable insights to forecast these costs effectively. By analyzing historical data, market trends, raw material prices, supply and demand dynamics, and economic indicators, we hope that we can develop a predictive model that helps businesses estimate future plastic costs with greater accuracy. This data-driven approach could empower Schneider Electric to make informed decisions, optimize its budgeting, and strategically plan its procurement strategies, ultimately maximizing profitability and minimizing financial risks associated with plastic materials.

This exercise will try to tackle this issue by making a model to accurately predict the plastic raw material prices leveraging various data sources and AI. The goal of this project is to predict the best compound price for the months of April, July and October of 2023 based on a dataset containing various time series.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For development run the following line to setup pre-commit hooks.
```bash
pre-commit install
```

## Run the Pipeline
```bash
python src/main.py
```
