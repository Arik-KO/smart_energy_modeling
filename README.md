# Month-ahead forecasting and battery scheduling for residential buildings

Repository for the manuscript

> **Month-ahead forecasting and battery scheduling for residential buildings: A leakage-free
> evaluation of wavelet–Transformer forecasts and genetic-algorithm scheduling**
> A. A. Anik, H. Esmaiel, S. S. Ba Hashwan, A. O. Baarimah, Y. H. Naif
> *Journal of Engineering Research* (under review)

## Status

The complete code will be released in this repository upon publication of the article.

## Overview

The project evaluates month-ahead forecasting of building electricity demand and PV supply,
and the value of these forecasts for daily battery scheduling under a two-step residential
tariff, for an aggregate of five residential buildings in British Columbia, Canada.

- **Leakage-free preprocessing:** min–max scaling fitted on the training period only, and the
  discrete wavelet transform (DWT) applied separately to each model input window.
- **Forecasting models:** LSTM, DWT-LSTM, Transformer (TNN) and DWT-TNN with identical inputs,
  prediction head and tuning budget, each trained with five random seeds.
- **Recursive forecasts:** month-ahead, week-ahead and one-hour-ahead, compared with naive
  baselines using the same information.
- **Scheduling:** genetic algorithm, rule-based scheduler, non-scheduled rule and an exact
  perfect-information optimum (linear programming), with every schedule scored on observed data.
- **Analyses:** DWT ablation, evaluation on several test months, value of forecast accuracy and
  battery economics.

## Planned repository structure
configs/ experiment settings
src/efs/ library code
data/ data loading, time alignment, calendar features, splits
preprocessing/ train-only scaling and causal wavelet denoising
forecasting/ windowing, models, training, recursive forecasting, tuning
evaluation/ metrics, hold-out evaluation, figures and tables
scheduling/ tariff, battery model, genetic algorithm, rule-based scheduler, LP
scripts/ command-line entry points for each step of the workflow
tests/ unit tests, including leakage and energy-conservation tests
tuned_params/ selected hyperparameters

## Data

The study uses publicly available data, which will not be redistributed here:

- **Demand:** HUE dataset, Harvard Dataverse, doi:10.7910/DVN/N3HGRN
- **PV:** renewables.ninja (MERRA-2, 2019; settings given in the paper)

## Contact

Arik Anjum Anik — arikanik40@gmail.com
