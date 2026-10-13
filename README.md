# EEG-Based Cognitive Load Decoding with Classical Machine Learning

Research code and analysis notebooks for a master's thesis at Kazakh-British Technical University (2026).

## Research question
How reliably can cognitive workload be decoded from EEG recordings across previously unseen participants, and which physiological features contribute most to model predictions?

## Overview
The project analyzes the COG-BCI dataset (29 participants, three sessions) with N-back and MATB-II tasks. It examines three-level cognitive-load classification and a more separable binary low-versus-high setting using engineered EEG features and covariance-based representations.

## Methods
- EEG preprocessing, epoching, feature engineering, and exploratory analysis.
- Spectral, temporal, regional/channel, and Hjorth-related descriptors.
- Classical machine-learning classifiers including XGBoost and random forests, plus optional Riemannian covariance approaches.
- Leave-one-subject-out (LOSO) evaluation with grouped inner cross-validation and model selection.
- Statistical analysis, visualizations, and feature interpretation (including SHAP in the broader thesis workflow).

## Repository contents
- `cog_bci_analysis.ipynb`: exploratory EEG analysis notebook.
- `loso_classification_final.py`: cross-subject model training and evaluation.
- `3class_cogbci_statistical_analysis_visualizations.ipynb`: three-class results analysis.
- `binary_cogbci_statistical_analysis_visualizations.ipynb`: binary results analysis.
- `features_metadata_final.json` and `common_eeg_channels.json`: feature and channel metadata.
- `mdm_covs_memmap/`: covariance-related artifacts and/or index files (inspect before use).

## Main findings
Three-class cognitive-load decoding was challenging across subjects. The more contrasting binary setting (easier versus harder task conditions) performed better. Spectral and Hjorth-related descriptors were among the influential features. Exact results and evaluation details are discussed in the thesis; numerical scores should be copied from verified result tables, not inferred from this summary.

## Reproducibility
This repository is currently a **research archive**, rather than a one-command reproducible software package. The classification script contains machine-specific Windows paths in its `USER CONFIG` section. To run it, obtain the dataset under its original access conditions, generate the expected feature CSV and covariance inputs, and replace the path settings with local paths. The workflow uses Python and packages including MNE-Python, NumPy, pandas, scikit-learn, XGBoost, joblib, and optional pyRiemann.

A pinned environment, full upstream data-preparation instructions, and example execution commands are planned improvements. No claim of full end-to-end reproducibility is made yet.

## Associated thesis
*EEG-Based Cognitive Load Decoding with Classical Machine Learning: Feature Selection, Evaluation, and Neurophysiological Interpretation.* Danial Daniyaruly, KBTU, 2026.

[Thesis PDF](https://drive.google.com/file/d/1Ws68acnoQ1VsAsALkANOPpWR87g3bk0x/view) · [Defense slides](https://drive.google.com/file/d/13BxwLY13GPAdyNgAe9W6DKWJ2sB1pI01/view)

## Author
Danial Daniyaruly · [GitHub](https://github.com/daniyarulydanial) · [LinkedIn](https://www.linkedin.com/in/danial-daniyaruly/)
