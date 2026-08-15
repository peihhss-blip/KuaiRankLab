# KuaiRankLab: Ranking on KuaiRand

KuaiRankLab is a recommendation ranking project based on the **KuaiRand** dataset.

The project focuses on learning and experimenting with classic and deep ranking models, including CTR prediction, feature interaction, and user behavior sequence modeling.

## Dataset

Dataset: **KuaiRand-1K**

Main information includes:

* User features
* Video features
* User behavior sequences
* Click / Like / Long View and other feedback
* Timestamp information

The dataset is stored locally and is not uploaded to GitHub.

## Current Plan

* [ ] Data overview and EDA
* [ ] Data preprocessing
* [ ] LR baseline
* [ ] DeepFM
* [ ] DIN
* [ ] Sequence length experiments
* [ ] Multi-behavior experiments
* [ ] Time-aware experiments
* [ ] Model comparison and ablation study

## Project Structure

```text
KuaiRankLab/
├── notebooks/      # Data analysis and visualization
├── scripts/        # Training and preprocessing scripts
├── src/
│   ├── data/       # Dataset and feature processing
│   ├── models/     # Ranking models
│   └── utils/
├── learning/       # Small model implementations for learning
├── experiments/    # Experiment records
├── results/        # Results and figures
└── data/           # Local dataset (ignored by Git)
```

## Models

Initial models:

* Logistic Regression
* DeepFM
* DIN

More models may be added later.

## Evaluation

Main offline metrics:

* AUC
* LogLoss

## Status

🚧 Work in progress.
