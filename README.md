# FL Fraud Detection — Results Dashboard

Interactive Streamlit dashboard for the Federated Learning for Cross-Bank Fraud Detection graduation project.

**[Live Demo →](https://your-app.streamlit.app)** *(update after deployment)*

## What this shows
- Federation setup: 4 synthetic heterogeneous banks
- Architecture: BiLSTM trunk + personalization head
- Main results: personalization head impact (C3: 0.18 → 0.60 AUC-PR)
- Ablation study: 5-configuration cumulative analysis
- Negative transfer finding

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Main repository
[federated-learning-fraud-detection](https://github.com/erencebeci/federated-learning-fraud-detection)
