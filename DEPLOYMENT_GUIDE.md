# Advanced Deployment & Quick Reference Guide: Insider Threat Detection Framework

This guide provides a quick reference for the 8-week sprint roadmap, common troubleshooting workflows, hyperparameter configs, and deployment checklists.

---

## 1. The 8-Week Sprint Timeline

### Week 1-2: Label Expansion & Splitting
Run label extraction to scale the minority class pool:
```bash
python scripts/label_extractor.py
```
*   **Target**: 30-50 malicious users.
*   **Verification**: Run `python training/stratified_splitter.py` to create splits and check that each fold contains $\ge 5$ positives in val/test.

### Week 3-4: Ablation Studies
Run ablation model variants (Logistic Regression, LSTM, TGN, TGN+GAT):
```bash
python training/ablation_study.py
```
*   **Verification**: Confirm TGN+GAT improves AUPRC significantly over Logistic Regression.

### Week 5: Loss Function Comparison
Train GNN models using Weighted BCE, Adaptive BCE, and Focal Loss variants to address class imbalance.

### Week 6-8: Results Reporting & Documentation
Use the results reporter to generate tables for your paper:
```bash
python training/results_reporter.py
```

---

## 2. Critical Validation Checklist

*   **Data Integrity**:
    *   Confirm $N_{\text{malicious}} \ge 30$ in the labels set.
    *   Verify that both train, val, and test splits have $\ge 1$ positive threat user across all folds.
    *   No data leakage between split indices.
*   **Evaluation Validity**:
    *   Use **AUPRC** (Area Under Precision-Recall Curve) as the primary evaluation metric.
    *   Report metrics as `mean +/- std` across all cross-validation folds.
*   **Ablation Quality**:
    *   Verify that baselines are weaker than the proposed GNN model.
    *   Perform statistical significance t-tests to verify $p < 0.05$.

---

## 3. Cost-Benefit Optimization Formula
Find the decision threshold that minimizes investigation overhead:
$$\text{Cost} = \text{False Positives} \times C_{\text{FP}} - \text{True Positives} \times V_{\text{TP}}$$
where:
*   $C_{\text{FP}}$ is the investigation cost per false alarm (e.g., $160, representing 2 hours of analyst investigation).
*   $V_{\text{TP}}$ is the value of a caught insider (e.g., preventing $1,000,000 in damages).

---

## 4. Hyperparameter Reference

### TGN Config
```python
tgn_config = {
    'memory_dimension': 128,
    'embedding_dimension': 128,
    'time_embedding_dimension': 32,
    'num_gru_layers': 1,
    'dropout': 0.1,
    'message_aggregation': 'attention'
}
```

### GAT Config
```python
gat_config = {
    'num_heads': 4,
    'num_layers': 2,
    'attention_dropout': 0.2,
    'residual_connection': True,
    'layer_normalization': True
}
```
