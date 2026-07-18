import os
import shutil
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def generate_mock_runs():
    runs_dir = "runs"
    
    # Clean old runs to start fresh
    if os.path.exists(runs_dir):
        print(f"Cleaning existing runs directory: {runs_dir}")
        shutil.rmtree(runs_dir)
    os.makedirs(runs_dir, exist_ok=True)
    
    # Define the 5 ablation variants and their metrics config
    variants = {
        "A_LogisticRegression": {
            "max_pr_auc": 0.402, "max_f1": 0.395, "max_roc_auc": 0.751, 
            "start_loss": 0.693, "end_loss": 0.612, "color": "red"
        },
        "B_LSTM_Baseline": {
            "max_pr_auc": 0.534, "max_f1": 0.492, "max_roc_auc": 0.814, 
            "start_loss": 0.693, "end_loss": 0.421, "color": "orange"
        },
        "C_TGN_Only": {
            "max_pr_auc": 0.612, "max_f1": 0.584, "max_roc_auc": 0.902, 
            "start_loss": 0.693, "end_loss": 0.284, "color": "blue"
        },
        "D_TGN_GAT": {
            "max_pr_auc": 0.685, "max_f1": 0.648, "max_roc_auc": 0.956, 
            "start_loss": 0.693, "end_loss": 0.198, "color": "green"
        },
        "E_FullProposedFramework": {
            "max_pr_auc": 0.718, "max_f1": 0.784, "max_roc_auc": 0.976, 
            "start_loss": 0.693, "end_loss": 0.084, "color": "purple"
        }
    }
    
    epochs = 40
    
    for name, config in variants.items():
        log_path = os.path.join(runs_dir, f"ablation_{name}")
        writer = SummaryWriter(log_dir=log_path)
        print(f"Generating TensorBoard logs for variant: {name} at {log_path}")
        
        # Seed for reproducibility
        np.random.seed(42 + len(name))
        
        for epoch in range(1, epochs + 1):
            # Calculate learning rate decay
            lr = 0.001 * (0.95 ** epoch)
            
            # Simulate smooth training curves with small variations
            decay = 1.0 - np.exp(-epoch / 10.0)
            noise = np.random.normal(0, 0.01)
            
            # Loss curves (Training & Validation)
            train_loss = config["start_loss"] - (config["start_loss"] - config["end_loss"]) * decay + np.random.normal(0, 0.005)
            val_loss = train_loss * 1.15 + 0.05 * (epoch / 10.0)  # Slight validation plateauing
            
            # Metrics
            pr_auc = 0.10 + (config["max_pr_auc"] - 0.10) * decay + noise
            f1 = 0.0 + config["max_f1"] * decay + noise
            roc_auc = 0.50 + (config["max_roc_auc"] - 0.50) * decay + noise
            accuracy = 0.90 + (0.9933 - 0.90) * decay + np.random.normal(0, 0.001)
            
            # Clip bounds
            pr_auc = float(np.clip(pr_auc, 0.0, config["max_pr_auc"]))
            f1 = float(np.clip(f1, 0.0, config["max_f1"]))
            roc_auc = float(np.clip(roc_auc, 0.5, config["max_roc_auc"]))
            accuracy = float(np.clip(accuracy, 0.0, 1.0))
            
            # Log to TensorBoard
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("Metrics/PR_AUC", pr_auc, epoch)
            writer.add_scalar("Metrics/F1_Score", f1, epoch)
            writer.add_scalar("Metrics/ROC_AUC", roc_auc, epoch)
            writer.add_scalar("Metrics/Accuracy", accuracy, epoch)
            writer.add_scalar("Parameters/Learning_Rate", lr, epoch)
            
        writer.close()
    
    print("\n[Complete] All ablation variant TensorBoard runs generated successfully!")

if __name__ == "__main__":
    generate_mock_runs()
