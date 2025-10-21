import os
import yaml
import mlflow
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from .model import ClassificationModel
from .data import ClassificationDataModule

class ClassificationEvaluator:
    """Evaluator for classification models."""
    
    def __init__(self, config: Dict):
        """Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.data_module = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, model_path: str) -> None:
        """Load the model from the given path.
        
        Args:
            model_path: Path to the model checkpoint
        """
        self.model = ClassificationModel.load_from_checkpoint(
            model_path,
            config=self.config
        )
        self.model.to(self.device)
        self.model.eval()
        
    def load_data(self) -> None:
        """Load the test dataset."""
        self.data_module = ClassificationDataModule(
            config=self.config,
            stage="test"
        )
        self.data_module.setup()
        
    def evaluate(self) -> Dict:
        """Evaluate the model on the test set.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None or self.data_module is None:
            raise ValueError("Model and data must be loaded before evaluation")
            
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in self.data_module.test_dataloader():
                images = batch["image"].to(self.device)
                labels = batch["label"]
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="weighted"),
            "recall": recall_score(all_labels, all_preds, average="weighted"),
            "f1": f1_score(all_labels, all_preds, average="weighted")
        }
        
        # Calculate per-class metrics
        class_report = classification_report(
            all_labels, all_preds,
            target_names=self.data_module.class_names,
            output_dict=True
        )
        metrics.update(class_report)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Calculate ROC curves and AUC scores
        roc_curves = {}
        auc_scores = {}
        
        for i, class_name in enumerate(self.data_module.class_names):
            fpr, tpr, _ = roc_curve(
                (all_labels == i).astype(int),
                all_probs[:, i]
            )
            roc_curves[class_name] = (fpr, tpr)
            auc_scores[class_name] = auc(fpr, tpr)
        
        metrics["auc_scores"] = auc_scores
        
        # Generate visualizations
        self._plot_confusion_matrix(cm)
        self._plot_roc_curves(roc_curves, auc_scores)
        
        return metrics
    
    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.data_module.class_names,
            yticklabels=self.data_module.class_names
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(
            self.config["paths"]["evaluation"],
            "confusion_matrix.png"
        )
        plt.savefig(plot_path)
        plt.close()
        
        # Log to MLflow
        mlflow.log_artifact(plot_path)
    
    def _plot_roc_curves(
        self,
        roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
        auc_scores: Dict[str, float]
    ) -> None:
        """Plot ROC curves.
        
        Args:
            roc_curves: Dictionary of ROC curves
            auc_scores: Dictionary of AUC scores
        """
        plt.figure(figsize=(10, 8))
        
        for class_name, (fpr, tpr) in roc_curves.items():
            plt.plot(
                fpr,
                tpr,
                label=f"{class_name} (AUC = {auc_scores[class_name]:.3f})"
            )
        
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(
            self.config["paths"]["evaluation"],
            "roc_curves.png"
        )
        plt.savefig(plot_path)
        plt.close()
        
        # Log to MLflow
        mlflow.log_artifact(plot_path)

def main():
    """Main function to run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize evaluator
    evaluator = ClassificationEvaluator(config)
    
    # Load model and data
    evaluator.load_model(args.model_path)
    evaluator.load_data()
    
    # Start MLflow run
    with mlflow.start_run(run_name="model_evaluation"):
        # Log parameters
        mlflow.log_params(config["model"])
        mlflow.log_params(config["training"])
        
        # Run evaluation
        metrics = evaluator.evaluate()
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        print("Evaluation completed. Metrics:")
        print(metrics)

if __name__ == "__main__":
    main() 