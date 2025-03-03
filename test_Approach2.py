import torch
from torchvision import models
#from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
from Focal_loss import FocalLoss
import pandas as pd
from sklearn.model_selection import train_test_split
from dataloader import ChestXrayDataSet
from torch.utils.data import DataLoader
from approach2_model import Approach2_Baseline
from torch.cuda.amp import autocast
from train_Approach2 import validate, plot_roc_curves, visualize_training

def load_model_from_checkpoint(
                               model_path, 
                               learner_weight_path, 
                               checkpoint_path,
                               DEVICE
                               ):
    # Load the model
    model = Approach2_Baseline(
                                model_path, 
                                learner_weight_path,
                                DEVICE)

    # Load the trained weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=False) 

    # Set to evaluation mode
    model.eval()

    return model

if __name__ == "__main__":

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # Global configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", DEVICE)
    batch_size = 64
    epochs = 3

    print("DATA LOADING")

    train_df_main = pd.read_csv('/kaggle/input/chestxray8-dataframe/train_df.csv')
    train_df_main.drop(['No Finding'], axis=1, inplace=True)
    labels = train_df_main.columns[2:-1].tolist()

    # Split data
    train_df, discard = train_test_split(train_df_main, test_size=0.7, random_state=1993)
    train_and_valid_set, test_set = train_test_split(train_df, test_size=0.2, random_state=1993)
    
    test_dataset = ChestXrayDataSet(
        dataframe=test_set,
        mode="test"
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=32,
        pin_memory=True
    )
   
    print("DATA LOADING COMPLETE")

    print("LOADING MODEL")
    # Create model
    # pre-trained model path
    model_path = "/kaggle/input/weight-ppad/weight/best_64_0.0001_original_35000_0.864.pt"

    # checkpoint path
    learner_weight_path = "/kaggle/input/weight-ppad/weight/PPAD_CheXpert_auc_0.896288_acc_0.842_f1_0.8301075268817204_ap_0.9075604434206554.pt"

    checkpoint_path = "/kaggle/input/weight-approach2/weight_Approach2/chest_xray_model_final.pth"
    # Load Approach2_Baseline model
    model = load_model_from_checkpoint(
        model_path, 
        learner_weight_path, 
        checkpoint_path,
        DEVICE
    )
   
    model = model.to(torch.float32)
    # Process all input tensors
    def process_batch(batch):
      return {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    

    print("MODEL LOADING COMPLETE")

    print("EVALUATION")

    # Evaluate on test set
    _, _, test_outputs, test_labels = validate(
        model, test_loader, FocalLoss(alpha=0.25, gamma=2.0, reduction='mean'), DEVICE
    )

    # Plot ROC curves
    auc_roc_vals = plot_roc_curves(labels, test_outputs, test_labels, when='after training')