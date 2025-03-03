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

# Function to train model for one epoch
def train_one_epoch(model, train_loader, criterion, optimizer, DEVICE):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.cuda.amp.autocast():
        for idx, (img, labels, masks, position_names) in enumerate(tqdm(train_loader)):
            #print(f"Batch {idx}: Image shape {img.shape}, Labels shape {labels.shape}, Masks shape {masks.shape}, Position names shape {len(position_names)}")
            image = img.to(DEVICE)
            labels = labels.to(dtype=image.dtype, device=image.device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            anomaly_features = []
            for mask, position_name in zip(masks, position_names):
                
                mask = mask.to(dtype=image.dtype, device=image.device)

                anomaly_feature = model.extract_anomaly_feature(image, mask, position_name)
                anomaly_features.append(anomaly_feature)
                #print("Done First position in 1st testloader loop")
            anomaly_features = torch.stack(anomaly_features, dim=0)
            anomaly_features = anomaly_features.permute(1, 0, 2)
            #print("Done extracting anomaly features in 1st testloader loop")
            #print(anomaly_features.shape)
            outputs = model(img, anomaly_features)
            #print(output.shape)
            #print(output)
            loss = criterion(outputs, labels)
            
            # Backward + optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * img.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# Function to validate model
def validate(model, valid_loader, criterion, DEVICE):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad() and autocast():
        for idx, (img, labels, masks, position_names) in enumerate(tqdm(valid_loader)):
            #print(f"Batch {idx}: Image shape {img.shape}, Labels shape {labels.shape}, Masks shape {masks.shape}, Position names shape {len(position_names)}")
            image = img.to(DEVICE)
            labels = labels.to(dtype=image.dtype, device=image.device)
            anomaly_features = []
            for mask, position_name in zip(masks, position_names):
                
                mask = mask.to(dtype=image.dtype, device=image.device)

                anomaly_feature = model.extract_anomaly_feature(image, mask, position_name)
                anomaly_features.append(anomaly_feature)
                #print("Done First position in 1st testloader loop")
            anomaly_features = torch.stack(anomaly_features, dim=0)
            anomaly_features = anomaly_features.permute(1, 0, 2)
            #print("Done extracting anomaly features in 1st testloader loop")
            #print(anomaly_features.shape)
            outputs = model(img, anomaly_features)
            #print(output.shape)
            #print(output)
            
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * img.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
            
            all_outputs.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    
    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = correct / total
    all_outputs = np.vstack(all_outputs)
    all_labels = np.vstack(all_labels)
    
    return epoch_loss, epoch_acc, all_outputs, all_labels

# Learning rate scheduler function
def get_lr_scheduler(optimizer):
    # Lambda function to simulate the custom learning rate function from the original code
    lr_lambda = lambda epoch: 1.0 if epoch < 8 else (0.8 ** (epoch - 8))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Function to plot ROC curves
def plot_roc_curves(labels_list, predicted_vals, true_labels, when=''):
    auc_roc_vals = []
    
    for i in range(len(labels_list)):
        try:
            gt = true_labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf, label=f"{labels_list[i]} ({round(auc_roc, 3)})")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title(f'ROC curve {when}')
            plt.legend(loc='best')
            plt.show()
        except Exception as e:
            print(f"Error in generating ROC curve for {labels_list[i]}: {str(e)}")
            auc_roc_vals.append(float('nan'))
    
    return auc_roc_vals

# Main training function
def train_model(model, train_loader, valid_loader, num_epochs, DEVICE):
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = get_lr_scheduler(optimizer)
    
    history = {
        'train_loss': [],
        'valid_loss': [],
        'train_acc': [],
        'valid_acc': []
    }
    
    best_valid_loss = float('inf')
    
    with torch.cuda.amp.autocast():
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            
            # Validate
            valid_loss, valid_acc, outputs, labels = validate(
                model, valid_loader, criterion, DEVICE
            )
            
            # Update learning rate
            scheduler.step()
            
            # Save history
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_acc'].append(train_acc)
            history['valid_acc'].append(valid_acc)
            
            # Print statistics
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
            print(f'LR: {scheduler.get_last_lr()[0]:.6f}')

            with open(f"/kaggle/working/Epoch {epoch+1}/{num_epochs}.txt", "w") as file:
                file.write(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                file.write(f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
                file.write(f'LR: {scheduler.get_last_lr()[0]:.6f}')
            
            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f'best_chest_xray_model_on_epoch_{epoch+1}.pth')
                print('Model saved!')
        
    return model, history
# Function to visualize training progress
def visualize_training(history, lw=3):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='training', marker='*', linewidth=lw)
    plt.plot(history['valid_acc'], label='validation', marker='o', linewidth=lw)
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='x-large')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='training', marker='*', linewidth=lw)
    plt.plot(history['valid_loss'], label='validation', marker='o', linewidth=lw)
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize='x-large')
    plt.show()

# Get label string from binary vector
def get_label(y, labels_list):
    ret_labels = []
    for i, value in enumerate(y):
        if value:
            ret_labels.append(labels_list[i])
    if not ret_labels:
        return 'No Label'
    else:
        return '|'.join(ret_labels)


if __name__ == "__main__":

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # Global configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 10

    print("DATA LOADING")

    train_df_main = pd.read_csv('/kaggle/input/chestxray8-dataframe/train_df.csv')
    train_df_main.drop(['No Finding'], axis=1, inplace=True)
    labels = train_df_main.columns[2:-1].tolist()

    # Split data
    train_df, discard = train_test_split(train_df_main, test_size=0.7, random_state=1993)
    train_and_valid_set, test_set = train_test_split(train_df, test_size=0.2, random_state=1993)
    train_set, valid_set = train_test_split(train_and_valid_set, test_size=0.2, random_state=1993)

    # Create data loaders
    train_labels = train_df.iloc[:, 2:-1].values
    
    # Create datasets
    train_dataset = ChestXrayDataSet(
        dataframe=train_set, 
        mode='train'
    )
    
    valid_dataset = ChestXrayDataSet(
        dataframe=valid_set,
        mode="val"
    )
    
    test_dataset = ChestXrayDataSet(
        dataframe=test_set,
        mode="test"
    )
    print("Train dataset size: ", len(train_dataset))
    print(train_dataset[0])
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=32,
        pin_memory=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=32,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=32,
        pin_memory=True
    )
   
    print("DATA LOADING COMPLETE")
    # Open a new file in write mode
    with open("/kaggle/working/DATA_LOADING_COMPLETE.txt", "w") as file:
        file.write(f"Train dataset size: {len(train_dataset)}\n")
        file.write(f"Test dataset size: {len(test_dataset)}\n")
        file.write(f"Valid dataset size: {len(valid_dataset)}\n")


    print("LOADING MODEL")
    # Create model
    # pre-trained model path
    model_path = "/kaggle/input/weight-ppad/weight/best_64_0.0001_original_35000_0.864.pt"

    # checkpoint path
    learner_weight_path = "/kaggle/input/weight-ppad/weight/PPAD_CheXpert_auc_0.896288_acc_0.842_f1_0.8301075268817204_ap_0.9075604434206554.pt"

    # Load Approach2_Baseline model
    model = Approach2_Baseline(model_path, learner_weight_path, DEVICE)
    model.to(DEVICE)
    model = model.to(torch.float32)
    # Process all input tensors
    def process_batch(batch):
      return {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    
    # Summary
    #summary(model, (3, 224, 224))
    # Verify parameters to train
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    learnable_params = sum(p.numel() for p in trainable_params)
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    print(type(model))  # Should be <class '__main__.Approach2_Baseline'>
    print(type(train_loader))  # Should be <class 'torch.utils.data.DataLoader'> 

    print("MODEL LOADING COMPLETE")
    with open("/kaggle/working/MODEL_LOADING_COMPLETE.txt", "w") as file:
        file.write(f"Number of trainable parameters: {learnable_params}")

    print("TRAINING AND EVALUATION")
    # Train model
    model, history = train_model(
        model, 
        train_loader, 
        valid_loader, 
        epochs, 
        DEVICE
    )

    # Visualize training
    visualize_training(history)

    # Evaluate on test set
    _, _, test_outputs, test_labels = validate(
        model, test_loader, FocalLoss(alpha=0.25, gamma=2.0, reduction='mean'), DEVICE
    )

    # Plot ROC curves
    auc_roc_vals = plot_roc_curves(labels, test_outputs, test_labels, when='after training')

    # Save model
    torch.save(model.state_dict(), '/kaggle/working/chest_xray_model_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'labels': labels,
    }, '/kaggle/working/chest_xray_model_complete.pth')

    # Save training history
    pd.DataFrame.from_dict(history).to_csv('/kaggle/working/Approach_2_training_history.csv', index=False)

    print("TRAINING AND EVALUATION COMPLETE")