import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Focal_loss import FocalLoss
from model_1_branch import Approach1BranchBaseline
import random
from torchvision.models import vit_b_16 


class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, num_tokens=5, feature_dim=512, num_labels=10):
        super(DummyDataset, self).__init__()
        self.num_samples = num_samples
        self.num_tokens = num_tokens
        self.feature_dim = feature_dim
        self.num_labels = num_labels
        # Randomly generated image and anomaly features.
        self.img_features = torch.randn(num_samples, num_tokens, feature_dim)
        self.anomaly_features = torch.randn(num_samples, num_tokens, 2, feature_dim)
        # Each sample's label is provided as a list of class indices (e.g., [1,2,3]).
        self.labels_list = []
        for _ in range(num_samples):
            # Randomly select between 1 and num_labels classes.
            k = random.randint(1, num_labels)
            label_indices = random.sample(range(num_labels), k)
            self.labels_list.append(label_indices)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img_feat = self.img_features[idx]
        anomaly_feat = self.anomaly_features[idx]
        # Convert list of indices to a multi-hot vector.
        target = torch.zeros(self.num_labels)
        target[self.labels_list[idx]] = 1.0
        return img_feat, anomaly_feat, target


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for img_feats, anomaly_feats, targets in dataloader:
        img_feats = img_feats.to(device)
        anomaly_feats = anomaly_feats.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(img_feats, anomaly_feats)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * img_feats.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for img_feats, anomaly_feats, targets in dataloader:
            img_feats = img_feats.to(device)
            anomaly_feats = anomaly_feats.to(device)
            targets = targets.to(device)
            logits, _ = model(img_feats, anomaly_feats)
            loss = criterion(logits, targets)
            total_loss += loss.item() * img_feats.size(0)
    return total_loss / len(dataloader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    batch_size = 32
    learning_rate = 1e-4
    
    base_model_extract_image = vit_b_16(pretrained=True)
    STATUS = ['normal', 'pneumonia']
    backbone_name = 'ViT-B/32'
    pretrained_dir = "/content/ChestXray_Classification/weight/best_64_0.0001_original_35000_0.864.pt"
    learner_weight_path = "/content/ChestXray_Classification/weight/PPAD_CheXpert_auc_0.896288_acc_0.842_f1_0.8301075268817204_ap_0.9075604434206554.pt"
    # Create training and validation datasets
    train_dataset = DummyDataset(num_samples=1000)
    val_dataset = DummyDataset(num_samples=200)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize the model
    model = Approach1BranchBaseline(base_model_extract_image, 
                                    STATUS, 
                                    backbone_name, 
                                    pretrained_dir, 
                                    learner_weight_path,
                                    feature_dim=512, 
                                    num_heads=8, 
                                    num_labels=10, 
                                    fusion_type="concat")
    model.to(device)
    
    # Initialize Focal Loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2, reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model.")

if __name__ == "__main__":
    main()