import torch
import sys
sys.path.append('..')
import torch.nn as nn
from CrossAttn_Classifier import MultiLabelClassifierWithAnomaly
from ViT_model import VitModel
from Anomaly_feature import AnomalyFeature
from torchvision.models import vit_b_16
from torchvision import transforms
from PIL import Image
from pseudo_zhang import MaskZhangTrain

class Approach1BranchBaseline(nn.Module):
    """
    Base line for Approach 2 from input to final output
    """

    def __init__(self, 
                base_model_extract_image, 
                STATUS, 
                backbone_name,
                pretrained_dir,
                learner_weight_path,
                feature_dim,  
                num_heads, 
                num_labels,
                fusion_type="concat"):
        """
        Args:
            base_model_extract_image: pretrained weight for extract features, ex: vit_b16
            num_heads: number of head attention
            num_classes: Number of dataset classes
        """
        super(Approach1BranchBaseline, self).__init__()
        self.extract_anomaly_feature = AnomalyFeature(
                                                    STATUS,
                                                    backbone_name, 
                                                    pretrained_dir,
                                                    learner_weight_path,
                                                    n_ctx=16, 
                                                    class_specify=False, 
                                                    class_token_position="end", 
                                                    pos_embedding=True,
                                                    return_tokens=False)
        self.cross_attt_classifier = MultiLabelClassifierWithAnomaly(feature_dim, num_heads, num_labels, fusion_type)

    def forward(self, testloader):
        img_features, text_features = self.extract_anomaly_feature(testloader)  # [bs, 5, 512],  [bs, 5, 2, 512]
        logits, attn_weights = self.cross_attt_classifier(img_features, text_features)   # [bs, num_classes]
        print(logits.shape)
        return logits
    
if __name__ == "__main__":
    base_model_extract_image = vit_b_16(pretrained=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STATUS = ['normal', 'pneumonia']
    batch_size = 4
    num_tokens = 5  # 5 image tokens per sample
    feature_dim = 512
    num_labels = 15  # Number of multi-label outputs
    num_heads = 2
    backbone_name='ViT-B/32'
    model_path = "/content/ChestXray_Classification/weight/best_64_0.0001_original_35000_0.864.pt"
    learner_weight_path = "/content/ChestXray_Classification/weight/PPAD_CheXpert_auc_0.896288_acc_0.842_f1_0.8301075268817204_ap_0.9075604434206554.pt"

    pipeline = Approach1BranchBaseline(
                                base_model_extract_image, 
                                STATUS, 
                                backbone_name,
                                model_path,
                                learner_weight_path,
                                feature_dim,
                                num_heads, 
                                num_labels)
    
    test_transform = transforms.Compose([
            transforms.Resize(size=224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.39799, 0.39799, 0.39799], std=[0.32721349, 0.32721349, 0.32721349])
        ])
    test_dataset = MaskZhangTrain("/content/ChestXray_Classification/our_test_256_pa", train=False, transforms=test_transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=32)

    output = pipeline(testloader)    # [bs, num_classes]

