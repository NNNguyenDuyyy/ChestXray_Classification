import torch
import sys
sys.path.append('..')
import torch.nn as nn
from classifier import Classifier
from CrossAttention import CrossAttention
from ViT_model import VitModel
from Anomaly_feature import AnomalyFeature
from torchvision.models import vit_b_16
from torchvision import transforms
from PIL import Image
from pseudo_zhang import MaskZhangTrain

class Approach2_Baseline(nn.Module):
    """
    Base line for Approach 2 from input to final output
    """

    def __init__(self, 
                base_model_extract_image, 
                STATUS, 
                backbone_name,
                pretrained_dir,
                learner_weight_path,
                dim_q, 
                dim_kv, 
                num_heads, 
                num_classes):
        """
        Args:
            base_model_extract_image: pretrained weight for extract features, ex: vit_b16
            dim_q: dimension of anomaly features
            dim_kv: dimension of features extracted from base_model
            num_heads: number of head attention
            num_classes: Number of dataset classes
        """
        super(Approach2_Baseline, self).__init__()
        self.extract_input_feature = VitModel(base_model_extract_image)
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
        self.cross_attn = CrossAttention(dim_q, dim_kv, num_heads)
        self.classifier = Classifier(dim_q, num_classes)

    def forward(self, testloader1, testloader2):
        input_feature = self.extract_input_feature(testloader1)  # [bs, 1, 768]
        anomaly_feature = self.extract_anomaly_feature(testloader2)  # [bs, 2, 512],  [bs, 1, 1024]

        attn_out = self.cross_attn(anomaly_feature, input_feature)   # [bs, 1, 1024]

        output = self.classifier(attn_out)   # [bs, num_classes]
        print(output.shape)
        return output
    
if __name__ == "__main__":
    base_model_extract_image = vit_b_16(pretrained=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STATUS = ['normal', 'pneumonia']
    dim_q = 1024
    dim_kv = 768
    num_heads = 2
    num_classes = 15
    backbone_name='ViT-B/32'
    model_path = "/content/ChestXray_Classification/weight/best_64_0.0001_original_35000_0.864.pt"
    learner_weight_path = "/content/ChestXray_Classification/weight/PPAD_CheXpert_auc_0.896288_acc_0.842_f1_0.8301075268817204_ap_0.9075604434206554.pt"

    pipeline = Approach2_Baseline(
                                base_model_extract_image, 
                                STATUS, 
                                backbone_name,
                                model_path,
                                learner_weight_path,
                                dim_q, 
                                dim_kv, 
                                num_heads, 
                                num_classes)
    
    testloader1 = torch.randn(1, 3, 224, 224).to(device)
    testloader2 = torch.randn(1, 3, 256, 256).to(device)
    output = pipeline(testloader1, testloader2)     # [bs, num_classes]

