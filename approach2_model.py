import torch
import sys
sys.path.append('..')
import torch.nn as nn
from classifier import Classifier
from CrossAttention import CrossAttention
from ViT_model import VitModel
from Anomaly_feature import AnomalyFeature

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

    def forward(self, testloader):
        input_feature = self.extract_input_feature(testloader)
        anomaly_feature = self.extract_anomaly_feature(testloader)

        attn_out = self.cross_attn(anomaly_feature, input_feature)

        output = self.classifier(attn_out)
        print(output.shape)
        return output
