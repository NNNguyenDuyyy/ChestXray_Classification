import torch
import sys
sys.path.append('..')
import torch.nn as nn
from CrossAttn_Classifier import MultiLabelClassifierWithAnomaly
from ppad import AnomalyEncoder
from torchvision.models import vit_b_16
from torchvision import transforms
from PIL import Image
from pseudo_zhang import MaskZhangTrain

class Approach1BranchBaseline(nn.Module):
    """
    Base line for Approach 2 from input to final output
    """

    def __init__(self, 
                STATUS, 
                backbone_name,
                model_path,
                learner_weight_path,
                feature_dim,  
                num_heads, 
                num_labels,
                dropout,
                fusion_type="concat"):
        """
        Args:
            base_model_extract_image: pretrained weight for extract features, ex: vit_b16
            num_heads: number of head attention
            num_classes: Number of dataset classes
        """
        super(Approach1BranchBaseline, self).__init__()
        self.extract_anomaly_feature = AnomalyEncoder(
                                                    STATUS,
                                                    backbone_name, 
                                                    n_ctx=16,
                                                    class_specify=False, 
                                                    class_token_position="end", 
                                                    pretrained_dir=model_path,
                                                    pos_embedding=True,
                                                    return_tokens=False)
        
        learner_weights = torch.load(learner_weight_path)

        for name, _ in self.extract_anomaly_feature.named_parameters():
            if name == "customclip.image_mask_learner.mask_embedding":
                self.extract_anomaly_feature.state_dict()[name].copy_(learner_weights["image_mask_learner"]["mask_embedding"])
            elif name == "customclip.prompt_learner.ctx":
                self.extract_anomaly_feature.state_dict()[name].copy_(learner_weights["prompt_learner_ctx"]["ctx"])


        self.extract_anomaly_feature.to(device).eval()
        self.cross_attt_classifier = MultiLabelClassifierWithAnomaly(feature_dim, num_heads, num_labels, dropout, fusion_type)

    def forward(self, image, mask, position_name):
        img_features, text_features = self.extract_anomaly_feature(image, mask, position_name)  # [bs, 512],  [bs, 2, 512]
        logits, _ = self.cross_attt_classifier(img_features, text_features)   # [bs, num_classes]
        print(logits.shape)
        return logits
    
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    STATUS = ['normal', 'pneumonia']
    feature_dim = 512
    num_labels = 15  # Number of multi-label outputs
    num_heads = 2
    dropout = 0.1
    fusion_type = "concat"
    backbone_name='ViT-B/32'
    model_path = "/content/ChestXray_Classification/weight/best_64_0.0001_original_35000_0.864.pt"
    learner_weight_path = "/content/ChestXray_Classification/weight/PPAD_CheXpert_auc_0.896288_acc_0.842_f1_0.8301075268817204_ap_0.9075604434206554.pt"

    pipeline = Approach1BranchBaseline(
                                STATUS, 
                                backbone_name,
                                model_path,
                                learner_weight_path,
                                feature_dim,
                                num_heads, 
                                num_labels,
                                dropout,
                                fusion_type)
    pipeline.to(device)
    
    test_transform = transforms.Compose([
            transforms.Resize(size=224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.39799, 0.39799, 0.39799], std=[0.32721349, 0.32721349, 0.32721349])
        ])
    test_dataset = MaskZhangTrain("/content/ChestXray_Classification/our_test_256_pa", train=False, transforms=test_transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=32)

    all_results = []
    with torch.no_grad():
        for i, (img, labels, masks, position_names) in enumerate(testloader):
        
            image = img.to(device)
            for mask, position_name in zip(masks, position_names):
            
                mask = mask.to(dtype=image.dtype, device=image.device)

                logits = pipeline(image, mask, position_name)
    
                print(f"logits: {logits.shape}")
            
            all_results.append(logits)
            print(f"all_results: {all_results.shape}")


