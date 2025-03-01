import torch
import sys
sys.path.append('..')
import torch.nn as nn
from mutual_cross_attention import MutualCrossAttentionModel
from torchvision.models import vit_b_16
from torchvision import transforms
from PIL import Image
from pseudo_zhang import MaskZhangTrain
from build_model_swin import build_model
from ppad import AnomalyEncoder
import warnings
warnings.filterwarnings("ignore")

class Approach2_Baseline(nn.Module):
    """
    Base line for Approach 2 from input to final output
    """

    def __init__(self, model_path, learner_weight_path):
        """
        Args:
            base_model_extract_image: pretrained weight for extract features, ex: vit_b16
            dim_q: dimension of anomaly features
            dim_kv: dimension of features extracted from base_model
            num_heads: number of head attention
            num_classes: Number of dataset classes
        """
        super(Approach2_Baseline, self).__init__()
        self.model_path = model_path
        self.learner_weight_path = learner_weight_path
        self.extract_global_image_feature = build_model('swin')
        self.extract_anomaly_feature = self.load_extract_anomaly_feature(model_path, learner_weight_path)
        self.mutual_cross_attn = MutualCrossAttentionModel()
        # Define a linear projection layer
        self.proj = nn.Linear(768, 1024)

    def forward(self, image, mask, position_name):
        """
        image: [bs, 3, 224, 224]
        """
        global_image_feature = self.extract_global_image_feature(image)  # [bs, 768]
        global_image_feature = self.proj(global_image_feature) # [bs, 1024]
        anomaly_feature = self.extract_anomaly_feature(image, mask, position_name)  # [bs, 5, 1024]
        output = self.mutual_cross_attn(anomaly_feature, global_image_feature)  # [bs, num_classes]

        print(output.shape)
        return output
    
    def load_extract_anomaly_feature(model_path, learner_weight_path):

        model = AnomalyEncoder(
                                classnames = ['normal', 'pneumonia'],
                                class_token_position="end", 
                                pretrained_dir=model_path
                                )
        learner_weights = torch.load(learner_weight_path)
        for name, param in model.named_parameters():
            if name == "customclip.image_mask_learner.mask_embedding":
                model.state_dict()[name].copy_(learner_weights["image_mask_learner"]["mask_embedding"])
            elif name == "customclip.prompt_learner.ctx":
                model.state_dict()[name].copy_(learner_weights["prompt_learner_ctx"]["ctx"])

        model.to(device)
        return model
    
if __name__ == "__main__":
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # pre-trained model path
    model_path = "/content/ChestXray_Classification/weight/best_64_0.0001_original_35000_0.864.pt"

    # checkpoint path
    learner_weight_path = "/content/ChestXray_Classification/weight/PPAD_CheXpert_auc_0.896288_acc_0.842_f1_0.8301075268817204_ap_0.9075604434206554.pt"

    # Dataloader
    test_transform = transforms.Compose([
        transforms.Resize(size=224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.39799, 0.39799, 0.39799], std=[0.32721349, 0.32721349, 0.32721349])
    ])


    test_dataset = MaskZhangTrain("/content/ChestXray_Classification/3_images_to_test", train=False, transforms=test_transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=32)

    # Load Approach2_Baseline model
    model = Approach2_Baseline(model_path, learner_weight_path)
    model.eval()

    with torch.no_grad():
        for i, (img, labels, masks, position_names) in enumerate(testloader):
            
            image = img.to(device)
            #labels = labels.to(dtype=image.dtype, device=image.device)

            for mask, position_name in zip(masks, position_names):
                
                mask = mask.to(dtype=image.dtype, device=image.device)

                output = model(image, mask, position_name)

                print("Done First Loop in testloader")


