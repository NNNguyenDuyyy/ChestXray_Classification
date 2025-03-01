import sys
sys.path.append('..')
import torch
from PIL import Image

from torchvision import transforms
from ppad import AnomalyEncoder
from pseudo_zhang import MaskZhangTrain

import warnings
import torch.nn as nn

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"


class AnomalyFeature(nn.Module):
    """
    Return anomaly feature based on pretrained weight
    """

    def __init__(self, 
                STATUS,
                backbone_name, 
                pretrained_dir,
                learner_weight_path,
                n_ctx=16, 
                class_specify=False, 
                class_token_position="end", 
                pos_embedding=True,
                return_tokens=False):
        super(AnomalyFeature, self).__init__()

        # model_path = "./best_64_0.0001_original_35000_0.864.pt"
        self.model = AnomalyEncoder(STATUS,
                          backbone_name=backbone_name, 
                          n_ctx=n_ctx,
                          class_specify=class_specify,
                          class_token_position=class_token_position,
                          pretrained_dir=pretrained_dir,
                          pos_embedding=pos_embedding,
                          return_tokens=False
                         )


        # checkpoint path
        #learner_weight_path = './PPAD_CheXpert_auc_0.896288_acc_0.842_f1_0.8301075268817204_ap_0.9075604434206554.pt'
        self.learner_weights = torch.load(learner_weight_path)

        for name, param in self.model.named_parameters():
            if name == "customclip.image_mask_learner.mask_embedding":
                self.model.state_dict()[name].copy_(self.learner_weights["image_mask_learner"]["mask_embedding"])
            elif name == "customclip.prompt_learner.ctx":
                self.model.state_dict()[name].copy_(self.learner_weights["prompt_learner_ctx"]["ctx"])


        self.model.to(device)


        # self.test_transform = transforms.Compose([
        #     transforms.Resize(size=224, interpolation=Image.BICUBIC),
        #     transforms.CenterCrop(size=(224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.39799, 0.39799, 0.39799], std=[0.32721349, 0.32721349, 0.32721349])
        # ])


        # test_dataset = MaskZhangTrain('./chexpert/our_test_256_pa', train=False, transforms=self.test_transform)
        # testloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=2048, shuffle=False, num_workers=32)


        self.model.eval()

    def forward(self, img, labels, masks, position_names):
        
        image = img.to(device)
        labels = labels.to(dtype=image.dtype, device=image.device)

        anomaly_feature = []
        for mask, position_name in zip(masks, position_names):
    
            mask = mask.to(dtype=image.dtype, device=image.device)

            combined_feature = self.model(image, mask, position_name)

            anomaly_feature.append(combined_feature)
        anomaly_feature = torch.stack(anomaly_feature, dim=0)
        print(anomaly_feature.shape)
                
        return anomaly_feature
if __name__ == "__main__":

    model_path = "/content/ChestXray_Classification/weight/best_64_0.0001_original_35000_0.864.pt"
    learner_weight_path = "/content/ChestXray_Classification/weight/PPAD_CheXpert_auc_0.896288_acc_0.842_f1_0.8301075268817204_ap_0.9075604434206554.pt"
    test_transform = transforms.Compose([
            transforms.Resize(size=224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.39799, 0.39799, 0.39799], std=[0.32721349, 0.32721349, 0.32721349])
        ])
    test_dataset = MaskZhangTrain("/content/ChestXray_Classification/3_images_to_test", train=False, transforms=test_transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    STATUS = ['normal', 'pneumonia']
    backbone_name='ViT-B/32'
    anomaly_extract = AnomalyFeature(STATUS, backbone_name=backbone_name, pretrained_dir=model_path, learner_weight_path=learner_weight_path,n_ctx=16, 
                class_specify=False, 
                class_token_position="end", 
                pos_embedding=True,
                return_tokens=False)
    img_features, text_features = anomaly_extract(testloader)










