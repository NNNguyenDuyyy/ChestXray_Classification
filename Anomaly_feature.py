import sys
sys.path.append('..')
import torch
from PIL import Image

from torchvision import transforms
from ppad import PPAD
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
        self.model = PPAD(STATUS,
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

    def forward(self, testloader):
        all_img_features = []
        all_text_features = []
        #all_labels = []
        with torch.no_grad():
            for i, (img, labels, masks, position_names) in enumerate(testloader):
        
                image = img.to(device)
                labels = labels.to(dtype=image.dtype, device=image.device)

                img_features = []
                text_features = []
                for mask, position_name in zip(masks, position_names):
            
                    mask = mask.to(dtype=image.dtype, device=image.device)

                    #logits_per_image = self.model(image, mask, position_name)
                    img_feature, text_feature = self.model(image, mask, position_name)

                    # new_logits_per_image = torch.zeros((logits_per_image.shape[0],2))
                    # logits_per_image = logits_per_image.cpu()

                    # new_logits_per_image[:,0] = logits_per_image[:,0]
                    # new_logits_per_image[:,1] = logits_per_image[:,1]
                    # probs = new_logits_per_image.softmax(dim=1)
            
                    # abnormal_probs = probs[:,1]

                    img_features.append(img_feature)
                    text_features.append(text_feature)
                #temp_results = torch.stack(temp_results, dim=0)
                all_img_features.append(torch.stack(img_features, dim=0))
                all_text_features.append(torch.stack(text_features, dim=0))
                #max_results = temp_results.max(dim=0)[0]
                #mean_results = temp_results.mean(dim=0)

                # for max_result, mean_result in zip(max_results, mean_results):
                #     if max_result > 0.8:
                #         all_results.append(max_result)
                #     else:
                #         all_results.append(mean_result)


                # labels = labels.argmax(dim=-1).cpu().numpy()
                # all_labels.append(labels)
        print(all_img_features[0].shape)
        print(all_text_features[0].shape)
        return all_img_features, all_text_features


# all_results = np.array(all_results)
# all_labels = np.concatenate(all_labels)
# ap = metrics.average_precision_score(all_labels, all_results)
# auc = metrics.roc_auc_score(all_labels, all_results)
# f1 = metrics.f1_score(all_labels, all_results>0.5)
# acc = metrics.accuracy_score(all_labels, all_results>0.5)

# print("acc:", acc, "auc:", auc,  "f1:", f1, "ap:", ap )
if __name__ == "__main__":
    model_path = "/content/ChestXray_Classification/weight/best_64_0.0001_original_35000_0.864.pt"
    learner_weight_path = "/content/ChestXray_Classification/weight/PPAD_CheXpert_auc_0.896288_acc_0.842_f1_0.8301075268817204_ap_0.9075604434206554.pt"
    test_transform = transforms.Compose([
            transforms.Resize(size=224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.39799, 0.39799, 0.39799], std=[0.32721349, 0.32721349, 0.32721349])
        ])
    test_dataset = MaskZhangTrain("/content/ChestXray_Classification/our_test_256_pa", train=False, transforms=test_transform)
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










