import sys
sys.path.append('..')
import torch
import numpy as np
import sklearn.metrics as metrics
from PIL import Image

from torchvision import transforms
from ppad import PPAD

import warnings
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import random
import datetime
from pseudo_zhang import MaskZhangTrain


warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

STATUS = ['normal', 'pneumonia']


# pre-trained model path
model_path = "/content/ChestXray_Classification/weight/best_64_0.0001_original_35000_0.864.pt"


model = PPAD(STATUS, backbone_name='ViT-B/32', n_ctx=16, class_specify=False, class_token_position="end", pretrained_dir=model_path, pos_embedding=True, return_tokens=False)

# checkpoint path
learner_weight_path = "/content/ChestXray_Classification/weight/PPAD_CheXpert_auc_0.896288_acc_0.842_f1_0.8301075268817204_ap_0.9075604434206554.pt"


learner_weights = torch.load(learner_weight_path)

for name, param in model.named_parameters():
    if name == "customclip.image_mask_learner.mask_embedding":
        model.state_dict()[name].copy_(learner_weights["image_mask_learner"]["mask_embedding"])
    elif name == "customclip.prompt_learner.ctx":
        model.state_dict()[name].copy_(learner_weights["prompt_learner_ctx"]["ctx"])


model.to(device)


test_transform = transforms.Compose([
    transforms.Resize(size=224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.39799, 0.39799, 0.39799], std=[0.32721349, 0.32721349, 0.32721349])
])


test_dataset = MaskZhangTrain("/content/ChestXray_Classification/3_images_to_test", train=False, transforms=test_transform)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=32)


model.eval()

all_results = []
all_labels = []
with torch.no_grad():
    for i, (img, labels, masks, position_names) in enumerate(testloader):
        
        image = img.to(device)
        labels = labels.to(dtype=image.dtype, device=image.device)

        temp_results = []
        for mask, position_name in zip(masks, position_names):
            
            mask = mask.to(dtype=image.dtype, device=image.device)

            logits_per_image = model(image, mask, position_name)

            new_logits_per_image = torch.zeros((logits_per_image.shape[0],2))
            logits_per_image = logits_per_image.cpu()

            new_logits_per_image[:,0] = logits_per_image[:,0]
            new_logits_per_image[:,1] = logits_per_image[:,1]
            probs = new_logits_per_image.softmax(dim=1)
            print(f"probs: {probs}")
            
            abnormal_probs = probs[:,1]
            print(f"abnormal_probs: {abnormal_probs}")

            temp_results.append(abnormal_probs)
            print(f"temp_results: {temp_results}")
        temp_results = torch.stack(temp_results, dim=0)
        max_results = temp_results.max(dim=0)[0]
        print(f"max_results: {max_results}")
        mean_results = temp_results.mean(dim=0)
        print(f"mean_results: {mean_results}")

        for max_result, mean_result in zip(max_results, mean_results):
            if max_result > 0.8:
                all_results.append(max_result)
            else:
                all_results.append(mean_result)


        labels = labels.argmax(dim=-1).cpu().numpy()
        all_labels.append(labels)



all_results = np.array(all_results)
print(f"all_results: {all_results}")
all_labels = np.concatenate(all_labels)
print(f"all_labels: {all_labels}")
ap = metrics.average_precision_score(all_labels, all_results)
auc = metrics.roc_auc_score(all_labels, all_results)
f1 = metrics.f1_score(all_labels, all_results>0.5)
acc = metrics.accuracy_score(all_labels, all_results>0.5)

print("acc:", acc, "auc:", auc,  "f1:", f1, "ap:", ap )

