import torch
from torchvision import transforms
from pseudo_zhang import MaskZhangTrain
from PIL import Image

if __name__ == "__main__":  
    test_transform = transforms.Compose([
            transforms.Resize(size=224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.39799, 0.39799, 0.39799], std=[0.32721349, 0.32721349, 0.32721349])
        ])
    test_dataset = MaskZhangTrain("3_images_to_test", train=False, transforms=test_transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=32)

    print(len(testloader))
    for i, (img, labels, masks, position_names) in enumerate(testloader):
        print(i)
        print(img.shape)
        print(labels.shape)
        print(masks.shape)
        print(position_names.shape)