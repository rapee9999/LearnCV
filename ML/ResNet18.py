import torch
import torchvision
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import typing

if __name__ == "__main__":
    
    # init resnet model
    model : torchvision.models.ResNet = torchvision.models.resnet18(pretrained=True)
    
    # get labels
    with open("data\\LOC_synset_mapping.txt", 'r') as f:
        labels = f.readlines()
    
    # load image
    org_img : Image.Image = Image.open("data\\n01443537_10213.JPEG  ")
    org_img : torch.Tensor = transforms.ToTensor()(org_img)
    img : torch.Tensor = transforms.Resize((64,64))(org_img)
    img = torch.Tensor.unsqueeze(img, 0)
    
    # inference
    model.zero_grad()
    model.eval()
    output : torch.Tensor = model(img)
    output = torch.softmax(output, 1)
    prob : float = torch.max(output).item()
    class_idx : int = torch.argmax(output).item()
    pred_class : str = labels[class_idx][labels[class_idx].index(' ')+1:].split(',')[0]

    # show
    show_img : np.array = org_img.numpy()
    show_img = show_img.transpose(1,2,0)
    show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
    show_img = cv2.putText(show_img, f"{pred_class} {prob*100:.0f}%", (5,20), 1, 1.5, (0,0,0), 1)
    cv2.imshow("Result", show_img)
    cv2.waitKey(0)
