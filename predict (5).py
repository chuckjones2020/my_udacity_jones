import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
import os 
import random
from utils import load_checkpoint, load_cat_names
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath')
    parser.add_argument('checkpoint',action='store',default='checkpoint.pth')
    parser.add_argument('--top_k',dest='top_k',default='3')
    parser.add_argument('--filepath',dest='filepath',default='flowers/test/1/image_06743.jpg')
    parser.add_argument('--category_names',dest='category names',default='cat_to_nme.json')
    parser.add_argument('--gpu',action='store',default='gpu')
    return parser.parse_args()
def process_image(image):
    img_pil = Image.open(image)
    adjustments = transforms.Compose([tranforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
                                     ])
    image = adjustments(img_pil)
    return image
def predict (image_path, model, topk=3,gpu='gpu'):
    if gpu == True:
        model = model.cuda()
    else:
        model = model.cpu()
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze(0)
    img_torch = img_torch.float()
    if gpu == True:
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output = model.forward(img_torch)
    probability = F.softmax(output.data,dim=1)
    probos, indices = probability.topk(topk)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]
    idx_to_class = {v:k for k,v in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
    return probs, classes
def main():
    args = parse_args()
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_name = load_cat_names(args.category_names)
    img_path = args.filepath
    probs, classes = predict(img_path, model, int(args.top_k),gpu)
    lables = [cat_to_name[str(index)] for index in classes]
    probability = probs
    print ('File selected: ' + img_path)
    print (labels)
    print (probability)
    i=0
    while i < len(labels):
       print ("{} with a probability of {}".format(labels[i],probability[i]))
       i = i + 1
if __name__ == "__main__":
    main()
              
                         
                                    