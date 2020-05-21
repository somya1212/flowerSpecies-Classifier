import argparse

import torch
from torch import nn
from torch import optim
from torchvision import transforms, datasets, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import json

#Take inputs from user
parser = argparse.ArgumentParser()
parser.add_argument('path_to_image', type=str, help='Set path to image', default='./flowers/test/1/image_06743.jpg')
parser.add_argument('checkpoint', type=str, help='Load checkpoint', default='./checkpoint.pth')
parser.add_argument('--top_k', type=int, help='Return top k most likely classes', default=5)
parser.add_argument('--category_names', type=str, help='Use a mapping of categories to real names', default='cat_to_name.json')
parser.add_argument('--gpu', type=str, help='Use GPU for inference', default='cpu')
args = parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['model'] == "vgg16":
        model = models.vgg16(pretrained=True)
    
    model.eval()
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict']) 
    model.class_to_idx = checkpoint['class_to_idx']
    epoch = checkpoint['epoch']
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #Perform transformations, convert to tensor and normalize
    transform = transforms.Compose([transforms.Resize(256), 
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    #Open image and apply transformation
    pil_image = Image.open(image)
    pil_image = transform(pil_image)
    
    #Convert to numpy array
    np_image = np.array(pil_image)    
            
    return np_image

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = load_checkpoint(model)
    model.eval()
    model.to(device)
    
    np_image = process_image(image_path) #numpy array returned
    torch_image = torch.from_numpy(np_image).to(device) #convert to tensor
    torch_image = torch_image.unsqueeze_(0)
    torch_image = torch_image.float() #returns float tensor of single dimension (1 column) 
    
    with torch.no_grad(): 
        output = model.forward(torch_image) 
        ps = torch.exp(output) 
    
    #taking top 5 probabilities and their indices 
    if topk is None:
        probs, indices = torch.topk(ps, 1)
    else:
        probs, indices = torch.topk(ps, topk)
    
    #invert class_to_idx
    inv_class_to_idx = {index: cls for cls, index in model.class_to_idx.items()}
    
    classes = []
    for index in indices.cpu().numpy()[0]: #iterating through indices
        classes.append(inv_class_to_idx[index])
    
    return probs.cpu().numpy()[0], classes

# Print the most likely image class and it's associated probability
# map with json

if args.gpu == "gpu":
    device = "cpu"
elif args.gpu == "cpu":
    device = "cpu"

probs, classes = predict(args.path_to_image, args.checkpoint, args.top_k, device)

if args.category_names is not None:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)    
        classes = [cat_to_name[c] for c in classes]
        
        
print("Most probable class:", classes[0])
print("Probability :", probs[0])

if args.top_k is not None:
    print("\nTop",args.top_k,"probable classes and their probabilities are")
    for index in range(len(classes)):
        print(classes[index],":",probs[index])