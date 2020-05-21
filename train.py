import argparse

import torch
from torch import nn
from torch import optim
from torchvision import transforms, datasets, models
from collections import OrderedDict
import json

#Take inputs from user
parser = argparse.ArgumentParser()
parser.add_argument('data_directory', type=str, help='Train a new network on a data set')
parser.add_argument('--save_dir', type=str, help='Set directory to save checkpoint', default='./')
parser.add_argument('--category', type=str, help='Load categories', default='cat_to_name.json')
parser.add_argument('--arch', type=str, help='Choose architecture', default='vgg16')
parser.add_argument('--learning_rate', type=float, help='Set learning rate', default=0.0001)
parser.add_argument('--hidden_units', type=int, help='Set hidden units', default=510)
parser.add_argument('--epochs', type=int, help='Set epochs', default=8)
parser.add_argument('--dropout', type=float, help='Set dropout', default=0.1)
parser.add_argument('--gpu', type=str, help='Use GPU for training', default='gpu')
args = parser.parse_args()

#Start training! :)

# Load data (assuming folders are available for training, validation and testing)
data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transformations
data_transforms = {'train_transforms': transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(30),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
                   'valid_transforms': transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
                   'test_transforms': transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
                  }

# Load the datasets 
image_datasets = {'train_data': datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
                   'valid_data': datasets.ImageFolder(valid_dir, transform=data_transforms['valid_transforms']),
                   'test_data': datasets.ImageFolder(valid_dir, transform=data_transforms['test_transforms'])
                  }

# Define the dataloaders
dataloaders= {'trainloader': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
              'validloader': torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=32, shuffle=True),
              'testloader': torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=32, shuffle=True)
             }

# Load a model and create a classifier
def load_train_model(arch, category, hidden_units, dropout, gpu, learning_rate, epochs, save_dir):
    
    # Load category
    with open(category, 'r') as f:
        cat_to_name = json.load(f)    
    
    # Choose from vgg16 and densenet121 architecture
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        in_features = 25088
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
        in_features = 1024 
      
    # Output is number of types of images
    out_features = len(cat_to_name)
    
    #Freeze parameters 
    for param in model.parameters():
        param.requires_grad = False
    
    #Create classifier
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(in_features, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(dropout)),
                              ('fc2', nn.Linear(hidden_units, out_features)),
                              ('output', nn.LogSoftmax(dim=1))
                            ]))
    
    #Replace vgg16 classifier with self built classifier
    model.classifier = classifier    
    
    #Use gpu if given as parameter
    if gpu == "gpu":
        device = "cpu"
    elif gpu == "cpu":
        device = "cpu"
    model.to(device)
    
    #Criterion and optimizer
    #Update weights of parameters (use momentum)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    #Define loss (Negative log likelihood loss)
    criterion = nn.NLLLoss()
  
    #Train classifier layers using backpropogation, track loss and accuracy
    epoch = 3
    print_every = 5
    steps = 0

    for e in range(epoch):
        running_loss = 0
        for images, labels in iter(dataloaders['trainloader']):

            #Training mode- dropout on
            model.train()

            steps += 1

            #Shift to gpu
            images, labels = images.to(device), labels.to(device)

            #Reset optimizer to 0
            optimizer.zero_grad()

            #Forward and backward passes
            output = model.forward(images) 
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            #Track loss
            running_loss += loss.item()

            #Print loss every 50 steps
            if steps % print_every == 0:

                #Evaluation mode- dropout turned off
                model.eval()

                #Turn off gradients
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, dataloaders['validloader'], criterion, device)

                print("Epoch: {}/{}..".format(e+1,epoch),
                     "Training Loss: {:.3f}..".format(running_loss/print_every),
                     "Validation Loss: {:.3f}..".format(valid_loss/len(dataloaders['validloader'])),
                     "Validation Accuracy: {:.3f}%..".format(100*(accuracy/len(dataloaders['validloader']))))

                running_loss = 0

                #Training mode- dropout truned on
                model.train()
    model.to('cpu')
    save(arch, classifier, epoch, optimizer.state_dict(), model.state_dict(), image_datasets['train_data'].class_to_idx, save_dir)

#Function for validation    
def validation(model, loader, criterion, device):
    loss = 0
    accuracy = 0
    
    for images, labels in iter(loader):

        #Shift to gpu
        images, labels = images.to(device), labels.to(device)
        
        #Move to next image and calculate loss 
        output = model.forward(images)
        loss += criterion(output, labels).item()

        #Take exponent of log output
        ps = torch.exp(output)
        #Compare label with predicted class
        equality = (labels.data == ps.max(dim=1)[1])
        #Correct predictions/Total predictions
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return loss, accuracy

def save(arch, classifier, epoch, optimizer, state_dict, class_to_idx, save_dir):
    checkpoint = {'model': arch,
                  'classifier': classifier,
                  'epoch': epoch,  
                  'optimizer': optimizer,
                  'state_dict': state_dict,
                  'class_to_idx': class_to_idx
                 }
    torch.save(checkpoint, save_dir + 'checkpoint.pth')

load_train_model(args.arch, args.category, args.hidden_units, args.dropout, args.gpu, args.learning_rate, args.epochs, args.save_dir)
