import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from tqdm import tqdm
import random
import numpy as np

class AddNoise(object):
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def __call__(self, img):
        img_tensor = transforms.functional.to_tensor(img)
        noise = torch.rand_like(img_tensor) * self.noise_level
        noisy_img = img_tensor + noise
        return transforms.functional.to_pil_image(noisy_img)

# Applying Transforms to the Data
SIZE = 512
COLOR_DEVIATION = 0.01
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=SIZE),
        AddNoise(0.01),
        transforms.ColorJitter(brightness=(1.0-COLOR_DEVIATION,1.0+COLOR_DEVIATION),contrast=(1.0-COLOR_DEVIATION,1.0+COLOR_DEVIATION),saturation=(1.0-COLOR_DEVIATION,1.0+COLOR_DEVIATION),hue=(-1*COLOR_DEVIATION,COLOR_DEVIATION)),
        transforms.RandomRotation(degrees=0),
        transforms.RandomAffine(degrees=0),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'validation': transforms.Compose([
        transforms.Resize(size=SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}

# Set train and valid directory paths
root = 'eyepac-light-v2-512-jpg/'
train_directory = root + 'train'
test_directory = root + 'test'
validation_directory = root + 'validation'

# batch size
batch_size = 4

# Load Data from folders
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'validation': datasets.ImageFolder(root=validation_directory, transform=image_transforms['validation']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}

# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
validation_data_size = len(data['validation'])
test_data_size = len(data['test'])

# Create iterators for the Data loaded using DataLoader module
train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
validation_data = DataLoader(data['validation'], batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

# Print the train, validation and test set data sizes
train_data_size, validation_data_size, test_data_size

print("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Create a progress bar for the epoch
        epoch_progress = tqdm(total=len(train_data), desc=f'Epoch {epoch}/{num_epochs - 1}', position=0, leave=True)

        for phase in ['train', 'validation']:
            running_loss = 0.0
            running_corrects = 0

            # Set model mode
            if phase == 'train':
                model.train()
                dataloader = train_data
            else:
                model.eval()
                dataloader = validation_data

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar
                epoch_progress.set_postfix(phase=phase, loss=running_loss / len(data[phase]), acc=running_corrects.double() / len(data[phase]))
                epoch_progress.update()

            # Adjust learning rate
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(data[phase])
            epoch_acc = running_corrects.double() / len(data[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Update best accuracy
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        epoch_progress.close()

    # Test the model
    model.load_state_dict(best_model_wts)
    model.eval()
    test_corrects = 0

    for inputs, labels in test_data:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)

    test_acc = test_corrects.double() / test_data_size
    print('Test Accuracy: {:.4f}'.format(test_acc))

    return model


def run():
    torch.multiprocessing.freeze_support()
    print('loop')
    # load pretrained model
    model_ft = models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT, progress=True)
    
    model_ft.classifier[-1] = nn.Linear(1280, 2)

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 2 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

    # Calculate the number of parameters
    total_params = sum(p.numel() for p in model_ft.parameters())
    print("Total number of parameters: ", total_params)
    
    # Train the model
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=6)

    torch.save(model_ft.state_dict, 'model-eyepac.pth')
    

if __name__ == '__main__':
    run()