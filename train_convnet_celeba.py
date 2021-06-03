import argparse
import os
import csv

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils import data
import time
import torch.nn as nn
from torchvision import models

from diagan.utils.settings import set_seed
from diagan.datasets.image_loader_with_attr import get_celeba_with_attr

def get_dataloader(dataset, batch_size=128):
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True)

    return dataloader

def validate(model, test_dataloader, criterion, device):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    for int, data in enumerate(test_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        output = model(data)
        loss = criterion(output, target)
        
        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == target).sum().item()
    
    val_loss = val_running_loss/len(test_dataloader.dataset)
    val_accuracy = 100. * val_running_correct/len(test_dataloader.dataset)
    
    return val_loss, val_accuracy

def fit(model, optimizer, train_dataloader, criterion, device):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in enumerate(train_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss/len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)
    
    return train_loss, train_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vgg16", type=str, help="network model")
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--attr', default='Bald', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True
    else:
        device = "cpu"

    # load dataset
    print('Load data')
    train_dataset = get_celeba_with_attr(attr=args.attr, split='train', size=64)
    valid_dataset = get_celeba_with_attr(attr=args.attr, split='valid', size=64)
    test_dataset = get_celeba_with_attr(attr=args.attr, split='test', size=64)

    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size)
    valid_loader = get_dataloader(valid_dataset, batch_size=args.batch_size)
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size)


    # load model
    print('Load model')
    if args.model == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.model == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif args.model == 'inception':
        model = models.inception_v3(pretrained=True)
    else:
        raise ValueError('model should be vgg16 or resnet18 or inception')

    # change the number of classes 
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 2, bias=True)

    # freeze convolution weights
    for param in model.features.parameters():
        param.requires_grad = False

    # optimizer
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
    # loss function
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    
    save_path = f'./convnet_celeba'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    csv_file = os.path.join(save_path, 'loss_acc.csv')
    if os.path.exists(csv_file):
        f = open(csv_file, 'a', newline='')
        wr = csv.writer(f)
    else:
        f = open(csv_file, 'w', newline='')
        wr = csv.writer(f)
        wr.writerow(['', 'Train Acc', 'Valid Acc', 'Test Acc', 'Train Loss', 'Valid Loss', 'Test Loss'])

    train_loss , train_accuracy = [], []
    val_loss , val_accuracy = [], []
    start = time.time()
    print('Start training')
    for epoch in range(args.num_epochs):
        print(f'Epoch: {epoch+1}')
        train_epoch_loss, train_epoch_accuracy = fit(model, optimizer, train_loader, criterion, device)
        print(f'Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}')
        val_epoch_loss, val_epoch_accuracy = validate(model, valid_loader, criterion, device)
        print(f'Valid Loss: {val_epoch_loss:.4f}, Valid Acc: {val_epoch_accuracy:.2f}')
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
    end = time.time()
    print((end-start)/60, 'minutes')


    test_loss, test_accuracy = validate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}')

    # save loss and accuracy
    wr.writerow([args.attr, train_epoch_loss, val_epoch_loss, test_loss, train_epoch_accuracy, val_epoch_accuracy, test_accuracy])
    f.close()

    # save model
    print('Save model')
    torch.save(model.state_dict(), os.path.join(save_path, f'{args.attr}.pth'))

if __name__ == '__main__':
    main()