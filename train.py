import os
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

import seaborn as sns
from matplotlib.ticker import MaxNLocator

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
num = 16  #images name number

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn

def plot_accuracy(train_acc_list, val_acc_list):
    # TODO plot training and testing accuracy curve
    # print("ACC: ", train_acc_list, ", ", val_acc_list)
    plt.plot(train_acc_list)
    plt.plot(val_acc_list)
    plt.title('Accuracy Curve')
    plt.ylabel("Accuracy")
    plt.xlabel('Epoch')
    
    plt.legend(['train','test'],loc='upper left')
    
    plt.savefig('result_image/acc_' + str(num) + '.png')
    plt.show()

def plot_f1_score(f1_score_list):
    # TODO plot testing f1 score curve
    # print("F1: ", f1_score_list)
    plt.plot(f1_score_list)
    plt.title('F1 Score Curve')
    plt.ylabel("F1 score")
    plt.xlabel('Epoch')
    
    plt.legend(['S1 score'],loc='upper left')
    
    plt.savefig('result_image/f1_' + str(num) + '.png')
    plt.show()

def plot_confusion_matrix(confusion_matrix):
    # TODO plot confusion matrix
    # print("Confusion Matrix: ", confusion_matrix)
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len([0, 1]))
    plt.xticks(tick_marks, [0, 1])
    plt.yticks(tick_marks, [0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.savefig('result_image/confusion_matrix_' + str(num) + '.png')
    plt.show()

def train(device, train_loader, model, criterion, optimizer):
    print("Start train!")
    best_acc = 0.0
    best_model_wts = None
    train_acc_list = []
    val_acc_list = []
    f1_score_list = []
    best_c_matrix = []

    for epoch in range(1, args.num_epochs+1):

        with torch.set_grad_enabled(True):
            avg_loss = 0.0
            train_acc = 0.0
            tp, tn, fp, fn = 0, 0, 0, 0     
            for _, data in enumerate(tqdm(train_loader)):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                outputs = torch.max(outputs, 1).indices
                sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
                tp += sub_tp
                tn += sub_tn
                fp += sub_fp
                fn += sub_fn          

            avg_loss /= len(train_loader.dataset)
            train_acc = (tp+tn) / (tp+tn+fp+fn) * 100
            print(f'Epoch: {epoch}')
            print(f'↳ Loss: {avg_loss}')
            print(f'↳ Training Acc.(%): {train_acc:.2f}%')

        val_acc, f1_score, c_matrix = test(test_loader, model)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        f1_score_list.append(f1_score)

        if val_acc > best_acc:
            best_acc = val_acc
            best_c_matrix = c_matrix
            torch.save(model.state_dict(), 'best_model_weights.pt')

    return train_acc_list, val_acc_list, f1_score_list, best_c_matrix

def test(test_loader, model):
    val_acc = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        c_matrix = [[int(tp), int(fn)],
                    [int(fp), int(tn)]]
        
        val_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score = (2*tp) / (2*tp+fp+fn)
        print (f'↳ Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')
        print (f'↳ Test Acc.(%): {val_acc:.2f}%')

    return val_acc, f1_score, c_matrix

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=2)

    # for training
    parser.add_argument('--num_epochs', type=int, required=False, default=35)
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)  #1e-5
    parser.add_argument('--wd', type=float, default=0.9)

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='chest_xray')

    # for data augmentation
    parser.add_argument('--degree', type=int, default=90)
    parser.add_argument('--resize', type=int, default=224)

    args = parser.parse_args()

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    args.degree = (-90, 90)
    mean = [0.5, 0.5, 0.5]
    std = [0.1, 0.1, 0.1]
    # set dataloader
    train_dataset = ImageFolder(root=os.path.join(args.dataset, 'train'),
                                transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                                transforms.RandomRotation(args.degree, resample=False),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.RandomVerticalFlip(),
                                                                #transforms.ColorJitter(brightness=(3, 5), saturation=(3, 5)),
                                                                transforms.ToTensor(),
                                                                ]))  #transforms.Normalize(mean, std)
    test_dataset = ImageFolder(root=os.path.join(args.dataset, 'test'),
                               transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                               transforms.ToTensor(),
                                                               ]))  #transforms.Normalize(mean, std)
    # print(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # define model
    model = models.resnext50_32x4d(pretrained=True)
    num_neurons = model.fc.in_features
    model.fc = nn.Linear(num_neurons, args.num_classes)
    #print("Train model: ", model)
    model = model.to(device)

    # define loss function, optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))
    criterion = criterion.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)  #Adam

    # training
    train_acc_list, val_acc_list, f1_score_list, best_c_matrix = train(device, train_loader, model, criterion, optimizer)

    # plot
    plot_accuracy(train_acc_list, val_acc_list)
    plot_f1_score(f1_score_list)
    plot_confusion_matrix(best_c_matrix)