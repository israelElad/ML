import sys
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt


class ModelA(nn.Module):
    def __init__(self, image_size):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class ModelB(nn.Module):
    def __init__(self, image_size):
        super(ModelB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(self.dropout(x)))
        return F.log_softmax(self.fc2(self.dropout(x)), dim=1)


class ModelC(nn.Module):
    def __init__(self, image_size):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.batch1 = nn.BatchNorm1d(100)
        self.fc1 = nn.Linear(100, 50)
        self.batch2 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.batch1(self.fc0(x)))
        x = F.relu(self.batch2(self.fc1(x)))
        return F.log_softmax(self.fc2(x), dim=1)


class ModelD(nn.Module):
    def __init__(self, image_size):
        super(ModelD, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.log_softmax(self.fc5(x), dim=1)


class ModelE(nn.Module):
    def __init__(self, image_size):
        super(ModelE, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return F.log_softmax(self.fc5(x), dim=1)


def train(model, train_loader, optimizer, epochs):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        train_loss += F.nll_loss(output, labels, reduction='sum').item()  # sum up batch loss
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()

    # calculate average train loss and average accuracy for the current epoch
    train_loss /= len(train_loader.dataset)
    train_accuracy = (float(correct) / len(train_loader.dataset)) * 100
    return train_loss, train_accuracy


def validation(model, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # calculate average validation loss and average accuracy for the current epoch
    val_loss /= len(val_loader.dataset)
    val_accuracy = (float(correct) / len(val_loader.dataset)) * 100

    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     val_loss, correct, len(val_loader.dataset),
    #     100. * correct / len(val_loader.dataset)))
    return val_loss, val_accuracy


def train_validate_plot(model, train_loader, val_loader, optimizer, epochs):
    train_losses = {}
    val_losses = {}
    train_accuracies = {}
    val_accuracies = {}

    for epoch in range(1, epochs + 1):
        train_losses[epoch], train_accuracies[epoch] = train(model, train_loader, optimizer, epochs)
        val_losses[epoch], val_accuracies[epoch] = validation(model, val_loader)

    plot(train_losses, val_losses, "Loss", "Average loss per epoch for the validation and training set")
    plot(train_accuracies, val_accuracies, "Accuracy", "Average accuracy per epoch for the validation and training set")


def plot(train_loss_or_acc, val_loss_or_acc, plot_ylabel, plot_title):
    print("train:" + str(train_loss_or_acc))
    print("val:" + str(val_loss_or_acc))

    plt.title(plot_title)
    plt.xlabel("Epoch")
    plt.xlim(1, 10)
    plt.ylabel(plot_ylabel)

    plt.plot(list(train_loss_or_acc.keys()), list(train_loss_or_acc.values()), color='blue', linewidth=3,
             marker='.', markerfacecolor='blue', markersize=10)
    plt.plot(list(val_loss_or_acc.keys()), list(val_loss_or_acc.values()), color='cyan', linewidth=2, marker='.',
             markerfacecolor='cyan', markersize=10)

    plt.legend(("Train", "Validation"))

    plt.minorticks_on()
    plt.tick_params(color='black', direction='inout')
    plt.grid(which='both', color='grey', alpha=0.4)

    plt.show()


# classify the samples
def classify(test_x_file, chosen_model):
    class_predictions = []
    chosen_model.eval()
    for x in test_x_file:
        output = chosen_model(x)
        # the biggest result is x's predicted classification.
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        class_predictions.append(int(pred))
    return class_predictions


# writes the test predictions to a file
def write_test_predictions(predictions):
    test_y_file = open("test_y", "w")
    for prediction in predictions:
        test_y_file.write(np.str(prediction) + "\n")
    test_y_file.close()


def main():
    test_x_filename = "test_x"
    if len(sys.argv) >= 2:
        test_x_filename = sys.argv[1]

    # load fashion-mnist.
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0], [1])])
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=trans)

    # number of rows(number of samples)
    all_samples_num = len(train_dataset)
    train_num = int(all_samples_num * 0.8)
    val_num = all_samples_num - train_num
    # split to 80% training set and 20% validation(testing) set
    train_set, val_set = torch.utils.data.random_split(train_dataset, (train_num, val_num))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data', train=False, transform=trans),
                                              batch_size=64, shuffle=True)

    img_size = 28 * 28
    epochs = 10

    # print("modelA")
    # model_a = ModelA(img_size)
    # eta = 0.001
    # optimizer = optim.Adam(model_a.parameters(), lr=eta)
    # train_validate_plot(model_a, train_loader, val_loader, optimizer, epochs)

    # print("modelB")
    # model_b = ModelB(img_size)
    # eta = 0.001
    # optimizer = optim.Adam(model_b.parameters(), lr=eta)
    # train_validate_plot(model_b, train_loader, val_loader, optimizer, epochs)

    print("modelC")
    model_c = ModelC(img_size)
    eta = 0.001
    optimizer = optim.Adam(model_c.parameters(), lr=eta)
    train_validate_plot(model_c, train_loader, val_loader, optimizer, epochs)

    # print("modelD")
    # model_d = ModelD(img_size)
    # eta = 0.001
    # optimizer = optim.Adam(model_d.parameters(), lr=eta)
    # train_validate_plot(model_d, train_loader, val_loader, optimizer, epochs)

    # print("modelE")
    # model_e = ModelE(img_size)
    # eta = 0.01
    # optimizer = optim.Adam(model_e.parameters(), lr=eta)
    # train_validate_plot(model_e, train_loader, val_loader, optimizer, epochs)

    test_x_file = torch.from_numpy(np.loadtxt(test_x_filename) / 255).float()
    chosen_model = model_c
    predictions = classify(test_x_file, chosen_model)
    write_test_predictions(predictions)


main()
