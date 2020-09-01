from torch import nn

from gcommand_loader import GCommandLoader

import torch
import collections

input_height = 161
input_width = 101
num_of_classes = 30

classes = {0: "bed", 1: "bird", 2: "cat", 3: "dog", 4: "down", 5: "eight", 6: "five", 7: "four", 8: "go", 9: "happy",
           10: "house", 11: "left", 12: "marvin", 13: "nine", 14: "no", 15: "off", 16: "on", 17: "one", 18: "right",
           19: "seven", 20: "sheila", 21: "six", 22: "stop", 23: "three", 24: "tree", 25: "two", 26: "up", 27: "wow",
           28: "yes", 29: "zero"}


class convolutionNet(nn.Module):
    def __init__(self, num_of_input_layers, num_of_output_layers, filter_size, stride, padding, pooling_size,
                 pooling_stride):
        super(convolutionNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(num_of_input_layers[0], num_of_output_layers[0], kernel_size=filter_size,
                                              stride=stride, padding=padding),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_stride))
        self.layer2 = nn.Sequential(nn.Conv2d(num_of_input_layers[1], num_of_output_layers[1], kernel_size=filter_size,
                                              stride=stride, padding=padding),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_stride))
        self.drop_out = nn.Dropout()

        num_of_model_layers = len(num_of_input_layers)
        after_pooling_input_height = input_height
        after_pooling_input_width = input_width
        for i in range(num_of_model_layers):
            # after_pooling_input_height=40, after_pooling_input_width=25
            after_pooling_input_height, after_pooling_input_width = self.after_pooling_size_calculate(
                after_pooling_input_height, after_pooling_input_width, pooling_size)
        fc1_input_size = after_pooling_input_height * after_pooling_input_width * num_of_output_layers[
            num_of_model_layers - 1]  # fc1_input_size=40X25X9

        self.fc1 = nn.Linear(fc1_input_size, 1000)
        self.fc2 = nn.Linear(1000, num_of_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    # after pooling's size calculation
    def after_pooling_size_calculate(self, input_height, input_width, pooling_size):
        return int(input_height / pooling_size), int(input_width / pooling_size)


def train(model, train_loader, num_of_epochs, criterion, optimizer):
    model.train()
    loss_list = []
    acc_list = []
    correct = 0
    total = 0
    for epoch in range(num_of_epochs):
        for i, (audio, label) in enumerate(train_loader):
            output = model(audio)
            loss = criterion(output, label)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total = label.size(0)
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == label).sum().item()
            acc_list.append(correct / total)

        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_of_epochs, loss.item(),
                                                                      (correct / total) * 100))


def validate(model, val_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for audio, label in val_loader:
            output = model(audio)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        print('Test Accuracy of the model on the validation: {} %'.format((correct / total) * 100))


def test(model, test_loader):
    classifications = {}
    model.eval()
    with torch.no_grad():
        i = 0
        for audio, label in test_loader:

            output = model(audio)
            _, predicted = torch.max(output.data, 1)

            for p in predicted:
                path_of_example = test_loader.dataset.spects[i][0].split('/')
                name = path_of_example[len(path_of_example) - 1]
                name = name.split('\\')
                name = name[len(name) - 1]
                name = name.split('.')
                name = name[len(name) - 2]
                name = int(name)

                class_number = p.item()
                class_name = classes[class_number]
                classifications[name] = class_name
                i += 1
    return classifications


def train_and_test_model(model, learning_rate, train_loader, num_of_epochs, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model, train_loader, num_of_epochs, criterion, optimizer)
    validate(model, val_loader)


# write classifications to file
def write_classifications_to_file(classifications):
    output_file = open("test_y", "w")
    for name in classifications:
        output_file.write(str(name) + ".wav" + ',' + str(classifications[name]) + '\n')
    output_file.close()


def load_data(batch_size):
    train_dataset = GCommandLoader('./gcommands/train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

    val_dataset = GCommandLoader('./gcommands/valid')

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True,
        num_workers=20, pin_memory=True, sampler=None)

    test_dataset = GCommandLoader('./gcommands/test')

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=20, pin_memory=True, sampler=None)

    return train_loader, val_loader, test_loader


# padding calculation according to the formula we saw in class
def padding_calculate(input_size, output_size, filter_size, stride):
    padding = int((stride * output_size - stride - input_size + filter_size) / 2)
    return padding


def main():
    # hyper parameters
    num_of_epochs = 8
    learning_rate = 0.001
    batch_size = 100

    train_loader, val_loader, test_loader = load_data(batch_size)

    num_of_input_layers = [1, 5]
    num_of_output_layers = [5, 10]
    filter_size = 5
    stride = 1

    padding_height = padding_calculate(input_height, input_height, filter_size, stride)  # padding_height=2
    padding_width = padding_calculate(input_width, input_width, filter_size, stride)  # padding_width=2
    padding = (padding_height, padding_width)  # padding=2X2

    pooling_size = 2
    pooling_stride = 2

    model = convolutionNet(num_of_input_layers, num_of_output_layers, filter_size, stride, padding, pooling_size,
                           pooling_stride)

    train_and_test_model(model, learning_rate, train_loader, num_of_epochs, val_loader)
    classifications = test(model, test_loader)
    write_classifications_to_file(classifications)


if __name__ == '__main__':
    main()

# todo: params, report
