# Authors:
# Ziqiao Gao 2157371827
# Rui Hu 2350308289
# He Chang 5670527576
# Fanlin Qin 5317973858
#
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
origin_file_path = '../data/weatherAUS_APP_NORM.csv'


class WeatherAusDataset(Dataset):
    """WeatherAus Dataset"""

    def __init__(self, csv_file, transform=None):
        """
            Args:
                csv_file (string): Path to the csv file with annotations.
                transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.weatherDataset = pd.read_csv(csv_file, sep=',', header=0, dtype=np.float)
        self.transform = transform

    def __len__(self):
        return len(self.weatherDataset)

    def attributes_size(self):
        return self.weatherDataset.shape[1] - 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        weatherData = torch.tensor(np.array(self.weatherDataset.iloc[idx, :-1]), dtype=torch.float32)
        label = torch.tensor(np.array(self.weatherDataset.iloc[idx, -1]), dtype=torch.int64)
        sample = {"sample": weatherData, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hid_layer, output_dim):
        super(FeedForwardNN, self).__init__()
        # calculate hidden layer dimension
        descent_rate = 0.75
        self.hidden_layer = hid_layer
        self.hidden_dim_list = [np.int(np.floor(hidden_dim * pow(descent_rate, times))) for times in range(0, self.hidden_layer)]
        print(self.hidden_dim_list)
        # Linear function
        self.fc1 = nn.Linear(input_dim, self.hidden_dim_list[0])

        # Linear function
        self.fc2 = [nn.Linear(self.hidden_dim_list[layer], self.hidden_dim_list[layer + 1]) for layer in range(self.hidden_layer - 1)]

        # Linear function (readout)
        self.fc3 = nn.Linear(self.hidden_dim_list[-1], output_dim)

        # Non-linearity
        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        self.ReLU = nn.ReLU()
        self.CELU = nn.CELU()

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.ReLU(out)

        # Hidden Layer
        for layer in range(self.hidden_layer - 1):
            # Linear function
            out = self.fc2[layer](out)
            # Non-linearity
            out = self.Tanh(out)

        # Linear function (readout)  # LINEAR
        out = self.fc3(out)
        out = self.CELU(out)
        return out


def create_datasets(file_path=origin_file_path):
    weatherAus_Dataset = WeatherAusDataset(file_path)
    attribute_size = weatherAus_Dataset.attributes_size()
    print("number of attributes", attribute_size)
    """
    for i in range(len(weatherAus_Dataset)):
        sample = weatherAus_Dataset[i]
        print(i, sample.shape)
        print(sample)
        break
    """
    whole_size = len(weatherAus_Dataset)
    train_size = np.int(np.floor(0.8 * whole_size))
    test_size = whole_size - train_size
    print(train_size, test_size)
    train_dataset, test_dataset = random_split(weatherAus_Dataset, [train_size, test_size])
    return train_dataset, test_dataset, attribute_size


def train_model(model, num_epochs, batch_size, train_ds, test_ds, device=torch.device('cpu')):
    # set the device to cpu
    model = model.to(device)

    # Instantiate Loss Class
    criterion = nn.CrossEntropyLoss()
    # Instantiate Optimizer Class
    learning_rate = 0.35
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.95)
    # max_iteration: num_epochs * num_batches
    for epoch in range(num_epochs):
        for index, sample_batched in enumerate(train_ds):
            #print(index)
            #print(sample_batched)
            #print(sample_batched["sample"])
            #print(sample_batched["label"])
            #print(sample_batched['sample'].dtype, sample_batched['label'].dtype)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/log_prob
            outputs = model(sample_batched['sample'])
            #print(outputs.dtype)
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, sample_batched['label'])
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
            # Updating
            scheduler.step()
        # Calculate Accuracy for each epoch
        correct = 0
        total = 0
        Y_pred = torch.tensor((), dtype=torch.long)
        Y_true = torch.tensor((), dtype=torch.long)
        # Iterate through test dataset
        for i, sample_b in enumerate(test_ds):
            # Forward pass to get output/log_prob
            opts = model(sample_b['sample'])
            # Get predictions from the maximum value
            # # output.data get the tensor
            _, predicted = torch.max(opts.data, 1)
            # Total number of labels
            total += sample_b['label'].size(0)
            # Total correct predictions
            correct += (predicted == sample_b['label']).sum()
            # concat label
            Y_pred = torch.cat((Y_pred, predicted), 0)
            Y_true = torch.cat((Y_true, sample_b['label']), 0)
        # calculate the accuracy
        accuracy = 100 * correct / total
        # F1 Measure
        pred_series = pd.Series(Y_pred)
        true_series = pd.Series(Y_true)
        true_series_vlaue_counts = true_series.value_counts()
        null_accuracy = (true_series_vlaue_counts[0] / (true_series_vlaue_counts[0] + true_series_vlaue_counts[1]))
        print('Null Acuuracy: ', null_accuracy)
        conf_matrx = confusion_matrix(true_series, pred_series)
        print('Confusion matrix\n\n', conf_matrx)
        TP = conf_matrx[0, 0]
        TN = conf_matrx[1, 1]
        FP = conf_matrx[0, 1]
        FN = conf_matrx[1, 0]
        print('\nTrue Positives(TP) = ', TP)

        print('\nTrue Negatives(TN) = ', TN)

        print('\nFalse Positives(FP) = ', FP)

        print('\nFalse Negatives(FN) = ', FN)

        precision = TP / float(TP + FP)

        print('Precision : {0:0.4f}'.format(precision))
        recall = TP / float(TP + FN)

        print('Recall or Sensitivity : {0:0.4f}'.format(recall))
        f1 = f1_score(true_series, pred_series)

        print('f1 : {0:0.4f}'.format(f1))
        # Print Loss
        print('Epoch: {}. Test Set Accuracy: {}'.format(epoch + 1, accuracy))
    return


if __name__ == '__main__':
    batch_size = 1000

    train_ds, test_ds, attributes_num = create_datasets()
    """
    shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
    drop_last (bool, optional) – set to True to drop the last incomplete batch, 
        if the dataset size is not divisible by the batch size. 
        If False and the size of dataset is not divisible by the batch size, 
            then the last batch will be smaller. (default: False)
    """
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    # number of epoch
    num_epochs = 100
    # model layer structure
    input_dimension = attributes_num
    hidden_dimension = 200
    hidden_layer = 2
    output_dimension = 2
    # Instantiating FNN model class
    FNN_model = FeedForwardNN(input_dim=input_dimension, hidden_dim=hidden_dimension, hid_layer=hidden_layer, output_dim=output_dimension)
    train_model(FNN_model, num_epochs=num_epochs, batch_size=batch_size, train_ds=train_dataloader, test_ds=test_dataloader)

