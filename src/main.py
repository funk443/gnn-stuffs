# Example GCN/GNN code for vibration analysis.
# Copyright (C) 2024  CToID <funk443@yahoo.com.tw>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import csv

import torch
import pywt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GCNConv



class GCNModel(torch.nn.Module):
    def __init__(self,
                 data,
                 optimizer = None,
                 criterion = None,
                 activation = None,
                 dropout = None,
                 hidden1 = 8,
                 hidden2 = 4):
        super().__init__()
        self.data = data
        self.conv1 = GCNConv(data.x.shape[1], hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.conv3 = GCNConv(hidden2, data.class_count)

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters())

        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()

        if activation is None:
            activation = torch.nn.ReLU()

        if dropout is None:
            dropout = torch.nn.Dropout()

        self.optimizer = optimizer
        self.criterion = criterion
        self.activation = activation
        self.dropout = dropout

    def forward(self, x = None):
        if x is None:
            x = self.data.x

        x = self.conv1(
            x,
            self.data.edge_index,
            self.data.edge_attr)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(
            x,
            self.data.edge_index,
            self.data.edge_attr)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv3(
            x,
            self.data.edge_index,
            self.data.edge_attr)
        return x

    def go_training(self, epoch):
        def train_once():
            self.optimizer.zero_grad()
            out = self()
            loss = self.criterion(
                out[self.data.train_mask],
                self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()

            return loss

        self.train()
        for i in range(epoch):
            loss = train_once()
            print(f"Epoch {i + 1}: loss = {loss:.5f}")

    def go_testing(self):
        self.eval()

        out = self()
        pred = out.argmax(dim = 1)[self.data.test_mask]
        labels = self.data.y[self.data.test_mask]
        correct = pred == labels

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i, val in enumerate(correct):
            true_label = labels[i]
            if val and true_label == 0:
                tn += 1
            elif val and true_label == 1:
                tp += 1
            elif true_label == 0:
                fp += 1
            else:
                fn += 1

        accuracy = (tp+tn) / len(correct)
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1_score = 2*precision*recall / (recall+precision)

        return {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1_score}



def read_signals_from_tsv(
        path,
        delimiter = "\t",
        transformer = lambda x: x):
    with open(path, "r", encoding = "utf-8") as f:
        result = []
        reader = csv.reader(
            f,
            delimiter = delimiter)
        for row in reader:
            result.append([transformer(x) for x in row])

    return result


def split_signals(signals, n = 200):
    result = []
    temp = []

    for i, signal in enumerate(signals):
        if i > 0 and i % n == 0:
            result.append(temp)
            temp = []

        temp.append(signal)

    if temp:
        result.append(temp)

    return result


def calc_feature(signal):
    def calc_helper(signal):
        signal = np.array([
            sum(xyz) / 3
            for xyz in signal])
        std_deviation = np.std(signal)
        peak = max(signal)
        skewness = sp.stats.skew(signal)
        kurtosis = sp.stats.kurtosis(signal)
        rms = np.sqrt(np.sum(signal**2) / len(signal))
        crest_factor = peak / rms
        shape_factor = rms / np.sum(np.abs(signal))
        impulse_factor = (
            peak / (np.sum(np.abs(signal))/len(signal)))

        wp = pywt.WaveletPacket(
            data = signal,
            wavelet = "db20",
            maxlevel = 3)
        datas = [np.sum(np.abs(node.data))
                     for node in wp.get_level(3)]
        total = sum(datas)
        wpd = [x / total for x in datas]

        return np.concatenate((
            [
                std_deviation,
                peak,
                skewness,
                kurtosis,
                rms,
                crest_factor,
                shape_factor,
                impulse_factor],
            wpd))

    return np.array(
        [calc_helper(group) for group in signal])


def calc_adj_matrix(features, k = 5):
    def calc_distance(a, b):
        return np.sum(np.abs(a - b))

    distances = np.array(
        [[calc_distance(a, b) for b in features]
        for a in features])

    result = np.zeros(distances.shape)
    for i, dists in enumerate(distances):
        result[i][i] = 1

        sorted_indexes = np.argsort(dists)
        assert sorted_indexes[0] == i

        needed_indexes = sorted_indexes[1 : k + 1]
        needed_dists = dists[needed_indexes]
        needed_dists /= np.sum(needed_dists)

        result[i][needed_indexes] = needed_dists

    return result


def calc_edge_index_and_attr(
        adj_matrix,
        undirected = True):
    edge_index = [[], []]
    edge_attr = []

    for i, weights in enumerate(adj_matrix):
        for j, weight in enumerate(weights):
            if j == i or weight <= 0:
                continue

            edge_index[0].append(i)
            edge_index[1].append(j)
            edge_attr.append([weight])

    edge_index = torch.tensor(edge_index, dtype = int)
    edge_attr = torch.tensor(edge_attr, dtype = torch.double)

    if undirected:
        return to_undirected(edge_index, edge_attr)
    else:
        return edge_index, edge_attr



def plot_raw_signals(
        signals,
        fig = None,
        file_path = None):
    if fig is None:
        fig = plt.figure(layout = "tight")

    fig.set_title("Raw signal")
    xyzs = [
        [p[0] for p in signals],
        [p[1] for p in signals],
        [p[2] for p in signals]]
    subplots = [
        fig.add_subplot(2, 2, i + 1)
        for i in range(3)]

    stuffs = zip(("x", "y", "z"), xyzs, subplots)
    for title, data, subplot in stuffs:
        subplot.set_title(title)
        subplot.plot(data)

    if file_path is not None:
        fig.savefig(file_path)
    else:
        fig.show()


def plot_features(
        features,
        fig = None,
        file_path = None):
    if fig is None:
        fig = plt.figure(layout = "tight")

    ax = fig.add_subplot()
    ax.set_title("Features")
    cax = ax.imshow(
        features,
        aspect = "auto",
        interpolation = "none")
    cbar = fig.colorbar(cax)

    if file_path is not None:
        fig.savefig(file_path)
    else:
        fig.show()


def plot_adj_matrix(
        adj_matrix,
        fig = None,
        file_path = None):
    if fig is None:
        fig = plt.figure(layout = "tight")

    ax = fig.add_subplot()
    ax.set_title("Adjacent Matrix")
    cax = ax.imshow(
        adj_matrix,
        aspect = "auto",
        interpolation = "none")
    cbar = fig.colorbar(cax)

    if file_path is not None:
        fig.savefig(file_path)
    else:
        fig.show()


def plot_confusion_matrix(
        model_result,
        fig = None,
        file_path = None):
    if fig is None:
        fig = plt.figure(layout = "tight")

    ax = fig.add_subplot()
    ax.set_title("Confusion matrix")
    confusion_matrix = [
        [model_result["tp"], model_result["fn"]],
        [model_result["fp"], model_result["tn"]]]
    cax = ax.imshow(confusion_matrix)
    cbar = fig.colorbar(cax)

    ticks = ["Positive", "Negative"]
    ax.set_xticks(np.arange(len(ticks)), labels = ticks)
    ax.set_yticks(np.arange(len(ticks)), labels = ticks)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    if file_path is not None:
        fig.savefig(file_path)
    else:
        fig.show()



torch.set_default_dtype(torch.double)

signal_file = "../signals/signal-new.tsv"
signal_raw = read_signals_from_tsv(
    signal_file,
    transformer = float)
signal_splited = split_signals(signal_raw)
features = calc_feature(signal_splited)

status_labels = {
    "standby": 0,
    "grinding": 1}
feature_labels = []
for i, _ in enumerate(features):
    if i >= 0 and i < 15:
        label = status_labels['standby']
    elif i >= 15 and i < 200:
        label = status_labels['grinding']
    elif i >= 200 and i < 300:
        label = status_labels['standby']
    elif i >= 300 and i < 360:
        label = status_labels['grinding']
    elif i >= 360 and i < 450:
        label = status_labels['standby']
    elif i >= 450 and i < 500:
        label = status_labels['grinding']
    elif i >= 500 and i < 600:
        label = status_labels['standby']
    elif i >= 600 and i < 840:
        label = status_labels['grinding']
    else:
        label = status_labels['standby']

    feature_labels.append(label)
feature_labels = torch.tensor(feature_labels, dtype = int)

adj_matrix = calc_adj_matrix(features)

# plot_raw_signals(
#     signal_raw,
#     fig = plt.figure(dpi = 300, layout = "tight"),
#     file_path = "../plots/raw_signals")
# plot_features(
#     features,
#     fig = plt.figure(dpi = 300, layout = "tight"),
#     file_path = "../plots/features")
# plot_adj_matrix(
#     adj_matrix,
#     fig = plt.figure(dpi = 300, layout = "tight"),
#     file_path = "../plots/adj_matrix")

features = torch.tensor(features, dtype = torch.double)
edge_index, edge_attr = calc_edge_index_and_attr(adj_matrix)

train_mask = [False for _ in features]
test_mask = [False for _ in features]
for i, _ in enumerate(features):
    if i >= 0 and i < 10:
        train_mask[i] = True
    elif i >= 10 and i < 15:
        test_mask[i] = True
    elif i >= 15 and i < 150:
        train_mask[i] = True
    elif i >= 150 and i < 200:
        test_mask[i] = True
    elif i >= 200 and i < 275:
        train_mask[i] = True
    elif i >= 275 and i < 300:
        test_mask[i] = True
    elif i >= 300 and i < 345:
        train_mask[i] = True
    elif i >= 345 and i < 360:
        test_mask[i] = True
    elif i >= 360 and i < 428:
        train_mask[i] = True
    elif i >= 427 and i < 450:
        test_mask[i] = True
    elif i >= 450 and i < 488:
        train_mask[i] = True
    elif i >= 487 and i < 500:
        test_mask[i] = True
    elif i >= 500 and i < 575:
        train_mask[i] = True
    elif i >= 575 and i < 600:
        test_mask[i] = True
    elif i >= 600 and i < 780:
        train_mask[i] = True
    elif i >= 780 and i < 840:
        test_mask[i] = True
    elif i >= 840 and i < 966:
        train_mask[i] = True
    else:
        test_mask[i] = True

train_mask = torch.tensor(train_mask)
test_mask = torch.tensor(test_mask)

data = Data(
    x = features,
    y = feature_labels,
    edge_index = edge_index,
    edge_attr = edge_attr,
    train_mask = train_mask,
    test_mask = test_mask,
    class_count = 2)

epoch = 500
model = GCNModel(data)
model.go_training(epoch)
model_result = model.go_testing()

# plot_confusion_matrix(
#     model_result,
#     fig = plt.figure(dpi = 300, layout = "tight"),
#     file_path = "../plots/confusion_matrix")
