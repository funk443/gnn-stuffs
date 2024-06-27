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
import pywt
import torch
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GraphConv,
)
from torch_geometric.utils import to_undirected

def tsv_file_to_list(
        path,
        delimiter = "\t",
        transform_fn = lambda x: x
):
    result = []

    with open(path) as tsv:
        for row in csv.reader(tsv, delimiter = delimiter):
            result.append([transform_fn(x) for x in row])

    return result

def split_signals(signals, n = 200):
    result = []
    temp = []

    for i, signal in enumerate(signals):
        if i != 0 and i % n == 0:
            result.append(temp)
            temp = []
        temp.append(signal)

    if len(temp) != 0:
        result.append(temp)

    return result

def signal_to_feature(signal):
    def helper_fn(sig):
        std_deviation = np.std(sig)
        peak = max(sig)
        skewness = sp.stats.skew(sig)
        kurtosis = sp.stats.kurtosis(sig)
        rms = np.sqrt(np.sum(sig ** 2) / len(sig))
        crest_factor = peak / rms
        shape_factor = rms / np.sum(np.abs(sig))
        impulse_factor = (
            peak / (np.sum(np.abs(sig)) / len(sig))
        )

        wp = pywt.WaveletPacket(
            data = sig,
            wavelet = "db20",
            maxlevel = 3
        )
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
                impulse_factor,
            ],
            wpd
        ))

    return sum(map(helper_fn, np.array(signal).T))

def calc_adj_mat(features, top_k = 5):
    def calc_distance(a, b):
        return np.sum(np.abs(a - b))

    distance_mat = [
        [calc_distance(a, b) for b in features]
        for a in features
    ]

    adj_mat = []
    for i, distances in enumerate(distance_mat):
        sorted_args = np.argsort(distances)
        assert i == sorted_args[0]
        sorted_args = sorted_args[1:1 + top_k]

        total_dist = np.sum([
            distances[x] for x in sorted_args
        ])

        result = np.zeros(len(distances))
        result[i] = 1

        for j in sorted_args:
            result[j] = distances[j] / total_dist

        adj_mat.append(result)

    return np.array(adj_mat)

def calc_edge_index_and_attr(
        adj_mat,
        need_undirected = False,
        **kwargs
):
    edge_index = [[], []]
    edge_attr = []

    for i, weights in enumerate(adj_mat):
        for j, weight in enumerate(weights):
            if j == i or weight <= 0:
                continue

            edge_index[0].append(i)
            edge_index[1].append(j)
            edge_attr.append([weight])

    edge_index = torch.tensor(edge_index, dtype = int)
    edge_attr = torch.tensor(edge_attr, dtype = torch.double)

    if need_undirected:
        return to_undirected(edge_index, edge_attr, **kwargs)
    else:
        return edge_index, edge_attr

def train(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = criterion(
        out[data.train_mask],
        data.y[data.train_mask]
    )
    loss.backward()
    optimizer.step()

    return loss

def test(data, model):
    model.eval()

    out = model(data)
    pred = out.argmax(dim = 1)

    correct = pred[data.test_mask] == data.y[data.test_mask]

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, val in enumerate(correct):
        signal_class = data.y[data.test_mask][i]
        if val and signal_class == 0:
            tn += 1
        elif val and signal_class == 1:
            tp += 1
        elif signal_class == 0:
            fp += 1
        else:
            fn += 1

    accuracy = (tp + tn) / len(correct)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (recall + precision)

    return {
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
    }

def train_and_test(
        data,
        model,
        optimizer = None,
        criterion = None,
        epoch = 500,
        print_epoch = False,
):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    for i in range(epoch):
        loss = train(data, model, optimizer, criterion)
        if print_epoch:
            print(f"Epoch {i + 1}: loss = {loss:.5f}")

    return test(data, model)

def run_model_test(
        model_constructor,
        data,
        test_n,
        **kwargs
):
    scores = {
        k: 0 for k in ['accuracy', 'precision', 'recall', 'f1_score']
    }

    for _ in range(test_n):
        model = model_constructor(data)
        test_result = train_and_test(
            model = model,
            data = data,
            **kwargs
        )
        for k in scores:
            scores[k] += test_result[k]

    return {
        k: scores[k] / test_n for k in scores
    }

def benchmark_model(config_number, **kwargs):
    start = time.time()
    result = run_model_test(**kwargs)
    duration = time.time() - start

    print(f"---- Config {config_number:02} ----")
    print(f"Training and testing took {duration:.1f} second(s).")

    for k, v in result.items():
        print(f"{k:<15} = {v:.5f}")

    print()
    return result

class GCNConvOnly(torch.nn.Module):
    def __init__(
            self,
            data,
            hidden1 = 8,
            hidden2 = 4
    ):
        super().__init__()
        self.conv1 = GCNConv(len(data.x[0]), hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.conv3 = GCNConv(hidden2, data.class_count)

    def forward(self, data):
        activation_fn = torch.nn.ELU()
        dropout_fn = torch.nn.Dropout(p = 0.25)

        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = activation_fn(x)
        x = dropout_fn(x)

        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = activation_fn(x)
        x = dropout_fn(x)

        x = self.conv3(x, data.edge_index, data.edge_attr)

        return x

class GraphConvOnly(torch.nn.Module):
    def __init__(
            self,
            data,
            hidden1 = 8,
            hidden2 = 4
    ):
        super().__init__()
        self.conv1 = GraphConv(len(data.x[0]), hidden1)
        self.conv2 = GraphConv(hidden1, hidden2)
        self.conv3 = GraphConv(hidden2, data.class_count)

    def forward(self, data):
        activation_fn = torch.nn.ELU()
        dropout_fn = torch.nn.Dropout(p = 0.25)

        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = activation_fn(x)
        x = dropout_fn(x)

        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = activation_fn(x)
        x = dropout_fn(x)

        x = self.conv3(x, data.edge_index, data.edge_attr)

        return x

class GATConvOnly(torch.nn.Module):
    def __init__(
            self,
            data,
            hidden1 = 8,
            hidden2 = 4
    ):
        super().__init__()
        self.conv1 = GATConv(len(data.x[0]), hidden1)
        self.conv2 = GATConv(hidden1, hidden2)
        self.conv3 = GATConv(hidden2, data.class_count)

    def forward(self, data):
        activation_fn = torch.nn.ELU()
        dropout_fn = torch.nn.Dropout(p = 0.25)

        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x = activation_fn(x)
        x = dropout_fn(x)

        x = self.conv2(x, data.edge_index, data.edge_attr)
        x = activation_fn(x)
        x = dropout_fn(x)

        x = self.conv3(x, data.edge_index, data.edge_attr)

        return x

# ----------------

torch.set_default_dtype(torch.double)
plt.figure(figsize = (16, 16), dpi = 512)
plt.tight_layout()

signals = tsv_file_to_list(
    path = "../signals/signal-new.tsv",
    transform_fn = float
)

xyzs = [[p[i] for p in signals] for i in range(3)]

signals = split_signals(signals)

features = np.array([
    signal_to_feature(signal) for signal in signals
])

x_labels = [x + 1 for x in range(len(features[0]))]

status_labels = {
    'standby': 0,
    'grinding': 1,
}

feature_labels = []
for i, _ in enumerate(features):
    label = None

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

    assert label is not None
    feature_labels.append(label)

feature_labels = torch.tensor(feature_labels, dtype = int)

adj_mat = calc_adj_mat(features)

for axis, points in zip(['x', 'y', 'z'], xyzs):
    plt.clf()
    plt.title(f"Raw signal - {axis}")
    plt.plot(points)
    plt.savefig(f"../plots/raw-{axis}.png", bbox_inches = 'tight')

# plt.clf()
# plt.title("Feature matrix")
# plt.pcolormesh(features, cmap = 'viridis')
# plt.colorbar()
# plt.xticks(
#     ticks = [x - 0.5 for x in x_labels],
#     labels = x_labels,
# )
# plt.gca().invert_yaxis()
# plt.savefig("../plots/features.png", bbox_inches = 'tight')

# plt.clf()
# plt.title("Labels for features")
# plt.plot(feature_labels)
# plt.savefig("../plots/feature_labels.png", bbox_inches = 'tight')

# plt.clf()
# plt.title("Adjacent matrix")
# plt.pcolormesh(adj_mat, cmap = 'viridis')
# plt.colorbar()
# plt.gca().invert_yaxis()
# plt.savefig("../plots/adj_mat.png", bbox_inches = 'tight')

features = torch.tensor(features, dtype = torch.double)
edge_index, edge_attr = calc_edge_index_and_attr(
    adj_mat,
    need_undirected = True,
)

train_mask = []
test_mask = []
for i, _ in enumerate(features):
    is_train = False
    is_test = False

    if i >= 0 and i < 10:
        is_train = True
    elif i >= 10 and i < 15:
        is_test = True
    elif i >= 15 and i < 150:
        is_train = True
    elif i >= 150 and i < 200:
        is_test = True
    elif i >= 200 and i < 275:
        is_train = True
    elif i >= 275 and i < 300:
        is_test = True
    elif i >= 300 and i < 345:
        is_train = True
    elif i >= 345 and i < 360:
        is_test = True
    elif i >= 360 and i < 428:
        is_train = True
    elif i >= 427 and i < 450:
        is_test = True
    elif i >= 450 and i < 488:
        is_train = True
    elif i >= 487 and i < 500:
        is_test = True
    elif i >= 500 and i < 575:
        is_train = True
    elif i >= 575 and i < 600:
        is_test = True
    elif i >= 600 and i < 780:
        is_train = True
    elif i >= 780 and i < 840:
        is_test = True
    elif i >= 840 and i < 966:
        is_train = True
    else:
        is_test = True

    assert is_train or is_test
    train_mask.append(is_train)
    test_mask.append(is_test)

train_mask = torch.tensor(train_mask, dtype = bool)
test_mask = torch.tensor(test_mask, dtype = bool)

data = Data(
    x = features,
    y = feature_labels,
    edge_index = edge_index,
    edge_attr = edge_attr,
)

data.train_mask = train_mask
data.test_mask = test_mask
data.class_count = 2

epoch = 500
test_n = 20

model = GCNConvOnly
benchmark_model(
    config_number = 1,
    test_n = test_n,
    data = data,
    model_constructor = model,
    epoch = epoch
)

model = GraphConvOnly
benchmark_model(
    config_number = 2,
    test_n = test_n,
    data = data,
    model_constructor = model,
    epoch = epoch
)

model = GATConvOnly
benchmark_model(
    config_number = 3,
    test_n = test_n,
    data = data,
    model_constructor = model,
    epoch = epoch
)
