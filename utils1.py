import torch as tr
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_hidden, indim=2, outdim=1, sigmoid_output=False):
        super(Net, self).__init__()
        self.sigmoid_output = sigmoid_output
        self.fc1 = nn.Linear(indim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, outdim)

    def forward(self, x):
        h1 = self.fc1(x).relu()
        h2 = self.fc2(h1).relu()
        y_hat = self.fc3(h2)
        if self.sigmoid_output:
            y_hat = y_hat.sigmoid()
        return y_hat

    def get_ops(self):
        if self.sigmoid_output:
            loss_op = nn.BCELoss()
        else:
            loss_op = nn.MSELoss()
        optim_op = tr.optim.SGD(self.parameters(), lr=0.01)
        return loss_op, optim_op
