import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(5, 1)
        self.input_size = (5,)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def loss_acc(self, x, y):
        output = self(x)
        loss = nn.BCELoss()(output, y)
        pred = output.round()
        acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
        return {"loss": loss, "acc": acc}

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(5, 1)
        self.input_size = (5,)

    def forward(self, x):
        return self.linear(x)

    def loss_acc(self, x, y):
        output = self(x)
        loss = ((output - y)**2).mean()
        return {"loss": loss, "acc": 0}

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_size = (5,)
        self.linear1 = nn.Linear(5, 5)
        self.linear2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return torch.sigmoid(self.linear2(x))

    def loss_acc(self, x, y):
        output = self(x)
        loss = nn.BCELoss()(output, y)
        pred = output.round()
        acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
        return {"loss": loss, "acc": acc}

class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        self.input_size = (28 * 28,)
        self.linear1 = nn.Linear(28 * 28, 100)
        self.linear2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

    def loss_acc(self, x, y):
        output = self(x)
        output = self.softmax(output)
        loss = self.criterion(output, y)
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
        return {"loss": loss, "acc": acc}

class MNIST_MLP_500(nn.Module):
    def __init__(self):
        super(MNIST_MLP_500, self).__init__()
        self.input_size = (28 * 28,)
        self.linear1 = nn.Linear(28 * 28, 500)
        self.linear2 = nn.Linear(500, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

    def loss_acc(self, x, y):
        output = self(x)
        output = self.softmax(output)
        assert not torch.any(torch.isnan(output))
        loss = self.criterion(output, y)
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
        return {"loss": loss, "acc": acc}

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.input_size = (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 64, 3, 3)
        self.conv2 = nn.Conv2d(64, 64, 3, 3)
        self.conv3 = nn.Conv2d(64, 64, 3, 3)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.head = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(-1, 64)
        return self.head(x)
    
    def loss_acc(self, x, y):
        output = self(x)
        output = self.softmax(output)
        loss = self.criterion(output, y)
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
        return {"loss": loss, "acc": acc}

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss = nn.CrossEntropyLoss(label_smoothing=0.2)

    dims = [3,64,128,128,128,256,512,512,512]
    seq = []
    for i in range(len(dims)-1):
      c_in,c_out = dims[i],dims[i+1]
      seq.append( nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False) )
      if c_out == c_in * 2:
        seq.append( nn.MaxPool2d(2) )
      seq.append( nn.BatchNorm2d(c_out) )
      seq.append( nn.CELU(alpha=0.075) )
    self.seq = nn.Sequential(*seq, nn.MaxPool2d(4), nn.Flatten(), nn.Linear(dims[-1], 10, bias=False))

  def forward(self, x):
    x = self.seq(x) / 8
    return x

  def loss_acc(self, x, y):
    output = self(x)
    loss = self.loss(output, y)
    pred = output.argmax(dim=1, keepdim=True)
    acc = pred.eq(y.view_as(pred)).sum().item() / len(y)
    return {"loss": loss, "acc": acc}

def get_model(model_name):
    if model_name == "logistic":
        return LogisticRegression()
    elif model_name == "linear":
        return LinearRegression()
    elif model_name == "mlp":
        return MLP()
    elif model_name == "mnist_mlp":
        return MNIST_MLP()
    elif model_name == "mnist_mlp_500":
        return MNIST_MLP_500()
    elif model_name == "mnist_cnn":
        return MNIST_CNN()
    elif model_name == "cnn":
        return CNN()
    else:
        raise ValueError("Invalid model name")