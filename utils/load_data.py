import torchvision
import torch
import pandas as pd
from torch import nn

def loadMNISTData(device):
    train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    data = train.train_data.float().to(device)
    # data = torch.nn.functional.interpolate(data.view(-1, 1, 28, 28), size=(14, 14)).view(-1, 14 * 14)
    data = data.view(-1, 28 * 28)
    train_data = data[:59000]
    test_data = data[59000:]
    train_labels = train.train_labels[:59000].to(device)
    test_labels = train.train_labels[59000:].to(device)
    return train_data, train_labels, test_data, test_labels

def loadCIFAR10(device):
  train = torchvision.datasets.CIFAR10(root="./data/cifar10", train=True, download=True)
  test  = torchvision.datasets.CIFAR10(root="./data/cifar10", train=False, download=True)
  ret = [torch.tensor(i, device=device) for i in (train.data, train.targets, test.data, test.targets)]
  std, mean = torch.std_mean(ret[0].float(),dim=(0,1,2),unbiased=True,keepdim=True)
  for i in [0,2]: ret[i] = ((ret[i]-mean)/std).to(torch.float32).permute(0,3,1,2)
  return ret

def loadLinearData(device):
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_openml

    boston = fetch_openml(name='boston', version=1)
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    boston_df['PRICE'] = boston.target

    features = ['CRIM', 'RM', 'AGE', 'DIS', 'LSTAT']
    X = boston_df[features].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = boston_df['PRICE'].values
    y = (y - y.mean()) / y.std() 

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (torch.tensor(X_train, dtype=torch.float32).to(device), 
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device), 
            torch.tensor(X_test, dtype=torch.float32).to(device), 
            torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device))

def loadSmallTableData(device):
    train_data = pd.read_csv('data/train_data.csv')
    X_train = train_data.drop('purchased', axis=1).values
    y_train = train_data['purchased'].values

    # Load the test data
    test_data = pd.read_csv('data/test_data.csv')
    X_test = test_data.drop('purchased', axis=1).values
    y_test = test_data['purchased'].values

    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train_tensor.to(device), y_train_tensor.to(device), X_test_tensor.to(device), y_test_tensor.to(device)

def get_data_loader(data_name):
    if data_name == "mnist":
        return loadMNISTData
    elif data_name == "cifar10":
        return loadCIFAR10
    elif data_name == "linear":
        return loadLinearData
    else:
        return loadSmallTableData

def getBatches(X, y, batch_size):
    length = len(X)
    if batch_size > length:
        repeat = (batch_size + length - 1) // length
        X = X.repeat(repeat, 1)
        y = y.repeat(repeat, 1)
        length = len(X)
    idx = torch.randperm(length)
    X = X[idx]
    y = y[idx]
    for i in range(0, length, batch_size):
        if i + batch_size > length:
            break
        yield X[i:i+batch_size], y[i:i+batch_size]

def augment_data(X, y, crop_size=4, cut_size=8, image_width=32, image_height=32):
    perm = torch.randperm(len(X), device=X.device)
    X, y = X[perm], y[perm]

    Crop = ([(y0,x0) for x0 in range(crop_size+1) for y0 in range(crop_size+1)], 
        lambda img, y0, x0 : nn.ReflectionPad2d(crop_size)(img)[..., y0:y0+image_height, x0:x0+image_width])
    FlipLR = ([(True,),(False,)], 
        lambda img, choice : torch.flip(img,[-1]) if choice else img)
    def cutout(img,y0,x0):
        img[..., y0:y0+cut_size, x0:x0+cut_size] = 0
        return img
    Cutout = ([(y0,x0) for x0 in range(image_width+1-cut_size) for y0 in range(image_height+1-cut_size)], cutout)

    for options, transform in (Crop, FlipLR, Cutout):
        optioni = torch.randint(len(options),(len(X),), device=X.device)
        for i in range(len(options)):
            X[optioni==i] = transform(X[optioni==i], *options[i])

    return X, y
