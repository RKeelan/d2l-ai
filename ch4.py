import argparse
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from d2l import torch as d2l

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    figure, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    # I can't get the plot to stay visible, but I can use this breakpoint to call `figure.show()`, which works for some
    # reason
    breakpoint()


class FashionMNIST(d2l.DataModule):
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(root=self.root, train=False, transform=trans, download=True)

    def text_labels(self, indices):
        """Return text labels."""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bad', 'ankle boot']
        return [labels[int(i)] for i in indices]
    
    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train, num_workers=self.num_workers)

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        show_images(X.squeeze(1), nrows, ncols, titles=labels)


def fn2(args):
    d2l.use_svg_display()
    data = FashionMNIST(resize=(32, 32))
    # print(len(data.train), len(data.val))
    X, y = next(iter(data.train_dataloader()))
    # print(X.shape, X.dtype, y.shape, y.dtype)
    # tic = time.time()
    # for X, y in data.train_dataloader():
    #     continue
    # print(f'{time.time() - tic:.2f} sec')
    batch = next(iter(data.train_dataloader()))
    data.visualize(batch)


class Classifier(d2l.Module):
    """The case class of classificaiton models."""
    def validation_step(self, batch):
        y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(y_hat, batch[-1]), train=False)
        print(f'loss: {self.loss(y_hat, batch[-1]):.3f}, acc: {self.accuracy(y_hat, batch[-1]):.3f}')
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def accuracy(self, Y_hat, Y, averaged=True):
        """Computer the number of correct predictions"""
        Y_hat = d2l.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(axis=1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition # The broadcasting mechanism is applied here


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()


class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]

    def forward(self, X):
        X = X.reshape((-1, self.W.shape[0]))
        return softmax(torch.matmul(X, self.W) + self.b)

    def loss(self, y_hat, y):
        return cross_entropy(y_hat, y)


def fn4(args):
    X = torch.rand((2, 5))
    X_prob = softmax(X)
    # print(X_prob, X_prob.sum(1))
    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    # print(y_hat[[0, 1], y])
    # print(cross_entropy(y_hat, y))

    data = d2l.FashionMNIST(batch_size=256)
    model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model, data)

    X, y = next(iter(data.val_dataloader()))
    preds = model(X).argmax(axis=1)
    preds.shape()
    wrong = preds.type(y.dtype) != y
    X, y, preds = X[wrong], y[wrong], preds[wrong]
    labels = [a+'\n'+b for a, b in zip(data.text_labels(y), data.text_labels(preds))]
    data.visualize((X, y), nrows=1, ncols=8, titles=labels)


class SoftmaxRegression(d2l.Classifier):
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)

    def loss(self, Y_hat, Y, average=True):
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape((-1,))
        return F.cross_entropy(Y_hat, Y, reduction='mean' if average else 'none')


def fn5(args):
    data = d2l.FashionMNIST(batch_size=256)
    model = SoftmaxRegression(num_outputs=10, lr=0.1)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model, data)


def main(args):
    parser = argparse.ArgumentParser(description="Dive Into Deep Learning, Chapter 4")
    commands = parser.add_subparsers(dest="cmd")

    cmd2 = commands.add_parser("4.2"); cmd2.set_defaults(function=fn2)
    cmd4 = commands.add_parser("4.4"); cmd4.set_defaults(function=fn4)
    cmd5 = commands.add_parser("4.5"); cmd5.set_defaults(function=fn5)
    
    args = parser.parse_args()
    if not hasattr(args, "function"):
        parser.print_help()
        return 1

    args.function(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
