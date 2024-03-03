import argparse
import math
import random
import sys

from d2l import torch as d2l
import numpy as np
import torch
from torch import nn


# Functions ###########################

# def show_plot(x_vector, y_matrix, title=None, xlabel=None, ylabel=None, figsize=None, legend=None):
#     plt.figure(figsize=figsize)
#     for y_vector in y_matrix:
#         plt.plot(x_vector, y_vector)

#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend(legend)
#     plt.grid(True)
#     plt.show()


def fn1(args):
    def normal(x, mu, sigma):
        p = 1 / math.sqrt(2 * math.pi * sigma**2)
        return p * np.exp(-0.5 * (x - mu)**2 / sigma**2)
    
    x = np.arange(-7, 7, 0.01)
    print(x)
    
    params = [(0, 1), (0, 2), (3, 1)]
    print([normal(x, mu, sigma) for mu, sigma in params])
    # show_plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x', ylabel='p(x)', figsize=(4.25, 2.5),
    #           legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
    d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x', ylabel='p(x)', figsize=(4.25, 2.5),
              legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])


class SyntheticRegressionData(d2l.DataModule):
    """Synthetic data for linear regression"""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise
    
    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)


def fn2(args):
    data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    print('features:', data.X[0], '\nlabel:', data.y[0])

    X, y = next(iter(data.train_dataloader()))
    print('X shape:', X.shape, '\ny shape:', y.shape)


class SGD(d2l.HyperParameters):
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class Trainer(d2l.HyperParameters):
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'
    
    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader) if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def prepare_batch(self, batch):
        return batch

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoc()
    
    def fit_epoc(self):
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1


class LinearRegressionScratch(d2l.Module):
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        l = (y_hat - y)**2 / 2
        return l.mean()

    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)


def fn4(args):
    model = LinearRegressionScratch(2, lr=0.03)
    data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    trainer = Trainer(max_epochs=3)
    trainer.fit(model, data)
    with torch.no_grad():
        print(f'error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
        print(f'error in estimating b: {data.b - model.b}')


class LinearRegression(d2l.Module):
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)
    
    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizer(self):
        return torch.optim.SGD(self.parameters(), self.lr)
    
    def get_w_b(self):
        return (self.net.weight.data, self.net.bias.data)
    

def fn5(args):
    model = LinearRegression(lr=0.03)
    data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    trainer = Trainer(max_epochs=3)
    trainer.fit(model, data)
    w, b = model.get_w_b()
    print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')
    print(f'error in estimating b: {data.b - b}')


class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)


def l2_penalty(w):
    return (w ** 2).sum() / 2


class WeightDecayScratch(LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
    
    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) + self.lambd * l2_penalty(self.w))


class WeightDecay(LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizer(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)


def fn7(args):
    data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
    trainer = Trainer(max_epochs=10)

    def train_scratch(lambd):
        model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
        model.board.yscale = 'log'
        trainer.fit(model, data)
        print(f'L2 norm of w: {torch.norm(model.w).item()}')

    train_scratch(3)


def main(args):
    parser = argparse.ArgumentParser(description="Dive Into Deep Learning, Chapter 3")
    commands = parser.add_subparsers(dest="cmd")

    cmd1 = commands.add_parser("3.1"); cmd1.set_defaults(function=fn1)
    cmd2 = commands.add_parser("3.2"); cmd2.set_defaults(function=fn2)
    cmd3 = commands.add_parser("3.4"); cmd3.set_defaults(function=fn4)
    cmd4 = commands.add_parser("3.5"); cmd4.set_defaults(function=fn5)
    cmd4 = commands.add_parser("3.7"); cmd4.set_defaults(function=fn7)
    
    args = parser.parse_args()
    if not hasattr(args, "function"):
        parser.print_help()
        return 1

    args.function(args)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
