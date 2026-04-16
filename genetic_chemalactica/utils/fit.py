import torch
from torch.nn import HuberLoss
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product


def lbfgs_fit(
    data_, pred, A, C, E, Alpha, Gamma1, Gamma2, logsumexp=True
):
    # print(X)
    data = data_.detach().clone()
    best_p = None
    best_loss = np.inf

    l = list(product(A, C, E, Alpha, Gamma1, Gamma2))

    for init_p in tqdm(l):
        p = torch.tensor(init_p, requires_grad=True)
    
        X = data[:, :3].detach().clone()
        Y = data[:, 3].detach().clone()
        
        # print(f"Initial parameter: {p}")
        loss_fn = HuberLoss(delta=0.001)
        
        # Define the optimizer (L-BFGS)
        optimizer = torch.optim.LBFGS(
            [p], lr=0.1, max_iter=100,
            tolerance_grad=1e-5, tolerance_change=1e-9,
            line_search_fn="strong_wolfe"
        )
        
        # Optimization step function (required for L-BFGS)
        def closure():
            optimizer.zero_grad() # Zero out gradients

            if logsumexp:
                pr = torch.logsumexp(pred(X, p, train=True), dim=0)
                target = torch.log(Y)
            else:
                pr = torch.log(pred(X, p))
                target = Y
            
            
            # target = torch.log(Y)
            # pr = pred(X, p)
            # target = Y
            # print(pr, target)
            # print(f"calling closure with p: {p}")
            loss = loss_fn(pr, target)  # Compute the loss
            # print(f"loss {loss:.4f}")
            loss.backward()  # Compute gradients
            # print(p, p.grad)
            return loss
        
        # Perform optimization
        for i in range(50):  # Optional manual loop for monitoring
            loss = optimizer.step(closure)
            # if i % 100 == 0:
            # print(f"Step {i+1}: p = {p}, Loss = {loss.item():.6f}")
            if loss.item() < 1e-9:  # Break if converged
                break
        
        # Final result
        # print(f"\nOptimized loss: {loss}")
        # and torch.all(p[:3] < -1)
        if loss < best_loss and p[5] > 0 and p[0] < 60:
            best_p = p.detach().clone()
            best_loss = loss.detach().clone()
            print(f"Better loss found {best_loss}, with params {best_p}")
    
    return best_p.detach().clone()


def infer1(sample, p, train=False):
    n = sample.shape[0]
    A, C, E, alpha, gamma1, gamma2 = p
    x = sample[:, 0]
    # x = torch.log(x)
    # x = torch.sqrt(x)
    x = x / 1000
    N = sample[:, 1]
    N = N * (10 ** 9)
    if train:
        return torch.stack(
            [
                torch.ones(n) * E,
                A - alpha * torch.log(N),
                C - x * torch.log(gamma1)
            ],
            dim=0
        )
    return 1 - (E + A / (N ** alpha) + C / (gamma1 ** x))


def infer2(sample, p, train=False):
    n = sample.shape[0]
    A, C, E, alpha, gamma1, gamma2 = p
    x = sample[:, 0]
    N = sample[:, 1]
    N = N * (10 ** 9)
    if train:
        return torch.stack(
            [
                torch.ones(n) * E,
                A - alpha * torch.log(N),
                C - gamma1 * torch.log(x)
            ],
            dim=0
        )
    return 1 - (E + A / (N ** alpha) + C / (x ** gamma1))


def infer3(sample, p, train=False):
    n = sample.shape[0]
    A, C, E, alpha, gamma1, alpha1 = p
    x = sample[:, 0]
    N = sample[:, 1]
    N = N * (10 ** 9)
    if train:
        return torch.stack(
            [
                torch.ones(n) * E,
                A - alpha * torch.log(N),
                C - gamma1 * torch.log(x) - alpha1 * torch.log(N)
            ],
            dim=0
        )
    return 1 - (E + A / (N ** alpha) + C / ((x ** gamma1) * (N ** alpha1)))


def infer4(sample, p, train=False):
    n = sample.shape[0]
    A, C, E, alpha, gamma1, alpha1 = p
    x = sample[:, 0]
    N = sample[:, 1]
    N = N * (10 ** 9)
    log_denom = torch.stack([
        gamma1 * torch.log(x),
        alpha1 * torch.log(N)
    ], dim=0)
    # print(log_denom.shape)
    if train:
        return torch.stack(
            [
                torch.ones(n) * E,
                A - alpha * torch.log(N),
                C - torch.logsumexp(log_denom, dim=0)
            ],
            dim=0
        )
    return 1 - (E + A / (N ** alpha) + C / (x ** gamma1 + N ** alpha1))


def infer5(sample, p, train=False):
    n = sample.shape[0]
    A, C, E, alpha, gamma1, alpha1 = p
    x = sample[:, 0]
    N = sample[:, 1]
    L = sample[:, 2]
    N = N * (10 ** 9)
    if train:
        return torch.stack(
            [
                torch.ones(n) * E,
                A - alpha * torch.log(N),
                C - gamma1 * torch.log(N ** alpha1 + x)
            ],
            dim=0
        )
    return 1 - (E + A / (N ** alpha) + C / ((N ** alpha1 + x) ** gamma1))


def infer6(sample, p, train=False):
    n = sample.shape[0]
    A, C, E, alpha, gamma1, alpha1 = p
    x = sample[:, 0]
    N = sample[:, 1]
    N = N * (10 ** 9)
    if train:
        return torch.stack(
            [
                torch.ones(n) * E,
                A - alpha * torch.log(N),
                C - (x ** gamma1 * alpha1) * torch.log(N)
            ],
            dim=0
        )
    return 1 - (E + A / (N ** alpha) + C / ((N) ** (alpha1 * x ** gamma1)))


def infer7(sample, p, train=False):
    n = sample.shape[0]
    A, C, E, alpha, gamma1, alpha1 = p
    x = sample[:, 0]
    N = sample[:, 1]
    L = sample[:, 2]
    N = N * (10 ** 9)
    if train:
        return torch.stack(
            [
                torch.ones(n) * E,
                A - alpha * torch.log(N),
                C - (gamma1) * torch.log(x * N)
            ],
            dim=0
        )
    return 1 - (E + A / (N ** alpha) + C / ((x * N) ** (gamma1)))


def infer8(sample, p, train=False):
    n = sample.shape[0]
    A, C, E, alpha, gamma1, alpha1 = p
    x = sample[:, 0]
    N = sample[:, 1]
    L = sample[:, 2]
    N = N * (10 ** 9)
    D = 40 * (10 ** 9)
    # e = torch.tensor([0.12])
    if train:
        return torch.stack(
            [
                torch.ones(n) * E,
                C - (x ** gamma1) * torch.log(D ** alpha1 * N ** alpha)
            ],
            dim=0
        )
    return 1 - (E + C / ((D ** alpha1 * N ** alpha) ** (x ** gamma1)))


def infer9(sample, p, train=False):
    n = sample.shape[0]
    A, C, E, alpha, gamma1, alpha1 = p
    x = sample[:, 0]
    N = sample[:, 1]
    L = sample[:, 2]
    N = N * (10 ** 9)
    D = 40 * (10 ** 9)
    if train:
        return torch.stack(
            [
                torch.ones(n) * E,
                C - gamma1 * torch.log(N ** alpha * x)
            ],
            dim=0
        )
    return 1 - (E + C / ((N ** alpha * x) ** gamma1))


def infer10(sample, p, train=False):
    n = sample.shape[0]
    A, C, E, alpha, gamma1, alpha1 = p
    x = sample[:, 0]
    N = sample[:, 1]
    L = sample[:, 2]
    N = N * (10 ** 9)
    D = 40 * (10 ** 9)

    return C / (1 + torch.exp(-alpha * L + alpha1 * x)) + E


def infer11(sample, p, train=False):
    n = sample.shape[0]
    A, C, E, alpha, gamma1, alpha1 = p
    x = sample[:, 0]
    N = sample[:, 1]
    L = sample[:, 2]
    N = N * (10 ** 9)
    D = 40 * (10 ** 9)
    # e = torch.tensor([0.12])
    if train:
        return torch.stack(
            [
                torch.ones(n) * E,
                C - (gamma1 * x) * torch.log(D ** alpha1 * N ** alpha)
            ],
            dim=0
        )
    return 1 - (E + C / ((D ** alpha1 * N ** alpha) ** (gamma1 * x)))
