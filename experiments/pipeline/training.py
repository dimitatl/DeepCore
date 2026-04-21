"""Training loop for a coreset-selected model — minimal and deterministic."""
import torch
import torch.nn as nn
import numpy as np
import deepcore.nets as nets
from torchvision import transforms


def build_network(model_name, channel, num_classes, im_size, device):
    net = nets.__dict__[model_name](channel, num_classes, im_size).to(device)
    return net


def _augment_cifar(dst, im_size):
    dst.transform = transforms.Compose([
        transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        dst.transform,
    ])
    return dst


def train_model(network, train_loader, test_loader, args, num_epochs, verbose=True):
    criterion = nn.CrossEntropyLoss(reduction="none").to(args.device)

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr)

    if args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max(1, len(train_loader)) * num_epochs, eta_min=args.min_lr)
    elif args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, len(train_loader)) * int(args.step_size), gamma=args.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(num_epochs):
        network.train()
        losses = []
        correct = 0
        total = 0
        for contents in train_loader:
            if isinstance(contents[0], (list, tuple)):
                (input_, target), weights = contents
                weights = weights.to(args.device)
            else:
                input_, target = contents
                weights = None
            input_ = input_.to(args.device)
            target = target.to(args.device)

            optimizer.zero_grad()
            output = network(input_)
            sample_loss = criterion(output, target)
            if weights is not None:
                loss = (sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)
            else:
                loss = sample_loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        train_loss = float(np.mean(losses)) if losses else 0.0
        train_acc = correct / max(1, total)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if test_loader is not None and args.test_interval > 0 and (epoch + 1) % args.test_interval == 0:
            test_acc = _quick_test(network, test_loader, args.device)
            history["test_acc"].append(test_acc)
            if verbose:
                print(f"[epoch {epoch+1}/{num_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
        elif verbose and (epoch + 1) % max(1, num_epochs // 5) == 0:
            print(f"[epoch {epoch+1}/{num_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

    return network, history


@torch.no_grad()
def _quick_test(network, test_loader, device):
    network.eval()
    correct = 0
    total = 0
    for input_, target in test_loader:
        input_ = input_.to(device)
        target = target.to(device)
        out = network(input_)
        correct += (out.argmax(dim=1) == target).sum().item()
        total += target.size(0)
    return correct / max(1, total)
