import torch
import torch.nn as nn
from torchvision.transforms import v2
from .model import IMAGENET_MEAN, IMAGENET_STD, IMAGE_HEIGHT, IMAGE_WIDTH

TRAIN_TRANSFORM = v2.Compose(
    [
        v2.Resize((int(IMAGE_HEIGHT * 1.1), int(IMAGE_WIDTH * 1.1))),
        v2.RandomCrop((IMAGE_HEIGHT, IMAGE_WIDTH)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        v2.CenterCrop((IMAGE_HEIGHT, IMAGE_WIDTH)),
        v2.ToTensor(),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

TEST_TRANSFORM = v2.Compose(
    [
        v2.CenterCrop((IMAGE_HEIGHT, IMAGE_WIDTH)),
        v2.ToTensor(),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device) -> float:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy = 100 * (correct / size)
    print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy
