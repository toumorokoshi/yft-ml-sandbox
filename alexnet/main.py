import sys
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import pdb  # TODO remove

BATCH_SIZE = 16
# this is the number of classifications within the dataset.
# I believe this has to match the number of classifications in the original
# data set to enable validation
#
# Imagenette has 10 simple classifications.
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 400
NUM_CLASSIFICATIONS = 10
# not all images from the Imagenette dataset are the same size,
# so we have to crop them with a a custom transform, and request a s
# specific size
SIZE = "320px"
IMAGE_TRANSFORM = v2.Compose(
    [
        v2.CenterCrop((IMAGE_HEIGHT, IMAGE_WIDTH)),
        v2.ToTensor(),
    ]
)


def main(argv: list[str]) -> None:
    print("This is the main function of AlexNet.")

    # Download training data from open datasets.
    # this produces tensors of size BATCH_SIZE, 3 (per color channel), IMAGE_HEIGHT, IMAGE_WIDTH
    training_data = datasets.Imagenette(
        root="data",
        split="train",
        size=SIZE,
        download=True,
        transform=IMAGE_TRANSFORM,
    )

    ## Download test data from open datasets.
    test_data = datasets.Imagenette(
        root="data",
        split="val",
        size=SIZE,
        download=True,
        transform=IMAGE_TRANSFORM,
    )

    # shuffling the dataset is critical here, otherwise
    # it overfits to one at a time and does not progress past 10% accuracy.
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    loss_fn = nn.CrossEntropyLoss()
    print(f"Using device: {device}")
    model = NeuralNetwork().to(device)
    print(f"Model structure: {model}")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    epochs = 20
    model.train()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
    model.eval()
    test(test_dataloader, model, loss_fn, device)
    print("Done!")


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


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    # torch.
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # pred.shape = 64, 10
            # prd.argmax takes the value with the highest score, and
            # therefore handles the final transform to a single value per batch.
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        middle_layer_size = 28 * 28 * 28 // 32  # results in ~ 2gb
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(IMAGE_HEIGHT * IMAGE_WIDTH * 3, middle_layer_size),
            nn.ReLU(),
            nn.Linear(middle_layer_size, 512),
            nn.ReLU(),
            # The last layer generally should be a linear layer, not an
            # activation function like a ReLU, which will bound the output
            # unnescessarily and impact the fidelity into a function like
            # softmax, which can consume between negative and positive infinity.
            nn.Linear(512, NUM_CLASSIFICATIONS),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    main(sys.argv)
