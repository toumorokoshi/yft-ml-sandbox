import sys
import torch
import torch.nn as nn
import asciichartpy
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
    # TODO: add weight decay of 0.0005
    epochs = 60
    model.train()
    test_accuracies = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test_accuracy = test(test_dataloader, model, loss_fn, device)
        test_accuracies.append(test_accuracy)
    print(f"Test accuracies: {test_accuracies}")
    ascii_chart = asciichartpy.plot(test_accuracies, {"height": 20})
    print(ascii_chart)
    model.eval()
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


def test(dataloader, model, loss_fn, device) -> float:
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
    accuracy = 100 * (correct / size)
    print(
        f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return accuracy


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # middle_layer_size = 28 * 28 * 28 // 32  # 686, results in ~ 2gb
        # conv_output_height = ((IMAGE_HEIGHT - 11) // 4 + 1) - 4
        # conv_output_width = ((IMAGE_WIDTH - 11) // 4 + 1) - 4
        conv_features = [96, 256, 384, 384, 256]
        # conv_output_size = 52224 # with maxpool
        conv_output_size = 38400 # with maxpool * 2
        # conv_output_size = conv_features[-1] * conv_output_height * conv_output_width
        #print(f"Conv output size: {conv_output_size}")
        # TO TRY:
        # - maxpooling between conv layers
        self.conv_stack = nn.Sequential(
            # the output of a conv layer is a 3D tensor, where:
            # - the input tensor is shrunk by the kernel size minus 1 (since the convolution
            #   has to act on a fill matrix).
            # - the input tensor is divided by the stride, since you will get that many
            #   fewer points.
            # - the output is the number of channels desired.
            nn.Conv2d(3, conv_features[0], kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(conv_features[0], conv_features[1], kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(conv_features[1], conv_features[2], kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(conv_features[2], conv_features[3], kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(conv_features[3], conv_features[4], kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(conv_output_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # The last layer generally should be a linear layer, not an
            # activation function like a ReLU, which will bound the output
            # unnescessarily and impact the fidelity into a function like
            # softmax, which can consume between negative and positive infinity.
            nn.Linear(4096, NUM_CLASSIFICATIONS),
        )

    def forward(self, x):
        conv_output = self.conv_stack(x)
        # print(f"Conv output shape: {conv_output.shape}")
        logits = self.linear_stack(conv_output)
        return logits


if __name__ == "__main__":
    main(sys.argv)
