import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import os
import sys
import logging
import time

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

import smdebug.pytorch as smd

checkpoints_path = "./earlystopping"

def test(model, test_loader, loss_criterion, device, hook):
    print("Testing model on whole testing dataset")
    model.eval()
    hook.set_mode(smd.modes.PREDICT)
    running_loss = 0
    running_corrects = 0
    # (actual labels, predicted labels)
    y, y_hat = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Load data and target to device
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predictions == labels.data).item()
            y.extend(labels.tolist())
            y_hat.extend(predictions.tolist())

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    logger.info(f"Testing Accuracy: {100 * total_acc}%; Testing Loss: {total_loss}")

    logger.info("Classification report:")
    logger.info(classification_report(y, y_hat, target_names=["1", "2", "3", "4", "5"]))

    logger.info("Confusion matrix:")
    logger.info(confusion_matrix(y, y_hat))


def train(args, model, train_loader, valid_loader, loss_criterion, optimizer, device, hook):
    best_loss = 1e6
    image_dataset = {'train': train_loader, 'valid': valid_loader}
    loss_counter = 0
    for epoch in range(args.epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if (phase == 'train'):
                model.train()
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)
            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            # Iterate through each batch of images and their labels in the DataLoader
            for batch_idx, (inputs, labels) in enumerate(image_dataset[phase]):
                # Load model data and target to device
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)
                # Compute loss
                loss = loss_criterion(outputs, labels)

                if (phase == 'train'):
                    # Reset gradients
                    optimizer.zero_grad()
                    # Calculate gradients in backward pass
                    loss.backward()
                    # Increment the optimizer
                    optimizer.step()

                _, predictions = torch.max(outputs, 1)
                running_corrects += torch.sum(predictions == labels.data).item()
                # Compute total loss for the batch and add to the running loss
                running_loss += loss.item() * inputs.size(0)
                # Keeps track of the number of images we've processed
                running_samples += len(inputs)
                # print("Phase: {} Loss: {:.2f}".format(phase, loss.item()))
                if (batch_idx % args.log_interval == 0):
                    # Calculate and display the accuracy and other metrics
                    accuracy = running_corrects / running_samples
                    print("Images [{}/{} ({:.0f}%)] Phase: {} Loss: {:.2f} Accuracy: {}/{} ({:.2f}%) Time: {}".format(
                        running_samples,
                        len(image_dataset[phase].dataset),
                        100.0 * (running_samples / len(image_dataset[phase].dataset)),
                        phase,
                        loss.item(),
                        running_corrects,
                        running_samples,
                        100.0 * accuracy,
                        time.asctime()
                    )
                    )

                # For only training and validating on part of the dataset
                if (phase == 'train') and (running_samples > (args.train_proportion * len(image_dataset[phase].dataset))):
                    break
                elif (phase == 'valid') and (running_samples > (args.valid_proportion * len(image_dataset[phase].dataset))):
                    break

            epoch_loss = running_loss / running_samples

            if (phase == 'valid'):
                if (epoch_loss < best_loss):
                    best_loss = epoch_loss
                    torch.save({'state_dict': model.state_dict()}, checkpoints_path)
                    # Reset loss counter
                    loss_counter = 0
                else:
                    loss_counter += 1
                    logger.info(f"Loss not improved for {loss_counter} epochs")
        # Break if loss doesn't improve after args.patience consecutive epochs
        if (loss_counter == args.patience):
            logger.info(f"Loss not improved for {loss_counter} epochs: early stopping.")
            break
    return model


def net():

    model = models.resnet18(pretrained=True)

    # Freezes convolutional layers of model
    for param in model.parameters():
        param.requires_grad = False

    # Number of features in the output of the pretrained model
    num_features = model.fc.in_features

    # Output 5 channels, one for each class of image
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 5)
    )

    return model


def create_data_loaders(data_dir, batch_size, training=True):
    """
    Loads training, validation and test data into a DataLoader to use as input for
    the CNN. If training, applies some augmentation to each image.
    """
    if training:
        transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomAdjustSharpness(2, p=0.2),
            transforms.RandomAutocontrast(p=0.1),
            transforms.RandomApply([
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)
            ], p=0.15),
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomRotation((90, 90)),
                    transforms.RandomRotation((-90, -90)),
                ])
            ], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4758, 0.4253, 0.3760],
                                 std=[0.1616, 0.1491, 0.1424])
        ])

        shuffle = True
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4758, 0.4253, 0.3760],
                                 std=[0.1616, 0.1491, 0.1424])
        ])

        shuffle = False

    data = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size,
                                              shuffle=shuffle)
    return data_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-proportion", type=float, default=0.3)
    parser.add_argument("--valid-proportion", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--model-dir", type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()

    # Initialise model
    model = net()
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    # Check if GPU is available and if so select it.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device {device}")

    model = model.to(device)

    # Define loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), 
                           lr=args.lr, 
                           betas=(args.beta1, 0.999),
                           weight_decay=args.weight_decay)

    # Get data loaders
    train_loader = create_data_loaders(os.environ["SM_CHANNEL_TRAIN"], args.batch_size, training=True)
    valid_loader = create_data_loaders(os.environ["SM_CHANNEL_VALID"], args.batch_size, training=False)
    test_loader = create_data_loaders(os.environ["SM_CHANNEL_TEST"], args.batch_size, training=False)

    logger.info("Training model...")
    model = train(args,
                  model,
                  train_loader,
                  valid_loader,
                  loss_criterion,
                  optimizer,
                  device,
                  hook)

    logger.info('Loading model before testing.')
    model_checkpoint = torch.load(checkpoints_path)

    model.load_state_dict(model_checkpoint['state_dict'])

    test(model, test_loader, loss_criterion, device, hook)

    logger.info("Saving model...")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pt"))
    logger.info(f"Model saved at {args.model_dir}")


if __name__ == '__main__':
    main()
