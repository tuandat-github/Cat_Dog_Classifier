import torchvision
import torch.nn as nn
import torch
from dataset import create_dataloader
from tqdm import tqdm

ROOT = "data/train"
IMAGE_SIZE = 448
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 1
LEARNING_RATE = 5e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(net, train_loader, val_loader, trainer, criterion, num_epochs):
    print("Training!")
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in range(num_epochs):

        # Train Step.
        net.train()
        train_cur_loss = 0.0
        train_cur_correct = 0
        counter = 0
        for _, data in tqdm(enumerate(train_loader)):
            inputs, labels = data
            counter += 1
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1).float()

            trainer.zero_grad()
            # Forward pass.
            outputs = net(inputs)
            # Calculate loss
            loss = criterion(outputs, labels)
            train_cur_loss += loss.item()
            # Calculate Accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_cur_correct += (predictions == labels).sum().item()
            # Backpropagation.
            loss.backward()
            # Update weights.
            trainer.step()

        # Train loss and accuracy for each epoch
        epoch_train_loss = train_cur_loss/counter
        epoch_train_acc = train_cur_correct/len(train_loader.dataset)

        # Append to list to visualize
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)

        # Validation Step.
        net.eval()
        val_cur_correct = 0
        val_cur_loss = 0.0
        counter = 0
        with torch.no_grad():
            for _, data in tqdm(enumerate(val_loader)):
                inputs, labels = data
                counter += 1
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).unsqueeze(1).float()

                outputs = net(inputs)

                # loss
                loss = criterion(outputs, labels)
                val_cur_loss += loss.item()
                # Calculate Accuracy
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_cur_correct += (predictions == labels).sum().item()

            epoch_val_loss = val_cur_loss / counter
            epoch_val_acc = val_cur_correct / len(val_loader.dataset)

        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)

        print(f"Epoch {epoch}, train_loss: {epoch_train_loss}, train_accuracy: {epoch_train_acc}, val_loss: {epoch_val_loss}, val_accuracy:{epoch_val_acc}")
    torch.save(net.state_dict(), "../Cat_Dog_Classifier/models/efficientnet_b7.pth")

    return {"train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc}


if __name__ == "__main__":
    # create train_dataloader, val_dataloader
    train_dataloader, val_dataloader = create_dataloader(ROOT,
                                                         img_size=IMAGE_SIZE,
                                                         batch_size=BATCH_SIZE,
                                                         num_workers=NUM_WORKERS)

    # Create model Efficientnet_b0 with pretrained weights
    model = torchvision.models.efficientnet_b7(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Linear(in_features=2560, out_features=1))
    model = model.to(DEVICE)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    results = train(model, train_dataloader,val_dataloader, optimizer, loss_function, NUM_EPOCHS)
