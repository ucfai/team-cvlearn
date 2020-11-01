import pandas as pd
import torch
from IPython import get_ipython

# Switch between console and notebook progress bar
if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def validate(model, val_loader, criterion=None, device="cuda"):
    """Get accuracy and loss on validation data"""
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    val_loss = 0
    val_correct = 0
    n_samples = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            n_samples += inputs.shape[0]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Get batch loss/acc
            loss = criterion(outputs, labels)
            preds = torch.max(outputs, 1)[1]

            val_loss += loss.item()
            val_correct += torch.sum(preds == labels).item()

    return val_loss / len(val_loader), val_correct / n_samples


def fit(model, train_loader, val_loader, epochs, criterion, optimizer, device="cuda"):
    """Fit a model on a dataset, with a keras like progress bar

    Args:
        model: pytorch model
        train_loader: Training data in a dataloader
        val_loader: Validation data in a dataloader
        epoch: number of epochs
        criterion:
        optimizer:
        device: 'cpu' or 'cuda' (default)
    """
    history = []
    epoch_loop = tqdm(range(epochs))
    for epoch in epoch_loop:
        train_loss = 0.0
        train_acc = 0

        model.train()
        batch_loop = tqdm(train_loader)
        batch_loop.set_description("Epoch {}/{}".format(epoch + 1, epochs))
        for inputs, labels in batch_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Get accuracy
            preds = torch.max(outputs, 1)[1]
            batch_acc = torch.sum(preds == labels).item() / labels.shape[0]

            # Print loss and acc in inner loop
            batch_loop.set_postfix(loss=loss.item(), acc=batch_acc)

        # Record time of training
        elapsed = float(batch_loop.format_dict["elapsed"])

        # Get loss/metrics on training/val data. Append loss/metrics
        train_loss, train_acc = validate(model, train_loader, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history.append([train_loss, train_acc, val_loss, val_acc, elapsed])

        # Update outer loop
        epoch_loop.set_postfix(train_acc=train_acc, val_acc=val_acc)

    return pd.DataFrame(
        history, columns=["train_loss", "train_acc", "val_loss", "val_acc", "time"]
    )
