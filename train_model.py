import torch

def train_autoencoder(model: torch.nn.Module,
                      loss_fn: torch.nn.Module,
                      optimizer: torch.optim.Optimizer,
                      epochs: int,
                      X_train: torch.Tensor,
                      X_test: torch.Tensor,
                      updates: bool = False, 
                      updates_epochs: int = 100):
    """
    Trains an autoencoder model.

    Args:
        model (torch.nn.Module): The autoencoder model.
        loss_fn (torch.nn.Module): The loss function (e.g., MSELoss).
        optimizer (torch.optim.Optimizer): The optimizer (e.g., Adam).
        epochs (int): Number of training epochs.
        X_train (torch.Tensor): Training data.
        X_test (torch.Tensor): Test data.
        updates (bool, optional): Whether to print updates. Defaults to False.
        updates_epochs (int, optional): Frequency of updates. 

    Returns:
        tuple: (trained model, train losses, test losses)
    """
    
    train_loss = []
    test_loss = []

    for epoch in range(epochs):
        # Training loop
        model.train()
        data_decoded = model(X_train)
        loss = loss_fn(data_decoded, X_train)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Testing loop
        model.eval()
        with torch.no_grad():  # Use torch.no_grad() instead of torch.inference_mode() (for wider compatibility)
            test_decoded = model(X_test)
            test_loss_value = loss_fn(test_decoded, X_test).item()
            test_loss.append(test_loss_value)

        # Print updates
        if updates and epoch % updates_epochs == 0:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss_value:.4f}")

    return model, train_loss, test_loss

              
