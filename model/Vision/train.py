import torch
import argparse
import os
import time
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import VitModel


def yolo_loss(preds, targets, grid_size, boxes_per_cell, num_classes, lambda_coord=5.0, lambda_noobj=0.5):
    return 0

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    model = VitModel(args.num_classes, args.boxes_per_cell, args.grid_size, args.hidden_size, args.vit_model_name)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    x_train = torch.load(args.train_data_x)
    y_train = torch.load(args.train_data_y)

    x_test = torch.load(args.test_data_x)
    y_test = torch.load(args.test_data_y)

    batch_size = args.batch_size

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    epochs = args.epochs

    model_path = os.path.join(args.model_dir, "vit_yolo_model.pth")

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print(f"epoch: {epoch}, batch: {batch_idx}")            

            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Reshape outputs to match the target format
            outputs = model.reshape(outputs)

            # Compute the loss
            loss = yolo_loss() # **** TO BE IMPLEMENTED

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Track loss
            running_loss += loss.item()

        # Calculate average loss for this epoch
        avg_loss = running_loss / len(train_loader)
        elapsed_time = time.time() - start_time

        print(f"epoch: {epoch}, loss: {avg_loss:.4f}, time: {elapsed_time:.2f}s")

        # Validation (optional, but recommended for tracking performance on test set)
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            total_test_loss = 0.0
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs = model.reshape(outputs)
                total_test_loss += yolo_loss().item() # **** TO BE IMPLEMENTED

            avg_test_loss = total_test_loss / len(test_loader)

            print(f"loss: {avg_test_loss:.4f}")

        # Save the checkpoint
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_classes", type=int, required=True, help="The number of classes")
    parser.add_argument("--boxes_per_cell", type=int, required=True, help="The number of boxes per cell")
    parser.add_argument("--grid_size", type=int, required=True, help="The size of the grid for x and y")
    parser.add_argument("--hidden_size", type=int, required=True, help="Hidden size of the neural network")
    parser.add_argument("--vit_model_name", type=str, required=True, help="Hidden size of the neural network")

    parser.add_argument("--learning_rate", type=float, required=True, help="Training learning rate")
    parser.add_argument("--batch_size", type=int, required=True, help="Training batch size")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")

    parser.add_argument("--train_data_x", type=str, required=True, help="S3 location of the train data x")
    parser.add_argument("--train_data_y", type=str, required=True, help="S3 location of the train data y")
    parser.add_argument("--test_data_x", type=str, required=True, help="S3 location of the test data x")
    parser.add_argument("--test_data_y", type=str, required=True, help="S3 location of the test data y")

    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save the model")

    args = parser.parse_args()

    main(args)