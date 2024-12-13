import torch
import argparse
import torch.utils.tensorboard as tb
import numpy as np

from pathlib import Path
from datetime import datetime
from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric

def transformer_trainer(
        # Export directory for tensorboard logs and model checkpoints
        exp_dir: str = "transformer_logs", # Defualt dir for testing
        model_name: str = "mlp_planner",
        num_epoch: int = 50,
        # Learning rate for the optimizer
        lr: float = 1e-3,
        batch_size: int = 64,
        num_workers: int = 2,
        loss_function: str="L1",
        # Random seed for reproducibility
        seed: int = 2024,
        patience: int = 10,
        # Additional keyword arguments to pass to the model
        **kwargs,
):
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("Cuda is not available. Training on CPU")
        device = torch.device("cpu")
    
    # Set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Grader uses default kwargs, you can customize them; set model to training mode
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # Load the data, can augment using road_dataset module
    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_data=load_data("drive_data/val", shuffle=False, num_workers=num_workers)

    # Set the loss function to be either L1 or MSELoss
    if (loss_function == "L1"):
        loss_func = torch.nn.L1Loss()
    else:
        loss_func = torch.nn.MSELoss()
  
    # loss_func = lambda preds, targets: weighted_loss(preds, targets, alpha=0.8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Used to for tracking; keeps track of the x axis in tensorboard plot
    global_step = 0

    # Used to calculate metrics
    train_metrics = PlannerMetric()
    val_metrics = PlannerMetric()

    # Used for early stoppage
    best_long_error = torch.tensor(float("inf"))
    best_lat_error = torch.tensor(float("inf"))
    early_stop_counter = 0

    for epoch in range(num_epoch):
        # Set model to training mode
        model.train()

        # Reset metrics
        train_metrics.reset()

        for batch in train_data:
            # Seperate and move data to GPU
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            # Forward pass
            preds = model(track_left, track_right)
            
            # Compute loss
            loss = loss_func(preds, waypoints)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            train_metrics.add(preds, waypoints, waypoints_mask)

            # Increment global step
            global_step += 1
        
        # Compute epoch-wide metrics for training
        train_epoch_metrics = train_metrics.compute()

        # Disable gradient compution and switch to evaluation mode
        model.eval()
        with torch.inference_mode():
            # Reset metrics
            val_metrics.reset()

            for batch in val_data:
                # Seperate and move data to GPU
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                # Forward pass
                preds = model(track_left, track_right)

                # Compute validation loss (optional for logging)
                loss = loss_func(preds, waypoints)

                # Update metrics
                val_metrics.add(preds, waypoints, waypoints_mask)
            
            # Compute epoch-wide metrics for validation
            val_epoch_metrics = val_metrics.compute()
        
        # Early stopping
        val_long_e = val_epoch_metrics["longitudinal_error"]
        val_lat_e = val_epoch_metrics["lateral_error"]

        long_check = val_long_e < 0.15
        lat_check = val_lat_e < 0.5

        if val_long_e < best_long_error and val_lat_e < best_lat_error:
            best_long_error = val_long_e
            best_lat_error = val_lat_e
            early_stop_counter = 0
            save_model(model)
            torch.save(model.state_dict(), log_dir / f"{model_name}.th")
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience or (long_check and lat_check):
            print(
                f"Early stopping at epoch {epoch + 1}\n"
                f"Longitudinal Error: {val_epoch_metrics['longitudinal_error']:.4f} | "
                f"Lateral Error: {val_epoch_metrics['lateral_error']:.4f}\n\n"
                f"Saved Longitudinal Error: {best_long_error:.4f} | "
                f"Saved Lateral Error: {best_lat_error:.4f}"
            )
            break

        # Log metrics to tensorboard
        logger.add_scalar("train/longitudinal_error", train_epoch_metrics["longitudinal_error"], global_step)
        logger.add_scalar("train/lateral_error", train_epoch_metrics["lateral_error"], global_step)
        logger.add_scalar("val/longitudinal_error", val_epoch_metrics["longitudinal_error"], global_step)
        logger.add_scalar("val/lateral_error", val_epoch_metrics["lateral_error"], global_step)
        
        # Print on first, last and every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}: "
                f"Longitudinal Error: {val_epoch_metrics['longitudinal_error']:.4f} | "
                f"Lateral Error: {val_epoch_metrics['lateral_error']:.4f}"
            )
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="transfromer_logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--num_workers", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--loss_function", type=str, default="L1")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2024)

    transformer_trainer(**vars(parser.parse_args()))
