import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ZINCDataset, collate_graphs
from model import GraphAF

# pyright: reportPossiblyUnboundVariable=false


def main():
    # Setup
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! This script requires GPU.")

    device = torch.device("cuda")

    # GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    os.makedirs("checkpoints", exist_ok=True)

    # TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/graphaf_{timestamp}")

    # Log config to TensorBoard
    config = {
        "gpu": torch.cuda.get_device_name(0),
        "epochs": 10,
        "batch_size": 32,
        "lr": 0.001,
        "grad_clip": 1.0,
    }
    writer.add_text("config", str(config))

    # Model and optimizer
    model = GraphAF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    writer.add_text("model/total_params", f"{total_params:,}")

    # Hyperparameters
    num_epochs = config["epochs"]
    batch_size = config["batch_size"]
    grad_clip = config["grad_clip"]

    # Dataset
    dataset = ZINCDataset("data/250k_rndm_zinc_drugs_clean_3.csv")
    writer.add_text("data/dataset_size", str(len(dataset)))

    # Data loader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # Training loop
    best_loss = float("inf")
    global_step = 0

    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(train_loader):
                # Move to GPU
                X = batch["X"].to(device, non_blocking=True)
                A = batch["A"].to(device, non_blocking=True)

                # Forward pass
                optimizer.zero_grad()
                loss = model(X, A)

                # Check for NaN
                if torch.isnan(loss):
                    writer.add_text("warnings", f"NaN loss at step {global_step}")
                    continue

                # Backward pass
                loss.backward()

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                # Optimizer step
                optimizer.step()

                # Metrics
                loss_val = loss.item()
                epoch_loss += loss_val
                global_step += 1

                # TensorBoard logging
                writer.add_scalar("loss/batch", loss_val, global_step)
                writer.add_scalar("gradients/norm", grad_norm.item(), global_step)
                writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)

            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / len(train_loader)
            throughput = len(train_loader) * batch_size / epoch_time

            # TensorBoard epoch metrics
            writer.add_scalar("loss/epoch", avg_epoch_loss, epoch + 1)
            writer.add_scalar("time/epoch_seconds", epoch_time, epoch + 1)
            writer.add_scalar("throughput/graphs_per_sec", throughput, epoch + 1)

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_epoch_loss,
                "config": config,
            }

            # Save latest
            torch.save(checkpoint, "checkpoints/graphaf_latest.pt")

            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(checkpoint, "checkpoints/graphaf_best.pt")
                writer.add_text(
                    "checkpoints", f"New best at epoch {epoch + 1}: {avg_epoch_loss:.4f}"
                )

            # Save periodic checkpoints
            torch.save(checkpoint, f"checkpoints/graphaf_epoch_{epoch + 1}.pt")

            # Clear GPU cache
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        writer.add_text("status", f"Training interrupted at epoch {epoch + 1}")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "checkpoints/graphaf_interrupt.pt",
        )

    except Exception as e:
        writer.add_text("errors", f"Error at epoch {epoch + 1}: {str(e)}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "checkpoints/graphaf_error.pt",
        )
        raise

    finally:
        writer.add_text("status", f"Training completed. Best loss: {best_loss:.4f}")
        writer.close()


if __name__ == "__main__":
    main()
