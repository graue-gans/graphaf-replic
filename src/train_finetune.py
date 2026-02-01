import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import PolyInfoDataset, collate_graphs
from model import GraphAF

# pyright: reportPossiblyUnboundVariable=false


def main():
    # Get project root (parent of src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        device_name = torch.cuda.get_device_name(0)

    # Create directories
    os.makedirs(os.path.join(project_root, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "runs"), exist_ok=True)

    # TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(project_root, "runs", f"graphaf_finetune_{timestamp}"))

    # Fine-tuning config
    config = {
        "device": device_name,
        "epochs": 5,  # Fewer epochs for fine-tuning
        "batch_size": 32,
        "lr": 0.0001,  # 10x lower learning rate than pretraining
        "grad_clip": 1.0,
        "pretrained_checkpoint": "graphaf_best.pt",
    }
    writer.add_text("config", str(config))

    # Initialize model
    model = GraphAF().to(device)

    # Load pretrained checkpoint
    checkpoint_path = os.path.join(project_root, "checkpoints", config["pretrained_checkpoint"])
    print(f"Loading pretrained checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")

    # Log pretrained info
    writer.add_text("pretrained/epoch", str(checkpoint["epoch"]))
    writer.add_text("pretrained/loss", f"{checkpoint['loss']:.4f}")
    writer.add_text("pretrained/checkpoint", config["pretrained_checkpoint"])

    # Optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Optionally load optimizer state (commented out - starting fresh optimizer is often better)
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    writer.add_text("model/total_params", f"{total_params:,}")

    # Hyperparameters
    num_epochs = config["epochs"]
    batch_size = config["batch_size"]
    grad_clip = config["grad_clip"]

    # PolyInfo Dataset
    dataset_path = os.path.join(project_root, "data", "Tg_SMILES_class_pid_polyinfo_median.csv")
    dataset = PolyInfoDataset(dataset_path)
    writer.add_text("data/dataset_size", str(len(dataset)))
    writer.add_text("data/dataset_type", "PolyInfo")

    # Data loader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    # Training loop
    best_loss = float("inf")
    global_step = checkpoint.get("global_step", 0)  # Continue from pretrained global step
    start_epoch = 0  # Fine-tuning starts from epoch 0

    try:
        for epoch in range(start_epoch, num_epochs):
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

            print(
                f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_epoch_loss:.4f} - Time: {epoch_time:.2f}s"
            )

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_epoch_loss,
                "config": config,
                "finetuned": True,  # Mark as fine-tuned checkpoint
            }

            # Save latest
            torch.save(
                checkpoint, os.path.join(project_root, "checkpoints", "graphaf_finetune_latest.pt")
            )

            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(
                    checkpoint,
                    os.path.join(project_root, "checkpoints", "graphaf_finetune_best.pt"),
                )
                writer.add_text(
                    "checkpoints", f"New best at epoch {epoch + 1}: {avg_epoch_loss:.4f}"
                )

            # Save periodic checkpoints
            torch.save(
                checkpoint,
                os.path.join(
                    project_root, "checkpoints", f"graphaf_finetune_epoch_{epoch + 1}.pt"
                ),
            )

            # Clear cache
            if device.type == "cuda":
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        writer.add_text("status", f"Fine-tuning interrupted at epoch {epoch + 1}")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "finetuned": True,
            },
            os.path.join(project_root, "checkpoints", "graphaf_finetune_interrupt.pt"),
        )

    except Exception as e:
        writer.add_text("errors", f"Error at epoch {epoch + 1}: {str(e)}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "finetuned": True,
            },
            os.path.join(project_root, "checkpoints", "graphaf_finetune_error.pt"),
        )
        raise

    finally:
        writer.add_text("status", f"Fine-tuning completed. Best loss: {best_loss:.4f}")
        writer.close()


if __name__ == "__main__":
    main()
