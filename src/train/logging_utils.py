from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from utils.config import PATHS, CLASSES


def get_writer(run_name: str) -> SummaryWriter:
    log_dir = PATHS["runs"] / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(log_dir))


def log_train_step(writer, loss, step, lr=None):
    writer.add_scalar("train/loss", loss, step)
    if lr is not None:
        writer.add_scalar("train/lr", lr, step)


def log_val_epoch(writer, results, epoch):
    """
    results : dict from BEVIoUMetric.compute()
              e.g. {"drivable_area": 0.4, "vehicle": 0.2, "mIoU": 0.3}
    """
    writer.add_scalar("val/mIoU", results["mIoU"], epoch)
    for name in CLASSES["names"]:
        if name in results:
            writer.add_scalar(f"val/{name}_iou", results[name], epoch)


def print_epoch(epoch, total_epochs, loss, results):
    print(f"Epoch {epoch}/{total_epochs}: "
          f"loss={loss:.4f}  val_mIoU={results['mIoU']:.4f}")
    for name in CLASSES["names"]:
        print(f"  {name}: {results.get(name, 0.0):.4f}")