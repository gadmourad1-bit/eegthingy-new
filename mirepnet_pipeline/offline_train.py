import os
import sys
import time
import copy

import numpy as np
import mne
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def get_device(use_gpu=True):
    """Get the best available device (GPU or CPU)."""
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n{'='*60}")
        print(f"GPU Training Enabled")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"{'='*60}\n")
        return device
    else:
        print("\nUsing CPU training (set use_gpu=True to use GPU if available)\n")
        return torch.device('cpu')


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CLASSIFIER_DIR = os.path.join(REPO_ROOT, "classifier")
if CLASSIFIER_DIR not in sys.path:
    sys.path.insert(0, CLASSIFIER_DIR)

from config import TARGET_MAPPINGS, DATA_DIR, EEG_CHANNELS_TARGETS
from classifier.run import discover_files, select_files, process_data_mirepnet
from mirepnet_pipeline.model import MIRepNet
from mirepnet_pipeline.preprocess import prepare_for_mirepnet


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------
def augment_batch(X, max_shift=20, noise_std=0.03, scale_range=(0.9, 1.1),
                   channel_dropout_p=0.05, mixup_alpha=0.0, y=None,
                   n_classes=None):
    """Apply a bundle of light augmentations to a batch of EEG windows.

    X: (n, c, t) numpy array
    Returns augmented X (and, if mixup is used, soft targets).
    """
    n, c, t = X.shape
    X_aug = X.copy()

    # 1) random circular-ish time shift (zero padded at the edge, as before)
    if max_shift > 0:
        shifts = np.random.randint(-max_shift, max_shift + 1, size=n)
        for i, shift in enumerate(shifts):
            if shift == 0:
                continue
            elif shift > 0:
                X_aug[i] = np.roll(X_aug[i], shift, axis=-1)
                X_aug[i, :, :shift] = 0.0
            else:
                X_aug[i] = np.roll(X_aug[i], shift, axis=-1)
                X_aug[i, :, shift:] = 0.0

    # 2) Gaussian noise
    if noise_std > 0:
        X_aug = X_aug + np.random.randn(*X_aug.shape).astype(np.float32) * noise_std

    # 3) random per-trial amplitude scaling
    if scale_range is not None:
        scales = np.random.uniform(scale_range[0], scale_range[1], size=(n, 1, 1)).astype(np.float32)
        X_aug = X_aug * scales

    # 4) random channel dropout (zero out a few channels per trial)
    if channel_dropout_p > 0:
        mask = (np.random.rand(n, c) > channel_dropout_p).astype(np.float32)
        X_aug = X_aug * mask[:, :, None]

    y_soft = None
    if mixup_alpha > 0 and y is not None and n_classes is not None:
        lam = np.random.beta(mixup_alpha, mixup_alpha, size=n).astype(np.float32)
        perm = np.random.permutation(n)
        X_mix = lam[:, None, None] * X_aug + (1 - lam[:, None, None]) * X_aug[perm]

        y_onehot = np.eye(n_classes, dtype=np.float32)[y]
        y_soft = lam[:, None] * y_onehot + (1 - lam[:, None]) * y_onehot[perm]
        X_aug = X_mix

    return X_aug.astype(np.float32), y_soft


# --------------------------------------------------------------------------
# Fine-tuning
# --------------------------------------------------------------------------
def fine_tune_mirepnet(
    train_files,
    epochs=60,
    batch_size=32,
    head_lr=5e-4,
    backbone_lr=5e-5,
    weight_decay=1e-2,
    val_ratio=0.2,
    early_stop_patience=15,
    use_augment=True,
    mixup_alpha=0.2,
    label_smoothing=0.05,
    freeze_encoder=True,
    unfreeze_last_n_blocks=0,
    warmup_epochs=5,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """Fine-tune MIRepNet on the selected recordings.

    Key differences from the original script:
    - Defaults to freezing the pretrained backbone and only training the
      classification head, since fine-tuning a full transformer on a
      few hundred trials tends to overfit / destroy pretrained features.
      Set freeze_encoder=False (or unfreeze_last_n_blocks>0) to relax this.
    - Discriminative learning rates: a bigger LR for the head, a much
      smaller LR for any unfrozen backbone parameters.
    - Cosine LR schedule with linear warmup instead of a flat LR.
    - Richer augmentation (noise, amplitude scaling, channel dropout,
      optional mixup) instead of just time-shift.
    - Early stopping patience is actually reachable (epochs > patience
      by default), and the best checkpoint (by val loss, not just val
      acc) is restored at the end.
    - Weight decay and label smoothing added for regularization.

    Parameters
    ----------
    device : str
        'cuda' for GPU training or 'cpu' for CPU training.
    """

    all_X, all_y = [], []
    for f in train_files:
        X, y = process_data_mirepnet(
            f, TARGET_MAPPINGS,
            tmin=0.5, tmax=2.5,
            apply_car=True, apply_laplacian=False,
        )
        all_X.append(X)
        all_y.append(y)
        print(f"  {os.path.basename(f)}: {len(y)} epochs")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"\nTotal training windows: {X.shape[0]}")
    classes, counts = np.unique(y, return_counts=True)
    print(f"Classes: {classes}")
    print(f"Counts:  {counts}")
    if len(classes) < 2:
        raise ValueError("Only one class found. Check your annotations.")
    if X.shape[0] < 150:
        print(f"WARNING: only {X.shape[0]} trials total. With this little data, "
              f"strongly prefer freeze_encoder=True and heavier augmentation/regularization. "
              f"Also consider whether all trials came from a single session — session-to-session "
              f"drift is a very common cause of good CV / poor held-out-test performance in MI-EEG.")

    label_map = {old: new for new, old in enumerate(classes)}
    y_mapped = np.array([label_map[yy] for yy in y])
    n_classes = len(classes)

    raw_tmp = mne.io.read_raw_fif(train_files[0], preload=False)
    sfreq = float(raw_tmp.info['sfreq'])
    print(f"Source sfreq: {sfreq} Hz")
    print(f"Native channel count: {len(EEG_CHANNELS_TARGETS)}")

    # Preprocess: resample to 250 Hz, Euclidean Alignment, normalization, no 45-ch interpolation
    X_prep, W, ea_whitener = prepare_for_mirepnet(
        X,
        EEG_CHANNELS_TARGETS,
        sfreq,
        fit_ea=True,
        project_to_template=False,  # keep native 15 channels
    )
    print(f"Preprocessed shape: {X_prep.shape}")
    print(f"Preprocessed data range: min={X_prep.min():.4f}, max={X_prep.max():.4f}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_prep, y_mapped, test_size=val_ratio, random_state=42, stratify=y_mapped
    )
    print(f"Train set: {X_train.shape[0]}, Validation set: {X_val.shape[0]}")

    pretrain_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'MIRepNet', 'weight', 'MIRepNet.pth'
    )
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(f"Pretrained weight not found at {pretrain_path}.")

    num_channels = len(EEG_CHANNELS_TARGETS)  # 15
    model = MIRepNet(
        pretrain_path=pretrain_path,
        n_classes=n_classes,
        num_channels=num_channels,
    )

    # ---- Freezing strategy -------------------------------------------------
    # Default: freeze everything except the classification head. Optionally
    # unfreeze the last N transformer blocks for a lighter form of full
    # fine-tuning once the head-only model is working.
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if 'clshead' in name:
            param.requires_grad = True
            head_params.append(param)
        else:
            param.requires_grad = False  # will be overridden if unfreezing

    if not freeze_encoder:
        # Unfreeze all backbone params
        for name, param in model.named_parameters():
            if 'clshead' not in name:
                param.requires_grad = True
                backbone_params.append(param)
    elif unfreeze_last_n_blocks > 0:
        # Unfreeze only the last N transformer blocks
        block_ids = set()
        for name, _ in model.named_parameters():
            if 'clshead' in name:
                continue
            # Extract block index from parameter names like
            # transformer.blocks.3.0.fn.0.norm1.weight
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part.isdigit():
                    block_ids.add(int(part))
        if block_ids:
            keep_blocks = sorted(block_ids)[-unfreeze_last_n_blocks:]
            for name, param in model.named_parameters():
                if 'clshead' in name:
                    continue
                parts = name.split('.')
                block_index = None
                for part in parts:
                    if part.isdigit():
                        block_index = int(part)
                        break  # use first numeric token (block index)
                if block_index in keep_blocks:
                    param.requires_grad = True
                    backbone_params.append(param)

    trainable_params = head_params + backbone_params
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,} "
          f"(head: {sum(p.numel() for p in head_params):,}, "
          f"backbone: {sum(p.numel() for p in backbone_params):,})")

    model = model.to(device)
    print(f"Model moved to device: {device}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Discriminative learning rates: head trains faster than backbone.
    param_groups = [{'params': head_params, 'lr': head_lr}]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': backbone_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    hard_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def soft_ce(logits, y_soft_t):
        log_probs = torch.log_softmax(logits, dim=1)
        return -(y_soft_t * log_probs).sum(dim=1).mean()

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None
    global_step = 0

    print(f"\nFine-tuning for up to {epochs} epochs "
          f"(head_lr={head_lr}, backbone_lr={backbone_lr}, batch_size={batch_size}, "
          f"freeze_encoder={freeze_encoder}, unfreeze_last_n_blocks={unfreeze_last_n_blocks}, "
          f"aug={use_augment}, mixup_alpha={mixup_alpha})...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for xb, yb in train_loader:
            xb_np = xb.cpu().numpy()
            yb_np = yb.cpu().numpy()

            y_soft_np = None
            if use_augment:
                xb_np, y_soft_np = augment_batch(
                    xb_np,
                    max_shift=20,
                    noise_std=0.03,
                    scale_range=(0.9, 1.1),
                    channel_dropout_p=0.05,
                    mixup_alpha=mixup_alpha,
                    y=yb_np,
                    n_classes=n_classes,
                )

            xb = torch.tensor(xb_np, dtype=torch.float32).to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            _, logits = model(xb)

            if y_soft_np is not None:
                y_soft_t = torch.tensor(y_soft_np, dtype=torch.float32).to(device)
                loss = soft_ce(logits, y_soft_t)
            else:
                loss = hard_criterion(logits, yb)

            if not torch.isfinite(loss):
                print(f"  WARNING: non-finite loss at epoch {epoch}, batch skipped.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            running_loss += loss.item() * xb.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / total if total > 0 else 0.0
        avg_loss = running_loss / total if total > 0 else float('inf')

        model.eval()
        with torch.no_grad():
            _, val_logits = model(X_val_t)
            val_loss = hard_criterion(val_logits, y_val_t).item()
            _, val_preds = torch.max(val_logits, 1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        cur_lr_head = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
              f"train_acc={train_acc:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr_head={cur_lr_head:.2e}  ({time.time()-t0:.1f}s)")

        # Track best by val_loss (more stable signal than acc on small val sets)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val_loss improvement for {early_stop_patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best val_loss={best_val_loss:.4f}  (val_acc at that point: {best_val_acc:.4f})")

    return model


def main():
    files = discover_files()
    if not files:
        print(f"No .fif files found in {DATA_DIR}")
        return

    print("\nAvailable training files:")
    for i, f in enumerate(files, 1):
        print(f"  {i}) {os.path.basename(f)}")
    print()

    train_files = select_files("training files for fine-tuning (e.g. 1,2,3): ", files)
    if not train_files:
        print("No files selected.")
        return

    print(f"\nFine-tuning on {len(train_files)} file(s):")
    device = get_device(use_gpu=True)

    # Stage 1: head-only fine-tuning (safe default for small datasets).
    model = fine_tune_mirepnet(
        train_files,
        epochs=60,
        batch_size=32,
        head_lr=5e-4,
        backbone_lr=5e-5,
        weight_decay=1e-2,
        early_stop_patience=15,
        use_augment=True,
        mixup_alpha=0.2,
        label_smoothing=0.05,
        freeze_encoder=True,
        unfreeze_last_n_blocks=0,
        warmup_epochs=5,
        device=device,
    )
    if model is None:
        print("Training failed. Exiting.")
        return

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "MIRepNet_finetuned.pth")
    torch.save(model.state_dict(), out_path)
    print(f"\nFine-tuned model saved to: {out_path}")


if __name__ == "__main__":
    main()