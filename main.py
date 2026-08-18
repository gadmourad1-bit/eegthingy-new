import os
import ctypes
import shlex
import subprocess
import sys
from platform import system


def run_collector():
    os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    if system() == "Windows":
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    from gui.window import Window
    Window().start()


def run_classifier():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier", "run.py")
    subprocess.run([sys.executable, script])


def run_simulation():
    sim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation")
    src_dir = os.path.join(sim_dir, "src")
    try:
        subject = input("subject id> ").strip()
        test = input("test id> ").strip()
        print("simulation args (blank prints the sim's help):")
        print("  --ws-url URL              decision server (default ws://127.0.0.1:8765)")
        print("  --no-ws                   disable the websocket client")
        print("  --no-manual-keys          disable arrow-key decisions")
        print("  --view {third,first}      camera (default third)")
        raw = input("args> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    id_args = []
    if subject:
        id_args += ["--subject", subject]
    if test:
        id_args += ["--test", test]

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src_dir, env.get("PYTHONPATH")) if p)
    subprocess.run([sys.executable, "-m", "tiago_maze", *shlex.split(raw), *id_args],
                   cwd=sim_dir, env=env)


def run_mirepnet_training():
    """Run MIRepNet training with proper feedback and error handling."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mirepnet_pipeline", "offline_train.py")
    
    print("\n" + "=" * 60)
    print("MIRepNet Training")
    print("=" * 60)
    print("\nImportant notes:")
    print("  - Data should be balanced (equal left/right hand trials)")
    print("  - Training requires at least 2 classes")
    print("  - Use multiple training files for better generalization")
    print("  - After training, evaluate with Classifier option 2")
    print()
    
    try:
        result = subprocess.run([sys.executable, script], check=False)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("✅ Training completed successfully!")
            print("=" * 60)
            print("\nNext steps:")
            print("  1. Run Classifier (option 2) to evaluate the model")
            print("  2. Select option 3 (MIRepNet) when prompted for decoder")
            print("  3. Use different files for training vs testing")
            print()
        else:
            print("\n" + "=" * 60)
            print("❌ Training failed with error code:", result.returncode)
            print("=" * 60)
            print("\nCheck the error messages above for details.")
            print("Common issues:")
            print("  - No .fif files found in ./data directory")
            print("  - Only 1 class in training data")
            print("  - Pretrained weights not found")
            print()
            
    except FileNotFoundError:
        print("\n❌ Error: Could not find training script at:")
        print(f"   {script}")
        print("\nMake sure you're in the project root directory.")
        
    except Exception as e:
        print(f"\n❌ Unexpected error during training: {e}")


def run_mirepnet_evaluation():
    """Quick evaluation of the fine-tuned model."""
    import numpy as np
    import mne
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, REPO_ROOT)
    
    from config import TARGET_MAPPINGS, DATA_DIR, EEG_CHANNELS_TARGETS
    from classifier.run import discover_files, select_files, process_data_mirepnet
    from mirepnet_pipeline.inference import MIRepNetDecoder
    
    print("\n" + "=" * 60)
    print("MIRepNet Quick Evaluation")
    print("=" * 60)
    
    files = discover_files()
    if not files:
        print(f"\n❌ No .fif files found in {DATA_DIR}")
        return
    
    print("\nAvailable files:")
    for i, f in enumerate(files, 1):
        print(f"  {i}) {os.path.basename(f)}")
    
    train_files = select_files("\nSelect training files (e.g., 1,2,3): ", files)
    if not train_files:
        return
    
    # Load data
    print("\nLoading training data...")
    all_X, all_y = [], []
    for f in train_files:
        X, y = process_data_mirepnet(f, TARGET_MAPPINGS, tmin=0.5, tmax=1.5)
        all_X.append(X)
        all_y.append(y)
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"\nTotal epochs: {len(y)}")
    classes, counts = np.unique(y, return_counts=True)
    print(f"Classes: {classes}")
    print(f"Counts: {counts}")
    
    if len(classes) < 2:
        print("\n❌ Need at least 2 classes for evaluation!")
        return
    
    # Get sampling frequency
    raw_tmp = mne.io.read_raw_fif(train_files[0], preload=False)
    sfreq = float(raw_tmp.info['sfreq'])
    
    # Initialize decoder
    finetuned_path = os.path.join(REPO_ROOT, "mirepnet_pipeline", "weights", "MIRepNet_finetuned.pth")
    if not os.path.exists(finetuned_path):
        print(f"\n❌ Fine-tuned model not found at: {finetuned_path}")
        print("Run training first (option 4)!")
        return
    
    print(f"\nLoading fine-tuned model from: {finetuned_path}")
    decoder = MIRepNetDecoder(sfreq=sfreq, pretrain_path=finetuned_path)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_acc = []
    
    print("\nRunning 5-fold cross-validation...")
    for fold, (tr_i, va_i) in enumerate(cv.split(X, y), 1):
        decoder.fit(X[tr_i], y[tr_i])
        y_pred = decoder.predict(X[va_i])
        acc = accuracy_score(y[va_i], y_pred)
        fold_acc.append(acc)
        print(f"  Fold {fold}: {acc*100:.1f}%")
    
    fold_acc = np.array(fold_acc)
    print(f"\nMean CV accuracy: {fold_acc.mean()*100:.2f}% ± {fold_acc.std()*100:.2f}%")
    
    if fold_acc.mean() < 0.6:
        print("\n⚠️  Accuracy is below 60%. Consider:")
        print("  - Using more training data")
        print("  - Training with different files")
        print("  - Checking data quality")
    else:
        print("\n✅ Model performance looks good!")


def menu():
    print("\n" + "=" * 35)
    print(" 1) Data Collector")
    print(" 2) Classifier")
    print(" 3) Robot Simulation")
    print(" 4) Train MIRepNet Classifier")
    print(" 5) Quick Evaluate MIRepNet")
    print(" q) Quit")
    print("=" * 35)
    return input("> ").strip().lower()


def main():
    while True:
        try:
            choice = menu()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        
        if choice in ("1", "collector", "data"):
            run_collector()
            return
        
        if choice in ("2", "classifier"):
            run_classifier()
            return
        
        if choice in ("3", "simulation", "sim"):
            run_simulation()
            return
        
        if choice in ("4", "train", "mirepnet"):
            run_mirepnet_training()
            return
        
        if choice in ("5", "eval", "evaluate"):
            run_mirepnet_evaluation()
            return
        
        if choice in ("q", "quit", "exit"):
            return
        
        print("invalid choice\n")


if __name__ == "__main__":
    main()