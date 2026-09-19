"""Laptop-friendly Alpha profile for roughly 6GB VRAM."""
import sys
from train_alpha import main

DEFAULTS = [
    "--vocab-size", "16000", "--block-size", "512", "--dim", "384",
    "--heads", "6", "--layers", "8", "--micro-batch-size", "1",
    "--grad-accum", "32", "--checkpoint-dir", "alpha-laptop",
]

if __name__ == "__main__":
    sys.argv[1:1] = DEFAULTS
    main()

