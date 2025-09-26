from pathlib import Path

def find_root():
    return str(Path(__file__).resolve().parent.parent.parent)