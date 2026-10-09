import io
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


def confirm_setup(message):
    setup_confirmed = input(message).strip().lower()
    if setup_confirmed not in {"yes", "y"}:
        open_file("text_files/Pre-requisites.txt")
        raise SystemExit(0)


def open_file(file_name):
    os.startfile(str(Path.cwd() / file_name))


def printm(msg):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    if not isinstance(msg, str):
        msg_str = str(msg)
    else:
        msg_str = msg
        
    border = "-" * (max(len(line) for line in msg_str.split('\n')) + 50 if msg_str else 50)
    border = border[:120]
    
    print(f"\n{border}")
    print(f"[{current_time}]\n{msg_str} ")
    print(f"{border}\n")

def print_info(msg, df):
    """Helper function to capture df.info() output and send it to printm."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    printm(f'{msg} {buffer.getvalue()}')


def load_dataset():
    repo_root = Path(__file__).resolve().parents[3]
    csv_path = repo_root / 'data' / 'demand_forecasting.csv'
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"1)Error: The file at {csv_path} was not found. Please update the path.")
        return None

    print("\n1)File loaded successfully!")
    return df
