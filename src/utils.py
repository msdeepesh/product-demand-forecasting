import io
from datetime import datetime

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
