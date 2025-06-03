import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def parse_log_file(filepath):
    """Parse a log file and extract step, val_loss, train_time, and step_avg from lines containing val_loss."""
    data = {'step': [], 'val_loss': [], 'train_time': [], 'step_avg': []}
    # Regex to match lines with val_loss
    pattern = r'step:(\d+)/\d+\s+val_loss:([\d.]+)\s+train_time:(\d+)ms\s+step_avg:([\d.]+)ms'
    
    with open(filepath, 'r') as f:
        for line in f:
            match = re.match(pattern, line.strip())
            if match:
                data['step'].append(int(match.group(1)))
                data['val_loss'].append(float(match.group(2)))
                data['train_time'].append(int(match.group(3)))
                data['step_avg'].append(float(match.group(4)))
    
    return data

def main(logdir):
    """Process all log files in the directory and create a train_time vs val_loss plot."""
    # Ensure the directory exists
    if not os.path.isdir(logdir):
        print(f"Error: {logdir} is not a valid directory")
        sys.exit(1)
    
    plt.figure(figsize=(10, 6))
    
    # Process each file in the directory
    for filename in os.listdir(logdir):
        if filename.endswith('.txt'):  # Adjust extension if needed
            filepath = os.path.join(logdir, filename)
            data = parse_log_file(filepath)
            if data['train_time']:  # Only plot if data was found
                plt.plot([x/1000 for x in data['train_time']], data['val_loss'], label=filename)
    
    # Set up the plot
    plt.xlabel('Wall Time (s)')
    plt.ylabel('Validation Loss')
    plt.title('Wall Time vs Validation Loss')
    plt.ylim(3.27, 6)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(logdir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_path = os.path.join(output_dir, f'{timestamp}.png')
    
    # Save the plot
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <logdir>")
        sys.exit(1)
    
    logdir = sys.argv[1]
    main(logdir)