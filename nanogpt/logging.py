import logging
import os
import platform
import subprocess
import uuid
import sys

import torch

INCLUDE_EXTENSIONS = [".py"]
INCLUDE_DIRS = ["nanogpt"]
EXCLUDE_DIRS = ["__pycache__"]


class DistributedLogger:
    def __init__(self, name: str | None = None, log_dir: str = "./logs"):
        self.name = name
        self.log_dir = log_dir

        self.rank = int(os.getenv("RANK", 0))

        os.makedirs(self.log_dir, exist_ok=True)
        self.logfile = os.path.join(self.log_dir, f"{self.name + '_' if self.name else ''}{uuid.uuid4()}.txt")

        self.logger = logging.getLogger("nanogpt")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            file_handler = logging.FileHandler(self.logfile, mode="a")
            file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            self.logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            console_handler.setLevel(logging.CRITICAL)  # Disable console logging by default
            self.logger.addHandler(console_handler)

        if self.rank != 0:
            self.logger.disabled = True
        else:
            print(self.logfile)

    def log(self, message: str, print_to_console: bool = False):
        if self.rank == 0:
            if print_to_console:
                for handler in self.logger.handlers:
                    if isinstance(handler, logging.StreamHandler):
                        handler.setLevel(logging.INFO)
                self.logger.info(message)
                for handler in self.logger.handlers:
                    if isinstance(handler, logging.StreamHandler):
                        handler.setLevel(logging.CRITICAL)
            else:
                self.logger.info(message)


def generate_file_tree(base_dir, include_extensions=None, include_dirs=None, exclude_dirs=None):
    """Create file tree for run logging purposes.

    Note: this is only because not all machines may have `tree` installed.

    Args:
        base_dir (str):
            Base directory to start generating the tree from.

        include_extensions (list[str]):
            List of file extensions to include in the tree.

        include_dirs (list[str]):
            List of directory names or relative paths to include.

        exclude_dirs (list[str]):
            List of directory names or relative paths to exclude.

    Returns:
        str:
            Plain text representation of the file tree.
    """

    from rich.console import Console
    from rich.text import Text
    from rich.tree import Tree

    if include_extensions is None:
        include_extensions = INCLUDE_EXTENSIONS
    if include_dirs is None:
        include_dirs = INCLUDE_DIRS
    if exclude_dirs is None:
        exclude_dirs = EXCLUDE_DIRS

    def build_tree(path, tree_node):
        for entry in os.scandir(path):
            relative_path = os.path.relpath(entry.path, base_dir)
            if not any(relative_path.startswith(include_dir) for include_dir in include_dirs):
                continue
            if entry.name in exclude_dirs or any(exclude_dir in relative_path.split(os.sep) for exclude_dir in exclude_dirs):
                continue
            if entry.is_dir():
                subdir_node = tree_node.add(Text(f"{entry.name}/"))
                build_tree(entry.path, subdir_node)
            elif entry.is_file() and any(entry.name.endswith(ext) for ext in include_extensions):
                tree_node.add(Text(entry.name))

    tree = Tree(Text(f"{os.path.basename(base_dir)}/"))
    build_tree(base_dir, tree)

    console = Console(record=True)
    # To prevent rich from logging to terminal, only to logfile.
    with open(os.devnull, "w") as devnull:
        original_stdout = sys.stdout
        sys.stdout = devnull
        try:
            console.print(tree)
        finally:
            sys.stdout = original_stdout
    return console.export_text()


def collect_code_snapshot(base_dir, include_extensions=None, include_dirs=None, exclude_dirs=None) -> str:
    """Log all code for a run.

    Args:
        base_dir (str):
            Base directory to start collecting files from.

        include_extensions (list[str]):
            List of file extensions to include in the snapshot.

        include_dirs (list[str]):
            List of directory names or relative paths to include.

        exclude_dirs (list[str]):
            List of directory names or relative paths to exclude.

    Returns:
        str:
            String containing the snapshot of the code.
    """
    if include_extensions is None:
        include_extensions = INCLUDE_EXTENSIONS
    if include_dirs is None:
        include_dirs = INCLUDE_DIRS
    if exclude_dirs is None:
        exclude_dirs = EXCLUDE_DIRS

    snapshot = []

    file_tree = generate_file_tree(base_dir, include_extensions=include_extensions, include_dirs=include_dirs, exclude_dirs=exclude_dirs)
    snapshot.append(f"{'#' * 80}\n## File Tree\n{'#' * 80}\n")
    snapshot.append(file_tree)
    snapshot.append("\n\n")

    for root, _, files in os.walk(base_dir):
        relative_root = os.path.relpath(root, base_dir)

        if any(exclude_dir in relative_root.split(os.sep) for exclude_dir in exclude_dirs):
            continue

        if not any(relative_root.startswith(include_dir) for include_dir in include_dirs):
            continue

        for file in files:
            if any(file.endswith(ext) for ext in include_extensions):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, base_dir)

                snapshot.append(f"{'#' * 80}\n## {relative_path}\n{'#' * 80}\n")

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        snapshot.append(f.read())
                except Exception as e:
                    snapshot.append(f"Error reading file: {e}\n")

                snapshot.append("\n\n")

    return "".join(snapshot)


def log_system_info():
    """
    Gathers detailed system information using PyTorch CUDA utilities and nvidia-smi for GPU info.

    Returns:
        str: A formatted string containing system information.
    """
    # Create a table for better formatting
    from rich.console import Console
    from rich.table import Table

    table = Table(title="System Information", show_header=True)
    # Add system information to the table
    table.add_row("System", platform.system())
    table.add_row("Release", platform.release())
    table.add_row("Version", platform.version())
    table.add_row("Python Version", sys.version)
    table.add_row("Torch Version", torch.version.__version__)
    table.add_row("CUDA", torch.version.cuda)

    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
        table.add_row("nvidia-smi Output", nvidia_smi_output.strip())
    except FileNotFoundError:
        table.add_row("nvidia-smi Output", "nvidia-smi not found. Ensure NVIDIA drivers are installed.")
    except Exception as e:
        table.add_row("nvidia-smi Output", f"Error running nvidia-smi: {e}")

    console = Console(record=True)

    # To prevent rich from logging to terminal, only to logfile.
    with open(os.devnull, "w") as devnull:
        original_stdout = sys.stdout
        sys.stdout = devnull
        try:
            console.print(table)
        finally:
            sys.stdout = original_stdout

    return console.export_text()
