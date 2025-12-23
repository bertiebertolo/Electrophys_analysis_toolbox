#!/usr/bin/env python3
"""
Mosquito Frequency Range Analysis - GUI Application

Comprehensive tool for analyzing mechanoreceptor frequency tuning characteristics
from mosquito electrophysiology recordings.

FEATURES
========

1. FREQUENCY ANALYSIS
   - High-resolution power spectral density analysis (cubic spline interpolation to 1 Hz)
   - Automatic tuning curve feature extraction (peak frequency, bandwidth, Q-factor)
   - Adjustable threshold detection (std dev, percentile, or absolute methods)
   - Interactive visualisation with hover tooltips showing exact frequency values
   - Export high-quality plots (PNG, PDF, SVG, JPEG at 300 DPI)

2. SMR FILE MANAGEMENT
   - Transfer: Copy .smr files from source to destination (flat or preserve structure)
   - Convert: Transform Spike2 .smr files to WAV format (raw data extraction)
   - Smart species detection and automatic folder organization
   - Direct conversion without filtering or normalisation
   - 32-bit floating-point WAV output

3. BATCH PROCESSING
   - Process entire folders of WAV files with configurable sampling rate
   - Multi-threaded execution keeps GUI responsive
   - Progress tracking with real-time status updates
   - Automatic CSV export with comprehensive tuning features
   - Summary statistics by species and direction

4. INTERACTIVE VISUALISATION
   - Dual-panel plots (dB and linear scales)
   - Colour-coded annotations (red=peak, orange=bandwidth, blue=baseline)
   - Hover tooltips display exact frequency and power values
   - Click for detailed frequency information
   - Save plots in multiple formats

TECHNICAL DETAILS
==================
- Sampling Rate: Adjustable 10-200 kHz (default 100 kHz)
- Frequency Resolution: Exact 1 Hz after cubic spline interpolation
- PSD Method: Welch's method with nperseg=4096 for high resolution
- Bandwidth: Primary measure is -3dB (half-power), secondary -10dB
- Q-factor: peak_frequency / bandwidth (measure of tuning sharpness)

Authors: Mosquito Electrophysiology Lab
Version: 2.0
"""

import os
import sys
import shutil
from pathlib import Path
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import seaborn as sns
import librosa
try:
    from neo.io import CedIO, Spike2IO
    NEO_AVAILABLE = True
except ImportError:
    CedIO = Spike2IO = None
    NEO_AVAILABLE = False
import soundfile as sf
from datetime import datetime

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# -------------------------
# Helper functions (transfer + convert)
# -------------------------
def transfer_smr_files(
    source_dir,
    dest_dir,
    flatten=True,
    overwrite=False,
    status_callback=None,
    progress_callback=None,
):
    """
    Recursively copy all .smr files from source directory tree to destination.
    
    Use Cases:
    ----------
    - Consolidate .smr files from multiple experiment folders
    - Backup raw electrophysiology data
    - Organize files before conversion
    
    Args:
        source_dir (str): Root directory to search recursively for .smr files
        dest_dir (str): Destination directory (created if doesn't exist)
        flatten (bool): 
            True = Copy all files into dest_dir root (flat structure)
            False = Preserve relative folder structure from source
        overwrite (bool): 
            True = Overwrite existing files
            False = Add numeric suffix to avoid overwriting (file_1.smr, file_2.smr, ...)
        status_callback (callable): Function(str) for logging status messages
        progress_callback (callable): Function(float) receiving progress 0.0-1.0
    
    Returns:
        dict: {
            "copied": list of successfully copied file paths,
            "errors": list of (filepath, error_message) tuples,
            "total_found": int count of .smr files found
        }
    """
    def emit_status(m):
        if status_callback:
            status_callback(m)
        else:
            print(m)

    src = Path(source_dir)
    dst = Path(dest_dir)
    if not src.is_dir():
        emit_status(f"[!] Source directory not found: {source_dir}")
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    dst.mkdir(parents=True, exist_ok=True)
    smr_files = list(src.rglob("*.smr"))
    total = len(smr_files)
    if total == 0:
        emit_status(f"[!] No .smr files found in {source_dir}")
        return {"copied": [], "errors": [], "total_found": 0}

    emit_status(f"[+] Found {total} .smr files to copy...")

    copied = []
    errors = []

    for i, src_path in enumerate(smr_files):
        try:
            if flatten:
                dest_path = dst / src_path.name
            else:
                rel = src_path.relative_to(src)
                dest_path = dst / rel
                dest_path.parent.mkdir(parents=True, exist_ok=True)

            if dest_path.exists():
                if overwrite:
                    shutil.copy2(src_path, dest_path)
                    copied.append(str(dest_path))
                    emit_status(f"[+] Overwritten: {dest_path.name}")
                else:
                    stem = dest_path.stem
                    suf = dest_path.suffix
                    counter = 1
                    new_dest = dest_path
                    while new_dest.exists():
                        new_dest = dest_path.with_name(f"{stem}_{counter}{suf}")
                        counter += 1
                    shutil.copy2(src_path, new_dest)
                    copied.append(str(new_dest))
                    emit_status(f"[+] Copied with suffix: {new_dest.name}")
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
                copied.append(str(dest_path))
                emit_status(f"[+] Copied: {dest_path.name}")

        except Exception as e:
            errors.append((str(src_path), str(e)))
            emit_status(f"[!] Error copying {src_path.name}: {e}")

        if progress_callback:
            progress_callback((i + 1) / total)

    final_count = len(list(Path(dest_dir).rglob("*.smr"))) if not flatten else len(list(Path(dest_dir).glob("*.smr")))
    emit_status(f"[✓] Transfer complete! {len(copied)} files copied to {dest_dir}/ (verification: {final_count})")
    return {"copied": copied, "errors": errors, "total_found": total}


def convert_smr_to_wav_no_default_species(
    input_dir,
    output_root,
    channel_indices=None,
    segment_indices=None,
    target_sampling_rate=None,
    status_callback=None,
    progress_callback=None,
    min_freq=0,
    max_freq=1000,
    filter_order=4,
    directionality_settings=None,
):
    """
    Convert Spike2 .smr electrophysiology files to WAV format with Butterworth filtering & normalization.
    
    Process:
    --------
    1. Recursively scans input_dir for all .smr files
    2. Reads analog signal data using Neo's CedIO reader
    3. Applies Butterworth bandpass filter (min_freq to max_freq)
    4. Normalizes filtered data to max absolute value
    5. Optionally resamples to target sampling rate
    6. Saves as 32-bit float WAV files
    7. Organizes output by species subfolder structure
    
    Species Detection:
    ------------------
    - Files in species subfolders → output to species subfolder
    - Files in root → defaults to "aedes aegypti"
    - Example: input/Culex/file.smr → output/Culex/file.wav
    
    Args:
        input_dir (str): Source directory containing .smr files
        output_root (str): Destination directory for WAV files
        channel_indices (list): List of channel indices to extract (default [0])
        segment_indices (list): List of segment indices to extract (default [0])
        target_sampling_rate (int): Optional target sampling rate (if None, uses original)
        status_callback (callable): Function for status messages
        progress_callback (callable): Function receiving progress 0.0-1.0
        min_freq (float): Minimum frequency for bandpass filter (Hz), default 0
        max_freq (float): Maximum frequency for bandpass filter (Hz), default 1000
        filter_order (int): Butterworth filter order, default 4
    
    Returns:
        dict: {
            "species_counts": dict of species -> file count,
            "out_files_by_species": dict of species -> [output paths],
            "errors": list of failed file paths,
            "total_files": int total processed
        }
    """
    if not NEO_AVAILABLE:
        raise ImportError("neo is required for SMR conversion; install neo and sonpy or skip conversion.")

    # Default to channel 0, segment 0 if not specified
    if channel_indices is None:
        channel_indices = [0]
    if segment_indices is None:
        segment_indices = [0]
    
    def emit_status(m):
        if status_callback:
            status_callback(m)
        else:
            print(m)

    if not os.path.isdir(input_dir):
        emit_status(f"[!] Directory not found: {input_dir}")
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    total_files = [
        os.path.join(root, fname)
        for root, _, files in os.walk(input_dir)
        for fname in files
        if fname.lower().endswith('.smr')
    ]

    if not total_files:
        emit_status(f"[!] No .smr files found in {input_dir}")
        return {"species_counts": {}, "out_files_by_species": {}, "errors": ["No .smr files found"], "total_files": 0}

    os.makedirs(output_root, exist_ok=True)
    emit_status(f"[+] Found {len(total_files)} files to process...")

    species_groups = {}
    for file_path in total_files:
        rel_dir = os.path.relpath(os.path.dirname(file_path), input_dir)
        if rel_dir == '.' or rel_dir == '':
            species_label = "aedes aegypti"  # Default for root-level files
        else:
            parts = rel_dir.split(os.sep)
            species_label = parts[0] if parts and parts[0] != '' else "aedes aegypti"
        species_groups.setdefault(species_label, []).append(file_path)

    emit_status(f"[+] Detected species: {', '.join(sorted(species_groups.keys()))}")
    for sp_label in sorted(species_groups.keys()):
        emit_status(f"    - {sp_label}: {len(species_groups[sp_label])} files")

    errors = []
    out_files_by_species = {label: [] for label in species_groups.keys()}

    for i, full_path in enumerate(total_files):
        base = os.path.splitext(os.path.basename(full_path))[0]
        rel_dir = os.path.relpath(os.path.dirname(full_path), input_dir)
        if rel_dir == '.' or rel_dir == '':
            species_label = "aedes aegypti"
        else:
            parts = rel_dir.split(os.sep)
            species_label = parts[0] if parts and parts[0] != '' else "aedes aegypti"

        species_dir = os.path.join(output_root, species_label)
        os.makedirs(species_dir, exist_ok=True)

        try:
            # Load SMR file
            reader = CedIO(filename=full_path)
            block = reader.read_block(lazy=False)
            
            num_segments = len(block.segments)
            emit_status(f"[+] {base}: Processing {len(segment_indices)} segment(s) × {len(channel_indices)} channel(s)")
            
            # Process each selected segment and channel
            for segment_idx in segment_indices:
                if segment_idx >= num_segments:
                    emit_status(f"[!] Segment {segment_idx} out of range (max {num_segments-1})")
                    continue
                
                seg = block.segments[segment_idx]
                
                if len(seg.analogsignals) == 0:
                    emit_status(f"[!] No analog signals in segment {segment_idx}")
                    errors.append(full_path)
                    continue
                
                for channel_idx in channel_indices:
                    if channel_idx >= len(seg.analogsignals):
                        emit_status(f"[!] Channel {channel_idx} out of range (max {len(seg.analogsignals)-1})")
                        continue
                    
                    analog = seg.analogsignals[channel_idx]
                    data = np.array(analog).flatten()
                    fs = float(analog.sampling_rate)
                    
                    # Apply Butterworth bandpass filter
                    nyq = fs / 2.0
                    if min_freq <= 0:
                        b, a = signal.butter(filter_order, max_freq / nyq, btype='low')
                    else:
                        b, a = signal.butter(filter_order, 
                                            [min_freq / nyq, max_freq / nyq], 
                                            btype='band')
                    filtered = signal.filtfilt(b, a, data)
                    
                    # Handle directionality splitting or full file
                    if directionality_settings:
                        # Split into forward and backward segments BEFORE normalizing
                        forward_range = directionality_settings['forward']
                        backward_range = directionality_settings['backward']
                        
                        # Convert time to samples using original fs
                        fwd_start_idx = int(forward_range[0] * fs)
                        fwd_end_idx = int(forward_range[1] * fs)
                        bwd_start_idx = int(backward_range[0] * fs)
                        bwd_end_idx = int(backward_range[1] * fs)
                        
                        # Extract segments from filtered data
                        forward_filtered = filtered[fwd_start_idx:fwd_end_idx]
                        backward_filtered = filtered[bwd_start_idx:bwd_end_idx]
                        
                        # Normalize each segment separately to its own max absolute value
                        fwd_max = np.max(np.abs(forward_filtered))
                        forward_normalized = forward_filtered / fwd_max if fwd_max > 0 else forward_filtered
                        
                        bwd_max = np.max(np.abs(backward_filtered))
                        backward_normalized = backward_filtered / bwd_max if bwd_max > 0 else backward_filtered
                        
                        # Now resample the segments if needed
                        if target_sampling_rate is not None and target_sampling_rate != fs:
                            fwd_num_samples = int(np.round(len(forward_normalized) * target_sampling_rate / fs))
                            bwd_num_samples = int(np.round(len(backward_normalized) * target_sampling_rate / fs))
                            forward_normalized = signal.resample(forward_normalized, fwd_num_samples)
                            backward_normalized = signal.resample(backward_normalized, bwd_num_samples)
                            fs_out = target_sampling_rate
                        else:
                            fs_out = fs
                        
                        # Create subdirectories for forward/backward
                        forward_dir = os.path.join(species_dir, "forward")
                        backward_dir = os.path.join(species_dir, "backward")
                        os.makedirs(forward_dir, exist_ok=True)
                        os.makedirs(backward_dir, exist_ok=True)
                        
                        # Save forward
                        fwd_out_name = f"{base}.wav"
                        fwd_out_path = os.path.join(forward_dir, fwd_out_name)
                        sf.write(fwd_out_path, forward_normalized, int(fs_out), subtype='FLOAT')
                        emit_status(f"    ✓ {fwd_out_name} [forward] (sr={int(fs_out)} Hz, len={len(forward_normalized)} samples, std={np.std(forward_normalized):.6f})")
                        out_files_by_species[species_label].append(fwd_out_path)
                        
                        # Save backward
                        bwd_out_name = f"{base}.wav"
                        bwd_out_path = os.path.join(backward_dir, bwd_out_name)
                        sf.write(bwd_out_path, backward_normalized, int(fs_out), subtype='FLOAT')
                        emit_status(f"    ✓ {bwd_out_name} [backward] (sr={int(fs_out)} Hz, len={len(backward_normalized)} samples, std={np.std(backward_normalized):.6f})")
                        out_files_by_species[species_label].append(bwd_out_path)
                    else:
                        # Normalize the full file to its max absolute value
                        max_val = np.max(np.abs(filtered))
                        normalized = filtered / max_val if max_val > 0 else filtered
                        
                        # Optionally resample
                        if target_sampling_rate is not None and target_sampling_rate != fs:
                            num_samples = int(np.round(len(normalized) * target_sampling_rate / fs))
                            normalized = signal.resample(normalized, num_samples)
                            fs_out = target_sampling_rate
                        else:
                            fs_out = fs
                        
                        # Save full file
                        out_name = f"{base}.wav"
                        out_path = os.path.join(species_dir, out_name)
                        sf.write(out_path, normalized, int(fs_out), subtype='FLOAT')
                        emit_status(f"    ✓ {out_name} (sr={int(fs_out)} Hz, std={np.std(normalized):.6f})")
                        out_files_by_species[species_label].append(out_path)

        except Exception as e:
            emit_status(f"[!] Error processing {full_path}: {str(e)}")
            traceback.print_exc()
            errors.append(full_path)

        if progress_callback:
            progress_callback((i + 1) / len(total_files))

    emit_status("[✓] Conversion complete!")

    species_counts = {label: len(files) for label, files in out_files_by_species.items()}
    total_out = sum(species_counts.values())
    return {
        "species_counts": species_counts,
        "out_files_by_species": out_files_by_species,
        "errors": errors,
        "total_files": total_out,
    }


# -------------------------
# GUI class
# -------------------------
class FrequencyAnalysisGUI:
    """Main GUI application for frequency range analysis"""

    def __init__(self, root):
        self.root = root
        self.root.title("Mosquito Frequency Range Analysis")
        self.root.geometry("1600x1000")

        # Current working directory for browse dialogs
        self.cwd = os.getcwd()
        
        # Cache for storing complete analysis features (including response_regions)
        self.features_cache = {}  # {filename: features_dict}

        # Analysis parameters
        self.input_dir = tk.StringVar(value='Wav_data')
        self.output_dir = tk.StringVar(value='frequency_range_analysis')
        self.sampling_rate = tk.IntVar(value=100000)
        self.min_freq = tk.DoubleVar(value=200)
        self.max_freq = tk.DoubleVar(value=1000)
        self.threshold_method = tk.StringVar(value='freq_range')
        self.std_multiplier = tk.DoubleVar(value=2.0)
        self.baseline_freq_std_multiplier = tk.DoubleVar(value=2.0)
        self.percentile_threshold = tk.DoubleVar(value=95)
        self.baseline_freq_min = tk.DoubleVar(value=800)
        self.baseline_freq_max = tk.DoubleVar(value=1000)

        # Data storage
        self.current_data = None
        self.current_freqs = None
        self.current_psd = None
        self.current_features = None
        self.current_filepath = None
        self.wav_files = []
        self.results_df = None

        # Create GUI
        self.setup_gui()
        self.scan_files()

    def setup_gui(self):
        """Create the GUI layout with menu bar"""
        # Add menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Flatten button borders to avoid gray boxes
        style = ttk.Style()
        style.configure("TButton", borderwidth=0, relief="flat")
        style.map("TButton", relief=[("pressed", "sunken"), ("active", "flat")])
        style.configure("Accent.TButton", borderwidth=0, relief="flat")
        style.map("Accent.TButton", relief=[("pressed", "sunken"), ("active", "flat")])
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="SMR Tools", menu=tools_menu)
        tools_menu.add_command(label="Transfer .smr Files...", command=self.transfer_smr_dialog)
        if NEO_AVAILABLE:
            tools_menu.add_command(label="Convert .smr to WAV...", command=self.convert_smr_dialog)
        else:
            tools_menu.add_command(
                label="Convert .smr to WAV... (neo required)",
                state="disabled"
            )
        
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=0)

        # Create horizontal paned window for controls and visualisation
        paned_h = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_h.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Left panel - Controls
        self.create_control_panel(paned_h)

        # Right panel - Visualisation and results
        self.create_visualization_panel(paned_h)

        # Bottom panel - Status and log
        self.create_status_panel(main_frame)

    def create_control_panel(self, parent):
        """Create left control panel"""
        control_container = ttk.LabelFrame(parent, text="Analysis Controls", padding="5")
        parent.add(control_container, weight=0)
        control_container.columnconfigure(0, weight=1)
        control_container.rowconfigure(0, weight=1)

        # Create scrollable canvas with both vertical and horizontal scrollbars
        canvas_frame = ttk.Frame(control_container)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        # Vertical scrollbar
        scrollbar_y = ttk.Scrollbar(canvas_frame, orient="vertical")
        scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Horizontal scrollbar
        scrollbar_x = ttk.Scrollbar(canvas_frame, orient="horizontal")
        scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Canvas
        canvas = tk.Canvas(canvas_frame, highlightthickness=0, yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar_y.config(command=canvas.yview)
        scrollbar_x.config(command=canvas.xview)

        scrollable_frame = ttk.Frame(canvas)

        # Configure scroll region when frame size changes
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        scrollable_frame.bind("<Configure>", configure_scroll_region)

        # Configure canvas scroll
        canvas_frame_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # Configure canvas width to match container
        def configure_canvas_width(event):
            canvas.itemconfig(canvas_frame_id, width=event.width)
        canvas.bind('<Configure>', configure_canvas_width)

        # Enable mousewheel scrolling for vertical
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        # Enable shift+mousewheel for horizontal
        def on_shift_mousewheel(event):
            canvas.xview_scroll(int(-1*(event.delta/120)), "units")

        def bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
            canvas.bind_all("<Shift-MouseWheel>", on_shift_mousewheel)

        def unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Shift-MouseWheel>")

        canvas.bind('<Enter>', bind_to_mousewheel)
        canvas.bind('<Leave>', unbind_from_mousewheel)

        # Now use scrollable_frame as our control_frame
        control_frame = scrollable_frame

        row = 0

        # Directory settings
        ttk.Label(control_frame, text="Input Directory:", font=('', 10, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1

        # Input type selection variable (store as instance variable for batch dialog)
        dir_frame = ttk.Frame(control_frame)
        dir_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Entry(dir_frame, textvariable=self.input_dir, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_frame, text="Browse", command=self.browse_input_dir, width=8).pack(side=tk.LEFT, padx=(5, 0))
        row += 1

        ttk.Label(control_frame, text="Output Directory:").grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        ttk.Entry(control_frame, textvariable=self.output_dir, width=30).grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        row += 1

        # Frequency range (fixed)
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10)
        row += 1
        ttk.Label(control_frame, text="Analysis Parameters:", font=('', 10, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1

        param_frame = ttk.Frame(control_frame)
        param_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(param_frame, text="Sampling Rate:").pack(side=tk.LEFT)
        ttk.Spinbox(param_frame, from_=10000, to=200000, increment=10000,
                   textvariable=self.sampling_rate, width=10).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(param_frame, text="Hz").pack(side=tk.LEFT, padx=(5, 0))
        row += 1

        freq_frame = ttk.Frame(control_frame)
        freq_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Label(freq_frame, text="Analysis Range:").pack(side=tk.LEFT)
        ttk.Spinbox(freq_frame, from_=0, to=1000, increment=10,
                   textvariable=self.min_freq, width=6).pack(side=tk.LEFT, padx=(5, 2))
        ttk.Label(freq_frame, text="-").pack(side=tk.LEFT)
        ttk.Spinbox(freq_frame, from_=0, to=1000, increment=10,
                   textvariable=self.max_freq, width=6).pack(side=tk.LEFT, padx=(2, 2))
        ttk.Label(freq_frame, text="Hz").pack(side=tk.LEFT)
        row += 1

        # Threshold settings (adjustable)
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10)
        row += 1
        ttk.Label(control_frame, text="Threshold Settings:", font=('', 10, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Label(control_frame, text="Detection Method:").grid(row=row, column=0, sticky=tk.W, pady=(0, 2))
        row += 1
        method_frame = ttk.Frame(control_frame)
        method_frame.grid(row=row, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Radiobutton(method_frame, text="Freq Range", variable=self.threshold_method,
                       value='freq_range', command=self.update_threshold_controls).pack(side=tk.LEFT)
        ttk.Radiobutton(method_frame, text="Std Dev", variable=self.threshold_method,
                       value='std', command=self.update_threshold_controls).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(method_frame, text="Percentile", variable=self.threshold_method,
                       value='percentile', command=self.update_threshold_controls).pack(side=tk.LEFT, padx=10)
        row += 1

        # Frequency range baseline
        freq_base_frame = ttk.Frame(control_frame)
        freq_base_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(freq_base_frame, text="Baseline Range:").pack(side=tk.LEFT)
        self.baseline_freq_min_spin = ttk.Spinbox(freq_base_frame, from_=0, to=1000, increment=10,
                                                  textvariable=self.baseline_freq_min, width=6)
        self.baseline_freq_min_spin.pack(side=tk.LEFT, padx=(5, 2))
        ttk.Label(freq_base_frame, text="-").pack(side=tk.LEFT)
        self.baseline_freq_max_spin = ttk.Spinbox(freq_base_frame, from_=0, to=1000, increment=10,
                                                  textvariable=self.baseline_freq_max, width=6)
        self.baseline_freq_max_spin.pack(side=tk.LEFT, padx=(2, 2))
        ttk.Label(freq_base_frame, text="Hz").pack(side=tk.LEFT)
        row += 1

        # Baseline frequency std multiplier
        base_std_frame = ttk.Frame(control_frame)
        base_std_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(base_std_frame, text="Baseline Std Multiplier:").pack(side=tk.LEFT)
        self.baseline_std_spin = ttk.Spinbox(base_std_frame, from_=0.0, to=5.0, increment=0.1,
                                             textvariable=self.baseline_freq_std_multiplier, width=8)
        self.baseline_std_spin.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(base_std_frame, text="σ (threshold = baseline + N×σ)").pack(side=tk.LEFT, padx=(2, 0))
        row += 1

        # Std multiplier
        std_frame = ttk.Frame(control_frame)
        std_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(std_frame, text="Std Multiplier:").pack(side=tk.LEFT)
        self.std_spin = ttk.Spinbox(std_frame, from_=0.0, to=5.0, increment=0.1,
                                    textvariable=self.std_multiplier, width=8)
        self.std_spin.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(std_frame, text="σ (0=baseline only, 1=baseline+1σ, etc)").pack(side=tk.LEFT, padx=(2, 0))
        row += 1

        # Percentile threshold
        perc_frame = ttk.Frame(control_frame)
        perc_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(perc_frame, text="Percentile:").pack(side=tk.LEFT)
        self.perc_spin = ttk.Spinbox(perc_frame, from_=50, to=99, increment=1,
                                     textvariable=self.percentile_threshold, width=8)
        self.perc_spin.pack(side=tk.LEFT, padx=(5, 0))
        row += 1

        self.update_threshold_controls()

        # File selection
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10)
        row += 1
        ttk.Label(control_frame, text="File Selection:", font=('', 10, 'bold')).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1

        # File/folder selection buttons
        file_btn_frame = ttk.Frame(control_frame)
        file_btn_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Button(file_btn_frame, text="📂 Add Folder", command=self.add_folder, width=12).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_btn_frame, text="📂 Add Folder", command=self.add_folder, width=12).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_btn_frame, text="📄 Add Files", command=self.add_files, width=12).pack(side=tk.LEFT)
        row += 1

        # Species filter
        filter_frame = ttk.Frame(control_frame)
        filter_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT)
        self.species_filter = tk.StringVar(value="All Species")
        self.species_combo = ttk.Combobox(filter_frame, textvariable=self.species_filter,
                                         state='readonly', width=15)
        self.species_combo['values'] = ["All Species"]
        self.species_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.species_combo.bind('<<ComboboxSelected>>', lambda e: self._update_file_listbox())
        row += 1

        # File listbox with scrollbar
        list_frame = ttk.Frame(control_frame)
        list_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        scrollbar_list = ttk.Scrollbar(list_frame)
        scrollbar_list.pack(side=tk.RIGHT, fill=tk.Y)

        self.file_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar_list.set, height=8, selectmode=tk.EXTENDED)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_list.config(command=self.file_listbox.yview)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)
        row += 1

        # File count label
        self.file_count_label = ttk.Label(control_frame, text="No files in list", foreground='gray')
        self.file_count_label.grid(row=row, column=0, sticky=tk.W, pady=(0, 10))
        row += 1

        # Analysis buttons
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10)
        row += 1

        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.analyze_btn = ttk.Button(btn_frame, text="▶ Analyze Selected",
                                      command=self.analyze_selected_file)
        self.analyze_btn.pack(fill=tk.X, pady=(0, 5))

        self.batch_btn = ttk.Button(btn_frame, text="▶▶ Batch Process Folder",
                                    command=self.batch_process_folder)
        self.batch_btn.pack(fill=tk.X, pady=(0, 5))

    def create_visualization_panel(self, parent):
        """Create right visualisation panel"""
        viz_frame = ttk.LabelFrame(parent, text="Visualisation & Results", padding="10")
        parent.add(viz_frame, weight=1)
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)

        # Notebook for tabs
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Tab 1: Results table
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results Table")

        # Create treeview for results
        tree_frame = ttk.Frame(self.results_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Vertical scrollbar
        v_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Horizontal scrollbar
        h_scroll = ttk.Scrollbar(self.results_frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.TOP, fill=tk.X, padx=5)

        self.results_tree = ttk.Treeview(tree_frame, yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.config(command=self.results_tree.yview)
        h_scroll.config(command=self.results_tree.xview)

        # Buttons frame for Export, Plot Selected, and Help
        buttons_frame = ttk.Frame(self.results_frame)
        buttons_frame.pack(pady=5)

        export_btn = ttk.Button(buttons_frame, text="💾 Export to CSV",
                               command=self.export_results)
        export_btn.pack(side=tk.LEFT, padx=5)

        plot_selected_btn = ttk.Button(buttons_frame, text="📊 Plot Selected",
                                       command=self.plot_selected_results)
        plot_selected_btn.pack(side=tk.LEFT, padx=5)

        waveform_btn = ttk.Button(buttons_frame, text="🔊 Plot Waveform",
                                  command=self.plot_waveform_with_analysis)
        waveform_btn.pack(side=tk.LEFT, padx=5)

        help_btn = ttk.Button(buttons_frame, text="❓ Column Help",
                             command=self.show_column_help)
        help_btn.pack(side=tk.LEFT, padx=5)

        # Tab 2: Features
        self.features_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.features_frame, text="Features")

        self.features_text = scrolledtext.ScrolledText(self.features_frame, wrap=tk.WORD,
                                                       font=('Courier', 10))
        self.features_text.pack(fill=tk.BOTH, expand=True)

    def create_status_panel(self, parent):
        """Create bottom status panel"""
        status_frame = ttk.LabelFrame(parent, text="Status Log", padding="5")
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        parent.rowconfigure(1, weight=0, minsize=160)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(status_frame, orient='horizontal', mode='determinate',
                                        variable=self.progress_var, maximum=1.0)
        self.progress.pack(fill=tk.X, padx=5, pady=(0, 6))

        # Status text
        self.status_text = scrolledtext.ScrolledText(status_frame, wrap=tk.WORD,
                                                     height=6, font=('Courier', 9))
        self.status_text.pack(fill=tk.BOTH, expand=True)

        self.log("Welcome to Mosquito Frequency Range Analysis GUI")
        self.log("Ready to analyze mechanoreceptor frequency tuning data")

    # -------------------------
    # Utility / UI helper methods
    # -------------------------
    def log(self, message):
        """Add message to status log (thread-safe via after)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        def append():
            self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.status_text.see(tk.END)
        self.root.after(0, append)

    def _set_progress(self, value):
        """Set progress bar value (0..1)"""
        def setter():
            self.progress_var.set(value)
        self.root.after(0, setter)

    def browse_input_dir(self):
        d = filedialog.askdirectory(title="Select Input Directory", initialdir=self.cwd)
        if d:
            self.input_dir.set(d)
            self.scan_files()

    # -------------------------
    # Transfer .smr UI flow
    # -------------------------
    def transfer_smr_dialog(self):
        """Show dialog for selecting source and destination and options for transfer"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Transfer .smr Files")
        dialog.geometry("520x220")
        dialog.transient(self.root)
        dialog.grab_set()

        src_var = tk.StringVar(value="Mech_data")
        src_type_var = tk.StringVar(value="mech")
        dst_var = tk.StringVar(value="Mech_data_smr")
        dst_type_var = tk.StringVar(value="mech")
        flatten_var = tk.BooleanVar(value=False)
        overwrite_var = tk.BooleanVar(value=False)

        def update_src_dir():
            """Update source directory based on selected type"""
            src_type = src_type_var.get()
            src_var.set(f"{src_type.capitalize()}_data")

        def update_dst_dir():
            """Update destination directory based on selected type"""
            dst_type = dst_type_var.get()
            dst_var.set(f"{dst_type.capitalize()}_data_smr")

        def choose_src():
            d = filedialog.askdirectory(title="Select Source Directory (search recursively)", initialdir=self.cwd)
            if d:
                src_var.set(d)

        def choose_dst():
            d = filedialog.askdirectory(title="Select Destination Directory (or create new)", initialdir=self.cwd)
            if d:
                dst_var.set(d)

        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Source Directory:", font=('', 9, 'bold')).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        src_frame = ttk.Frame(frame)
        src_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(2, 6))
        src_frame.columnconfigure(0, weight=1)
        ttk.Entry(src_frame, textvariable=src_var, width=35).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Radiobutton(src_frame, text="Mech", variable=src_type_var, value="mech", command=update_src_dir).grid(row=0, column=1, padx=3)
        ttk.Radiobutton(src_frame, text="Nerve", variable=src_type_var, value="nerve", command=update_src_dir).grid(row=0, column=2, padx=3)
        ttk.Button(src_frame, text="Browse", command=choose_src).grid(row=0, column=3, padx=(5, 0))

        ttk.Label(frame, text="Destination Directory:", font=('', 9, 'bold')).grid(row=2, column=0, columnspan=2, sticky=tk.W)
        dst_frame = ttk.Frame(frame)
        dst_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(2, 6))
        dst_frame.columnconfigure(0, weight=1)
        ttk.Entry(dst_frame, textvariable=dst_var, width=35).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Radiobutton(dst_frame, text="Mech", variable=dst_type_var, value="mech", command=update_dst_dir).grid(row=0, column=1, padx=3)
        ttk.Radiobutton(dst_frame, text="Nerve", variable=dst_type_var, value="nerve", command=update_dst_dir).grid(row=0, column=2, padx=3)
        ttk.Button(dst_frame, text="Browse", command=choose_dst).grid(row=0, column=3, padx=(5, 0))

        opt_frame = ttk.Frame(frame)
        opt_frame.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(6, 6))
        ttk.Checkbutton(opt_frame, text="Flatten (copy all into destination root)", variable=flatten_var).pack(anchor=tk.W)
        ttk.Checkbutton(opt_frame, text="Overwrite existing files", variable=overwrite_var).pack(anchor=tk.W)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(8, 0))
        def on_start():
            src = src_var.get().strip()
            dst = dst_var.get().strip()
            if not src or not os.path.isdir(src):
                messagebox.showerror("Invalid Source", "Please choose a valid source directory")
                return
            if not dst:
                messagebox.showerror("Invalid Destination", "Please choose a destination directory")
                return
            dialog.destroy()
            # Start transfer in background
            thread = threading.Thread(target=self._transfer_thread, args=(src, dst, flatten_var.get(), overwrite_var.get()))
            thread.daemon = True
            thread.start()

        ttk.Button(btn_frame, text="Start Transfer", command=on_start, width=16).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy, width=12).pack(side=tk.LEFT)

    def _transfer_thread(self, src, dst, flatten, overwrite):
        """Background thread performing transfer and updating UI callbacks"""
        self.log(f"▶ Starting transfer from {src} → {dst} (flatten={flatten}, overwrite={overwrite})")
        self._set_progress(0.0)

        def status_cb(msg):
            self.log(msg)

        def progress_cb(p):
            self._set_progress(p)

        try:
            summary = transfer_smr_files(src, dst, flatten=flatten, overwrite=overwrite,
                                         status_callback=status_cb, progress_callback=progress_cb)
            self.log(f"✓ Transfer finished: {len(summary.get('copied', []))} files copied")
            self._set_progress(0.0)
            # After transfer, refresh input_dir scan if destination is the current input_dir
            if os.path.abspath(dst) == os.path.abspath(self.input_dir.get()):
                self.scan_files()
        except Exception as e:
            self.log(f"❌ Transfer failed: {str(e)}")
            self._set_progress(0.0)

    # -------------------------
    # Convert .smr -> WAV UI flow
    # -------------------------
    def convert_smr_dialog(self):
        """Dialog to confirm conversion parameters and start conversion"""
        if not NEO_AVAILABLE:
            messagebox.showerror(
                "SMR conversion unavailable",
                "neo/sonpy not installed. WAV analysis works; install dependencies to enable conversion."
            )
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Convert .smr → WAV")
        dialog.geometry("550x450")
        dialog.transient(self.root)
        dialog.grab_set()

        in_var = tk.StringVar(value="Mech_data_smr")
        in_type_var = tk.StringVar(value="mech")
        out_var = tk.StringVar(value="Wav_data")
        out_type_var = tk.StringVar(value="mech")
        sampling_rate_var = tk.IntVar(value=self.sampling_rate.get())
        use_target_sr_var = tk.BooleanVar(value=False)
        info_var = tk.StringVar(value="")
        
        # Directionality options
        use_directionality_var = tk.BooleanVar(value=False)
        forward_start_var = tk.DoubleVar(value=0.05)
        forward_end_var = tk.DoubleVar(value=1.05)
        backward_start_var = tk.DoubleVar(value=2.65)
        backward_end_var = tk.DoubleVar(value=3.65)
        
        # Store scan results
        scan_results = {'segments': [], 'channels': [], 'files_scanned': 0}

        def update_input_dir():
            """Update input directory based on selected type"""
            in_type = in_type_var.get()
            in_var.set(f"{in_type.capitalize()}_data_smr")

        def choose_input():
            d = filedialog.askdirectory(title="Input directory with .smr files", initialdir=self.cwd)
            if d:
                in_var.set(d)

        def scan_directory():
            """Recursively scan directory for ALL .smr files in subdirectories"""
            d = in_var.get().strip()
            if not d or not os.path.isdir(d):
                info_var.set("Please select a valid input directory first")
                messagebox.showwarning("Invalid Directory", "Please select an input directory first")
                return
            
            # Scan ALL .smr files recursively
            try:
                smr_files = [
                    os.path.join(root, fname)
                    for root, _, files in os.walk(d)
                    for fname in files
                    if fname.lower().endswith('.smr')
                ]
                
                if not smr_files:
                    info_var.set("No .smr files found in directory tree")
                    messagebox.showinfo("No Files", f"No .smr files found in {d} or its subdirectories")
                    return
                
                info_var.set(f"Scanning {len(smr_files)} files recursively...")
                dialog.update()
                
                all_segments = set()
                all_channels = set()
                sampling_rates = []
                files_processed = 0
                
                for smr_file in smr_files:
                    try:
                        reader = CedIO(filename=smr_file)
                        block = reader.read_block(lazy=False)
                        
                        num_segments = len(block.segments)
                        for seg_idx in range(num_segments):
                            all_segments.add(seg_idx)
                            seg = block.segments[seg_idx]
                            num_channels = len(seg.analogsignals)
                            for ch_idx in range(num_channels):
                                all_channels.add(ch_idx)
                                # Get sampling rate from first channel
                                if seg_idx == 0 and ch_idx == 0:
                                    analog = seg.analogsignals[0]
                                    if hasattr(analog, 'sampling_rate'):
                                        fs = float(analog.sampling_rate.magnitude) if hasattr(analog.sampling_rate, 'magnitude') else float(analog.sampling_rate)
                                        sampling_rates.append(int(fs))
                        files_processed += 1
                    except Exception as e:
                        self.log(f"Warning: Could not read {os.path.basename(smr_file)}: {str(e)}")
                        continue
                
                scan_results['segments'] = sorted(all_segments)
                scan_results['channels'] = sorted(all_channels)
                scan_results['files_scanned'] = files_processed

                avg_sr = "unknown"
                if sampling_rates:
                    avg_sr = int(np.mean(sampling_rates))
                    sampling_rate_var.set(avg_sr)
                
                info_var.set(f"✓ Scanned {files_processed} files (in folders): Segments {scan_results['segments']}, Channels {scan_results['channels']}, ~{avg_sr} Hz")
                
                # Update the segment/channel selection widgets
                update_selection_widgets()
                
            except Exception as e:
                info_var.set(f"Error scanning files: {str(e)}")
                messagebox.showerror("Scan Error", f"Error scanning directory: {str(e)}")

        def choose_output():
            d = filedialog.askdirectory(title="Output directory for WAV files", initialdir=self.cwd)
            if d:
                out_var.set(d)
                
        def update_output_dir():
            """Update output directory based on selected type"""
            out_type = out_type_var.get()
            out_var.set(f"Wav_data_{out_type}")

        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        # I/O
        ttk.Label(frame, text="Input (.smr) Directory:", font=('', 9, 'bold')).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        input_row_frame = ttk.Frame(frame)
        input_row_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(2, 6))
        input_row_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(input_row_frame, textvariable=in_var, width=28).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Radiobutton(input_row_frame, text="Mech", variable=in_type_var, value="mech", command=update_input_dir).grid(row=0, column=1, padx=(8, 2))
        ttk.Radiobutton(input_row_frame, text="Nerve", variable=in_type_var, value="nerve", command=update_input_dir).grid(row=0, column=2, padx=2)
        ttk.Button(input_row_frame, text="Browse", command=choose_input).grid(row=0, column=3, padx=(8, 2))
        ttk.Button(input_row_frame, text="Scan Files", command=scan_directory, style='Accent.TButton').grid(row=0, column=4, padx=2)

        ttk.Label(frame, text="Output (WAV) Directory:", font=('', 9, 'bold')).grid(row=2, column=0, columnspan=2, sticky=tk.W)
        
        output_row_frame = ttk.Frame(frame)
        output_row_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(2, 6))
        output_row_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(output_row_frame, textvariable=out_var, width=28).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Radiobutton(output_row_frame, text="Mech", variable=out_type_var, value="mech", 
                       command=update_output_dir).grid(row=0, column=1, padx=(8, 2))
        ttk.Radiobutton(output_row_frame, text="Nerve", variable=out_type_var, value="nerve", 
                       command=update_output_dir).grid(row=0, column=2, padx=2)
        ttk.Button(output_row_frame, text="Browse", command=choose_output).grid(row=0, column=3, padx=(8, 0))

        # File info display
        info_label = ttk.Label(frame, textvariable=info_var, foreground='blue', font=('', 8))
        info_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(0, 6))

        # Options
        opt_frame = ttk.Frame(frame)
        opt_frame.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(6, 6))
        ttk.Label(opt_frame, text="Conversion options:", font=('', 9)).grid(row=0, column=0, sticky=tk.W, pady=(0, 6))

        # Selection widgets (will be populated after scan)
        selection_frame = ttk.Frame(opt_frame)
        selection_frame.grid(row=1, column=0, sticky=tk.W, pady=(0, 6))

        ttk.Label(selection_frame, text="Select Segments:").grid(row=0, column=0, sticky=tk.W)
        segments_list_frame = ttk.Frame(selection_frame)
        segments_list_frame.grid(row=1, column=0, sticky=tk.W, padx=(0, 10))

        ttk.Label(selection_frame, text="Select Channels:").grid(row=0, column=1, sticky=tk.W)
        channels_list_frame = ttk.Frame(selection_frame)
        channels_list_frame.grid(row=1, column=1, sticky=tk.W)

        # Store checkbutton variables
        segment_vars = {}
        channel_vars = {}

        def update_selection_widgets():
            """Update segment and channel selection widgets based on scan results"""
            for widget in segments_list_frame.winfo_children():
                widget.destroy()
            for widget in channels_list_frame.winfo_children():
                widget.destroy()

            segment_vars.clear()
            channel_vars.clear()

            if not scan_results['segments'] or not scan_results['channels']:
                ttk.Label(segments_list_frame, text="(scan files first)", foreground='gray').pack()
                ttk.Label(channels_list_frame, text="(scan files first)", foreground='gray').pack()
                return

            for seg_idx in scan_results['segments']:
                var = tk.BooleanVar(value=True)
                segment_vars[seg_idx] = var
                ttk.Checkbutton(segments_list_frame, text=f"Seg {seg_idx}", variable=var).pack(anchor=tk.W)

            for ch_idx in scan_results['channels']:
                var = tk.BooleanVar(value=True)
                channel_vars[ch_idx] = var
                ttk.Checkbutton(channels_list_frame, text=f"Ch {ch_idx}", variable=var).pack(anchor=tk.W)

        # Initial placeholder
        ttk.Label(segments_list_frame, text="(scan files first)", foreground='gray').pack()
        ttk.Label(channels_list_frame, text="(scan files first)", foreground='gray').pack()

        # Directionality mode
        dir_frame = ttk.LabelFrame(opt_frame, text="Splitting Mode", padding="10")
        dir_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 6))
        
        def toggle_directionality():
            state = tk.NORMAL if use_directionality_var.get() else tk.DISABLED
            fwd_start_spin.config(state=state)
            fwd_end_spin.config(state=state)
            bwd_start_spin.config(state=state)
            bwd_end_spin.config(state=state)
        
        ttk.Radiobutton(dir_frame, text="Full File (no splitting)", 
                       variable=use_directionality_var, value=False,
                       command=toggle_directionality).grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 5))
        ttk.Radiobutton(dir_frame, text="Split by Directionality (forward/backward)", 
                       variable=use_directionality_var, value=True,
                       command=toggle_directionality).grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(0, 10))
        
        # Time range inputs
        ttk.Label(dir_frame, text="Forward:", font=('', 9, 'bold')).grid(row=2, column=0, sticky=tk.W, padx=(20, 5))
        ttk.Label(dir_frame, text="Start:").grid(row=2, column=1, sticky=tk.E, padx=(0, 2))
        fwd_start_spin = ttk.Spinbox(dir_frame, from_=0.0, to=10.0, increment=0.05, 
                                     textvariable=forward_start_var, width=6, format="%.2f", state=tk.DISABLED)
        fwd_start_spin.grid(row=2, column=2, sticky=tk.W, padx=(0, 5))
        ttk.Label(dir_frame, text="End:").grid(row=2, column=3, sticky=tk.E, padx=(0, 2))
        fwd_end_spin = ttk.Spinbox(dir_frame, from_=0.0, to=10.0, increment=0.05, 
                                   textvariable=forward_end_var, width=6, format="%.2f", state=tk.DISABLED)
        fwd_end_spin.grid(row=2, column=4, sticky=tk.W)
        ttk.Label(dir_frame, text="sec").grid(row=2, column=5, sticky=tk.W, padx=(2, 0))
        
        ttk.Label(dir_frame, text="Backward:", font=('', 9, 'bold')).grid(row=3, column=0, sticky=tk.W, padx=(20, 5), pady=(5, 0))
        ttk.Label(dir_frame, text="Start:").grid(row=3, column=1, sticky=tk.E, padx=(0, 2))
        bwd_start_spin = ttk.Spinbox(dir_frame, from_=0.0, to=10.0, increment=0.05, 
                                     textvariable=backward_start_var, width=6, format="%.2f", state=tk.DISABLED)
        bwd_start_spin.grid(row=3, column=2, sticky=tk.W, padx=(0, 5))
        ttk.Label(dir_frame, text="End:").grid(row=3, column=3, sticky=tk.E, padx=(0, 2))
        bwd_end_spin = ttk.Spinbox(dir_frame, from_=0.0, to=10.0, increment=0.05, 
                                   textvariable=backward_end_var, width=6, format="%.2f", state=tk.DISABLED)
        bwd_end_spin.grid(row=3, column=4, sticky=tk.W)
        ttk.Label(dir_frame, text="sec").grid(row=3, column=5, sticky=tk.W, padx=(2, 0))

        sr_frame = ttk.Frame(opt_frame)
        sr_frame.grid(row=3, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Checkbutton(sr_frame, text="Resample to:", variable=use_target_sr_var).pack(side=tk.LEFT)
        ttk.Spinbox(sr_frame, from_=10000, to=200000, increment=10000, textvariable=sampling_rate_var, width=8).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(sr_frame, text="Hz (uncheck to keep original sampling rates)").pack(side=tk.LEFT, padx=(2, 0))

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=(8, 0))

        def on_start():
            in_d = in_var.get().strip()
            out_d = out_var.get().strip()
            if not in_d or not os.path.isdir(in_d):
                messagebox.showerror("Invalid Input", "Please choose a valid input directory containing .smr files")
                return
            if not out_d:
                messagebox.showerror("Invalid Output", "Please choose an output directory")
                return

            selected_segments = [idx for idx, var in segment_vars.items() if var.get()]
            selected_channels = [idx for idx, var in channel_vars.items() if var.get()]

            if not selected_segments:
                messagebox.showerror("No Selection", "Please select at least one segment")
                return
            if not selected_channels:
                messagebox.showerror("No Selection", "Please select at least one channel")
                return

            dialog.destroy()
            target_sr = sampling_rate_var.get() if use_target_sr_var.get() else None
            
            # Prepare directionality settings
            dir_settings = None
            if use_directionality_var.get():
                dir_settings = {
                    'forward': (forward_start_var.get(), forward_end_var.get()),
                    'backward': (backward_start_var.get(), backward_end_var.get())
                }
            
            thread = threading.Thread(target=self._convert_thread, args=(
                in_d, out_d, selected_channels, selected_segments, target_sr, dir_settings
            ))
            thread.daemon = True
            thread.start()

        ttk.Button(btn_frame, text="Start Conversion", command=on_start, width=16).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy, width=12).pack(side=tk.LEFT)

    def _convert_thread(self, input_dir, output_root, channel_indices, segment_indices, target_sampling_rate=None, directionality_settings=None):
        """Background thread to run SMR->WAV conversion with filtering and normalization"""
        self.log(f"▶ Starting conversion: {input_dir} → {output_root}")
        self.log(f"   Segments: {segment_indices}, Channels: {channel_indices}, target_sr={target_sampling_rate}")
        self.log(f"   Filter: {0}-{1000} Hz (order=4), Normalization: enabled")
        if directionality_settings:
            fwd = directionality_settings['forward']
            bwd = directionality_settings['backward']
            self.log(f"   Directionality: Forward {fwd[0]:.2f}-{fwd[1]:.2f}s, Backward {bwd[0]:.2f}-{bwd[1]:.2f}s")
        else:
            self.log(f"   Mode: Full file (no splitting)")
        self._set_progress(0.0)

        def status_cb(msg):
            self.log(msg)

        def progress_cb(p):
            self._set_progress(p)

        try:
            summary = convert_smr_to_wav_no_default_species(
                input_dir=input_dir,
                output_root=output_root,
                channel_indices=channel_indices,
                segment_indices=segment_indices,
                target_sampling_rate=target_sampling_rate,
                status_callback=status_cb,
                progress_callback=progress_cb,
                min_freq=0,
                max_freq=1000,
                filter_order=4,
                directionality_settings=directionality_settings
            )
            self.log(f"✓ Conversion finished. Wrote {summary.get('total_files', 0)} WAV files")
            # After conversion, rescan WAV input dir if it's set to output_root
            if os.path.abspath(self.input_dir.get()) == os.path.abspath(output_root):
                self.scan_files()
            self._set_progress(0.0)
        except Exception as e:
            self.log(f"❌ Conversion failed: {str(e)}")
            self._set_progress(0.0)

    # -------------------------
    # Existing functionality below (unchanged except small call backs)
    # -------------------------
    def add_folder(self):
        """Add all WAV files from a selected folder"""
        directory = filedialog.askdirectory(title="Select Folder with WAV Files")
        if not directory:
            return

        # Update input directory and type based on selected folder
        # Check if path contains 'mech' or 'nerve' to auto-detect type
        path_lower = directory.lower()
        if 'mech' in path_lower:
            self.input_type_var.set('mech')
            self.input_dir.set(directory)
        elif 'nerve' in path_lower:
            self.input_type_var.set('nerve')
            self.input_dir.set(directory)
        else:
            # Set the directory and trigger scan
            self.input_dir.set(directory)

        # Find WAV files in selected directory and subdirectories
        new_files = []
        for root, dirs, files in os.walk(directory):
            for fname in files:
                if fname.lower().endswith('.wav'):
                    full_path = os.path.join(root, fname)
                    if full_path not in self.wav_files:
                        new_files.append(full_path)

        if new_files:
            self.wav_files.extend(new_files)
            self._update_file_listbox()
            self.log(f"✓ Added {len(new_files)} WAV files from {os.path.basename(directory)}")
        else:
            self.log(f"⚠️  No WAV files found in {os.path.basename(directory)}")
            messagebox.showinfo("No Files", f"No WAV files found in:\n{directory}")

    def add_files(self):
        """Add individual WAV files"""
        filepaths = filedialog.askopenfilenames(
            title="Select WAV Files",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )

        if not filepaths:
            return

        new_files = []
        for filepath in filepaths:
            if filepath not in self.wav_files:
                new_files.append(filepath)

        if new_files:
            self.wav_files.extend(new_files)
            self._update_file_listbox()
            self.log(f"✓ Added {len(new_files)} WAV file(s)")
        else:
            self.log("⚠️  Files already in list")

    def clear_file_list(self):
        """Clear all files from the list"""
        if self.wav_files:
            response = messagebox.askyesno("Clear List",
                                          f"Remove all {len(self.wav_files)} files from the list?")
            if response:
                self.wav_files = []
                self.file_listbox.delete(0, tk.END)
                self.file_count_label.config(text="No files in list", foreground='gray')
                self.log("Cleared file list")

    def remove_selected_files(self):
        """Remove selected files from the list"""
        selections = self.file_listbox.curselection()
        if not selections:
            messagebox.showinfo("No Selection", "Please select files to remove")
            return

        # Remove in reverse order to maintain indices
        for idx in reversed(selections):
            if idx < len(self.wav_files):
                removed_file = self.wav_files.pop(idx)
                self.log(f"Removed: {os.path.basename(removed_file)}")

        self._update_file_listbox()

    def _update_file_listbox(self):
        """Update the file listbox with current wav_files"""
        self.file_listbox.delete(0, tk.END)

        # Get current species filter
        species_filter = self.species_filter.get()

        # Create mapping from listbox index to filepath
        self.displayed_files = []

        # Group by species/direction for display
        species_groups = {}
        for filepath in self.wav_files:
            species, direction, filename = self._parse_file_metadata(filepath)

            # Apply species filter
            if species_filter != "All Species" and species != species_filter:
                continue

            key = species if not direction else f"{species}/{direction}"

            if key not in species_groups:
                species_groups[key] = []
            species_groups[key].append(filepath)

        # Add grouped items to listbox
        for group_key in sorted(species_groups.keys()):
            files = species_groups[group_key]
            for filepath in sorted(files):
                self.displayed_files.append(filepath)
                display_name = f"{group_key}/{os.path.basename(filepath)}"
                self.file_listbox.insert(tk.END, display_name)

        # Update count label
        displayed_count = self.file_listbox.size()
        total_count = len(self.wav_files)

        if total_count > 0:
            if displayed_count < total_count:
                self.file_count_label.config(
                    text=f"{displayed_count} displayed / {total_count} total file(s)",
                    foreground='blue'
                )
            else:
                self.file_count_label.config(
                    text=f"{total_count} file(s) in list",
                    foreground='green'
                )
        else:
            self.file_count_label.config(text="No files in list", foreground='gray')

    def update_threshold_controls(self):
        """Enable/disable threshold controls based on method"""
        method = self.threshold_method.get()

        if method == 'freq_range':
            self.baseline_freq_min_spin.config(state='normal')
            self.baseline_freq_max_spin.config(state='normal')
            self.baseline_std_spin.config(state='normal')
            self.std_spin.config(state='disabled')
            self.perc_spin.config(state='disabled')
        elif method == 'std':
            self.baseline_freq_min_spin.config(state='disabled')
            self.baseline_freq_max_spin.config(state='disabled')
            self.baseline_std_spin.config(state='disabled')
            self.std_spin.config(state='normal')
            self.perc_spin.config(state='disabled')
        elif method == 'percentile':
            self.baseline_freq_min_spin.config(state='disabled')
            self.baseline_freq_max_spin.config(state='disabled')
            self.baseline_std_spin.config(state='disabled')
            self.std_spin.config(state='disabled')
            self.perc_spin.config(state='normal')

    def scan_files(self):
        """Scan input directory for WAV files"""
        input_dir = self.input_dir.get()

        if not os.path.exists(input_dir):
            self.log(f"❌ Directory not found: {input_dir}")
            self.file_count_label.config(text="Directory not found", foreground='red')
            return

        # Clear existing and rescan
        self.wav_files = []
        detected_species = set()

        for root, dirs, files in os.walk(input_dir):
            for fname in files:
                if fname.lower().endswith('.wav'):
                    full_path = os.path.join(root, fname)
                    self.wav_files.append(full_path)
                    
                    # Detect species from path
                    species, _, _ = self._parse_file_metadata(full_path)
                    if species != "Unknown":
                        detected_species.add(species)

        # Update species filter dropdown with detected species
        species_list = ["All Species"] + sorted(detected_species)
        self.species_combo['values'] = species_list
        
        # Update listbox
        self._update_file_listbox()

        # Log results
        if self.wav_files:
            # Group by species for reporting
            species_groups = {}
            for filepath in self.wav_files:
                species, direction, _ = self._parse_file_metadata(filepath)
                key = species if not direction else f"{species}/{direction}"
                if key not in species_groups:
                    species_groups[key] = []
                species_groups[key].append(filepath)

            self.log(f"✓ Found {len(self.wav_files)} WAV files in {len(species_groups)} groups")
            for group, files in sorted(species_groups.items()):
                self.log(f"  {group}: {len(files)} files")
        else:
            self.log(f"⚠️  No WAV files found in {input_dir}")

    def on_file_select(self, event):
        """Handle file selection in listbox"""
        selection = self.file_listbox.curselection()
        if selection:
            idx = selection[0]
            if idx < len(self.displayed_files):
                filepath = self.displayed_files[idx]
                self.log(f"Selected: {os.path.basename(filepath)}")

    def load_wav_file(self, filepath):
        """Load WAV file using librosa"""
        try:
            data, sr = librosa.load(filepath, sr=self.sampling_rate.get())

            if sr != self.sampling_rate.get():
                self.log(f"⚠️  Warning: Expected {self.sampling_rate.get()} Hz, got {sr} Hz")

            return data, sr
        except Exception as e:
            self.log(f"❌ Error loading {filepath}: {str(e)}")
            return None, None

    def compute_power_spectrum(self, data, fs, nperseg=4096):
        """Compute power spectral density using Welch's method
        Returns PSD interpolated to exactly 1 Hz resolution for precise frequency analysis.
        """
        freqs_raw, psd_raw = signal.welch(data, fs=fs, nperseg=nperseg)

        # Filter to frequency range
        mask = (freqs_raw >= self.min_freq.get()) & (freqs_raw <= self.max_freq.get())
        freqs_raw = freqs_raw[mask]
        psd_raw = psd_raw[mask]

        min_freq = int(np.ceil(self.min_freq.get()))
        max_freq = int(np.floor(self.max_freq.get()))
        freqs = np.arange(min_freq, max_freq + 1, 1.0)

        interpolator = interp1d(freqs_raw, psd_raw, kind='cubic', bounds_error=False, fill_value='extrapolate')
        psd = interpolator(freqs)
        psd = np.maximum(psd, 0)

        return freqs, psd

    def calculate_baseline_threshold(self, power_spectrum, species=None, freqs=None):
        """Calculate baseline and threshold for response detection"""
        method = self.threshold_method.get()
        std_mult = self.std_multiplier.get()
        baseline_freq_std_mult = self.baseline_freq_std_multiplier.get()
        percentile = self.percentile_threshold.get()

        if method == 'freq_range':
            # Use specified frequency range as baseline (noise floor)
            if freqs is None:
                # Fallback to std method if freqs not provided
                baseline = np.mean(power_spectrum)
                std = np.std(power_spectrum)
                threshold = baseline + 2.0 * std
            else:
                baseline_min = self.baseline_freq_min.get()
                baseline_max = self.baseline_freq_max.get()
                
                # Find indices for baseline frequency range
                baseline_mask = (freqs >= baseline_min) & (freqs <= baseline_max)
                
                if np.any(baseline_mask):
                    # Baseline is the mean power in the specified frequency range
                    baseline = np.mean(power_spectrum[baseline_mask])
                    # Threshold is baseline + N std of the baseline region (using baseline_freq_std_multiplier)
                    baseline_std = np.std(power_spectrum[baseline_mask])
                    threshold = baseline + baseline_freq_std_mult * baseline_std
                else:
                    # Fallback if no frequencies in range
                    baseline = np.mean(power_spectrum)
                    std = np.std(power_spectrum)
                    threshold = baseline + 2.0 * std

        return baseline, threshold

    def find_response_regions(self, freqs, power, threshold):
        """Find continuous frequency regions where power exceeds threshold"""
        above_threshold = power > threshold

        if not np.any(above_threshold):
            peak_power = np.max(power)
            peak_power_db = 10 * np.log10(peak_power + 1e-12)
            power_db = 10 * np.log10(power + 1e-12)
            threshold_3db = peak_power_db - 3
            above_threshold = power_db > threshold_3db

        regions = []
        in_region = False
        start_idx = None

        for i, is_above in enumerate(above_threshold):
            if is_above and not in_region:
                start_idx = i
                in_region = True
            elif not is_above and in_region:
                end_idx = i - 1
                if end_idx > start_idx:
                    freq_start = freqs[start_idx]
                    freq_end = freqs[end_idx]
                    mean_power = np.mean(power[start_idx:end_idx+1])
                    regions.append((freq_start, freq_end, mean_power))
                in_region = False

        if in_region and start_idx < len(freqs) - 1:
            freq_start = freqs[start_idx]
            freq_end = freqs[-1]
            mean_power = np.mean(power[start_idx:])
            regions.append((freq_start, freq_end, mean_power))

        return regions

    def extract_tuning_features(self, freqs, power, baseline, threshold):
        """Extract comprehensive tuning curve features"""
        features = {}
        peak_idx = np.argmax(power)
        features['peak_frequency'] = freqs[peak_idx]
        features['peak_power'] = power[peak_idx]
        features['baseline'] = baseline
        features['threshold'] = threshold
        regions = self.find_response_regions(freqs, power, threshold)
        features['num_response_regions'] = len(regions)
        features['response_regions'] = regions  # Store all regions
        
        # Store individual ranges as Range 1, Range 2, etc.
        for idx, (freq_start, freq_end, mean_power) in enumerate(regions):
            features[f'range_{idx+1}_start'] = freq_start
            features[f'range_{idx+1}_end'] = freq_end
            features[f'range_{idx+1}_bandwidth'] = freq_end - freq_start
        
        # Store detection parameters used for this analysis
        features['_detection_baseline'] = baseline
        features['_detection_threshold'] = threshold

        if features['peak_power'] > 0:
            peak_power_db = 10 * np.log10(features['peak_power'])
            with np.errstate(divide='ignore', invalid='ignore'):
                power_db = np.where(power > 0, 10 * np.log10(power), -np.inf)

            threshold_3db = peak_power_db - 3
            above_3db = power_db > threshold_3db

            if np.any(above_3db):
                indices_3db = np.where(above_3db)[0]
                features['freq_range_start'] = freqs[indices_3db[0]]
                features['freq_range_end'] = freqs[indices_3db[-1]]
                features['freq_range_bandwidth'] = freqs[indices_3db[-1]] - freqs[indices_3db[0]]
                features['bandwidth_3dB'] = features['freq_range_bandwidth']
                if features['freq_range_bandwidth'] > 0:
                    features['Q_factor'] = features['peak_frequency'] / features['freq_range_bandwidth']
                    features['Q3dB'] = features['Q_factor']
                else:
                    features['Q_factor'] = np.nan
                    features['Q3dB'] = np.nan
            else:
                features['freq_range_start'] = np.nan
                features['freq_range_end'] = np.nan
                features['freq_range_bandwidth'] = np.nan
                features['bandwidth_3dB'] = np.nan
                features['Q_factor'] = np.nan
                features['Q3dB'] = np.nan

            threshold_10db = peak_power_db - 10
            above_10db = power_db > threshold_10db
            if np.any(above_10db):
                indices_10db = np.where(above_10db)[0]
                bw_10db = freqs[indices_10db[-1]] - freqs[indices_10db[0]]
                features['bandwidth_10dB'] = bw_10db
                features['Q10dB'] = features['peak_frequency'] / bw_10db if bw_10db > 0 else np.nan
            else:
                features['bandwidth_10dB'] = np.nan
                features['Q10dB'] = np.nan

            if baseline > 0:
                features['response_intensity'] = features['peak_power'] / baseline
                features['response_intensity_db'] = 10 * np.log10(features['response_intensity'])
            else:
                features['response_intensity'] = np.nan
                features['response_intensity_db'] = np.nan

            if not np.isnan(features['freq_range_start']) and not np.isnan(features['freq_range_end']):
                in_range = (freqs >= features['freq_range_start']) & (freqs <= features['freq_range_end'])
                features['freq_range_mean_power'] = np.mean(power[in_range])
            else:
                features['freq_range_mean_power'] = np.nan

        else:
            for key in ['freq_range_start', 'freq_range_end', 'freq_range_bandwidth',
                        'freq_range_mean_power', 'Q_factor', 'bandwidth_3dB', 'Q3dB',
                        'bandwidth_10dB', 'Q10dB', 'response_intensity', 'response_intensity_db']:
                features[key] = np.nan

        return features

    def plot_tuning_analysis(self, freqs, power, features, filename=None):
        """Create interactive plot in a separate window"""
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"Frequency Tuning Plot - {os.path.basename(filename) if filename else 'Analysis'}")
        plot_window.geometry("1200x700")
        
        # Create figure
        fig = Figure(figsize=(12, 7), dpi=90)
        axes = fig.subplots(2, 1)

        power_db = 10 * np.log10(power + 1e-12)
        baseline_db = 10 * np.log10(features['baseline'] + 1e-12)
        threshold_db = 10 * np.log10(features['threshold'] + 1e-12)

        ax1 = axes[0]
        ax1.plot(freqs, power_db, 'k-', linewidth=2, label='Mechanical Response')
        ax1.fill_between(freqs, power_db, baseline_db, where=(power_db >= baseline_db),
                        alpha=0.3, color='gray', label='Response Area')

        ax1.axhline(baseline_db, color='blue', linestyle='--', linewidth=2.5,
                   label='Baseline', zorder=10)
        
        # Only plot threshold line if std method is selected
        if self.threshold_method.get() == 'std':
            ax1.axhline(threshold_db, color='cyan', linestyle='--', linewidth=2,
                       label=f'Threshold ({self.std_multiplier.get()}σ)', alpha=0.8)

        peak_freq = features['peak_frequency']
        peak_power_db = 10 * np.log10(features['peak_power'] + 1e-12)
        ax1.axvline(peak_freq, color='red', linestyle='--', linewidth=2.5,
                   label=f'Peak: {peak_freq:.1f} Hz', zorder=10)
        ax1.plot(peak_freq, peak_power_db, 'ro', markersize=12, zorder=15,
                markeredgecolor='darkred', markeredgewidth=2)

        # Plot multiple response regions if they exist
        if 'response_regions' in features and len(features['response_regions']) > 0:
            for idx, (freq_start, freq_end, mean_power) in enumerate(features['response_regions']):
                if idx == 0:
                    label = 'Range'
                else:
                    label = None
                ax1.axvline(freq_start, color='orange', linestyle='--',
                           linewidth=2.5, label=label, zorder=10)
                ax1.axvline(freq_end, color='orange', linestyle='--',
                           linewidth=2.5, zorder=10)
                ax1.axvspan(freq_start, freq_end, alpha=0.15, color='orange', zorder=5)
            
            # Show bandwidth of first (largest) region in text
            if len(features['response_regions']) > 0:
                freq_start, freq_end, _ = features['response_regions'][0]
                mid_freq = (freq_start + freq_end) / 2
                bw = freq_end - freq_start
                ax1.text(mid_freq, threshold_db + 5, f'BW: {bw:.1f} Hz',
                        ha='center', fontsize=10, bbox=dict(boxstyle='round',
                        facecolor='orange', alpha=0.3))
        elif not np.isnan(features.get('freq_range_start', np.nan)):
            # Fallback to single range for backward compatibility
            ax1.axvline(features['freq_range_start'], color='orange', linestyle='--',
                       linewidth=2.5, label='Range', zorder=10)
            ax1.axvline(features['freq_range_end'], color='orange', linestyle='--',
                       linewidth=2.5, zorder=10)
            ax1.axvspan(features['freq_range_start'], features['freq_range_end'],
                       alpha=0.15, color='orange', zorder=5)

            mid_freq = (features['freq_range_start'] + features['freq_range_end']) / 2
            bw = features['freq_range_bandwidth']
            ax1.text(mid_freq, threshold_db + 5, f'BW: {bw:.1f} Hz',
                    ha='center', fontsize=10, bbox=dict(boxstyle='round',
                    facecolor='orange', alpha=0.3))

        ax1.set_xlabel('Frequency (Hz)', fontsize=11)
        ax1.set_ylabel('Power (dB)', fontsize=11)
        ax1.set_title('Frequency Tuning Analysis (dB Scale)', fontsize=11, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(freqs, power, 'k-', linewidth=2)
        ax2.fill_between(freqs, power, features['baseline'],
                        where=(power >= features['baseline']),
                        alpha=0.3, color='gray')

        ax2.axhline(features['baseline'], color='blue', linestyle='--', linewidth=2, alpha=0.7)
        
        # Only plot threshold line in linear scale if std method is selected
        if self.threshold_method.get() == 'std':
            ax2.axhline(features['threshold'], color='cyan', linestyle='--', linewidth=2, alpha=0.7)
        ax2.axvline(peak_freq, color='red', linestyle='--', linewidth=2, alpha=0.7)

        # Plot multiple response regions if they exist
        if 'response_regions' in features and len(features['response_regions']) > 0:
            for freq_start, freq_end, _ in features['response_regions']:
                ax2.axvline(freq_start, color='orange', linestyle='--', linewidth=2, alpha=0.7)
                ax2.axvline(freq_end, color='orange', linestyle='--', linewidth=2, alpha=0.7)
                ax2.axvspan(freq_start, freq_end, alpha=0.15, color='orange')
        elif not np.isnan(features.get('freq_range_start', np.nan)):
            # Fallback to single range for backward compatibility
            ax2.axvline(features['freq_range_start'], color='orange', linestyle='--', linewidth=2, alpha=0.7)
            ax2.axvline(features['freq_range_end'], color='orange', linestyle='--', linewidth=2, alpha=0.7)
            ax2.axvspan(features['freq_range_start'], features['freq_range_end'],
                       alpha=0.15, color='orange')

        ax2.set_xlabel('Frequency (Hz)', fontsize=11)
        ax2.set_ylabel('Power (Linear)', fontsize=11)
        ax2.set_title('Linear Scale View', fontsize=11)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout(pad=1.5, h_pad=2.0)
        
        # Embed in window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add interactive toolbar
        toolbar_frame = ttk.Frame(plot_window)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        # Enable hover/click interactivity (tooltips + info dialog)
        self.canvas = canvas
        self.hover_annot = None
        self.plot_data = {
            'axes': axes,
            'freqs': freqs,
            'power': power,
            'power_db': 10 * np.log10(power + 1e-12),
            'features': features,
        }
        canvas.mpl_connect('motion_notify_event', self.on_plot_hover)
        canvas.mpl_connect('button_press_event', self.on_plot_click)

    def on_plot_hover(self, event):
        """Handle mouse hover over plot to show tooltips"""
        if event.inaxes is None or not hasattr(self, 'plot_data'):
            if self.hover_annot is not None:
                self.hover_annot.set_visible(False)
                self.canvas.draw_idle()
            return

        ax = event.inaxes
        if 'axes' not in self.plot_data or ax not in self.plot_data['axes']:
            return

        freqs = self.plot_data['freqs']
        x_data = event.xdata
        if x_data is None:
            return

        idx = np.argmin(np.abs(freqs - x_data))
        freq_val = freqs[idx]

        if ax == self.plot_data['axes'][0]:
            power_val = self.plot_data['power_db'][idx]
            label = f'Freq: {int(freq_val)} Hz\nPower: {power_val:.2f} dB'
        else:
            power_val = self.plot_data['power'][idx]
            label = f'Freq: {int(freq_val)} Hz\nPower: {power_val:.6e}'

        if self.hover_annot is None:
            self.hover_annot = ax.annotate(
                label,
                xy=(freq_val, power_val),
                xytext=(20, 20),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black'),
                fontsize=9,
                zorder=100
            )
        else:
            self.hover_annot.xy = (freq_val, power_val)
            self.hover_annot.set_text(label)
            self.hover_annot.set_visible(True)

        self.canvas.draw_idle()

    def on_plot_click(self, event):
        """Handle mouse click on plot to show detailed info"""
        if event.inaxes is None or not hasattr(self, 'plot_data'):
            return

        ax = event.inaxes
        if 'axes' not in self.plot_data or ax not in self.plot_data['axes']:
            return

        x_data = event.xdata
        if x_data is None:
            return

        freqs = self.plot_data['freqs']
        idx = np.argmin(np.abs(freqs - x_data))
        freq = freqs[idx]
        power = self.plot_data['power'][idx]
        power_db = self.plot_data['power_db'][idx]
        features = self.plot_data['features']

        info = f"Frequency Details at {freq:.2f} Hz\n\n"
        info += f"Power (dB): {power_db:.2f} dB\n"
        info += f"Power (Linear): {power:.6e}\n\n"
        info += f"Analysis Results:\n"
        info += f"  Peak Frequency: {features['peak_frequency']:.2f} Hz\n"
        info += f"  Bandwidth: {features['freq_range_bandwidth']:.2f} Hz\n"
        info += f"  Q-factor: {features['Q_factor']:.2f}\n"
        info += f"  Response Intensity: {features['response_intensity_db']:.2f} dB\n"

        if not np.isnan(features['freq_range_start']):
            if features['freq_range_start'] <= freq <= features['freq_range_end']:
                info += f"\n✓ Within response range"
            else:
                info += f"\n✗ Outside response range"

        messagebox.showinfo("Frequency Details", info)

    def display_features(self, features, filepath):
        """Display extracted features in text widget"""
        self.features_text.delete('1.0', tk.END)
        species, recording_type, filename = self._parse_file_metadata(filepath)

        text = f"{'='*60}\n"
        text += f"FREQUENCY TUNING FEATURES\n"
        text += f"{'='*60}\n\n"
        text += f"File: {filename}\n"
        text += f"Species: {species}\n"
        text += f"Recording Type: {recording_type}\n"
        text += f"\n{'-'*60}\n"
        text += f"TUNING CHARACTERISTICS\n"
        text += f"{'-'*60}\n\n"
        text += f"Peak Frequency (RED):      {features['peak_frequency']:.2f} Hz\n"
        text += f"Peak Power:                {features['peak_power']:.6e}\n"
        text += f"Peak Power (dB):           {10*np.log10(features['peak_power']) if features['peak_power'] > 0 else np.nan:.2f} dB\n"
        text += f"\n*** BANDWIDTH (Half-Power / -3dB) ***\n"
        text += f"  Frequency range where power ≥ 50% of peak\n"
        text += f"Range Start (ORANGE):      {features['freq_range_start']:.2f} Hz\n"
        text += f"Range End (ORANGE):        {features['freq_range_end']:.2f} Hz\n"
        text += f"Bandwidth (-3dB):          {features['freq_range_bandwidth']:.2f} Hz\n"
        text += f"\nQ-factor (Selectivity):    {features['Q_factor']:.2f}\n"
        text += f"  Higher Q = Sharper tuning\n"
        text += f"\n*** ADDITIONAL MEASURES ***\n"
        text += f"Bandwidth (-10dB):         {features['bandwidth_10dB']:.2f} Hz\n"
        text += f"  (Broader: power ≥ 10% of peak)\n"
        text += f"Q-factor (10dB):           {features['Q10dB']:.2f}\n"
        text += f"\nResponse Intensity (SNR):  {features['response_intensity_db']:.2f} dB\n"
        text += f"  Peak power vs baseline noise\n"
        text += f"\n{'-'*60}\n"
        text += f"DETECTION PARAMETERS\n"
        text += f"{'-'*60}\n\n"
        text += f"Baseline (BLUE):           {features['baseline']:.6e}\n"
        text += f"Threshold (CYAN):          {features['threshold']:.6e}\n"
        text += f"Number of Regions:         {features['num_response_regions']}\n"
        text += f"\n{'='*60}\n"

        self.features_text.insert('1.0', text)

    def plot_waveform_with_analysis(self):
        """Plot raw waveform with analysis annotations from selected results table row"""
        # Get selected row from results table
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a row from the Results Table to plot")
            return

        item = selection[0]
        values = self.results_tree.item(item, 'values')
        
        if not values:
            messagebox.showwarning("No Data", "Could not retrieve data from selected row")
            return

        # Get filename from first column
        filename = values[0]
        
        # Find the full filepath
        filepath = None
        for f in self.wav_files:
            if os.path.basename(f) == filename:
                filepath = f
                break

        if filepath is None:
            messagebox.showerror("File Not Found", f"Could not locate file: {filename}")
            return

        # Load WAV file
        data, sr = self.load_wav_file(filepath)
        if data is None:
            messagebox.showerror("Error", f"Could not load WAV file: {filename}")
            return

        # Compute frequency analysis to get features
        freqs, psd = self.compute_power_spectrum(data, sr)
        species, _, _ = self._parse_file_metadata(filepath)
        baseline, threshold = self.calculate_baseline_threshold(psd, species, freqs)
        features = self.extract_tuning_features(freqs, psd, baseline, threshold)

        # Create plot window
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"Waveform with Analysis - {filename}")
        plot_window.geometry("1400x600")

        # Create figure with raw waveform
        fig = Figure(figsize=(14, 6), dpi=90)
        ax = fig.add_subplot(111)
        
        # Time axis and plot raw signal
        time = np.arange(len(data)) / sr
        ax.plot(time, data, 'k-', linewidth=0.8, label='Response')
        
        # Get amplitude statistics
        baseline_level = np.mean(data)
        baseline_std = np.std(data)
        peak_freq = features['peak_frequency']
        
        # Show baseline range as shaded band (±0.5 std dev)
        ax.axhspan(baseline_level - baseline_std * 0.5, baseline_level + baseline_std * 0.5, 
                   alpha=0.2, color='blue', label='Baseline Range', zorder=1)
        ax.axhline(baseline_level, color='blue', linestyle='--', linewidth=2.5,
                   label='Baseline (BLUE)', zorder=10)
        
        # Use the SAME response regions from frequency analysis
        if 'response_regions' in features and len(features['response_regions']) > 0:
            # Display each frequency region's text label
            region_labels = []
            for idx, (freq_start, freq_end, _) in enumerate(features['response_regions']):
                region_labels.append(f"Range {idx+1}: {freq_start:.0f}-{freq_end:.0f} Hz")
            
            # Mark continuous regions above threshold with orange shading
            if np.any(np.abs(data - baseline_level) > (2 * baseline_std)):
                # Find continuous regions above threshold
                above_threshold = np.abs(data - baseline_level) > (2 * baseline_std)
                diff = np.diff(above_threshold.astype(int))
                starts = np.where(diff == 1)[0] + 1
                ends = np.where(diff == -1)[0] + 1
                
                if len(data) > 0 and above_threshold[0]:
                    starts = np.insert(starts, 0, 0)
                if len(data) > 0 and above_threshold[-1]:
                    ends = np.append(ends, len(above_threshold))
                
                # Shade all response regions in orange
                for start_idx, end_idx in zip(starts, ends):
                    ax.axvspan(time[start_idx], time[min(end_idx, len(time)-1)], 
                              alpha=0.15, color='orange', zorder=2)
        
        # Mark the main peak (RED) - from frequency analysis
        peak_idx = np.argmax(np.abs(data - baseline_level))
        peak_time = time[peak_idx]
        peak_value = data[peak_idx]
        ax.axvline(peak_time, color='red', linestyle='--', linewidth=2.5,
                   label=f'Peak: {peak_freq:.1f} Hz (RED)', zorder=10)
        ax.plot(peak_time, peak_value, 'ro', markersize=10, zorder=15,
                markeredgecolor='darkred', markeredgewidth=2)
        
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        
        # Add title with frequency analysis summary
        title_str = f"Raw Waveform - {filename}\n"
        title_str += f"Peak Frequency: {peak_freq:.1f} Hz | Bandwidth: {features['freq_range_bandwidth']:.1f} Hz | Q-factor: {features['Q_factor']:.2f} | SNR: {features['response_intensity_db']:.1f} dB"
        ax.set_title(title_str, fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add all response ranges as text annotation
        ranges_text = "Detected Ranges (ORANGE):\n"
        if 'response_regions' in features and len(features['response_regions']) > 0:
            for idx, (freq_start, freq_end, _) in enumerate(features['response_regions']):
                ranges_text += f"Range {idx+1}: {freq_start:.0f} - {freq_end:.0f} Hz\n"
        else:
            ranges_text += "None detected"
        
        ax.text(0.02, 0.97, ranges_text.strip(), 
                transform=ax.transAxes, ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3), zorder=20)
        
        # Add baseline range indicator (frequency range as text annotation)
        baseline_min = self.baseline_freq_min.get()
        baseline_max = self.baseline_freq_max.get()
        ax.text(0.98, 0.97, f'Baseline Range (Freq): {baseline_min:.0f} - {baseline_max:.0f} Hz', 
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3), zorder=20)
        
        ax.legend(loc='lower right', fontsize=9)

        fig.tight_layout()

        # Embed in window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add toolbar
        toolbar_frame = ttk.Frame(plot_window)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

    def analyze_selected_file(self):
        """Analyze the selected file"""
        selection = self.file_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a file to analyze")
            return

        idx = selection[0]
        if idx >= len(self.displayed_files):
            return

        filepath = self.displayed_files[idx]
        self._analyze_single_file(filepath)

    def _process_files_common(self, filepaths):
        """Common logic for processing files"""
        results_list = []

        for i, filepath in enumerate(filepaths, 1):
            filename = os.path.basename(filepath)
            self.log(f"[{i}/{len(filepaths)}] Processing: {filename}")

            try:
                species, direction, _ = self._parse_file_metadata(filepath)

                data, sr = self.load_wav_file(filepath)
                if data is None:
                    continue

                freqs, psd = self.compute_power_spectrum(data, sr)
                baseline, threshold = self.calculate_baseline_threshold(psd, species, freqs)
                features = self.extract_tuning_features(freqs, psd, baseline, threshold)
                
                # Cache the complete features for this file
                self.features_cache[filename] = {
                    'features': features.copy(),
                    'freqs': freqs.copy(),
                    'psd': psd.copy(),
                    'analysis_params': {
                        'min_freq': self.min_freq.get(),
                        'max_freq': self.max_freq.get(),
                        'baseline_freq_min': self.baseline_freq_min.get(),
                        'baseline_freq_max': self.baseline_freq_max.get(),
                        'threshold_method': self.threshold_method.get(),
                        'std_multiplier': self.std_multiplier.get(),
                        'baseline_freq_std_multiplier': self.baseline_freq_std_multiplier.get(),
                        'percentile_threshold': self.percentile_threshold.get()
                    }
                }

                results_list.append({
                    'filename': filename,
                    'species': species,
                    'direction': direction,
                    **features
                })
            except Exception as e:
                self.log(f"  ❌ Error: {str(e)}")

        return results_list

    def _parse_file_metadata(self, filepath):
        """Extract species, direction, and filename with mechanical handling"""
        parts = filepath.split(os.sep)
        filename = os.path.basename(filepath)

        species = "Unknown"
        direction = ""

        try:
            if "Wav_data_mech" in parts:
                idx = parts.index("Wav_data_mech")
                if idx + 1 < len(parts):
                    species = parts[idx + 1]
                direction = ""
            elif "Wav_data_nerve" in parts:
                idx = parts.index("Wav_data_nerve")
                if idx + 1 < len(parts):
                    species = parts[idx + 1]
                if idx + 2 < len(parts):
                    dir_candidate = parts[idx + 2].lower()
                    if dir_candidate in ['forward', 'backward']:
                        direction = parts[idx + 2]
            else:
                if len(parts) >= 3:
                    species = parts[-3]
                    dir_candidate = parts[-2].lower()
                    if dir_candidate in ['forward', 'backward']:
                        direction = parts[-2]
        except ValueError:
            pass

        return species, direction, filename

    def _analyze_single_file(self, filepath):
        """Internal method to analyze a single file and display results"""
        self.log(f"▶ Analyzing: {os.path.basename(filepath)}")

        data, sr = self.load_wav_file(filepath)
        if data is None:
            return

        self.log(f"  Loaded {len(data)} samples at {sr} Hz ({len(data)/sr:.2f}s)")

        species, direction, filename = self._parse_file_metadata(filepath)

        freqs, psd = self.compute_power_spectrum(data, sr)
        self.log(f"  Computed power spectrum: {len(freqs)} frequency bins")

        baseline, threshold = self.calculate_baseline_threshold(psd, species, freqs)
        self.log(f"  Baseline: {baseline:.6e}, Threshold: {threshold:.6e}")

        features = self.extract_tuning_features(freqs, psd, baseline, threshold)
        self.log(f"  Peak frequency: {features['peak_frequency']:.2f} Hz")
        self.log(f"  Bandwidth: {features['freq_range_bandwidth']:.2f} Hz")
        self.log(f"  Q-factor: {features['Q_factor']:.2f}")
        
        # Cache the complete features (including response_regions) for this file
        filename = os.path.basename(filepath)
        self.features_cache[filename] = {
            'features': features.copy(),
            'freqs': freqs.copy(),
            'psd': psd.copy(),
            'analysis_params': {
                'min_freq': self.min_freq.get(),
                'max_freq': self.max_freq.get(),
                'baseline_freq_min': self.baseline_freq_min.get(),
                'baseline_freq_max': self.baseline_freq_max.get(),
                'threshold_method': self.threshold_method.get(),
                'std_multiplier': self.std_multiplier.get(),
                'baseline_freq_std_multiplier': self.baseline_freq_std_multiplier.get(),
                'percentile_threshold': self.percentile_threshold.get()
            }
        }

        self.current_data = data
        self.current_freqs = freqs
        self.current_psd = psd
        self.current_features = features
        self.current_filepath = filepath

        # Infer recording type from path for single analysis
        rec_type = 'mechanical' if 'Wav_data_mech' in filepath else ('electrical' if 'Wav_data_nerve' in filepath else 'unknown')

        result = {
            'filename': os.path.basename(filepath),
            'species': species,
            'direction': direction,
            'recording_type': rec_type,
            **features
        }

        if self.results_df is None:
            self.results_df = pd.DataFrame([result])
        else:
            existing_mask = self.results_df['filename'] == result['filename']
            if existing_mask.any():
                for key, value in result.items():
                    self.results_df.loc[existing_mask, key] = value
            else:
                self.results_df = pd.concat([self.results_df, pd.DataFrame([result])], ignore_index=True)

        self._update_results_table()
        self.display_features(features, filepath)

        # Show annotated tuning plot
        try:
            self.plot_tuning_analysis(freqs, psd, features, filepath)
        except Exception as e:
            self.log(f"⚠️  Could not render plot: {e}")

        self.log("✓ Analysis complete")

    def batch_process_folder(self):
        """Batch process files from a selected folder"""
        # Use current input directory as default
        default_dir = self.input_dir.get() if os.path.isdir(self.input_dir.get()) else self.cwd
        
        # Show combined config dialog
        config = self._show_batch_config_dialog(default_dir)
        if config is None:
            return
        
        folder = config['folder']
        
        wav_files = []
        for root, dirs, files in os.walk(folder):
            for fname in files:
                if fname.lower().endswith('.wav'):
                    wav_files.append(os.path.join(root, fname))

        if len(wav_files) == 0:
            messagebox.showwarning("No Files", f"No WAV files found in:\n{folder}")
            return

        # Filter files by selected species if not "All"
        if config.get('species') != 'All':
            filtered_files = []
            for wav_file in wav_files:
                species, _, _ = self._parse_file_metadata(wav_file)
                if species == config['species']:
                    filtered_files.append(wav_file)
            wav_files = filtered_files
            
            if len(wav_files) == 0:
                messagebox.showwarning("No Files", f"No WAV files found for species: {config['species']}")
                return

        original_sr = self.sampling_rate.get()
        self.sampling_rate.set(config['sampling_rate'])

        self.log(f"▶▶ Starting batch processing of {len(wav_files)} files...")
        self.log(f"   Folder: {folder}")
        self.log(f"   Recording Type: {config['recording_type']}")
        self.log(f"   Species Filter: {config.get('species', 'All')}")
        self.log(f"   Sampling Rate: {config['sampling_rate']} Hz")

        self.analyze_btn.config(state=tk.DISABLED)
        self.batch_btn.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._batch_process_thread,
                                 args=(wav_files, original_sr, config['recording_type']))
        thread.daemon = True
        thread.start()

    def _show_batch_config_dialog(self, default_dir):
        """Show dialog to configure batch processing parameters"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Batch Processing Configuration")
        dialog.geometry("500x580")
        dialog.transient(self.root)
        dialog.grab_set()

        result = {'folder': default_dir, 'sampling_rate': self.sampling_rate.get(), 
                 'recording_type': 'mechanical', 'species': 'All'}

        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Batch Processing Configuration",
                 font=('', 11, 'bold')).pack(pady=(0, 15))

        # Folder selection
        folder_frame = ttk.LabelFrame(main_frame, text="Input Folder", padding="10")
        folder_frame.pack(fill=tk.X, pady=(0, 15))

        folder_var = tk.StringVar(value=default_dir)
        
        def browse_folder():
            folder = filedialog.askdirectory(title="Select Folder to Batch Process", initialdir=folder_var.get())
            if folder:
                folder_var.set(folder)
                update_file_count()

        def update_file_count():
            folder = folder_var.get()
            if not os.path.isdir(folder):
                file_count_label.config(text="Invalid folder")
                return
            
            # Count WAV files
            count = 0
            for root, dirs, files in os.walk(folder):
                for fname in files:
                    if fname.lower().endswith('.wav'):
                        count += 1
            
            # Detect species
            detected_species = set()
            for root, dirs, files in os.walk(folder):
                for fname in files:
                    if fname.lower().endswith('.wav'):
                        full_path = os.path.join(root, fname)
                        species, _, _ = self._parse_file_metadata(full_path)
                        if species != "Unknown":
                            detected_species.add(species)
            
            # Update species radio buttons
            for widget in species_control.winfo_children():
                widget.destroy()
            
            ttk.Radiobutton(species_control, text="All", variable=species_var, value='All').pack(side=tk.LEFT, padx=(0, 10))
            for species in sorted(detected_species):
                ttk.Radiobutton(species_control, text=species, variable=species_var, value=species).pack(side=tk.LEFT, padx=(0, 10))
            
            file_count_label.config(text=f"Found {count} WAV files")

        folder_entry_frame = ttk.Frame(folder_frame)
        folder_entry_frame.pack(fill=tk.X)
        ttk.Entry(folder_entry_frame, textvariable=folder_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(folder_entry_frame, text="Browse", command=browse_folder, width=10).pack(side=tk.LEFT)

        file_count_label = ttk.Label(folder_frame, text="", foreground='gray', font=('', 8))
        file_count_label.pack(pady=(5, 0))

        # Recording Type Selection - auto-set from input_type_var
        rec_frame = ttk.LabelFrame(main_frame, text="Recording Type", padding="10")
        rec_frame.pack(fill=tk.X, pady=(0, 15))

        # Default based on current input directory selection
        default_rec = 'mechanical' if self.input_type_var.get() == 'mech' else 'electrical'
        rec_var = tk.StringVar(value=default_rec)

        def update_for_recording():
            """Auto-adjust sampling rate based on recording type"""
            if rec_var.get() == 'mechanical':
                sr_var.set(100000)
            else:
                sr_var.set(20000)

        rec_control = ttk.Frame(rec_frame)
        rec_control.pack(fill=tk.X)
        ttk.Radiobutton(rec_control, text="Mechanical Tuning", variable=rec_var, 
                       value='mechanical', command=update_for_recording).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Radiobutton(rec_control, text="Electrical Tuning", variable=rec_var, 
                       value='electrical', command=update_for_recording).pack(side=tk.LEFT)

        ttk.Label(rec_frame, text="Applied to all files in batch",
                 foreground='gray', font=('', 8)).pack(pady=(5, 0))

        # Species Selection - dynamically populated
        species_frame = ttk.LabelFrame(main_frame, text="Species", padding="10")
        species_frame.pack(fill=tk.X, pady=(0, 15))

        species_var = tk.StringVar(value='All')
        species_control = ttk.Frame(species_frame)
        species_control.pack(fill=tk.X)
        
        # Sampling Rate
        sr_frame = ttk.LabelFrame(main_frame, text="Sampling Rate", padding="10")
        sr_frame.pack(fill=tk.X, pady=(0, 15))

        sr_var = tk.IntVar(value=self.sampling_rate.get())
        update_for_recording()

        sr_control = ttk.Frame(sr_frame)
        sr_control.pack(fill=tk.X)
        ttk.Spinbox(sr_control, from_=10000, to=200000, increment=10000,
                   textvariable=sr_var, width=12).pack(side=tk.LEFT)
        ttk.Label(sr_control, text="Hz").pack(side=tk.LEFT, padx=(5, 0))

        ttk.Label(sr_frame, text="Note: Must match your WAV file specifications",
                 foreground='gray', font=('', 8)).pack(pady=(5, 0))

        # Current analysis settings
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 15))
        ttk.Label(info_frame, text="Current analysis settings:",
                 font=('', 9)).pack(anchor=tk.W)
        
        # Frequency range
        freq_text = f"  • Frequency Range: {self.min_freq.get():.0f}-{self.max_freq.get():.0f} Hz"
        ttk.Label(info_frame, text=freq_text, foreground='blue').pack(anchor=tk.W)
        
        # Threshold method
        method_text = f"  • Baseline Method: {self.threshold_method.get()}"
        if self.threshold_method.get() == 'std':
            method_text += f" ({self.std_multiplier.get()}σ)"
        elif self.threshold_method.get() == 'freq_range':
            method_text += f" ({self.baseline_freq_min.get():.0f}-{self.baseline_freq_max.get():.0f} Hz)"
        ttk.Label(info_frame, text=method_text, foreground='blue').pack(anchor=tk.W)

        # Initialize file count
        update_file_count()

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=(10, 0))

        def on_ok():
            folder = folder_var.get()
            if not os.path.isdir(folder):
                messagebox.showerror("Invalid Folder", "Please select a valid folder")
                return
            result['folder'] = folder
            result['sampling_rate'] = sr_var.get()
            result['recording_type'] = rec_var.get()
            result['species'] = species_var.get()
            result['confirmed'] = True
            dialog.destroy()

        def on_cancel():
            result['confirmed'] = False
            dialog.destroy()

        ttk.Button(btn_frame, text="Start Processing", command=on_ok, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Cancel", command=on_cancel, width=15).pack(side=tk.LEFT)

        self.root.wait_window(dialog)

        return result if result.get('confirmed') else None

    def _batch_process_thread(self, wav_files, original_sr, recording_type='mechanical'):
        """Thread function for batch processing"""
        results_list = self._process_files_common(wav_files)
        # Override recording_type for all files in batch
        for result in results_list:
            result['recording_type'] = recording_type
        self.results_df = pd.DataFrame(results_list)

        output_dir = self.output_dir.get()
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, 'tuning_features_all.csv')
        self.results_df.to_csv(csv_path, index=False)
        self.log(f"✓ Saved results to: {csv_path}")

        groupby_cols = ['species', 'recording_type']
        summary = self.results_df.groupby(groupby_cols).agg({
            'peak_frequency': ['count', 'mean', 'std', 'min', 'max'],
            'freq_range_bandwidth': ['mean', 'std'],
            'Q_factor': ['mean', 'std'],
            'response_intensity_db': ['mean', 'std']
        }).round(2)

        summary_path = os.path.join(output_dir, 'tuning_summary_statistics.csv')
        summary.to_csv(summary_path)
        self.log(f"✓ Saved summary statistics to: {summary_path}")

        self.root.after(0, self._update_results_table)
        self.root.after(0, lambda: self.sampling_rate.set(original_sr))
        self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.batch_btn.config(state=tk.NORMAL))

        self.log(f"✓✓ Batch processing complete! Processed {len(results_list)}/{len(wav_files)} files")

    def _update_results_table(self):
        """Update the results treeview"""
        if self.results_df is None:
            return

        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        columns = ['filename', 'species', 'direction', 'peak_frequency',
                  'num_response_regions', 'freq_range_start', 'freq_range_end', 'freq_range_bandwidth',
                  'Q_factor', 'Q3dB', 'Q10dB', 'response_intensity_db']
        
        # Add individual range columns dynamically
        max_ranges = 0
        if 'num_response_regions' in self.results_df.columns:
            max_ranges = int(self.results_df['num_response_regions'].max())
        
        for r in range(1, max_ranges + 1):
            columns.extend([f'range_{r}_start', f'range_{r}_end', f'range_{r}_bandwidth'])
        self.results_tree['columns'] = columns
        self.results_tree['show'] = 'headings'

        heading_labels = {
            'filename': 'Filename',
            'species': 'Species',
            'direction': 'Direction',
            'peak_frequency': 'Peak (Hz)',
            'num_response_regions': 'Num Ranges',
            'freq_range_start': 'Start (Hz)',
            'freq_range_end': 'End (Hz)',
            'freq_range_bandwidth': 'BW (Hz)',
            'Q_factor': 'Q-factor',
            'Q3dB': 'Q-3dB',
            'Q10dB': 'Q-10dB',
            'response_intensity_db': 'Intensity (dB)'
        }
        
        # Add labels for individual ranges
        for r in range(1, max_ranges + 1):
            heading_labels[f'range_{r}_start'] = f'R{r} Start'
            heading_labels[f'range_{r}_end'] = f'R{r} End'
            heading_labels[f'range_{r}_bandwidth'] = f'R{r} BW'

        for col in columns:
            self.results_tree.heading(col, text=heading_labels.get(col, col))
            if col == 'filename':
                self.results_tree.column(col, width=200, minwidth=150)
            elif col in ['species', 'direction']:
                self.results_tree.column(col, width=100, minwidth=80)
            else:
                self.results_tree.column(col, width=80, minwidth=60)

        for idx, row in self.results_df.iterrows():
            values = [row[col] if col in row else '' for col in columns]
            formatted = []
            for i, val in enumerate(values):
                if i > 2 and isinstance(val, (int, float)) and not np.isnan(val):
                    formatted.append(f"{val:.2f}")
                else:
                    formatted.append(str(val))
            self.results_tree.insert('', tk.END, values=formatted)

    def show_column_help(self):
        """Show a window explaining what each results column means"""
        help_window = tk.Toplevel(self.root)
        help_window.title("Results Column Help")
        help_window.geometry("900x700")
        help_window.transient(self.root)
        help_window.grab_set()
        
        # Create scrollable text area
        help_text = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, font=('Courier', 10), padx=10, pady=10)
        help_text.pack(fill=tk.BOTH, expand=True)
        
        # Add explanations for each column
        explanations = """FREQUENCY TUNING RESULTS - COLUMN EXPLANATIONS
════════════════════════════════════════════════════════════════════════════════

BASIC INFO:
───────────
• filename: Name of the analyzed WAV file
• species: Mosquito species detected from folder structure (Culex, aedes aegypti, etc.)
• direction: Audio segment direction (forward/backward) or empty for mechanical recordings

PEAK FREQUENCY & POWER:
──────────────────────
• peak_frequency [Hz]:
  The frequency at which the sensory neuron responds MOST STRONGLY.
  This is the "preferred" frequency for that mechanoreceptor.
  Example: 287 Hz means the neuron fires strongest at 287 Hz.

• peak_power [linear]:
  The raw signal power (amplitude²) at the peak frequency.
  Higher values = stronger neural response at peak frequency.

FREQUENCY RANGE & BANDWIDTH:
────────────────────────────
• freq_range_start & freq_range_end [Hz]:
  The frequency range where the neuron RESPONDS (above threshold).
  These define the boundaries of the "tuning curve".
  Example: If range is 250-350 Hz, the neuron responds to wing strokes between 250-350 Hz.

• freq_range_bandwidth [Hz]:
  The WIDTH of the response range = (freq_range_end - freq_range_start)
  Narrower = more selective (sharp tuning), Wider = less selective (broad tuning)
  Example: 100 Hz bandwidth means the neuron responds across a 100 Hz window.

Q-FACTOR (TUNING SHARPNESS):
────────────────────────────
• Q_factor (dimensionless):
  Q = peak_frequency / bandwidth
  
  Measures how SHARP the tuning is:
  - Q > 5: SHARP tuning (highly selective for specific frequency)
  - Q = 2-5: MODERATE tuning (selective but responsive to range)
  - Q < 2: BROAD tuning (responds to many frequencies)
  
  Example: Q = 300 Hz / 50 Hz = 6.0 (sharp tuning)

• Q3dB & Q10dB:
  Q-factor calculated at -3 dB and -10 dB points (alternative measures)
  -3 dB = half-power point (50% of peak power)
  -10 dB = lower sensitivity threshold

BANDWIDTH VARIANTS:
────────────────────
• bandwidth_3dB [Hz]:
  Frequency range where power stays ABOVE 50% of peak.
  Narrower = sharper tuning curve.
  (Standard measure in electrophysiology)

• bandwidth_10dB [Hz]:
  Frequency range where power stays ABOVE 10% of peak.
  Broader than 3dB bandwidth, captures overall response width.

RESPONSE INTENSITY:
────────────────────
• response_intensity (dimensionless):
  Ratio = peak_power / baseline
  How much STRONGER the response is compared to background noise.
  Example: 50 = response is 50x stronger than baseline noise.
  Higher = clearer signal, more robust neuronal firing.

• response_intensity_db [dB]:
  10 × log₁₀(response_intensity)
  Logarithmic representation of signal strength.
  20 dB = 10x amplification
  40 dB = 100x amplification

BASELINE & THRESHOLD:
──────────────────────
• baseline [linear]:
  Average power in the background (when no strong response).
  Represents noise floor or spontaneous activity.

• threshold [linear]:
  Cutoff level for detecting a "response".
  Points above threshold count as neuron responding.
  Calculated as: baseline + (multiplier × standard_deviation)
  Adjustable via "Std Multiplier" in Analysis Parameters.

NUM_RESPONSE_REGIONS:
──────────────────────
• num_response_regions (integer):
  Number of SEPARATE frequency ranges where neuron responds.
  Usually 1, but can be >1 if response has gaps (bimodal/multimodal tuning).
  Example: 2 = neuron responds at 200-250 Hz AND 400-450 Hz (bandpass filter effect).

════════════════════════════════════════════════════════════════════════════════
TYPICAL VALUES FOR MOSQUITO MECHANORECEPTORS:

Culex pipiens J-organ neurons:
  • Peak frequency: 250-350 Hz (species-specific)
  • Q-factor: 3-8 (moderately sharp tuning)
  • Bandwidth: 50-150 Hz
  • Response intensity: 10-100 dB above baseline

Aedes aegypti Johnston's organ neurons:
  • Peak frequency: 200-400 Hz (varies by neuron type)
  • Q-factor: 2-6 (broader than Culex)
  • Bandwidth: 80-200 Hz
  • Response intensity: 5-80 dB above baseline
"""
        
        help_text.insert(tk.END, explanations)
        help_text.config(state=tk.DISABLED)  # Read-only
        
        # Add close button
        btn_frame = ttk.Frame(help_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Close", command=help_window.destroy).pack()

    def plot_selected_results(self):
        """Plot selected rows from results table in interactive window"""
        selected_items = self.results_tree.selection()
        
        if not selected_items:
            messagebox.showwarning("No Selection", "Please select one or more rows from the results table to plot.")
            return
        
        if self.results_df is None or len(self.results_df) == 0:
            messagebox.showinfo("No Data", "No results available to plot. Run batch processing first.")
            return

        # If a single file is selected, show the detailed annotated plot
        if len(selected_items) == 1:
            item = selected_items[0]
            values = self.results_tree.item(item)['values']
            if not values:
                messagebox.showerror("Plot Error", "Could not read selected row.")
                return

            filename = values[0]
            filepath = next((f for f in self.wav_files if os.path.basename(f) == filename), None)

            if not filepath:
                messagebox.showerror("File Missing", f"Could not locate file for {filename}")
                return

            # Check if we have cached features from the original analysis
            if filename in self.features_cache:
                cached = self.features_cache[filename]
                freqs = cached['freqs']
                psd = cached['psd']
                features = cached['features']
                
                # Check if analysis parameters have changed
                params_changed = False
                if 'analysis_params' in cached:
                    old_params = cached['analysis_params']
                    current_params = {
                        'min_freq': self.min_freq.get(),
                        'max_freq': self.max_freq.get(),
                        'baseline_freq_min': self.baseline_freq_min.get(),
                        'baseline_freq_max': self.baseline_freq_max.get(),
                        'threshold_method': self.threshold_method.get(),
                        'std_multiplier': self.std_multiplier.get(),
                        'baseline_freq_std_multiplier': self.baseline_freq_std_multiplier.get(),
                        'percentile_threshold': self.percentile_threshold.get()
                    }
                    
                    if old_params != current_params:
                        params_changed = True
                        changed_params = [k for k in old_params.keys() if old_params.get(k) != current_params.get(k)]
                        self.log(f"⚠️  Analysis parameters have changed since original analysis: {', '.join(changed_params)}")
                        self.log(f"   Using cached results with original parameters")
                
                self.log(f"✓ Using cached analysis for {filename}")
            else:
                # Recalculate if not in cache
                data, sr = self.load_wav_file(filepath)
                if data is None:
                    return

                freqs, psd = self.compute_power_spectrum(data, sr)
                species, _, _ = self._parse_file_metadata(filepath)
                baseline, threshold = self.calculate_baseline_threshold(psd, species, freqs)
                features = self.extract_tuning_features(freqs, psd, baseline, threshold)
                self.log(f"⚠️  Recalculated analysis for {filename} (not in cache)")

            self.plot_tuning_analysis(freqs, psd, features, filepath)
            self.log(f"✓ Opened annotated plot for {filename}")
            return
        
        # Create new window for interactive plotting
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Interactive Frequency Tuning Plots")
        plot_window.geometry("1200x700")
        
        # Create figure with interactive toolbar
        fig = Figure(figsize=(12, 7), dpi=90)
        ax = fig.add_subplot(111)
        
        # Plot each selected row
        colors = plt.cm.tab20(np.linspace(0, 1, len(selected_items)))
        plot_data = []  # Store plot data for hover annotations
        
        for idx, item in enumerate(selected_items):
            values = self.results_tree.item(item)['values']
            if not values:
                continue

            filename = values[0]
            filepath = next((f for f in self.wav_files if os.path.basename(f) == filename), None)
            if not filepath:
                self.log(f"[!] Could not find file for {filename}")
                continue

            try:
                data, sr = self.load_wav_file(filepath)
                if data is None:
                    continue

                # Compute PSD using the same helper to keep ranges consistent
                freqs, psd = self.compute_power_spectrum(data, sr)

                # Plot in dB scale
                power_db = 10 * np.log10(psd + 1e-12)
                label = os.path.basename(filepath)
                line, = ax.plot(freqs, power_db, linewidth=2.5, color=colors[idx],
                               label=label, marker='o', markersize=3, alpha=0.8)
                plot_data.append({'line': line, 'freqs': freqs, 'power_db': power_db, 'label': label})
            except Exception as e:
                self.log(f"[!] Could not plot {filename}: {str(e)}")
                continue
        
        if not plot_data:
            messagebox.showerror("Plot Error", "Could not load data for selected files.")
            plot_window.destroy()
            return
        
        # Format axes
        ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Power (dB)', fontsize=12, fontweight='bold')
        ax.set_title('Selected Frequency Tuning Curves', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        
        # Embed figure in window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add interactive toolbar
        toolbar_frame = ttk.Frame(plot_window)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        # Create hover annotation for interactive points
        annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                           bbox=dict(boxstyle="round,pad=0.5", fc='yellow', alpha=0.9),
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                           fontsize=9, family='monospace')
        annot.set_visible(False)
        
        def on_hover(event):
            """Show frequency/power values when hovering over plot lines"""
            if event.inaxes != ax or not plot_data:
                annot.set_visible(False)
                canvas.draw_idle()
                return
            
            # Find closest point across all lines
            min_distance = float('inf')
            closest_point = None
            closest_label = None
            
            for plot_info in plot_data:
                line = plot_info['line']
                freqs = plot_info['freqs']
                power_db = plot_info['power_db']
                
                # Find closest frequency to mouse x position
                if event.xdata is not None:
                    idx = np.argmin(np.abs(freqs - event.xdata))
                    # Calculate distance to line
                    data_x = freqs[idx]
                    data_y = power_db[idx]
                    
                    # Transform to display coordinates
                    display_x, display_y = ax.transData.transform((data_x, data_y))
                    mouse_x, mouse_y = canvas.get_tk_widget().winfo_pointerx() - canvas.get_tk_widget().winfo_rootx(), \
                                       canvas.get_tk_widget().winfo_pointery() - canvas.get_tk_widget().winfo_rooty()
                    
                    # Simple euclidean distance in display space
                    dist = np.sqrt((display_x - event.x)**2 + (display_y - event.y)**2)
                    
                    if dist < min_distance and dist < 20:  # Only show if within 20 pixels
                        min_distance = dist
                        closest_point = (data_x, data_y)
                        closest_label = plot_info['label']
            
            if closest_point is not None:
                annot.xy = closest_point
                annot.set_text(f"{closest_label}\nFreq: {closest_point[0]:.1f} Hz\nPower: {closest_point[1]:.2f} dB")
                annot.set_visible(True)
                canvas.draw_idle()
            else:
                annot.set_visible(False)
                canvas.draw_idle()
        
        # Bind hover event
        canvas.mpl_connect('motion_notify_event', on_hover)
        
        self.log(f"✓ Opened interactive plot with {len(selected_items)} selected file(s)")

    def export_results(self):
        """Export results to user-selected file"""
        if self.results_df is None:
            messagebox.showinfo("No Data", "No results to export. Run batch processing first.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="tuning_features_export.csv"
        )

        if filepath:
            self.results_df.to_csv(filepath, index=False)
            self.log(f"✓ Exported results to: {filepath}")
            messagebox.showinfo("Export Complete", f"Results exported to:\n{filepath}")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = FrequencyAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()