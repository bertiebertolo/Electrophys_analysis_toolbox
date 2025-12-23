# Mosquito Frequency Analysis Toolkit

Comprehensive toolkit for extracting frequency tuning features from mosquito mechanoreceptor electrophysiology recordings.

## Requirements
- **Windows or Linux** 
- Python 3.8+
- Conda or Miniconda

**Note**: macOS (especially Apple Silicon) is not supported due to dependencies on `neo`/`sonpy` libraries for .smr file conversion.

## Quick Start

### 1. Install
```bash
# Clone or download this repository
cd mosquito_electrophys_toolbox

# Create conda environment
conda env create -f environment.yml
conda activate mosquito_analysis
```

### 2. Launch GUI
```bash
python run_gui.py
```

The GUI provides:
- File browsing and selection
- Real-time parameter adjustment
- Frequency tuning visualization
- Batch processing
- CSV export
- Species comparison plots

### 3. Or Use Jupyter Notebooks

**For Analysis:**
```bash
jupyter lab
# Open Frequency_Range_Analysis.ipynb
```

**For .smr → WAV Conversion:**
```bash
jupyter lab
# Open smr_to_wav_converter.ipynb
```

## Data Structure

Create these folders in your working directory:

```
Wav_data_mech/{species}/          # Filtered WAV files (100 kHz, 0-1000 Hz)
Mech_data_smr/{species}/           # Optional: raw Spike2 .smr files
frequency_range_analysis/          # Auto-created: outputs (CSV, plots)
```

**Species folder examples:** `Culex`, `aedes aegypti`

## Workflow Options

### Option 1: GUI (No Coding)
1. Place filtered WAV files in `Wav_data_mech/{species}/`
2. Launch GUI: `python run_gui.py`
3. Select files → Adjust parameters → Analyze
4. Export results to CSV

### Option 2: Jupyter Analysis
1. Place filtered WAV files in `Wav_data_mech/{species}/`
2. Open `Frequency_Range_Analysis.ipynb`
3. Run all cells
4. Results saved to `frequency_range_analysis/`

### Option 3: Full Pipeline (from .smr)
1. Place raw .smr files in `Mech_data_smr/{species}/`
2. Convert using GUI (**SMR Tools → Convert .smr to WAV**)
   - OR run `smr_to_wav_converter.ipynb`
3. Analyze converted WAVs (GUI or notebook)

## What Gets Extracted

- **Peak Frequency** (Hz) - Maximum response
- **Frequency Range** (start, end, bandwidth)
- **Q-factor** - Tuning sharpness (peak/bandwidth)
- **-3dB/-10dB Bandwidths**
- **Response Intensity** (dB)
- **Multiple Response Regions** - All detected ranges

## Technical Details

### Signal Processing
- **Sampling Rate**: 100,000 Hz
- **Filter**: Butterworth bandpass 0-1000 Hz (order 4)
- **Normalization**: Max absolute value
- **Analysis Range**: 200-1000 Hz (default)
- **Baseline Band**: 800-1000 Hz (default)

### Analysis Method
- **PSD**: Welch's method (nperseg=4096)
- **Threshold**: Baseline band mean + 2.0σ (adjustable)
- **Detection**: Continuous regions above threshold + -3dB fallback

### Outputs
- `tuning_features_all.csv` - All extracted features per file
- `tuning_summary_statistics.csv` - Grouped statistics
- Individual tuning plots (frequency + waveform views)
- Species comparison plots (6-panel)

## File Descriptions

- **`frequency_analysis_gui.py`** - Main GUI application
- **`run_gui.py`** - Simple GUI launcher
- **`Frequency_Range_Analysis.ipynb`** - Analysis notebook (WAV → features)
- **`smr_to_wav_converter.ipynb`** - Conversion notebook (.smr → WAV)
- **`environment.yml`** - Conda environment specification

## Key Dependencies

- NumPy, Pandas, SciPy (signal processing)
- Librosa (audio analysis)
- Matplotlib, Seaborn (visualization)
- Neo, Sonpy (Spike2 file reading)
- Soundfile (WAV I/O)
- tkinter (GUI)

## Support

For issues or questions, check:
- In-GUI status panel and logs
- Notebook markdown cells for step-by-step explanations
- Parameter tooltips in GUI

## License

MIT License - Research tool provided as-is.

---

**Version**: December 2025  
**Platform**: Windows/Linux (x86_64)  
**Python**: 3.8+
