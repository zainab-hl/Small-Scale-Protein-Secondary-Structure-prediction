# Protein Secondary Structure Prediction

A deep learning approach to predict protein secondary structure (alpha-helix, beta-sheet, coil) from amino acid sequences using a 1D Convolutional Neural Network.

## Overview

This project implements a CNN-based classifier to predict the three main types of protein secondary structures:
- **H (Alpha-helix)**: Spiral structures stabilized by hydrogen bonds
- **E (Beta-sheet)**: Extended strands that form sheets through hydrogen bonding
- **C (Coil)**: Random coil regions with no regular structure

The model takes amino acid sequences as input and outputs per-residue secondary structure predictions, enabling researchers to understand protein folding patterns without performing computationally expensive 3D structure prediction.

## Project Structure

```
├── README.md
├── data/
│   ├── test.csv          # Test dataset
│   ├── train.csv         # Training dataset
│   └── valid.csv         # Validation dataset
├── requirements.txt      # Python dependencies
└── src/
    ├── app.py           # Main application interface
    ├── cnn.py           # CNN model implementation and training
    ├── dataset.py       # Data preprocessing and loading
    └── visualize.py     # Visualization utilities
```

## Model Architecture

The CNN model (`CNNSSPredictor`) features:
- **Input**: One-hot encoded amino acid sequences (20 features per position)
- **Layer 1**: 1D Convolution (kernel=7, channels=64) with ReLU activation
- **Layer 2**: 1D Convolution (kernel=5, channels=128) with ReLU activation  
- **Layer 3**: 1D Convolution (kernel=3, channels=64) with ReLU activation
- **Output**: 1D Convolution (kernel=1, channels=3) for classification
- **Sequence Length**: Fixed at 512 residues (padded/truncated as needed)

## Performance Metrics

The model achieves the following performance on the test set:

| Structure Type | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| **C (Coil)**   | 0.6690    | 0.7702 | 0.7160   | 58,858  |
| **E (Beta-sheet)** | 0.6607 | 0.4223 | 0.5153   | 29,902  |
| **H (Alpha-helix)** | 0.6975 | 0.7259 | 0.7114   | 46,162  |

**Overall Performance:**
- **Accuracy**: 67.79%
- **Macro Average**: Precision: 0.6757, Recall: 0.6395, F1-Score: 0.6476
- **Weighted Average**: Precision: 0.6769, Recall: 0.6779, F1-Score: 0.6700

The model shows strong performance for coil and alpha-helix prediction, with beta-sheets being more challenging to identify (lower recall at 42.23%).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zainab-hl/Small-Scale-Protein-Secondary-Structure-prediction.git
cd Small-Scale-Protein-Secondary-Structure-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the CNN training script:
```bash
cd src
python cnn.py
```

The training process:
- Runs for 20 epochs with Adam optimizer (lr=1e-3)
- Uses CrossEntropyLoss with padding token ignored
- Saves model weights to `cnn_ss_predictor.pth`
- Provides validation metrics after each epoch

### Using the Application Interface

```bash
python app.py
```

## Data Format

The CSV files should contain two columns:
- `seq`: Amino acid sequence (single letter codes)
- `sst3`: Secondary structure sequence (H/E/C labels)

## Model Features

- **One-hot encoding** of 20 standard amino acids
- **Sequence padding/truncation** to handle variable length inputs  
- **GPU acceleration** with CUDA support
- **Validation monitoring** during training
- **Per-residue predictions** maintaining sequence alignment

## Dependencies

- PyTorch: Deep learning framework
- Transformers: For potential ProtBert integration
- Scikit-learn: Metrics and preprocessing
- Pandas: Data manipulation
- Seqeval: Sequence labeling evaluation
- Matplotlib: Visualization
- Wandb: Experiment tracking

## Technical Notes

### Memory Optimization
The training is designed for resource-constrained environments (Google Colab):
- Batch size of 32 for efficient GPU memory usage
- Model checkpointing every 10 epochs
- Incremental training approach for limited compute resources

### Data Preprocessing
- Sequences longer than 512 residues are truncated
- Shorter sequences are padded with special tokens
- Invalid amino acids are ignored during encoding
- Labels use -1 for padding positions (ignored in loss calculation)

## Future Improvements

- **Attention mechanisms**: Add self-attention layers for long-range dependencies
- **Bidirectional processing**: Incorporate sequence context from both directions
- **Ensemble methods**: Combine multiple models for better accuracy
- **Transfer learning**: Fine-tune pre-trained protein language models (ProtBert)
- **Advanced architectures**: Explore Transformers and graph neural networks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for bugs, feature requests, or improvements.
