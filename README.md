# OCR Domain Adaptation

This project implements an OCR (Optical Character Recognition) system with domain adaptation capabilities. It allows training various pre-trained models on custom datasets while focusing on specific layer adaptations.

## Project Structure

```
.
├── config.yaml           # Configuration file
├── train.py             # Main training script
├── requirements.txt     # Python dependencies
├── data/               # Training data directory
│   ├── train/         
│   │   ├── imgs/      # Training images
│   │   └── texts/     # Training text files
│   └── val/           
│       ├── imgs/      # Validation images
│       └── texts/     # Validation text files
├── model/              # Model storage
│   ├── current/       # Current model checkpoint
│   └── best/          # Best model checkpoint
└── logs/              # Training logs
```

## Configuration

The `config.yaml` file contains all the necessary settings:

- Model selection from various pre-trained options
- Training parameters (batch size, learning rate, etc.)
- Domain adaptation mode selection
- Data paths
- Logging configuration

## Domain Adaptation Modes

- **low**: Trains only the lowest layers (20% of layers)
- **intermediate**: Trains a moderate number of layers (40% of layers)
- **best_practice**: Trains the commonly recommended number of layers (30% of layers)
- **high**: Trains up to half of the layers (50% of layers)

## Setup and Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your data:
   - Place training images in `data/train/imgs/`
   - Place corresponding text files in `data/train/texts/`
   - Place validation images in `data/val/imgs/`
   - Place corresponding text files in `data/val/texts/`

3. Configure the training:
   - Modify `config.yaml` according to your needs
   - Select the appropriate model and adaptation mode
   - Adjust training parameters if needed

4. Start training:
   ```bash
   python train.py
   ```

## Features

- Support for multiple pre-trained OCR models
- Configurable domain adaptation strategies
- Early stopping based on validation performance
- Learning rate scheduling
- Model checkpointing
- Basic logging
