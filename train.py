import yaml
import logging
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from tqdm import tqdm
from loguru import logger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/training.log')
    ]
)


class OCRDataset(Dataset):
    def __init__(self, image_dir, text_dir, processor, grayscale=False, enhance_contrast=True, target_size=None):
        self.image_dir = Path(image_dir)
        self.text_dir = Path(text_dir)
        self.processor = processor
        self.image_files = sorted(list(self.image_dir.glob("*")))
        self.text_files = sorted(list(self.text_dir.glob("*")))
        self.grayscale = grayscale
        self.enhance_contrast = enhance_contrast
        self.target_size = target_size
        
    def preprocess_image(self, image):
        # Convert to grayscale if specified
        if self.grayscale:
            image = image.convert('L').convert('RGB')
        
        # Resize if target size is specified
        if self.target_size:
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
        # Enhance contrast if specified
        if self.enhance_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)  # Increase contrast by 50%
            
        # Normalize the image
        image = ImageOps.autocontrast(image)
        
        return image
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        image = Image.open(self.image_files[idx]).convert("RGB")
        image = self.preprocess_image(image)
        
        # Load text
        with open(self.text_files[idx], 'r', encoding='utf-8') as f:
            text = f.read().strip()
            
        processed = self.processor(image, text, return_tensors="pt", padding="max_length", truncation=True)
        return {
            "pixel_values": processed.pixel_values.squeeze(),
            "labels": processed.labels.squeeze()
        }

def get_layer_freezing_config(mode):
    """Define which layers to freeze based on adaptation mode"""
    configs = {
        "low": 0.2,        # Freeze 80% of layers (train only 20% bottom layers)
        "intermediate": 0.4,  # Freeze 60% of layers
        "best_practice": 0.3,  # Freeze 70% of layers (common practice)
        "high": 0.5        # Freeze 50% of layers
    }
    return configs.get(mode, 0.3)  # Default to best_practice if mode not found

def freeze_layers(model, adaptation_mode):
    """Freeze layers based on adaptation mode"""
    trainable_ratio = get_layer_freezing_config(adaptation_mode)
    
    # Get all transformer layers
    encoder_layers = [module for name, module in model.named_modules() ]
    
    
    num_layers = len(encoder_layers)
    logger.info(num_layers)
    logger.info(trainable_ratio)
    num_trainable = int(num_layers * trainable_ratio)
    logger.info(num_trainable)
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the bottom layers according to mode
    for i, layer in enumerate(encoder_layers):
        if i < num_trainable:
            for param in layer.parameters():
                param.requires_grad = True
    
    logger.info(f"Adaptation mode: {adaptation_mode}")
    logger.info(f"Training {num_trainable} out of {num_layers} layers")

def get_scheduler(optimizer, scheduler_type, num_training_steps, num_warmup_steps):
    """Get the specified learning rate scheduler"""
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    elif scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}. Using cosine scheduler.")
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

def train():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create necessary directories
    os.makedirs(config['model']['save_path']['current'], exist_ok=True)
    os.makedirs(config['model']['save_path']['best'], exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Initialize model and processor
    model_id = config['model']['selected_model']
    logger.info(f"Loading model: {model_id}")
    
    model = AutoModelForVision2Seq.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Apply domain adaptation strategy
    freeze_layers(model, config['training']['adaptation_mode'])
    
    # Prepare datasets
    train_dataset = OCRDataset(
        config['data']['train_images'],
        config['data']['train_texts'],
        processor,
        grayscale=config['data']['grayscale'],
        enhance_contrast=config['data']['enhance_contrast'],
        target_size=config['data']['target_size']
    )
    val_dataset = OCRDataset(
        config['data']['val_images'],
        config['data']['val_texts'],
        processor,
        grayscale=config['data']['grayscale'],
        enhance_contrast=config['data']['enhance_contrast'],
        target_size=config['data']['target_size']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'])
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']['initial'])
    )
    
    num_training_steps = len(train_loader) * int(config['training']['max_epochs'])
    scheduler = get_scheduler(
        optimizer,
        config['training']['learning_rate']['scheduler'],
        num_training_steps,
        int(config['training']['learning_rate']['warmup_steps'])
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(int(config['training']['max_epochs']))):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(pixel_values=pixel_values, labels=labels)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{config['training']['max_epochs']}")
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        logger.info(f"Average validation loss: {avg_val_loss:.4f}")
        
        # Save current model
        model.save_pretrained(config['model']['save_path']['current'])
        
        # Early stopping check
        if avg_val_loss < best_val_loss +0.01:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            model.save_pretrained(config['model']['save_path']['best'])
            logger.info("Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= int(config['training']['early_stopping_patience']):
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break

if __name__ == "__main__":
    train()
