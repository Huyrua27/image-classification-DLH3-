# Fruit Classification Pipeline - Updated Documentation

## Overview
This updated pipeline implements a comprehensive fruit classification experiment with:
- **3 Models**: ResNet50, EfficientNet B6, ViT B16
- **2 Training Modes**: Original (no augmentation) + Data Augmentation
- **3 Fine-tuning Strategies**: Frozen (backbone), Linear Probe, Fine-tune All
- **Total Configurations**: 3 models × 2 modes × 3 strategies = 18 training runs
- **Test Set**: Always uses original (non-augmented) data for fair comparison

## Key Features

### 1. Model Support
- **ResNet50**: Classic CNN architecture
  - Config: `configs/models/resnet50.yaml`
  - Image size: 224×224
  
- **EfficientNet B6**: Efficient scaling architecture
  - Config: `configs/models/efficientnet_b6.yaml`
  - Image size: 528×528 (larger receptive field)
  
- **ViT B16**: Vision Transformer
  - Config: `configs/models/vit_b16.yaml`
  - Image size: 224×224

### 2. Fine-tuning Strategies

#### Frozen (Backbone)
- Backbone weights frozen, only classification head trained
- Learning rate: 0.001
- Epochs: 10
- Use case: When data is limited, prevents overfitting

#### Linear Probe
- Backbone frozen, but head gets higher learning rate
- Learning rate: 0.01
- Epochs: 15
- Use case: More aggressive training of the head

#### Fine-tune All
- All layers trainable with low learning rate
- Learning rate: 0.0001
- Epochs: 20
- Use case: Sufficient data available, adapt full model

### 3. Data Augmentation
Augmentation techniques (applied only to training set):
- Horizontal flip (50%)
- Random rotation (15°)
- Color jitter (brightness, contrast, saturation: 0.2)
- Random crop with resize (80-100% scale, 90-110% aspect ratio)

### 4. Training Features
- **TensorBoard Integration**: Real-time visualization during training
  - Tracks: Loss, accuracy, macro F1, learning rate per epoch
  - Found in: `runs/*/tensorboard/`
  
- **Early Stopping**: Stops training if validation macro F1 doesn't improve for 3 epochs

- **Checkpoint Saving**: Best model saved based on validation macro F1

### 5. Metrics & Evaluation
Computed on test set (always original/non-augmented):
- **Accuracy**: Overall classification accuracy
- **Macro F1**: Unweighted F1 score (good for imbalanced data)
- **Weighted F1**: Weighted by class frequency
- **Confusion Matrix**: Per-class performance analysis
- **Classification Report**: Per-class precision, recall, F1

## Usage

### Option 1: Run Full Pipeline (Recommended)
```python
# All cells execute sequentially:
# 1. Setup imports and data loaders
# 2. Configuration loading
# 3. Training setup
# 4. Train 6 base models (3 models × 2 modes) with 'frozen' strategy
# 5. Train with 'linear_probe' strategy (6 more runs)
# 6. Train with 'finetune_all' strategy (6 more runs)
# 7. Aggregate results and create comparison tables
# 8. Generate visualization charts
# 9. Compute model interpretability heatmaps
# 10. Generate summary report
```

### Option 2: Run Specific Configuration
```python
result = train_single_model(
    mode_name='original',           # or 'augmentation'
    model_name='resnet50',          # or 'efficientnet_b6', 'vit_b16'
    fine_tuning_strategy='frozen',  # or 'linear_probe', 'finetune_all'
    mode_config_rel='configs/experiments/original_no_aug.yaml',
    model_config_rel='configs/models/resnet50.yaml',
)
```

## Output Structure

### Directory: `experiments/runs/20260325_3models_ft/`

```
├── original__resnet50__frozen/
│   ├── best_model.pt              # Best checkpoint
│   ├── merged_config.yaml         # Combined config
│   ├── data_summary.json          # Train/val/test split info
│   ├── history.csv                # Epoch-wise metrics
│   ├── test_metrics.json          # Test results
│   ├── summary.json               # Best epoch & score
│   ├── confusion_matrix_test.png  # Confusion matrix
│   └── tensorboard/               # TensorBoard logs
│       └── events.out.tfevents.*
├── ... (more model runs)
└── reports/
    ├── all_results.csv              # All runs summary
    ├── all_history.csv              # Combined training history
    ├── metrics_by_model.csv         # Mean±std by model
    ├── metrics_by_mode.csv          # Mean±std by mode
    ├── metrics_by_finetuning.csv    # Mean±std by strategy
    ├── top_configurations.csv       # Top 10 runs
    ├── training_curves.png          # 4-panel training curves
    ├── accuracy_comparison.png      # 3-panel accuracy charts
    ├── macro_f1_comparison.png      # 3-panel F1 charts
    ├── heatmap_mode_model.png       # Mode×Model heatmap
    ├── heatmap_finetuning_model.png # Fine-tuning×Model heatmap
    ├── confusion_matrix_grid.png    # Confusion matrices grid
    ├── experiment_summary.txt       # Text summary
    └── interpretability/
        └── [Grad-CAM & Attention visualizations]
```

## TensorBoard Visualization

To visualize training in real-time:

```bash
# View all models' training
tensorboard --logdir=experiments/runs/20260325_3models_ft/*/tensorboard

# View specific model
tensorboard --logdir='experiments/runs/20260325_3models_ft/original__resnet50__frozen/tensorboard'
```

Metrics tracked:
- `train/batch_loss`: Per-batch training loss
- `val/loss`, `val/accuracy`, `val/macro_f1`, `val/weighted_f1`: Epoch validation metrics
- `test/loss`, `test/accuracy`: Test set metrics

## Model Interpretability

### Grad-CAM (for ResNet & EfficientNet)
Highlights which regions of the image influence the model's decision.

```python
from fruit_classification.utils.visualization import GradCAM

# Load best ResNet50
checkpoint = torch.load('runs/.../best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Create Grad-CAM for layer4 (last convolutional block)
grad_cam = GradCAM(model, model.layer4)

# Generate heatmap
heatmap = grad_cam(images, class_idx=0)

# Visualize on image
from fruit_classification.utils.visualization import visualize_heatmap
overlay = visualize_heatmap(image_np, heatmap, alpha=0.5)
```

### Attention Rollout (for ViT)
Aggregates attention weights from all transformer layers to show which patches are important.

```python
from fruit_classification.utils.visualization import AttentionRollout

# Create rollout for ViT
rollout = AttentionRollout(model)

# Generate attention heatmap
heatmap = rollout(images, class_idx=0)

# Visualize
overlay = visualize_heatmap(image_np, heatmap, alpha=0.4)
```

## Expected Results

### Baseline Performance (from paper/prior runs)
- ResNet50: ~92-94% accuracy
- EfficientNet: ~94-96% accuracy
- ViT: ~95-97% accuracy

### Fine-tuning Impact
- **Frozen**: Good for limited data, prevents overfitting
- **Linear Probe**: Better head adaptation, ~0-2% improvement
- **Fine-tune All**: Best if sufficient data, ~1-3% improvement

### Data Augmentation Impact
- +2-4% accuracy on validation set typically
- Helps with generalization
- More critical for smaller datasets

## Configuration Files

### Base Configuration
`configs/base_image_classification.yaml`: Default hyperparameters

### Mode Configs
- `configs/experiments/original_no_aug.yaml`: No augmentation
- `configs/experiments/augmentation_only.yaml`: Full augmentation

### Model Configs
- `configs/models/resnet50.yaml`: ResNet50 specific settings
- `configs/models/efficientnet_b6.yaml`: EfficientNet B6 specific settings
- `configs/models/vit_b16.yaml`: ViT B16 specific settings

### Fine-tuning Config (Reference)
`configs/base_finetuning.yaml`: Template for fine-tuning strategies

## Advanced Usage

### Custom Fine-tuning Strategy
```python
# Modify strategy in train_single_model or update FINETUNING_STRATEGIES
custom_strategy = {
    'frozen': {
        'model': {'freeze_backbone': True},
        'training': {'lr': 0.002, 'epochs': 15}
    }
}

# Or modify per-run:
result = train_single_model(...)
# config['model']['freeze_backbone'] = False
# config['training']['lr'] = 0.00005
```

### Extract Features (Pre-trained Embeddings)
```python
# Load trained model
checkpoint = torch.load('experiments/runs/.../best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Remove classification head to get embeddings
if model_name == 'resnet50':
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
elif model_name == 'efficientnet_b6':
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
elif model_name == 'vit_b16':
    feature_extractor = model.layers[:-1]  # Remove head

# Extract features
with torch.no_grad():
    embeddings = feature_extractor(images)
```

## Troubleshooting

### Out of Memory
- Reduce batch size in config: `data.loader.batch_size = 8`
- Use smaller model: EfficientNet B0/B2 instead of B6
- Reduce image size: `data.image_size = 128`

### Slow Training
- Reduce epochs: `training.epochs = 5`
- Skip certain fine-tuning strategies
- Use CPU if GPU memory is limited (slower but works)

### Poor Performance
- Increase epochs for 'linear_probe' and 'finetune_all'
- Try different learning rates
- Ensure data augmentation is enabled
- Check that test set matches expected class distribution

## Performance Monitoring

### During Training
- Watch Terminal output for loss and accuracy trends
- Use TensorBoard for real-time visualization
- Check validation macro F1 convergence

### After Training
- Compare results in `reports/all_results.csv`
- Check training curves in `reports/training_curves.png`
- Review best configurations in `reports/top_configurations.csv`
- Analyze per-class performance in confusion matrices

## References

### Papers
- ResNet: He et al. (2015) - Deep Residual Learning
- EfficientNet: Tan & Le (2019) - EfficientNet: Rethinking Model Scaling
- ViT: Dosovitskiy et al. (2020) - An Image is Worth 16x16 Words

### Libraries
- PyTorch: Deep learning framework
- TensorBoard: Training visualization
- torchvision: Computer vision utilities
- timm: PyTorch Image Models (for EfficientNet, ViT)

## Author Notes

This pipeline provides a production-ready, comprehensive framework for:
1. **Model Comparison**: Objective evaluation of different architectures
2. **Fine-tuning Analysis**: Understanding which training strategies work best
3. **Data Augmentation Impact**: Quantifying benefits of augmentation
4. **Reproducibility**: Fixed seeds and documented configurations
5. **Interpretability**: Visualization tools for understanding model decisions

The separation of training code and notebooks ensures maintainability and reusability.
