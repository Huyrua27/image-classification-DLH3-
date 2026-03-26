# Fruit Image Classification Pipeline
Link of Experiments on Drive (Checkpoint, Results, Notebook): https://drive.google.com/drive/u/1/folders/15UnGf6TsY0xUfFcNkIdnjw9Xeo2-lToa
**Progress: 50% Complete** ✓

This pipeline treats the task as 15-class classification:

- `Apple__Fresh`
- `Apple__Rotten`
- `Apple__Formalin-mixed`
- ...
- `Orange__Formalin-mixed`

## Progress Report (50% Milestone)

### Completed ✓
- Dataset preparation and validation (15 classes: 5 fruits × 3 conditions)
- Model baseline implementations (ResNet50, EfficientNet B0, ViT B16, DeiT Small)
- Initial model training on original dataset without augmentation
- Results evaluation and metrics collection

### Current Results (Original Dataset, No Augmentation)

| Model | Accuracy | Macro F1 | Weighted F1 | Best Epoch |
|-------|----------|----------|-------------|-----------|
| EfficientNet B0 | 99.61% | 99.60% | 99.61% | 6 |
| ResNet50 | 98.92% | 98.93% | 98.92% | 10 |

### Next Steps (50% Remaining)
- [ ] Train all 5 models with data augmentation
- [ ] Compare augmented vs non-augmented performance
- [ ] Test combined augmentation approach
- [ ] Final model selection and optimization
- [ ] Comprehensive results analysis and report

## Project Structure

```
fruit_classification/
├── configs/                          # Configuration files
│   ├── base_image_classification.yaml
│   ├── requirements_image_classification.txt
│   ├── models/                       # Model-specific configs
│   │   ├── resnet50.yaml
│   │   ├── efficientnet_b0.yaml
│   │   ├── efficientnet_b2.yaml
│   │   ├── vit_b16.yaml
│   │   └── deit_small.yaml
│   └── experiments/                  # Experiment configs
│       ├── original_no_aug.yaml
│       ├── original_with_aug.yaml
│       ├── augmentation_only.yaml
│       ├── combined_with_aug.yaml
│       └── debug_smoke.yaml
│
├── data/                             # Dataset folders
│   ├── Fruits Original/              # Original dataset (5 fruits × 3 conditions)
│   │   └── [Apple, Banana, Grape, Mango, Orange]/
│   │       └── [Fresh, Rotten, Formalin-mixed]/
│   └── Fruits Augmentation/          # Augmented dataset
│       └── [Apple, Banana, Grape, Mango, Orange]/
│
├── models/
│   └── model_factory.py              # Factory for model creation
│
├── training/                         # Training code
│   ├── dataset.py                    # Dataset loading & preprocessing
│   ├── engine.py                     # Training loop
│   └── transforms.py                 # Data augmentations
│
├── utils/                            # Utility functions
│   ├── config.py                     # Config management
│   ├── metrics.py                    # Evaluation metrics
│   ├── artifacts.py                  # Model/results I/O
│   └── reproducibility.py            # Random seed management
│
└── experiments/
    ├── run_image_classification.py   # Main training script
    ├── run_all.ps1                   # PowerShell script to run all experiments
    ├── colab_run_4models_2modes.ipynb # Jupyter notebook for Colab
    ├── README.md                     # This file
    └── runs/                         # Training outputs
        └── 20260325_individual_runs/
            ├── original__efficientnet_b0/
            ├── original__resnet50/
            └── original__vit_b16/
                ├── best_model.pt
                ├── summary.json
                ├── test_metrics.json
                ├── history.csv
                ├── merged_config.yaml
                └── confusion_matrix_test.png
```

## Main entrypoint

```powershell
python fruit_classification\experiments\run_image_classification.py `
  --config configs/base_image_classification.yaml `
  --config configs/models/resnet50.yaml `
  --config configs/experiments/original_no_aug.yaml
```

## Run multiple experiments

Run the default comparison set:

```powershell
powershell -ExecutionPolicy Bypass -File fruit_classification\experiments\run_all.ps1
```

Run only selected models:

```powershell
powershell -ExecutionPolicy Bypass -File fruit_classification\experiments\run_all.ps1 `
  -Models resnet50,vit_b16 `
  -Experiments original_no_aug,original_with_aug
```

Dry run only:

```powershell
powershell -ExecutionPolicy Bypass -File fruit_classification\experiments\run_all.ps1 -DryRun
```

## Available model configs

- `configs/models/resnet50.yaml`
- `configs/models/efficientnet_b0.yaml`
- `configs/models/efficientnet_b2.yaml`
- `configs/models/vit_b16.yaml`
- `configs/models/deit_small.yaml`

## Available experiment configs

- `configs/experiments/original_no_aug.yaml`
- `configs/experiments/original_with_aug.yaml`
- `configs/experiments/augmentation_only.yaml`
- `configs/experiments/combined_with_aug.yaml`
- `configs/experiments/debug_smoke.yaml`

## Suggested comparison plan

1. Run each model on `original_no_aug`.
2. Run each model on `original_with_aug`.
3. Compare `accuracy`, `macro_f1`, `weighted_f1`, and confusion matrix.
4. Optionally run `combined_with_aug` to test whether using the provided augmentation folder helps further.

## Output artifacts

Each run creates a folder in `fruit_classification/experiments/runs/<run_name>` with:

- `merged_config.yaml`
- `data_summary.json`
- `history.csv`
- `best_model.pt`
- `summary.json`
- `test_metrics.json`
- `confusion_matrix_test.png`

## Notes

- `pretrained: true` is supported. If pretrained weights are not available locally and download fails, the pipeline falls back to random init unless `strict_pretrained: true`.
- The augmentation dataset is read directly from `.zip` archives, so you do not need to extract everything first.
- For this environment, use `numpy<2` with Torch/Torchvision to avoid compatibility issues.
