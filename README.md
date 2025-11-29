# RTXFair

A PyTorch-based explainable AI framework for tabular data classification with built-in fairness considerations. The framework implements an attention-based neural network with integrated gradients for model interpretability.

## Overview

RTXFair provides a complete pipeline for training, explaining, and evaluating tabular classification models with a focus on transparency and interpretability. The core architecture uses feature-wise attention mechanisms combined with integrated gradients to produce human-understandable explanations.

## Project Structure

```
rtxfair/
├── rtxfair/                  # Core library
│   ├── model.py             # TabularAttentionNet architecture
│   ├── train.py             # Training pipeline and configuration
│   ├── explain.py           # Explainability methods (IG, fusion, metrics)
│   ├── export.py            # TorchScript export utilities
│   ├── latency.py           # Performance benchmarking
│   ├── metrix.py            # Evaluation metrics (AUC, Brier, ECE)
│   └── init.py              # Package initialization
├── scripts/                  # Command-line tools
│   ├── run_train.py         # Train model on HELOC dataset
│   ├── run_latency.py       # Benchmark inference latency
│   ├── run_explain_batch.py # Batch explanation generation (empty)
│   └── export_ts.py         # Export to TorchScript format
├── test/                     # Unit tests
│   └── test_shapes.py       # Shape validation tests
├── DEMO/                     # Demonstration notebooks
│   └── main_demo.ipynb      # Main demo notebook
├── datasets/                 # Data directory
├── heloc_dataset_v1.csv     # HELOC dataset
└── latency_log.csv          # Latency benchmark results
```

## Core Components

### Model Architecture (`model.py`)

**TabularAttentionNet**: A neural network that combines feature-wise attention with a multi-layer perceptron head.

- **Input**: Tabular features (batch_size × d_in)
- **Output**: Predictions (batch_size × 1) and attention weights (batch_size × d_in)
- **Features**:
  - Feature-wise attention mechanism with softmax normalization
  - 2-layer MLP with ReLU activations
  - Dropout regularization
  - Sigmoid output for binary classification

```python
model = TabularAttentionNet(d_in=23, hidden=64, dropout=0.1)
prediction, attention = model(x)  # Returns probabilities and feature importance
```

### Training (`train.py`)

**TrainConfig**: Dataclass for training hyperparameters
- `hidden`: Hidden layer size (default: 64)
- `dropout`: Dropout rate (default: 0.0)
- `batch_size`: Batch size (default: 256)
- `epochs`: Training epochs (default: 20)
- `lr`: Learning rate (default: 1e-3)
- `weight_decay`: L2 regularization (default: 0.0)
- `seed`: Random seed (default: 0)
- `device`: Device for training (default: "cpu")

**train_model()**: Complete training loop with validation
- Binary cross-entropy loss
- AdamW optimizer
- Early stopping based on validation loss
- Automatic best model checkpoint

### Explainability (`explain.py`)

**Integrated Gradients (`integrated_gradients()`)**:
- Computes feature attributions via path integral from baseline to input
- Parameters: `f` (prediction function), `x` (input), `baseline`, `steps` (default: 16)
- Returns: Attribution tensor of same shape as input

**Attribution Fusion (`fuse_attributions()`)**:
- Combines integrated gradients with attention weights
- Formula: `E = β × IG_norm + (1-β) × Attn_norm`
- Default β=0.7 balances gradient-based and attention-based explanations

**Explanation Quality Metrics**:
- **Infidelity**: Measures explanation accuracy via perturbation analysis (lower is better)
- **Stability**: Cosine similarity under noise perturbations (higher is better)

### Evaluation Metrics (`metrix.py`)

**summary_metrics()**: Comprehensive model evaluation
- **AUC**: Area Under ROC Curve - discrimination ability
- **Brier Score**: Calibration quality (lower is better)
- **ECE**: Expected Calibration Error - reliability of probabilities

**predict_prob()**: Batch prediction utility with GPU support

**calibration_error()**: 10-bin expected calibration error calculation

### Performance Benchmarking (`latency.py`)

**benchmark_latency()**: Measures inference and explanation latency
- Separate timing for prediction vs explanation
- Statistics: mean, p90, p99 latency
- Optional CSV logging with per-sample breakdown
- Supports both DataFrame and Tensor inputs

### Model Export (`export.py`)

**export_torchscript()**: Export trained model to TorchScript format
- Creates inference wrapper for deployment
- Traces model with example input
- Saves serialized model for production use

## Scripts

### Training (`scripts/run_train.py`)

Train a model on the HELOC dataset:

```bash
python scripts/run_train.py --csv heloc_dataset_v1.csv --device cpu --epochs 10 --hidden 64
```

Outputs:
- Training/validation loss per epoch
- Final test metrics (AUC, Brier, ECE)
- Saved model weights: `./artifacts/rtxfair_weights.pt`

### Latency Benchmarking (`scripts/run_latency.py`)

Benchmark inference and explanation performance:

```bash
python scripts/run_latency.py --csv heloc_dataset_v1.csv --device cpu --steps 16 --n 200
```

Outputs:
- Prediction latency statistics
- Explanation latency statistics
- Detailed CSV log: `latency_log.csv`

### TorchScript Export (`scripts/export_ts.py`)

Export model for production deployment:

```bash
python scripts/export_ts.py --d 23 --weights ./artifacts/rtxfair_weights.pt --out ./artifacts/model.pt
```

Parameters:
- `--d`: Input feature dimension
- `--weights`: Path to trained weights (optional)
- `--out`: Output path for TorchScript model

## Testing

Run unit tests to verify model and explanation shapes:

```bash
python -m pytest test/test_shapes.py
```

Tests validate:
- Forward pass output dimensions
- Integrated gradients computation
- Attribution fusion shapes

## Dataset

**HELOC Dataset** (`heloc_dataset_v1.csv`): Home Equity Line of Credit risk prediction
- Binary classification task
- 23 features representing credit risk factors
- Used for fairness-aware credit scoring research

## Key Features

✅ **Explainable**: Built-in integrated gradients + attention fusion  
✅ **Performant**: Optimized for low-latency inference  
✅ **Calibrated**: Built-in calibration metrics (ECE, Brier)  
✅ **Production-Ready**: TorchScript export for deployment  
✅ **Benchmarked**: Comprehensive latency profiling tools  
✅ **Validated**: Explanation quality metrics (infidelity, stability)  

## Requirements

- PyTorch
- NumPy
- scikit-learn
- pandas

## Usage Example

```python
from rtxfair.train import train_model, TrainConfig
from rtxfair.model import TabularAttentionNet
from rtxfair.explain import integrated_gradients, fuse_attributions
from rtxfair.metrix import summary_metrics

# Train
cfg = TrainConfig(hidden=64, epochs=20, device="cpu")
model = train_model(X_train, y_train, X_val, y_val, cfg)

# Evaluate
metrics = summary_metrics(model, X_test, y_test)
print(f"AUC: {metrics['AUC']:.4f}, Brier: {metrics['Brier']:.4f}")

# Explain
def pred_fn(x):
    y, _ = model(x)
    return y

ig = integrated_gradients(pred_fn, x_sample, steps=16)
_, attn = model(x_sample)
explanation = fuse_attributions(ig, attn, beta=0.7)
```

## Demo

See `DEMO/main_demo.ipynb` for an interactive walkthrough of training, evaluation, and explanation generation.

## Notes

- The `rtxfair.data` module (referenced in scripts) is not present in the codebase - ensure data loading is implemented
- `scripts/run_explain_batch.py` is currently empty and needs implementation
- The framework assumes tabular data with numerical features
- GPU acceleration supported via `device` parameter

## License

See repository for license information.
