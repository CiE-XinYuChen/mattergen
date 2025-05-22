# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MatterGen is a generative model for inorganic materials design that uses diffusion models to generate crystal structures. The codebase supports unconditional generation, property-conditioned generation, training from scratch, and fine-tuning on specific properties.

## Key Commands

### Installation and Setup
```bash
pip install uv
uv venv .venv --python 3.10 
source .venv/bin/activate
uv pip install -e .
```

### Data Preprocessing
```bash
# For mp_20 dataset
git lfs pull -I data-release/mp-20/ --exclude=""
unzip data-release/mp-20/mp_20.zip -d datasets
csv-to-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20 --cache-folder datasets/cache

# For alex_mp_20 dataset (larger, ~1h processing time)
git lfs pull -I data-release/alex-mp/alex_mp_20.zip --exclude=""
unzip data-release/alex-mp/alex_mp_20.zip -d datasets
csv-to-dataset --csv-folder datasets/alex_mp_20/ --dataset-name alex_mp_20 --cache-folder datasets/cache
```

### Training
```bash
# Train base model on mp_20
mattergen-train data_module=mp_20 ~trainer.logger

# Train base model on alex_mp_20 (larger dataset)
mattergen-train data_module=alex_mp_20 ~trainer.logger trainer.accumulate_grad_batches=4

# Crystal structure prediction mode (CSP)
mattergen-train --config-name=csp data_module=mp_20 ~trainer.logger
```

### Fine-tuning
```bash
# Single property fine-tuning
export PROPERTY=dft_mag_density
mattergen-finetune adapter.pretrained_name=mattergen_base data_module=mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY ~trainer.logger data_module.properties=["$PROPERTY"]

# Multi-property fine-tuning
export PROPERTY1=dft_mag_density
export PROPERTY2=dft_band_gap
mattergen-finetune adapter.pretrained_name=mattergen_base data_module=mp_20 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 ~trainer.logger data_module.properties=["$PROPERTY1","$PROPERTY2"]
```

### Generation
```bash
# Unconditional generation
export MODEL_NAME=mattergen_base
export RESULTS_PATH=results/
mattergen-generate $RESULTS_PATH --pretrained-name=$MODEL_NAME --batch_size=16 --num_batches 1

# Property-conditioned generation
export MODEL_NAME=dft_mag_density
mattergen-generate $RESULTS_PATH --pretrained-name=$MODEL_NAME --batch_size=16 --properties_to_condition_on="{'dft_mag_density': 0.15}" --diffusion_guidance_factor=2.0

# Multi-property conditioned generation
mattergen-generate $RESULTS_PATH --pretrained-name=chemical_system_energy_above_hull --batch_size=16 --properties_to_condition_on="{'energy_above_hull': 0.05, 'chemical_system': 'Li-O'}" --diffusion_guidance_factor=2.0
```

### Evaluation
```bash
# Evaluate generated structures using MatterSim MLFF
git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""
mattergen-evaluate --structures_path=$RESULTS_PATH --relax=True --structure_matcher='disordered' --save_as="$RESULTS_PATH/metrics.json"

# Evaluate with pre-computed energies
mattergen-evaluate --structures_path=$RESULTS_PATH --energies_path='energies.npy' --relax=False --structure_matcher='disordered' --save_as='metrics'
```

### Testing
```bash
# Run tests using pytest
pytest mattergen/tests/
pytest mattergen/common/tests/
pytest mattergen/diffusion/tests/

# Run specific test files
pytest mattergen/tests/test_generator.py
pytest mattergen/common/tests/gemnet_test.py
```

### Code Quality
```bash
# Format code with black
black mattergen/ --line-length 100

# Sort imports with isort
isort mattergen/ --profile black --line-length 100

# Lint with pylint
pylint mattergen/
```

## Architecture Overview

### Core Components

1. **Diffusion Module** (`mattergen/diffusion/`): Core diffusion model implementation
   - `diffusion_module.py`: Main diffusion lightning module
   - `corruption/`: Noise corruption strategies for diffusion
   - `sampling/`: Sampling algorithms and predictors/correctors
   - `d3pm/`: Discrete denoising diffusion for categorical variables

2. **GemNet Backbone** (`mattergen/common/gemnet/`): Graph neural network for crystal representation
   - `gemnet.py`: Main GemNet model
   - `layers/`: Neural network layers for atomic interactions

3. **Data Pipeline** (`mattergen/common/data/`): Data loading and processing
   - `datamodule.py`: PyTorch Lightning data module
   - `dataset.py`: Crystal dataset implementation
   - `transform.py`: Data transformations

4. **Property Embeddings** (`mattergen/property_embeddings.py`): Handles conditioning on material properties

### Configuration System

The project uses Hydra for hierarchical configuration management. Key configuration areas:

- **Data Modules**: `mattergen/conf/data_module/` - Dataset configurations
- **Training**: `mattergen/conf/trainer/` - Training parameters
- **Models**: `mattergen/conf/lightning_module/` - Model architectures
- **Property Embeddings**: `mattergen/conf/lightning_module/diffusion_module/model/property_embeddings/` - Property conditioning configurations

### Fine-tuning Configuration

Fine-tuning requires specific configuration patterns:

1. **Property Selection**: Properties must be in `PROPERTY_SOURCE_IDS` list in `mattergen/common/utils/globals.py`
2. **Property Config**: Each property needs a YAML config in `property_embeddings/` directory
3. **Data Module**: Dataset must include the target property columns
4. **Adapter Pattern**: Uses `adapter.pretrained_name` or `adapter.model_path` to load base model

### Key Configuration Files

- `mattergen/conf/default.yaml`: Base training configuration
- `mattergen/conf/finetune.yaml`: Fine-tuning specific settings (reduced learning rate, fewer epochs)
- `mattergen/conf/data_module/alex_mp_20.yaml`: Large dataset configuration with supported properties
- Property embedding configs: Float properties use `NoiseLevelEncoding`, categorical use custom embeddings

### Supported Properties for Fine-tuning

Available in `PROPERTY_SOURCE_IDS`:
- `dft_mag_density`: DFT magnetic density
- `dft_bulk_modulus`: DFT bulk modulus  
- `dft_band_gap`: DFT band gap
- `ml_bulk_modulus`: ML predicted bulk modulus
- `energy_above_hull`: Energy above convex hull
- `space_group`: Crystal space group
- `chemical_system`: Chemical composition
- `hhi_score`: Herfindahl-Hirschman Index score

### Apple Silicon Support

For Apple Silicon (experimental):
- Export `PYTORCH_ENABLE_MPS_FALLBACK=1` before training/generation
- Add `~trainer.strategy trainer.accelerator=mps` to training commands

### Model Checkpoints

Pre-trained models available:
- `mattergen_base`: Unconditional base model
- `chemical_system`: Conditioned on chemical system
- `space_group`: Conditioned on space group  
- `dft_mag_density`: Conditioned on magnetic density
- `dft_band_gap`: Conditioned on band gap
- `ml_bulk_modulus`: Conditioned on bulk modulus
- Multi-property models: `dft_mag_density_hhi_score`, `chemical_system_energy_above_hull`

### Output Structure

- Training outputs: `outputs/singlerun/${date}/${time}/`
- Generation outputs: Specified by `$RESULTS_PATH`
  - `generated_crystals_cif.zip`: Individual CIF files
  - `generated_crystals.extxyz`: Single file with all structures
  - `generated_trajectories.zip`: Full denoising trajectories (optional)