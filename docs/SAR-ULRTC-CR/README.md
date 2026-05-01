# SAR-ULRTC-CR

SAR-ULRTC-CR (SAR-ULRTC-Cloud-Removal) is a PyTorch implementation for cloud removal in satellite imagery using a multi-stage robust principal component analysis network with SAR and multispectral data fusion.

## 📌 Overview

- Model: `RPCA_Net_4_stages_SAR_Trans_RGB_8` in `main_net.py`
- Dataset: SEN12MS-CR (cloudy/cloud-free Sentinel 1/2). Data loader in `dataLoader.py`.
- Training script: `train_new_4_comp_trans_RGB_SEN12MSCR_8_5.py`
- Testing script: `test_new_4_comp_trans_RGB_real_M_8_1.py`

## 🧩 Repository structure

- `main_net.py`  : network definition
- `basicblock.py`: building blocks used by the model
- `dataLoader.py`: SEN12MSCR(S/2) dataset loader and preprocessing
- `train_new_4_comp_trans_RGB_SEN12MSCR_8_5.py`: training pipeline
- `test_new_4_comp_trans_RGB_real_M_8_1.py`: testing/evaluation pipeline
- `ckpts_SEN12MSCR/`: checkpoints folder
- `util/`: helper modules (visualization, cloud shadow detection, SSIM, etc.)
- `environment.yaml`: conda environment requirements

## ⚙️ Environment setup

Recommended: create conda env with provided YAML.

```bash
conda env create -f environment.yaml
conda activate SAR-ULRTC-CR
```

If your conda environment uses another name, adjust accordingly.

## 🗂️ Data organization

The dataset loader expects a SEN12MSCR folder tree like:

```
SEN12MS-CR_1/
    |- SEN12MSCR_train_input_npy/
    |- SEN12MSCR_train_gt_npy/
    |- SEN12MSCR_train_sar_npy/
    |- SEN12MSCR_train_mask_npy/
    |- SEN12MSCR_train_translated/
    |- SEN12MSCR_train_input_npy_full/
    |- SEN12MSCR_train_gt_npy_full/
    |- SEN12MSCR_train_sar_npy_full/
    |- SEN12MSCR_train_mask_npy_full/
    |- SEN12MSCR_train_translated_full/
    ...
```

Please follow the instructions provided in [dataset_preprocessing.ipynb](dataset_preprocessing.ipynb) file to download and arrange the data accordingly.
## ▶️ Training

Basic training command:

```bash
python train_new_4_comp_trans_RGB_SEN12MSCR_8_5.py \
  --train_path /path/to/SEN12MS-CR \
  --save_path ckpts_SEN12MSCR/SEN12MSCR_CRTC_ckpts_4_trans_RGB_10iter_8 \
  --num_epoch 200 \
  --batch_size 1 \
  --N_iter 10 \
  --data_mode partial
```

Common options:

- `--resume <checkpoint>`: resume training from checkpoint
- `--checkpoint_freq <int>`: save checkpoint every N epochs (default 1)
- `--loss_freq <int>`: progress print frequency (default 10)
- `--set_lr <float>`: overwrite learning rate if set

## 🔍 Testing / Evaluation

Run model evaluation using a checkpoint:

```bash
python test_new_4_comp_trans_RGB_real_M_8_1.py \
  --ckpt_dir ckpts_SEN12MSCR/SEN12MSCR_CRTC_ckpts_4_trans_RGB_10iter_8 \
  --epoch 200 \
  --N_iter 10 \
  --data_mode partial \
  --data_path /path/to/SEN12MS-CR \
  --fig_mode all \
  --test_mode single \
  --save_folder results_test
```

Relevant flags:
- `--fig_mode [I|I+M|I+M+C|X_hat|all|none]` select visualization mode
- `--test_mode [all|single]` decide whether test single epoch or all
- `--gpu <int>` choose GPU index
- `--params` print params/FLOPs (requires `thop`)

## 📈 Evaluation metrics

- RMSE
- MAE
- PSNR
- SAM
- SSIM

The test script computes these metrics inside `compute_metric()`.

## 🕵️ Notes

- Default paths are configured to `ckpts_SEN12MSCR/SEN12MSCR_CRTC_ckpts_4_trans_RGB_10iter_8` in scripts.
- `train_*` and `test_*` rely on `SEN12MSCR_0` class in `dataLoader.py`; this class performs target selection.
- If using your own dataset, mirror the SEN12MS path/naming conventions or modify `dataLoader.py`.
