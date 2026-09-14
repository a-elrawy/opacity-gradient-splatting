# Opacity-Gradient Splatting

Few-shot 3D Gaussian Splatting with **opacity-gradient densification** for compact models and higher rendering throughput.

Densification accumulates photometric opacity gradients `|∂L_photo/∂α|`. Depth correlation (if used) stays in the training loss only.

```bash
git clone https://github.com/a-elrawy/opacity-gradient-splatting.git
cd opacity-gradient-splatting
```

## Environment

```bash
python -m venv fsgs_env
source fsgs_env/bin/activate
# Install PyTorch (CUDA) then project deps / rasterizer submodules
# as in standard 3DGS / FSGS setups. See environment.yml if present.
```

## Data

Put LLFF / Mip-NeRF 360 under `dataset/` (gitignored). COLMAP helpers: `tools/`.

## Training (LLFF, 3 views)

```bash
python train.py \
  --source_path dataset/nerf_llff_data/fern \
  --model_path output/llff_default/fern \
  --eval --n_views 3 --images images_8 \
  --use_error_densification \
  --error_densify_threshold 0.0005 \
  --prune_from_iter 2000 \
  --prune_threshold 0.001 \
  --depth_weight 0.05 \
  --max_gaussians 32000 \
  --sample_pseudo_interval 10
```

Optional ablation: `--densify_grad_from_total_loss` densifies from photometric + depth grads.

```bash
python render.py --source_path dataset/nerf_llff_data/fern \
  -m output/llff_default/fern --iteration 10000
python metrics.py -s output/llff_default/fern -m output/llff_default/fern
```

### Mip-NeRF 360 (24 views)

```bash
python train.py \
  --source_path dataset/mipnerf360/garden \
  --model_path output/mip_default/garden \
  --eval --n_views 24 \
  --use_error_densification \
  --error_densify_threshold 0.001 \
  --prune_from_iter 2000 \
  --max_gaussians 250000 \
  --sample_pseudo_interval 10 \
  --depth_weight 0.05
```

## Cluster (SLURM)

Machine-local paths stay out of git:

```bash
cp slurm/env.sh.example slurm/env.sh   # set OGS_ROOT, OGS_VENV, OGS_ACCOUNT
sbatch --account=$OGS_ACCOUNT slurm/run_llff_default.sh
sbatch --account=$OGS_ACCOUNT slurm/run_mipnerf_default.sh
```

Details: [`slurm/README.md`](slurm/README.md).

## Repo layout

| Path | Role |
|------|------|
| `train.py` / `render.py` / `metrics.py` | Main entrypoints |
| `arguments/`, `scene/`, `utils/` | Method code |
| `slurm/` | Portable cluster job templates |
| `tools/` | Dataset / COLMAP helpers |
| `merge_results.py` | Aggregate `results.json` across runs |

## Citation

If you use this code, please cite the accompanying paper (title below). BibTeX will be updated on publication.

```bibtex
@article{elrawy2026opacity,
  title={Opacity-Gradient Driven Density Control for Compact and Efficient Few-Shot 3D Gaussian Splatting},
  author={Elrawy, Abdelrhman and Mohammed, Emad A.},
  year={2026}
}
```
