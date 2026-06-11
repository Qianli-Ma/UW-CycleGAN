# UW-CycleGAN — Thesis Reimplementation

This is a reimplementation of UW-CycleGAN (Du et al., 2022) used as a baseline in
my thesis on underwater image enhancement. It lives as a subdirectory of the parent
[UIE](https://github.com/Qianli-Ma/UIE) repo at
`experiments/models/baselines/UW-CycleGAN/`. The implementation follows the paper's
CycleGAN-based unpaired training objective with a DenseNet-block generator,
blur-promoting adversarial loss, and VGG19 content loss.

---

## Differences from the paper

| Component | Paper | This reimplementation |
|---|---|---|
| Discriminator architecture | Described as five fully convolutional layers with a "70x70" Markov discriminator, but kernel sizes, strides, channels, and padding are not specified | Standard CycleGAN 70x70 receptive-field PatchGAN: 4x4 conv layers with channels 64/128/256/512/1 and strides 2/2/2/1/1; output is about 30x30 for 256x256 input |
| Spectral normalisation | Not stated | Applied to every discriminator convolution as required by the thesis baseline specification |
| LSGAN sign convention | Equations write real as 0 and fake as 1, which would make the generator optimise toward the fake label | Uses the standard CycleGAN LSGAN convention: real=1, fake=0 for discriminators; generated images target real=1 for generators |
| LR schedule | Paper does not specify an LR schedule | Linear decay to 0 over the last 150 epochs, controlled by `decay_start: 150` in `config.yaml` |

The blur-promoting adversarial loss treats Gaussian-blurred clean images and generated
images as fake samples for `D_Y`. The content loss uses frozen pretrained VGG19 conv4_4
features, matching the paper's description.

---

## Expected folder structure

All paths are relative to `experiments/` in the parent UIE repo.

```
experiments/
├── datasets/
│   └── UIEB/
│       ├── raw-890/                ← underwater inputs
│       ├── reference-890/          ← reference images
│       ├── train.txt               ← train split filenames
│       ├── val.txt                 ← validation split filenames
│       └── test.txt                ← test split filenames
└── models/
    ├── transfuse-gan/
    │   └── src/
    │       ├── datasets/           ← build_uieb_loader used by eval.py
    │       └── evaluation/         ← evaluate_paired_loader metric code
    └── baselines/
        └── UW-CycleGAN/            ← REPO_ROOT (this repo)
            ├── config.yaml
            ├── environment.yml
            ├── train.py
            ├── eval.py
            ├── trainer.py
            ├── datasets/
            │   └── uieb_unpaired.py
            ├── losses/
            │   ├── adversarial.py
            │   ├── content.py
            │   └── cycle.py
            ├── models/
            │   ├── discriminator.py
            │   ├── generator.py
            │   └── vgg_content.py
            └── utils/
                └── blur.py
```

---

## Splits

Split files are expected under the UIEB dataset directory. This repo's default
`config.yaml` points to `experiments/datasets/UIEB/splits/`, and the loader falls back to
`experiments/datasets/UIEB/train.txt`, `val.txt`, and `test.txt` when that `splits/`
directory is not present.

---

## Environment

```bash
mamba env remove -n uw-cyclegan -y && \
mamba create -n uw-cyclegan python=3.11.15 -y && \
mamba activate uw-cyclegan && \
mamba install -c conda-forge numpy pillow scipy matplotlib pyyaml tqdm -y && \
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
pip install tensorboard torchmetrics && \
mamba env export > environment.yml
```

> PyTorch is installed via pip from the CUDA 12.1 wheel index; the remaining core
> scientific stack is installed with mamba.

---

## Training

```bash
python train.py --config config.yaml --device cuda
```

All paths are relative to REPO_ROOT (`experiments/models/baselines/UW-CycleGAN/`).
Checkpoints and TensorBoard logs are saved to `runs/`; the best checkpoint by validation
PSNR is `runs/best.pth`.

---

## Evaluation

Uses the shared TransFuse-GAN thesis evaluation infrastructure. Metrics: PSNR, SSIM,
CIEDE2000, chroma_ab, UCIQE, and UIQM, consistent with the thesis evaluation protocol.

```bash
python eval.py --checkpoint runs/best.pth --split test --device cuda
```

---

## Citation

UW-CycleGAN:

```bibtex
@article{duUnpairedUnderwaterImage2022,
  title = {Unpaired Underwater Image Enhancement Based on {{CycleGAN}}},
  author = {Du, Rong and Li, Weiwei and Chen, Shudong and Li, Congying and Zhang, Yong},
  year = 2022,
  journal = {Information},
  volume = {13},
  number = {1},
  pages = {1},
  doi = {10.3390/info13010001},
}

```

CycleGAN:

```bibtex
@inproceedings{zhuUnpairedImageImage2017,
  title = {Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks},
  author = {Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A.},
  year = 2017,
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
  pages = {2223--2232},
}

```

UIEB dataset:

```bibtex
@article{liUnderwaterImageEnhancement2020,
  title = {An {{Underwater Image Enhancement Benchmark Dataset}} and {{Beyond}}},
  author = {Li, Chongyi and Guo, Chunle and Ren, Wenqi and Cong, Runmin and Hou, Junhui and Kwong, Sam and Tao, Dacheng},
  year = 2020,
  journal = {IEEE Transactions on Image Processing},
  volume = {29},
  pages = {4376--4389},
  issn = {1941-0042},
  doi = {10.1109/TIP.2019.2955241},
  urldate = {2025-04-26},
}

```

---

## Acknowledgements

This reimplementation uses the shared UIEB loader and evaluation protocol from the
TransFuse-GAN thesis infrastructure to keep dataset splits and metric computation
consistent across all compared methods.
