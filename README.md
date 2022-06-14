
## Create experimental env

- Linux machine with four GPUs
- `Conda`
- Python's dependencies: [this file](./spec-file.txt):

```bash
conda create --name research --file spec-file.txt
conda activate research
```

### External dependency

#### Apex install

```bash
git clone git@github.com:NVIDIA/apex.git
cd apex
git checkout 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a  # to fix the exact library version
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Detailed version of PyTorch

```bash
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

## Preparation

## Training

Please run __content of all scripts__ in `./scripts/**/train/` under [`code`](./code).

## Evaluation

After training, please run [`./gather_weights.py`](./gather_weights.py) to generate text files for evaluation.

Please run __content__ of all scripts in `./scripts/**/eval/` under [`code`](./code) as well.

For the AG news dataset, please run [`code/notebooks/filter_ag_news.ipynb`](code/notebooks/filter_ag_news.ipynb) __after__ evaluation of mean classifier __before__ the other evaluation scripts such as linear classifier and bound computation.

To obtain all figures and tables in the paper, you run notebooks in [`code/notebooks/`](./code/notebooks). The codes save generated figures and tables into [`./doc/figs`](./doc/figs) and [`./doc/tabs`](./doc/tabs), respectively.
