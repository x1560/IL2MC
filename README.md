# IL2MC

The commands below use Windows PowerShell. Start in the repository root.

## Setup

```powershell
conda create -n il2mc python=3.8 -y
conda activate il2mc
python -m pip install -r requirements.txt
```

Some entries in `requirements.txt` point to local Linux files (`@ file:///home/...` or `@ file:///tmp/...`). Replace them with package names and matching versions before installing.

## Data

Download the datasets from the links below. Data files and training outputs are not included in this repository.

| Dataset | Download | Filenames used by the code |
| --- | --- | --- |
| Taobao UserBehavior | [Tianchi](https://tianchi.aliyun.com/dataset/649) | `UserBehavior.csv` → `taobao_user_behavior.csv` |
| MovieLens | [GroupLens](https://grouplens.org/datasets/movielens/) | Compatible ratings file → `movie.csv`; movie metadata → `movie_meta.csv` |
| Kuaishou / KuaiRec | [KuaiRec](https://kuairec.com/) | `big_matrix.csv` → `kuaishou_big_matrix.csv`; `item_categories.csv` → `kuaishou_item_feat.csv` |

Extract and rename the files, then place them in:

```text
Disentangling_Interest_Network_for_Sequential_Recommendation/
  tests/resources/deeprec/slirec/
```

## Preprocessing

Run from the repository root. Set `$dataset` to `taobao`, `ml`, or `kuaishou`.

```powershell
Set-Location "Disentangling_Interest_Network_for_Sequential_Recommendation"
$env:PYTHONPATH = (Get-Location).Path
$dataset = "taobao"

@'
import sys
from pathlib import Path
from recommenders.datasets.amazon_reviews import data_preprocessing

dataset = sys.argv[1]
data = Path("tests/resources/deeprec/slirec")
data.mkdir(parents=True, exist_ok=True)

inputs = {
    "taobao": ("taobao_user_behavior.csv", ""),
    "ml": ("movie.csv", "movie_meta.csv"),
    "kuaishou": ("kuaishou_big_matrix.csv", "kuaishou_item_feat.csv"),
}
reviews, metadata = inputs[dataset]

data_preprocessing(
    reviews_file=str(data / reviews),
    meta_file=str(data / metadata),
    train_file=str(data / "train_data"),
    valid_file=str(data / "valid_data"),
    test_file=str(data / "test_data"),
    user_vocab=str(data / "user_vocab.pkl"),
    item_vocab=str(data / "item_vocab.pkl"),
    cate_vocab=str(data / "category_vocab.pkl"),
    itemidx2cat=str(data / "itemidx2cat.pkl"),
    dataset=dataset,
    sample_rate=1.0,
    valid_num_ngs=4,
    test_num_ngs=99,
)
'@ | python - $dataset
```

The output directory contains `train_data`, `valid_data`, `test_data`, the vocabularies, and `itemidx2cat.pkl`. Running preprocessing again overwrites these files. Back them up before switching datasets.

## Training

Set `datatype` in `examples/00_quick_start/sequential_recsys_amazondataset.py` to the dataset you processed. From the project directory used above, run:

```powershell
Set-Location "examples\00_quick_start"
python sequential_recsys_amazondataset.py
```

## Baselines

The paper compares IL2MC with the following six methods (Section 6.1.3).

| Method | Description |
| --- | --- |
| GRU4Rec | Uses recurrent units to learn sequential behavior and short-term interests. |
| Caser | Applies convolutional filters to recent interactions to learn sequential patterns. |
| SLi-Rec | Combines time-aware and content-aware recurrent modeling with adaptive fusion of long-term and short-term interests. |
| CLSR | Uses separate encoders and self-supervised learning to disentangle long-term and short-term interests. |
| SAPM | Uses preference stability scores to combine long-term interests, short-term interests, and candidate features. |
| SUM | Maintains multiple interest channels to represent different aspects of user preferences in long sequences. |

## Experimental settings

These settings are reported in Sections 4.1 and 6.1 of *Calibrating Interest Mutation with Locality Modeling for Sequential Recommendation*.

### Data and evaluation

- Apply iterative 10-core filtering to users and items.
- Sort interactions globally by time, then split them into training, validation, and test sets at an 8:1:1 ratio.
- Sample negative items from those the user has not interacted with. The positive-to-negative ratio is 1:4 for training and validation, and 1:99 for testing.
- Evaluate with MRR, HR, NDCG, and Group AUC (GAUC). Table 3 reports MRR, HR@2, NDCG@2, and GAUC; Figure 9 also compares HR@K and NDCG@K at other cutoffs.

### Model and training

| Setting | Value reported in the paper |
| --- | --- |
| Framework | Microsoft Recommenders |
| Learning rate | `1e-3` |
| Batch size | `400` |
| Prediction MLP hidden layers | `[80, 40]` |
| L2 regularization strength | `1e-6` |
| Training environment | Linux, NVIDIA RTX 4090, CUDA 11.6 |

| Dataset setting | Maximum sequence length | Number of segmentation scales (`step_L`) |
| --- | --- | --- |
| Taobao | 50 | 3 |
| MovieLens | 50 | 3 |
| Kuaishou_50 | 50 | 3 |
| Kuaishou_100 | 100 | 4 |

`Kuaishou_50` and `Kuaishou_100` use the same dataset with different history truncation lengths.

The checked-in YAML files differ from the paper settings: `ILMN.yaml` and `AILMN.yaml` use `batch_size: 500` and prediction `layer_sizes: [100, 64]`; `ILMN.yaml` also has `step_L: 1`. Adjust these values and `max_seq_length` in the selected configuration when reproducing the paper. `att_fcn_layer_sizes` controls the attention networks, not the prediction MLP.
