# Running the Code

## 1. Create an environment and install dependencies

```powershell
conda create -n il2mc python=3.8 -y
conda activate il2mc
python -m pip install -r requirements.txt
```

**Dependency note:** The current `requirements.txt` contains references to local Linux files (`@ file:///home/...` and `@ file:///tmp/...`). Replace these references with installable package specifications and appropriate versions before running the installation command.

## 2. Download the datasets

The datasets are publicly available from their providers. Download them separately instead of committing raw data, processed data, training logs, or model checkpoints to GitHub.

| Dataset | Download page | Local input filenames |
| --- | --- | --- |
| Taobao UserBehavior | [Tianchi](https://tianchi.aliyun.com/dataset/649) | Rename the extracted `UserBehavior.csv` to `taobao_user_behavior.csv`. |
| MovieLens | [GroupLens](https://grouplens.org/datasets/movielens/) | Rename the compatible ratings file to `movie.csv` and the movie metadata file to `movie_meta.csv`. |
| Kuaishou / KuaiRec | [KuaiRec](https://kuairec.com/) | Rename `big_matrix.csv` to `kuaishou_big_matrix.csv` and `item_categories.csv` to `kuaishou_item_feat.csv`. |

Place the extracted input files in:

```text
Disentangling_Interest_Network_for_Sequential_Recommendation/
  tests/resources/deeprec/slirec/
```

## 3. Preprocess a dataset

From the directory containing this README, run the following commands. Set `$dataset` to `taobao`, `ml`, or `kuaishou`.

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

This calls the existing preprocessing function without launching model training. It generates `train_data`, `valid_data`, `test_data`, vocabulary files, and `itemidx2cat.pkl`. Each run rewrites the outputs in the same directory; keep separate copies if preparing multiple datasets. Processing large datasets requires sufficient memory and disk space.

## 4. Run the model

After preprocessing, while still in the project directory, set `datatype` in `examples/00_quick_start/sequential_recsys_amazondataset.py` to the dataset you processed (`taobao`, `ml`, or `kuaishou`). Then run:

```powershell
Set-Location "examples\00_quick_start"
python sequential_recsys_amazondataset.py
```
