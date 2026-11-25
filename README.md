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

## Model files

| Model | File location |
| --- | --- |
| IL2MC / ILMN (ours) | [Disentangling_Interest_Network_for_Sequential_Recommendation/recommenders/models/deeprec/models/sequential/ILMN.py](Disentangling_Interest_Network_for_Sequential_Recommendation/recommenders/models/deeprec/models/sequential/ILMN.py) |
| AILMN (variant) | [Disentangling_Interest_Network_for_Sequential_Recommendation/recommenders/models/deeprec/models/sequential/AILMN.py](Disentangling_Interest_Network_for_Sequential_Recommendation/recommenders/models/deeprec/models/sequential/AILMN.py) |
| GRU4Rec | [Disentangling_Interest_Network_for_Sequential_Recommendation/recommenders/models/deeprec/models/sequential/gru4rec.py](Disentangling_Interest_Network_for_Sequential_Recommendation/recommenders/models/deeprec/models/sequential/gru4rec.py) |
| Caser | [Disentangling_Interest_Network_for_Sequential_Recommendation/recommenders/models/deeprec/models/sequential/caser.py](Disentangling_Interest_Network_for_Sequential_Recommendation/recommenders/models/deeprec/models/sequential/caser.py) |
| SLi-Rec | [Disentangling_Interest_Network_for_Sequential_Recommendation/recommenders/models/deeprec/models/sequential/sli_rec.py](Disentangling_Interest_Network_for_Sequential_Recommendation/recommenders/models/deeprec/models/sequential/sli_rec.py) |
| SUM | [Disentangling_Interest_Network_for_Sequential_Recommendation/recommenders/models/deeprec/models/sequential/sum.py](Disentangling_Interest_Network_for_Sequential_Recommendation/recommenders/models/deeprec/models/sequential/sum.py) |

## Experimental settings

### 6.1.4 Implementation Details.

To ensure fairness and facilitate performance comparison, we conduct experiments under the Microsoft Recommenders framework following CLSR [53]. The learning rate is set to 1e-3. For the Taobao and Movielens datasets, the maximum sequence length is set to 𝑀𝑎𝑥𝐿𝑒𝑛= 50, while for the Kuaishou dataset, it is set to 𝑀𝑎𝑥𝐿𝑒𝑛= 50 and 𝑀𝑎𝑥𝐿𝑒𝑛= 100, respectively. The prediction layers of all models is MLP (Multilayer Perceptron) with two hidden layers of dimensions [80, 40]. The ratio of positive to negative examples in the training and validation set is 1:4, and in the testing set, it is 1:99, with the batch size being set to 400. We set the regularization strength 𝛾 to 1e-6. For the set size in the segmentation step |𝐿| in our model, we set it to 3 for the Taobao, Movielens, and Kuaishou_50 datasets, and to 4 for the Kuaishou_100 dataset. All experiments are conducted on Linux servers equipped with NVIDIA RTX 4090 GPU, using CUDA Version 11.6.
