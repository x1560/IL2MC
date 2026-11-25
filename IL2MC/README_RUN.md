# Running the Code

## 1. Create an environment and install dependencies

Run the following commands in PowerShell from the `SNA` directory:

```powershell
conda create -n il2mc python=3.8 -y
conda activate il2mc
python -m pip install -r requirements.txt
```

**Dependency note:** The current `requirements.txt` contains references to local Linux files (`@ file:///home/...` and `@ file:///tmp/...`). Replace these references with installable package specifications and appropriate versions before running the installation command.

## 2. Run the model

To run the implementation closest to the paper's IL2MC model, change `model_sel="AILMN"` to `model_sel="ILMN"` in `IL2MC/examples/00_quick_start/sequential_recsys_amazondataset.py`. Otherwise, the script runs AILMN.

From the `SNA` directory, run:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path "IL2MC"
Set-Location (Join-Path $env:PYTHONPATH "examples\00_quick_start")
python sequential_recsys_amazondataset.py
```

The script trains the model and evaluates the validation and test sets. By default, `train(needRetrain=True)` attempts to resume from an existing `best_model` checkpoint in the configured model directory.

These commands have not been verified with a complete training run on this machine.
