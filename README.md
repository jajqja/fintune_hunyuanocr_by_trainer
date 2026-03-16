# FINETUNE HUYNYANOCR BY TRAINER


## Dataset Preparation
The system automatically detects your data if it follows one of these three structures:

### Structure A: Image + TXT Pairs
Place a `.txt` file with the exact same name as the image in the same directory.

```
data/
  img_001.jpg
  img_001.txt      ← content: "ground truth text"
  img_002.png
  img_002.txt
```

### Structure B: Centralized Labels
A single file mapping filenames to their ground truth.

**Option 1**: `labels.txt` or `gt.txt`
Format: filename `<TAB>` ground_truth.

```
data/
  images/
    img_001.jpg
    img_002.jpg
  labels.txt       ← format: "img_001.jpg\tground truth text"
```

**Option 2**: `labels.json`
Supports both dictionary and list formats.
```
data/
  images/
    img_001.jpg
  labels.json      ← {"img_001.jpg": "ground truth text"}
```

### Structure C: Recursive Subdirectories
The loader will scan all folders within the data_path for valid pairs or labels.
```
data/
  folder_1/
    img_001.jpg
    img_001.txt
  folder_2/
    img_002.jpg
    img_002.txt
```

## Requirements
- Python: 3.12+ 
- CUDA: 12.9
- PyTorch: 2.7.1
- GPU: NVIDIA GPU support CUDA

## Installation
```bash
pip install accelerate trl
pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4
```

## Training
```bash
./train.sh
```
