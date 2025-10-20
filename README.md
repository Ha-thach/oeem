# OEEM (Online Easy Example Mining) for BCSS_WSSS

## Mục tiêu
Áp dụng **OEEM (Online Easy Example Mining)** cho **Weakly-Supervised Segmentation** trên dataset **BCSS_WSSS**, gồm hai giai đoạn:

1. **Stage 1 – Pseudo Mask Generation (Classification-based CAM)**
2. **Stage 2 – Segmentation Training with DeeplabV2**

---

## 1. Environment 
Create new virtual environment
```bash
conda create -n oeem python=3.10.19
conda activate oeem
```

Install dependents library
```bash
pip install -v requirements.txt
```

## 🧩 2. Directory Structure (simplified)

```
OEEM/
├── classification/
│   ├── train_stage1.py          
│   ├── generate_pseudo_masks.py 
│   └── weights/                 
│
├── segmentation/
│   ├── train_stage2.py          
│   ├── test_stage2.py           
│   ├── dataset.py               
│   ├── models/                  
│   ├── results/
│   ├── configuration_seg.yml    
│   └── demo/
│       └── oeem_vis.png
│
└── data/
    └── BCSS_WSSS/
        ├── train/
        ├── valid/
        │     └──img
        │      └──mask
        ├── valid/
              └──img
              └──mask     
```

---

## 🧠 3. Stage 1 — Pseudo Mask Generation

### (a) Train classification backbone
- Tải weights của ResNet18.pth.tar
- Cập nhật dataset_dir ở clasification/configuration.yaml
```bash
python classification/train_stage1.py --epochs 20 --batch 16 --lr 1e-4
```

### (b) Generate pseudo masks (CAM)
```bash
python classification/generate_pseudo_masks.py --model resnet18 --out_dir data/BCSS_WSSS/pseudo_mask/
```

Output là CAM được được lưu dạng data/BCSS_WSSS/pseudo_mask/*.png

---

## 🧩 4. Stage 2 — OEEM Segmentation

-  Tiếp tục cập nhật config data_dir ở`segmentation/configuration_seg.yml`)
- Train 

```bash
python segmentation/train_stage2.py --epoch 20
```

### (c) Test segmentation model
```bash
python segmentation/test_stage2.py
```

---

## 📈 5. Expected Metrics (on mask vs mask sanity check)

| Metric | Expected (mask vs mask) | Meaning |
|---------|-------------------------|----------|
| mIoU | 100% | correct metric setup |
| Dice | 100% | perfect overlap |
| FwIoU | 100% | class-weighted IoU correct |
| bIoU | 100% | boundary logic valid |

---

## 🧩 6. Palette for Visualization (BCSS-WSSS)

```python
LABEL_TO_COLOR = {
    0: [255, 0, 0],     # Tumor (TUM)
    1: [0, 255, 0],     # Stroma (STR)
    2: [0, 0, 255],     # Lymphocyte (LYM)
    3: [153, 0, 255]    # Necrosis (NEC)
}
```


## 🧠 8. Key Points for BCSS_WSSS

| Stage | Data Input | Output |
|--------|-------------|---------|
| Stage 1 | Patchs + class labels | Pseudo masks (CAM) |
| Stage 2 | Images + pseudo masks | Final segmentation model |
| Testing | Validation set | Predicted color masks + metrics |
