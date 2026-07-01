# GHGR: Malicious VPN Traffic Detection via Hierarchical Granularity Representation

> Chinese title: **GHGR: 分层粒度表征的恶意 VPN 流量检测**

This repository provides the experimental implementation of **GHGR**, a malicious VPN traffic detection framework based on **Hierarchical Granularity Representation**. GHGR is designed to detect malicious behaviors hidden in encrypted VPN tunnels without decrypting packet payloads or relying on application-layer content.

The implementation follows the main design of the paper and includes:

- payload-free traffic preprocessing;
- session-level statistical feature extraction;
- flow/window-level sequential feature extraction;
- adaptive hybrid filtering and damping-window weighting;
- GRU-TCN temporal representation learning;
- bidirectional cross-modal mutual attention;
- multi-branch supervision;
- improved center-loss regularization;
- NPZ-based data caching and training.

---

## 1. Project Overview

Malicious VPN traffic detection is challenging because VPN tunneling and encryption hide application-layer payloads. Instead of inspecting payload content, GHGR uses only payload-free metadata, including packet length, packet direction, inter-arrival time, protocol type, transport-layer information, port statistics, and session-level traffic statistics.

The complete workflow is:

```text
PCAP / PCAPNG traffic files
        |
        v
data_preprocessing.py
        |
        v
traffic_data_xxx.npz
        |
        v
train_ghgr.py
        |
        v
trained checkpoint + evaluation results
```

The project separates preprocessing and training. The preprocessing script generates an NPZ file once, and the training script directly loads this NPZ file for model training and evaluation.

---

## 2. Main Contributions Implemented in Code

This implementation covers the core components of GHGR:

1. **Payload-free traffic representation**  
   Only metadata and statistical/temporal side-channel features are used. Packet payloads are not decrypted, inspected, or stored.

2. **Hierarchical granularity modeling**  
   GHGR jointly models:
   - session-level statistical behavior;
   - flow/window-level temporal dynamics.

3. **GRU-TCN sequential encoder**  
   The sequential branch combines bidirectional GRU and TCN to capture both long-range temporal dependencies and local temporal patterns.

4. **Cross-modal mutual attention**  
   Statistical and sequential representations interact through bidirectional attention, allowing global session context and local temporal evidence to refine each other.

5. **Improved center loss**  
   The statistical branch is regularized by an improved center-loss module with L2 normalization, class-center initialization, and momentum-based center updates.

6. **Multi-branch training objective**  
   The model uses auxiliary losses for statistical and sequential branches, together with the main fusion loss and center loss.

---

## 3. Recommended Repository Structure

```text
GHGR/
├── README.md
├── requirements.txt
├── data_preprocessing.py
├── train_ghgr.py
├── data/
│   ├── benign/
│   └── malicious/
├── outputs/
│   ├── traffic_data_cache_strict.npz
│   ├── best_branches_optimized_model.pth
│   ├── confusion_matrices_branches.png
│   └── training_metrics_branches_optimized.png
└── scripts/
    ├── run_preprocess.sh
    └── run_train.sh
```

The folder names can be adjusted according to the actual dataset location and local running environment.

---

## 4. Environment Requirements

### Python Version

Python 3.8 or later is recommended.

### Python Dependencies

Install the required packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn pyshark torch
```

For GPU training, install a PyTorch version compatible with the local CUDA version.

### External Dependency

The preprocessing script requires **tshark**, which is included in Wireshark.

Typical tshark paths:

```text
Windows: C:/Program Files/Wireshark/tshark.exe
Linux:   /usr/bin/tshark
```

Before preprocessing, make sure tshark is installed and can be executed correctly.

---

## 5. Data Preprocessing

The preprocessing script converts labelled PCAP/PCAPNG files into an NPZ cache file.

### Input Format

Each input item should be written as:

```text
/path/to/traffic_files=label_name
```

Each path can be either:

- a directory containing PCAP/PCAPNG files; or
- a single PCAP/PCAPNG file.

Example:

```text
/data/benign_vpn=benign
/data/malicious_vpn=malicious
```

### Linux/macOS Example

```bash
python data_preprocessing.py \
  --data "/data/benign_vpn=benign" "/data/malicious_vpn=malicious" \
  --tshark "/usr/bin/tshark" \
  --window-length 80 \
  --cache-path "traffic_data_cache_strict.npz"
```

### Windows Example

```bash
python data_preprocessing.py ^
  --data "D:/data/benign_vpn=benign" "D:/data/malicious_vpn=malicious" ^
  --tshark "C:/Program Files/Wireshark/tshark.exe" ^
  --window-length 80 ^
  --cache-path "traffic_data_cache_strict.npz"
```

---

## 6. Generated NPZ File

The preprocessing script generates an NPZ file containing the following arrays:

```text
X_stats_train
X_seqs_train
y_train
X_stats_test
X_seqs_test
y_test
classes
scaler_stats
scaler_seq
train_sample_files
test_sample_files
```

The training script is compatible with three class-name storage formats:

```text
label_encoder
classes
class_names
```

This means NPZ files generated by older and newer preprocessing scripts can both be used.

The required arrays for training are:

```text
X_stats_train
X_seqs_train
y_train
X_stats_test
X_seqs_test
y_test
```

If any of these fields are missing, the training script will stop and report the missing keys.

---

## 7. Feature Representation

### 7.1 Statistical Features

For each capture/session, GHGR extracts a 23-dimensional statistical feature vector, including:

- total packet count;
- total duration;
- packet rate;
- packet length mean and standard deviation;
- inbound packet count, size, and rate;
- outbound packet count, size, and rate;
- TCP/UDP ratio;
- direction-change count;
- inbound/outbound ratio;
- TCP SYN and ACK flag statistics;
- source and destination port diversity.

The statistical vector is repeated for each sequential window generated from the same capture.

### 7.2 Sequential Features

For each sliding window, GHGR constructs a 10-channel sequential representation:

```text
0. inbound packet length after damping
1. outbound packet length after damping
2. inter-arrival time after damping
3. filtered inbound packet length
4. filtered outbound packet length
5. filtered inter-arrival time
6. first-order difference of inbound length
7. first-order difference of outbound length
8. second-order difference of inbound length
9. packet direction
```

The default window length is:

```text
L = 80
```

This setting follows the main experimental configuration of GHGR.

---

## 8. Model Architecture

The training script implements the GHGR model using the following modules.

### 8.1 Statistical Branch

The statistical branch models global session-level traffic behavior.

Main components:

```text
StatsSelfAttention
stats_branch
stats_classifier
```

The branch first applies self-attention to statistical features and then maps them into a 128-dimensional representation.

### 8.2 Sequential Branch

The sequential branch models local temporal dynamics.

Main components:

```text
GRU_TCN_Module
TemporalConvNet
TemporalBlock
MultiScaleConvBlock
ChannelSELayer
seq_classifier
```

The GRU-TCN module contains:

- a two-layer bidirectional GRU;
- a TCN with dilated convolutions;
- a learnable gate for fusing GRU and TCN outputs.

### 8.3 Cross-Modal Mutual Attention

The fusion module performs bidirectional interaction between statistical and sequential representations.

Main component:

```text
CrossModalMutualAttention
```

The attention directions are:

```text
statistics -> sequence
sequence -> statistics
```

The enhanced statistical and sequential representations are concatenated and projected into the final fused representation.

### 8.4 Improved Center Loss

The improved center-loss module improves intra-class compactness of statistical representations.

Main component:

```text
ImprovedCenterLoss
```

It includes:

- L2 normalization of features and centers;
- class-center initialization using batch class means;
- momentum-based center updates;
- numerical distance clamping for stable optimization.

---

## 9. Training

Before training, generate the NPZ file using `data_preprocessing.py`.

Then set the NPZ path in the training script:

```python
NPZ_FILE_PATH = "traffic_data_cache_strict.npz"
MODEL_SAVE_PATH = "best_branches_optimized_model.pth"
```

Run:

```bash
python train_ghgr.py
```

The training script will:

1. load the NPZ file;
2. check required NPZ fields;
3. read class names from `label_encoder`, `classes`, or `class_names`;
4. convert labels into one-hot format;
5. build the GHGR model;
6. train with multi-branch losses;
7. save the best checkpoint;
8. evaluate the statistical branch, sequential branch, and fusion branch.

---

## 10. Training Objective

The total training loss is:

```text
L_total = w_stats  * L_stats
        + w_seq    * L_seq
        + w_fused  * L_fused
        + w_center * L_center
```

The default loss weights are:

```python
LOSS_WEIGHTS = {
    "stats": 0.3,
    "seq": 0.3,
    "fused": 0.3,
    "center": 0.1
}
```

Where:

- `L_stats` is the auxiliary classification loss of the statistical branch;
- `L_seq` is the auxiliary classification loss of the sequential branch;
- `L_fused` is the main classification loss of the fusion branch;
- `L_center` is the improved center loss.

The training script also uses a warm-up strategy to adjust branch loss weights during early epochs.

---

## 11. Evaluation

After training, the best checkpoint is loaded for final evaluation.

The script reports:

- statistical branch accuracy;
- sequential branch accuracy;
- fusion branch accuracy;
- precision, recall, and F1-score;
- branch-wise classification reports;
- confusion matrices.

Generated output files include:

```text
best_branches_optimized_model.pth
confusion_matrices_branches.png
training_metrics_branches_optimized.png
```

---

## 12. Example Workflow

### Step 1: Generate NPZ Data

```bash
python data_preprocessing.py \
  --data "/data/benign=benign" "/data/malicious=malicious" \
  --tshark "/usr/bin/tshark" \
  --window-length 80 \
  --cache-path "traffic_data_cache_strict.npz"
```

### Step 2: Train GHGR

```bash
python train_ghgr.py
```

### Step 3: Check Results

```text
outputs/
├── best_branches_optimized_model.pth
├── confusion_matrices_branches.png
└── training_metrics_branches_optimized.png
```

---

## 13. Reproducibility Notes

For reproducible experiments, keep the following settings consistent:

- window length;
- dataset labels;
- train/test split strategy;
- random seed;
- tshark version;
- preprocessing cache path;
- feature normalization strategy;
- model hyperparameters;
- loss weights;
- batch size;
- gradient accumulation steps.

The preprocessing script performs capture-file-level splitting to reduce leakage risk. Windows generated from the same PCAP file are kept in the same split.

For stricter experimental protocols, a separate validation set can be added for checkpoint selection, while the test set should only be used for final evaluation.

---

## 14. Citation

If this implementation is used in academic work, please cite the corresponding GHGR paper:

```bibtex
@inproceedings{ghgr2026,
  title     = {GHGR: Malicious VPN Traffic Detection via Hierarchical Granularity Representation},
  author    = {Anonymous},
  booktitle = {Proceedings of Inscrypt},
  year      = {2026}
}
```

---

## 15. Disclaimer

This repository is intended for academic research and defensive security evaluation only. The implementation uses payload-free traffic metadata and does not decrypt, inspect, or store application-layer payload content.

Users should ensure that all traffic collection, processing, and analysis activities comply with applicable laws, institutional rules, and ethical requirements.
