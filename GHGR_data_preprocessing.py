#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_preprocessing.py

GHGR malicious VPN traffic detection data preprocessing module.

This file extracts the data-processing part from the provided experimental code:
1. Read PCAP/PCAPNG files with PyShark/tshark.
2. Extract payload-free packet metadata.
3. Generate session-level statistical features.
4. Generate flow/window-level sequential features.
5. Apply adaptive hybrid filtering and damping-window weighting.
6. Split data at capture-file level to reduce leakage risk.
7. Fit scalers only on the training split.
8. Save processed arrays to NPZ cache.

Dependencies:
    pip install pyshark pandas numpy scikit-learn scipy

External dependency:
    tshark / Wireshark must be installed.
"""

import argparse
import gc
import json
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyshark
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


PCAP_SUFFIXES = (".pcap", ".pcapng")


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------
def pad_sequence(seq: Sequence[float], length: int) -> np.ndarray:
    """Pad or truncate a 1-D sequence to fixed length."""
    arr = np.asarray(seq, dtype=np.float32)
    if len(arr) >= length:
        return arr[:length].astype(np.float32)
    return np.pad(arr, (0, length - len(arr)), mode="constant").astype(np.float32)


def median_filter(data: Sequence[float], window_size: int = 3) -> np.ndarray:
    """Simple median filter implemented with NumPy."""
    data = np.asarray(data, dtype=np.float32)
    if len(data) == 0:
        return np.array([], dtype=np.float32)

    window_size = min(window_size, len(data))
    if window_size % 2 == 0:
        window_size = max(1, window_size - 1)

    half = window_size // 2
    filtered = []
    for i in range(len(data)):
        start = max(0, i - half)
        end = min(len(data), i + half + 1)
        filtered.append(np.median(data[start:end]))
    return np.asarray(filtered, dtype=np.float32)


def gaussian_filter(data: Sequence[float], sigma: float = 1.0) -> np.ndarray:
    """Simple Gaussian smoothing implemented with NumPy."""
    data = np.asarray(data, dtype=np.float32)
    if len(data) == 0:
        return np.array([], dtype=np.float32)
    if len(data) < 3:
        return data.copy()

    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, len(data))
    if kernel_size % 2 == 0:
        kernel_size = max(1, kernel_size - 1)

    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    kernel = np.exp(-0.5 * (ax ** 2) / (sigma ** 2))
    kernel = kernel / np.sum(kernel)
    return np.convolve(data, kernel, mode="same").astype(np.float32)


def sliding_std(data: Sequence[float], window_size: int = 5) -> np.ndarray:
    """Compute local sliding-window standard deviation."""
    data = np.asarray(data, dtype=np.float32)
    if len(data) == 0:
        return np.array([], dtype=np.float32)

    window_size = min(window_size, len(data))
    if window_size % 2 == 0:
        window_size = max(1, window_size - 1)

    half = window_size // 2
    stds = []
    for i in range(len(data)):
        start = max(0, i - half)
        end = min(len(data), i + half + 1)
        window_data = data[start:end]
        stds.append(np.std(window_data) if len(window_data) > 1 else 0.0)
    return np.asarray(stds, dtype=np.float32)


def adaptive_hybrid_filter(
    data: Sequence[float],
    median_window: int = 5,
    gaussian_sigma: float = 0.6,
    threshold_ratio: float = 1.0,
) -> np.ndarray:
    """
    Adaptive hybrid filtering.

    Median filtering is used as a robust pre-filter. Gaussian smoothing is then
    applied. If the difference between the original value and the smoothed value
    is larger than the local threshold, the original value is retained; otherwise
    the smoothed value is used.
    """
    data = np.asarray(data, dtype=np.float32)
    if len(data) == 0:
        return np.array([], dtype=np.float32)

    median_window = min(median_window, len(data))
    if median_window < 3:
        median_window = 3 if len(data) >= 3 else len(data)

    median = median_filter(data, window_size=median_window)
    gauss = gaussian_filter(median, sigma=gaussian_sigma)

    diff = np.abs(data - gauss)
    local_std = sliding_std(data, window_size=max(3, median_window * 2))
    threshold = threshold_ratio * local_std

    return np.where(diff > threshold, data, gauss).astype(np.float32)


def calculate_multi_order_diff(data: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized first-order and second-order differences."""
    data = np.asarray(data, dtype=np.float32)
    if len(data) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    if len(data) == 1:
        zeros = np.zeros_like(data, dtype=np.float32)
        return zeros, zeros

    first = np.diff(data)
    first = np.pad(first, (1, 0), mode="edge")

    second = np.diff(first)
    second = np.pad(second, (1, 0), mode="edge")

    first_std = np.std(first)
    if first_std > 1e-6:
        first = (first - np.mean(first)) / first_std
    else:
        first = np.zeros_like(first)

    second_std = np.std(second)
    if second_std > 1e-6:
        second = (second - np.mean(second)) / second_std
    else:
        second = np.zeros_like(second)

    return first.astype(np.float32), second.astype(np.float32)


# ---------------------------------------------------------------------
# PCAP reading
# ---------------------------------------------------------------------
def read_pcap(
    pcap_file: Union[str, Path],
    tshark_path: Union[str, Path],
    max_packets: int = 100_000,
) -> pd.DataFrame:
    """
    Read a PCAP/PCAPNG file and extract payload-free TCP/UDP metadata.

    Returned columns:
        timestamp, src_ip, dst_ip, src_port, dst_port, length,
        direction, protocol, tcp_flags
    """
    pcap_file = str(pcap_file)
    tshark_path = str(tshark_path)

    if not os.path.exists(pcap_file):
        raise FileNotFoundError(f"PCAP file does not exist: {pcap_file}")
    if not os.path.exists(tshark_path):
        raise FileNotFoundError(f"tshark path does not exist: {tshark_path}")

    try:
        cap = pyshark.FileCapture(
            pcap_file,
            display_filter="ip",
            tshark_path=tshark_path,
            keep_packets=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to open PCAP file {pcap_file}: {exc}") from exc

    packets = []
    error_count = 0

    try:
        for idx, packet in enumerate(cap):
            if idx >= max_packets:
                print(f"[WARN] {os.path.basename(pcap_file)} exceeds {max_packets} packets; truncated.")
                break

            if not (hasattr(packet, "ip") and hasattr(packet, "transport_layer")):
                continue

            transport = packet.transport_layer
            if transport not in ("TCP", "UDP"):
                continue

            try:
                src_port = int(packet[transport].srcport)
                dst_port = int(packet[transport].dstport)
                length = int(packet.length)

                if not (0 <= src_port <= 65535 and 0 <= dst_port <= 65535 and length > 0):
                    error_count += 1
                    continue

                packets.append(
                    {
                        "timestamp": packet.sniff_time,
                        "src_ip": packet.ip.src,
                        "dst_ip": packet.ip.dst,
                        "src_port": src_port,
                        "dst_port": dst_port,
                        "length": length,
                        # This follows the original code. In a real gateway setting,
                        # you may replace it with a gateway-aware direction rule.
                        "direction": 1 if packet.ip.src < packet.ip.dst else 0,
                        "protocol": transport,
                        "tcp_flags": packet[transport].flags if transport == "TCP" else "",
                    }
                )
            except Exception:
                error_count += 1
                continue
    finally:
        cap.close()

    if error_count > 0:
        print(f"[INFO] {os.path.basename(pcap_file)} skipped {error_count} abnormal packets.")
    if not packets:
        raise ValueError(f"No valid TCP/UDP packets found in {pcap_file}")

    return pd.DataFrame(packets)


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------
def extract_features(
    df: pd.DataFrame,
    window_length: int = 80,
    damping_factor: float = 0.9,
    max_windows: int = 10_000,
    file_name: str = "unknown",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract statistical and sequential features from one parsed packet DataFrame.

    Statistical features:
        23-dimensional session-level vector repeated for each window.

    Sequential features:
        Shape = [num_windows, window_length, 10]
        Channels:
            0  inbound packet length after damping
            1  outbound packet length after damping
            2  inter-arrival time after damping
            3  filtered inbound length
            4  filtered outbound length
            5  filtered inter-arrival time
            6  first-order diff of inbound length
            7  first-order diff of outbound length
            8  second-order diff of inbound length
            9  packet direction
    """
    if len(df) < window_length:
        raise ValueError(f"[{file_name}] Too few packets: {len(df)} < window_length={window_length}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    total_pkts = len(df)

    in_pkts = df[df["direction"] == 0]
    out_pkts = df[df["direction"] == 1]
    tcp_pkts = df[df["protocol"] == "TCP"]

    total_duration = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
    total_duration = max(float(total_duration), 0.001)

    syn_count = 0
    ack_count = 0
    if not tcp_pkts.empty and "tcp_flags" in tcp_pkts.columns:
        for flag in tcp_pkts["tcp_flags"]:
            if isinstance(flag, str):
                syn_count += int("S" in flag)
                ack_count += int("A" in flag)

    direction_changes = int((df["direction"].diff() != 0).sum())

    global_stats = np.array(
        [
            # Basic traffic statistics
            float(total_pkts),
            float(total_duration),
            float(total_pkts / total_duration),
            float(df["length"].mean() if total_pkts > 0 else 0),
            float(df["length"].std() if total_pkts > 1 else 0),

            # Inbound statistics
            float(len(in_pkts)),
            float(in_pkts["length"].sum() if not in_pkts.empty else 0),
            float(in_pkts["length"].mean() if not in_pkts.empty else 0),
            float(in_pkts["length"].std() if len(in_pkts) > 1 else 0),
            float(len(in_pkts) / total_duration),

            # Outbound statistics
            float(len(out_pkts)),
            float(out_pkts["length"].sum() if not out_pkts.empty else 0),
            float(out_pkts["length"].mean() if not out_pkts.empty else 0),
            float(out_pkts["length"].std() if len(out_pkts) > 1 else 0),
            float(len(out_pkts) / total_duration),

            # Protocol and direction statistics
            float(len(tcp_pkts) / total_pkts if total_pkts > 0 else 0),
            float(1.0 - (len(tcp_pkts) / total_pkts if total_pkts > 0 else 0)),
            float(direction_changes),
            float(len(in_pkts) / len(out_pkts) if len(out_pkts) > 0 else 0),

            # TCP flag statistics
            float(syn_count / len(tcp_pkts) if not tcp_pkts.empty else 0),
            float(ack_count / len(tcp_pkts) if not tcp_pkts.empty else 0),

            # Port diversity
            float(df["dst_port"].nunique()),
            float(df["src_port"].nunique()),
        ],
        dtype=np.float32,
    )
    global_stats = np.nan_to_num(global_stats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if global_stats.shape != (23,):
        raise ValueError(f"[{file_name}] Statistical feature shape error: {global_stats.shape}, expected (23,)")

    seq_features = []
    damping_weights = np.array(
        [damping_factor ** (window_length - 1 - i) for i in range(window_length)],
        dtype=np.float32,
    )

    for start in range(0, len(df) - window_length + 1):
        if len(seq_features) >= max_windows:
            print(f"[WARN] {file_name} exceeds max_windows={max_windows}; truncated.")
            break

        window = df.iloc[start:start + window_length]
        in_win = window[window["direction"] == 0]
        out_win = window[window["direction"] == 1]

        in_len_raw = pad_sequence(in_win["length"].values, window_length)
        out_len_raw = pad_sequence(out_win["length"].values, window_length)

        timestamps = window["timestamp"].apply(lambda x: x.timestamp()).values
        inter_arrivals_raw = pad_sequence(np.diff(timestamps) if len(timestamps) > 1 else [], window_length)

        direction_raw = pad_sequence(window["direction"].values, window_length)

        # Damping window: applied to numerical features, not to categorical direction.
        in_len_raw = in_len_raw * damping_weights
        out_len_raw = out_len_raw * damping_weights
        inter_arrivals_raw = inter_arrivals_raw * damping_weights

        in_len_filtered = adaptive_hybrid_filter(in_len_raw)
        out_len_filtered = adaptive_hybrid_filter(out_len_raw)
        inter_arrivals_filtered = adaptive_hybrid_filter(inter_arrivals_raw, median_window=7)

        in_first_diff, in_second_diff = calculate_multi_order_diff(in_len_filtered)
        out_first_diff, _ = calculate_multi_order_diff(out_len_filtered)

        seq_instance = np.stack(
            [
                in_len_raw,
                out_len_raw,
                inter_arrivals_raw,
                in_len_filtered,
                out_len_filtered,
                inter_arrivals_filtered,
                in_first_diff,
                out_first_diff,
                in_second_diff,
                direction_raw,
            ],
            axis=-1,
        ).astype(np.float32)

        if seq_instance.shape != (window_length, 10):
            raise ValueError(
                f"[{file_name}] Sequential feature shape error: "
                f"{seq_instance.shape}, expected ({window_length}, 10)"
            )

        seq_features.append(seq_instance)

    if not seq_features:
        raise ValueError(f"[{file_name}] No sequential windows generated.")

    num_windows = len(seq_features)
    global_stats_repeated = np.tile(global_stats, (num_windows, 1)).astype(np.float32)
    seq_array = np.asarray(seq_features, dtype=np.float32)

    return global_stats_repeated, seq_array


# ---------------------------------------------------------------------
# Dataset loading, splitting, scaling, caching
# ---------------------------------------------------------------------
def collect_pcap_files(labelled_paths: Sequence[Tuple[Union[str, Path], str]]) -> List[Tuple[str, str]]:
    """
    Expand labelled paths into a list of (pcap_file, label).

    labelled_paths example:
        [
            ("/data/benign_vpn", "benign"),
            ("/data/malicious_vpn", "malicious"),
            ("/data/single_file.pcap", "malicious"),
        ]
    """
    all_files: List[Tuple[str, str]] = []

    for path, label in labelled_paths:
        path = Path(path)

        if path.is_dir():
            files = sorted(
                str(p) for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in PCAP_SUFFIXES
            )
            print(f"[INFO] Found {len(files)} PCAP files in {path} with label={label}")
            all_files.extend((f, label) for f in files)

        elif path.is_file() and path.suffix.lower() in PCAP_SUFFIXES:
            print(f"[INFO] Added file {path} with label={label}")
            all_files.append((str(path), label))

        else:
            print(f"[WARN] Skipped invalid path or unsupported suffix: {path}")

    if not all_files:
        raise ValueError("No PCAP/PCAPNG files found.")

    return all_files


def diagnose_split(
    X_stats_train: np.ndarray,
    X_stats_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Lightweight sanity check for split distribution."""
    print("\n" + "=" * 70)
    print("Data split diagnosis")
    print("=" * 70)
    print(f"Train samples: {X_stats_train.shape[0]}")
    print(f"Test samples : {X_stats_test.shape[0]}")

    total = X_stats_train.shape[0] + X_stats_test.shape[0]
    print(f"Train ratio  : {X_stats_train.shape[0] / total:.2%}")
    print(f"Test ratio   : {X_stats_test.shape[0] / total:.2%}")

    print(f"Train label distribution: {dict(Counter(y_train))}")
    print(f"Test label distribution : {dict(Counter(y_test))}")

    if X_stats_train.shape[0] > 0 and X_stats_test.shape[0] > 0:
        train_mean = np.mean(X_stats_train, axis=0)
        test_mean = np.mean(X_stats_test, axis=0)
        diff = np.mean(np.abs(train_mean - test_mean))
        print(f"Mean absolute difference between train/test statistical means: {diff:.6f}")


def load_mixed_data_strict(
    labelled_paths: Sequence[Tuple[Union[str, Path], str]],
    tshark_path: Union[str, Path],
    window_length: int = 80,
    test_size: float = 0.15,
    random_state: int = 42,
    cache_path: Union[str, Path] = "traffic_data_cache_strict.npz",
    max_packets_per_file: int = 100_000,
    max_windows_per_file: int = 10_000,
    use_cache: bool = True,
) -> Dict[str, object]:
    """
    Load and preprocess PCAP datasets.

    Important anti-leakage design:
        - Split is performed at capture-file level.
        - Windows from the same PCAP file will not appear in both train and test.
        - StandardScaler is fitted only on the training split.
    """
    cache_path = Path(cache_path)

    if use_cache and cache_path.exists():
        print(f"[INFO] Loading cache: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        required = {
            "X_stats_train", "X_seqs_train", "y_train",
            "X_stats_test", "X_seqs_test", "y_test",
            "classes", "scaler_stats", "scaler_seq",
        }
        if required.issubset(set(cached.files)):
            return {
                "X_stats_train": cached["X_stats_train"].astype(np.float32),
                "X_seqs_train": cached["X_seqs_train"].astype(np.float32),
                "y_train": cached["y_train"].astype(np.int64),
                "X_stats_test": cached["X_stats_test"].astype(np.float32),
                "X_seqs_test": cached["X_seqs_test"].astype(np.float32),
                "y_test": cached["y_test"].astype(np.int64),
                "classes": cached["classes"],
                "scaler_stats": cached["scaler_stats"].item(),
                "scaler_seq": cached["scaler_seq"].item(),
            }
        print("[WARN] Cache is incomplete; rebuilding.")

    all_files = collect_pcap_files(labelled_paths)
    print(f"[INFO] Total PCAP files: {len(all_files)}")

    if len(all_files) < 2:
        raise ValueError("At least two PCAP files are required for train/test splitting.")

    file_paths, file_labels = zip(*all_files)
    label_counts = Counter(file_labels)
    print(f"[INFO] File-level label distribution: {dict(label_counts)}")

    # Use stratified split only when every class has enough files.
    stratify = file_labels if min(label_counts.values()) >= 2 else None
    if stratify is None:
        print("[WARN] Some labels have fewer than two files; using non-stratified split.")

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        list(file_paths),
        list(file_labels),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    print(f"[INFO] Train files: {len(train_paths)} | Test files: {len(test_paths)}")

    def process_file_set(paths: Sequence[str], labels: Sequence[str], split_name: str):
        all_stats, all_seqs, all_y, sample_files = [], [], [], []
        processed = 0

        print(f"\n[INFO] Processing {split_name} split...")
        for idx, (pcap_file, label) in enumerate(zip(paths, labels), start=1):
            file_name = os.path.basename(pcap_file)
            try:
                print(f"  [{idx}/{len(paths)}] {file_name} -> {label}")
                df_pkt = read_pcap(
                    pcap_file,
                    tshark_path=tshark_path,
                    max_packets=max_packets_per_file,
                )
                stats, seqs = extract_features(
                    df_pkt,
                    window_length=window_length,
                    max_windows=max_windows_per_file,
                    file_name=file_name,
                )

                all_stats.append(stats)
                all_seqs.append(seqs)
                all_y.extend([label] * stats.shape[0])
                sample_files.extend([file_name] * stats.shape[0])
                processed += 1

                del df_pkt, stats, seqs
                gc.collect()

            except Exception as exc:
                print(f"  [WARN] Failed to process {file_name}: {exc}")

        if processed == 0:
            raise ValueError(f"No valid data generated for {split_name} split.")

        return (
            np.vstack(all_stats).astype(np.float32),
            np.vstack(all_seqs).astype(np.float32),
            np.asarray(all_y),
            np.asarray(sample_files),
        )

    X_stats_train, X_seqs_train, y_train_raw, train_sample_files = process_file_set(
        train_paths, train_labels, "train"
    )
    X_stats_test, X_seqs_test, y_test_raw, test_sample_files = process_file_set(
        test_paths, test_labels, "test"
    )

    # Fit scalers only on training data.
    print("\n[INFO] Scaling features. Scalers are fitted only on the training split.")
    scaler_stats = StandardScaler()
    X_stats_train = scaler_stats.fit_transform(X_stats_train).astype(np.float32)
    X_stats_test = scaler_stats.transform(X_stats_test).astype(np.float32)

    scaler_seq = StandardScaler()
    seq_train_flat = X_seqs_train.reshape(-1, X_seqs_train.shape[-1])
    seq_test_flat = X_seqs_test.reshape(-1, X_seqs_test.shape[-1])

    scaler_seq.fit(seq_train_flat)
    X_seqs_train = scaler_seq.transform(seq_train_flat).reshape(X_seqs_train.shape).astype(np.float32)
    X_seqs_test = scaler_seq.transform(seq_test_flat).reshape(X_seqs_test.shape).astype(np.float32)

    del seq_train_flat, seq_test_flat
    gc.collect()

    # Encode labels based on training classes.
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw).astype(np.int64)
    y_test = encoder.transform(y_test_raw).astype(np.int64)
    classes = encoder.classes_

    print(f"[INFO] Label mapping: {dict(zip(classes, encoder.transform(classes)))}")
    diagnose_split(X_stats_train, X_stats_test, y_train, y_test)

    save_dict = {
        "X_stats_train": X_stats_train,
        "X_seqs_train": X_seqs_train,
        "y_train": y_train,
        "X_stats_test": X_stats_test,
        "X_seqs_test": X_seqs_test,
        "y_test": y_test,
        "classes": classes,
        "train_sample_files": train_sample_files,
        "test_sample_files": test_sample_files,
        "scaler_stats": scaler_stats,
        "scaler_seq": scaler_seq,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **save_dict)
    print(f"[INFO] Saved processed cache to: {cache_path}")

    return save_dict


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------
def parse_labelled_paths(items: Iterable[str]) -> List[Tuple[str, str]]:
    """
    Parse CLI items in the form:
        /path/to/pcap_or_dir=label

    Example:
        --data "/data/benign=benign" "/data/mal=malicious"
    """
    labelled = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --data item: {item}. Expected format: path=label")
        path, label = item.split("=", 1)
        path = path.strip()
        label = label.strip()
        if not path or not label:
            raise ValueError(f"Invalid --data item: {item}. Expected non-empty path and label.")
        labelled.append((path, label))
    return labelled


def main() -> None:
    parser = argparse.ArgumentParser(description="GHGR data preprocessing from PCAP/PCAPNG files.")
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help='One or more labelled paths, e.g. "/data/benign=benign" "/data/mal=malicious"',
    )
    parser.add_argument(
        "--tshark",
        required=True,
        help="Path to tshark executable, e.g. /usr/bin/tshark or C:/Program Files/Wireshark/tshark.exe",
    )
    parser.add_argument("--window-length", type=int, default=80, help="Sliding window length. Default: 80")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split ratio. Default: 0.15")
    parser.add_argument("--cache-path", default="traffic_data_cache_strict.npz", help="Output NPZ cache path.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore existing cache and rebuild.")
    parser.add_argument("--max-packets-per-file", type=int, default=100_000)
    parser.add_argument("--max-windows-per-file", type=int, default=10_000)

    args = parser.parse_args()
    labelled_paths = parse_labelled_paths(args.data)

    result = load_mixed_data_strict(
        labelled_paths=labelled_paths,
        tshark_path=args.tshark,
        window_length=args.window_length,
        test_size=args.test_size,
        cache_path=args.cache_path,
        max_packets_per_file=args.max_packets_per_file,
        max_windows_per_file=args.max_windows_per_file,
        use_cache=not args.no_cache,
    )

    print("\nDone.")
    print(f"X_stats_train: {result['X_stats_train'].shape}")
    print(f"X_seqs_train : {result['X_seqs_train'].shape}")
    print(f"X_stats_test : {result['X_stats_test'].shape}")
    print(f"X_seqs_test  : {result['X_seqs_test'].shape}")
    print(f"Classes      : {list(result['classes'])}")


if __name__ == "__main__":
    main()
