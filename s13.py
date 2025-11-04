#!/usr/bin/env python3
"""
deep_vipc_data_diagnostics.py
Enhanced diagnostics for ViPC LMDB dataset.

Usage:
    python deep_vipc_data_diagnostics.py --lmdb_path <path_to_lmdb> --out_dir ./diag_out --sample_per_ep 3

What it does:
 - Scans LMDB keys ending with _subgoal_heatmaps
 - Decodes each entry (raw uint8 / pickled PNG list / pickled ndarray)
 - Computes per-episode stats: count, shape, dtype, min, max, mean, percent_zero
 - Saves N sample heatmaps per episode to disk, alongside current/goal images (if those modalities exist)
 - Attempts to unpickle and prints object types for deeper clues
 - Loads index JSON if present (same dir or parent) and prints metadata for subgoal_heatmaps key
 - Writes summary JSON and CSV
"""
import lmdb, os, argparse, json, pickle, traceback
from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path
import csv

# ---------- CONFIG ----------
DEFAULT_LMDB = "/content/drive/MyDrive/pda/data/validation/validation_dataset.lmdb"
OUT_DIR = "vipc_diag_out"
SAMPLES_PER_EP = 3
VERBOSE = True
# ----------------------------

def try_decode(raw_bytes):
    """Return tuple (format_name, decoded_np_array or None, info_str)"""
    # Try raw uint8 contiguous array
    try:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        if arr.size % (56*56) == 0 and arr.size > 0:
            n = arr.size // (56*56)
            decoded = arr.reshape(n, 56, 56)
            return ("raw_uint8", decoded, f"raw_uint8 n={n} shape={decoded.shape}")
    except Exception as e:
        pass

    # Try pickle
    try:
        obj = pickle.loads(raw_bytes)
        if isinstance(obj, list):
            imgs = []
            for idx, b in enumerate(obj):
                try:
                    arr = np.frombuffer(b, np.uint8)
                    im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                    if im is None:
                        # Could be raw bytes not PNG
                        continue
                    imgs.append(im)
                except Exception:
                    continue
            if imgs:
                decoded = np.stack(imgs)
                return ("pickle_png_list", decoded, f"pickle_png_list n={len(imgs)} shape={decoded.shape}")
            else:
                # maybe list of arrays
                try:
                    arrs = [np.asarray(x) for x in obj]
                    decoded = np.stack(arrs)
                    return ("pickle_list_arrays", decoded, f"pickle_list_arrays n={decoded.shape[0]}")
                except Exception:
                    return ("pickle_list_unknown", None, f"pickle_list_unknown len={len(obj)}")
        elif isinstance(obj, np.ndarray):
            return ("pickle_ndarray", obj, f"pickle_ndarray shape={obj.shape}")
        else:
            return ("pickle_other", None, f"pickle_other type={type(obj)}")
    except Exception as e:
        pass

    # fallback: attempt to interpret as image stream (single png)
    try:
        arr = np.frombuffer(raw_bytes, np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if im is not None:
            # single image: return as shape (1,H,W[,C])
            if im.ndim == 3 and im.shape[2] in (1,3,4):
                if im.shape[2] != 1:
                    # convert to gray
                    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                else:
                    im_gray = im.squeeze(-1)
                return ("single_png", im_gray[np.newaxis,...], f"single_png shape={im_gray.shape}")
            else:
                return ("single_png_other", im, f"single_png_other shape={im.shape}")
    except Exception:
        pass

    return ("unknown", None, "could not decode")

def save_img(arr, path):
    # Expect arr to be 2D or 3D
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if arr.dtype != np.uint8:
        # Normalize to 0-255 for saving
        m = arr.min() if arr.size else 0
        M = arr.max() if arr.size else 0
        if M - m > 0:
            out = ((arr - m) / (M - m) * 255.0).astype(np.uint8)
        else:
            out = (arr * 255.0).astype(np.uint8)
    else:
        out = arr
    # If single-channel, write directly
    if out.ndim == 2:
        cv2.imwrite(path, out)
    elif out.ndim == 3:
        if out.shape[2] == 1:
            cv2.imwrite(path, out.squeeze(-1))
        else:
            # assume BGR or RGB; cv2 expects BGR
            cv2.imwrite(path, out)
    else:
        raise RuntimeError("Unsupported array shape for save")

def find_index_json(lmdb_path):
    p = Path(lmdb_path)
    # try same dir for *_index.json
    candidates = list(p.parent.glob("*index*.json"))
    if candidates:
        return str(candidates[0])
    # also try p.parent / p.parent.parent
    for up in (p.parent, p.parent.parent):
        candidates = list(up.glob("*index*.json"))
        if candidates:
            return str(candidates[0])
    return None

def run_diagnostics(lmdb_path, out_dir, sample_per_ep=3):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
    summary = []
    keys_checked = []

    # Try to find index JSON and load if present
    idx_path = find_index_json(lmdb_path)
    idx_data = None
    if idx_path:
        try:
            with open(idx_path, "r") as f:
                idx_data = json.load(f)
            print(f"[INFO] Loaded index JSON: {idx_path}")
        except Exception as e:
            print("[WARN] Could not load index json:", e)

    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        # collect all heatmap keys
        entries = [(k.decode("ascii"), v) for k,v in cursor if k.decode("ascii").endswith("_subgoal_heatmaps")]
        entries.sort()
        print(f"[INFO] Found {len(entries)} subgoal_heatmaps keys")
        for key, _ in tqdm(entries, desc="episodes"):
            keys_checked.append(key)
            raw = txn.get(key.encode("ascii"))
            info = {"key": key, "status": None, "format": None, "message": None}
            if raw is None:
                info.update(status="MISSING")
                summary.append(info)
                continue

            fmt, decoded, msg = try_decode(raw)
            info.update(format=fmt, message=msg)

            if decoded is None:
                info.update(status="DECODE_FAIL")
                # attempt to unpickle to see type
                try:
                    obj = pickle.loads(raw)
                    info["pickle_type"] = str(type(obj))
                except Exception as e:
                    info["pickle_type"] = f"unpickle_failed: {e}"
                summary.append(info)
                continue

            # Ensure decoded is numpy array with shape (N,H,W) or (N,H,W,C)
            arr = np.asarray(decoded)
            if arr.ndim == 3:
                N,H,W = arr.shape
                C = 1
            elif arr.ndim == 4:
                N,H,W,C = arr.shape
                if C == 1:
                    arr = arr.reshape(N,H,W)
                else:
                    # convert to gray by average
                    arr = arr.mean(axis=-1)
                    N,H,W = arr.shape
            else:
                info.update(status="DECODE_SHAPE_UNEXPECTED", shape=list(arr.shape))
                summary.append(info)
                continue

            # compute stats
            total_pixels = N * H * W
            zeros = int(np.sum(arr == 0))
            nonzeros = int(np.sum(arr != 0))
            pct_zero = zeros / total_pixels * 100.0 if total_pixels>0 else 100.0
            mn = float(arr.min()) if arr.size else 0.0
            mx = float(arr.max()) if arr.size else 0.0
            mean = float(arr.mean()) if arr.size else 0.0
            info.update(status="OK", count=int(N), shape=[N,H,W], dtype=str(arr.dtype),
                        min=mn, max=mx, mean=mean, zeros=zeros, nonzeros=nonzeros, pct_zero=pct_zero)
            summary.append(info)

            # Save sample images from this episode for visual inspection
            ep_safe = key.replace("/", "_")
            ep_dir = out_dir / ep_safe
            ep_dir.mkdir(exist_ok=True, parents=True)
            # Save sample heatmaps (up to sample_per_ep)
            how_many = min(sample_per_ep, N)
            for i in range(how_many):
                hm = arr[i].astype(np.float32)
                # Save raw as grayscale PNG scaled for viewing
                save_img(hm, str(ep_dir / f"heatmap_{i:03d}.png"))
            # Try to also save corresponding current_image and goal_image if present in LMDB
            # guess keys: replace segment like 'ep_XXX_subgoal_heatmaps' -> 'ep_XXX_current_image' and 'ep_XXX_goal_image'
            if key.startswith("ep_") and "_subgoal_heatmaps" in key:
                base = key.replace("_subgoal_heatmaps", "")
                for modal in ("current_image", "goal_image"):
                    k2 = f"{base}_{modal}"
                    raw2 = txn.get(k2.encode("ascii"))
                    if raw2:
                        # Try decode similar to heatmaps: image bytes or pickled image arrays
                        fmt2, dec2, msg2 = try_decode(raw2)
                        # if dec2 is array(s), take first entry and save
                        if dec2 is not None:
                            im = dec2[0]
                            if im.ndim == 3 and im.shape[2] in (3,4):
                                # ensure uint8
                                if im.dtype != np.uint8:
                                    im2 = ((im - im.min())/(im.max()-im.min()+1e-8)*255).astype(np.uint8)
                                else:
                                    im2 = im
                                # if likely RGB, cv2 expects BGR; we won't swap unless necessary
                                save_img(im2, str(ep_dir / f"{modal}.png"))
                            elif im.ndim == 2:
                                save_img(im.astype(np.uint8), str(ep_dir / f"{modal}.png"))
                            else:
                                # fallback: attempt to save first channel
                                save_img(im[...,0].astype(np.uint8), str(ep_dir / f"{modal}.png"))
                        else:
                            # attempt unpickle raw2 to inspect
                            try:
                                obj2 = pickle.loads(raw2)
                                with open(ep_dir / f"{modal}_raw_repr.txt","w") as f:
                                    f.write(f"pickle_type: {type(obj2)}\nrepr: {repr(obj2)[:1000]}\n")
                            except Exception as e:
                                with open(ep_dir / f"{modal}_raw_repr.txt","w") as f:
                                    f.write(f"raw decode fail: {e}\n")
            # end of per-episode processing

    # Write summary JSON and CSV
    out_json = out_dir / "vipc_diag_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    out_csv = out_dir / "vipc_diag_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        # CSV header
        header = ["key","status","format","count","shape","dtype","min","max","mean","zeros","nonzeros","pct_zero","message","pickle_type"]
        w.writerow(header)
        for s in summary:
            row = [
                s.get("key"),
                s.get("status"),
                s.get("format"),
                s.get("count"),
                json.dumps(s.get("shape")),
                s.get("dtype"),
                s.get("min"),
                s.get("max"),
                s.get("mean"),
                s.get("zeros"),
                s.get("nonzeros"),
                s.get("pct_zero"),
                s.get("message"),
                s.get("pickle_type", "")
            ]
            w.writerow(row)

    print(f"[DONE] Wrote summary json: {out_json}")
    print(f"[DONE] Wrote summary csv: {out_csv}")
    if idx_data is not None:
        print("[INFO] Index JSON snippet (if has subgoal_heatmaps entry):")
        # attempt to print index entries referencing subgoal_heatmaps
        for k,v in idx_data.items():
            if "subgoal" in k.lower() or "heatmap" in k.lower():
                print(k, "->", v)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", type=str, default=DEFAULT_LMDB, help="Path to LMDB dir/file")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--sample_per_ep", type=int, default=SAMPLES_PER_EP)
    args = parser.parse_args()
    run_diagnostics(args.lmdb_path, args.out_dir, args.sample_per_ep)



# python -m s13 --lmdb_path "C:\Users\Hellx\Documents\Programming\python\Project\redhot\data\training\training_dataset.lmdb" --out_dir "visualize/data/v1"