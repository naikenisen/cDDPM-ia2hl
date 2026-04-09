import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from cleanfid import fid
from skimage.metrics import structural_similarity

try:
    import lpips
except ImportError as e:
    raise ImportError(
        "LPIPS dependency is missing. Install with: pip install lpips"
    ) from e


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images_by_stem(folder):
    folder = Path(folder)
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
    files = sorted(files)
    by_stem = {}
    duplicates = set()
    for p in files:
        stem = p.stem
        if stem in by_stem:
            duplicates.add(stem)
        by_stem[stem] = p
    if duplicates:
        raise ValueError(
            f"Duplicate stems found in {folder}: {sorted(list(duplicates))[:5]}"
            " (showing up to 5). Ensure one file per sample id."
        )
    return by_stem


def load_rgb_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def to_tensor_minus1_1(img_np):
    # Converts HWC [0,255] image to CHW tensor in [-1,1] for LPIPS.
    t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    return (t * 2.0) - 1.0


def psnr(img_a, img_b, max_val=255.0):
    mse = max(np.mean((img_a - img_b) ** 2), 1e-12)
    return 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)


def ssim_rgb(img_a, img_b):
    score = structural_similarity(img_a, img_b, data_range=255.0, channel_axis=2, full=False)
    if isinstance(score, tuple):
        score = score[0]
    return float(score)


def bootstrap_mean_ci(values, n_boot=1000, ci_level=95.0, seed=0):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = values[idx].mean()
    alpha = (100.0 - ci_level) / 2.0
    lo = np.percentile(boot_means, alpha)
    hi = np.percentile(boot_means, 100.0 - alpha)
    return float(values.mean()), float(lo), float(hi)


def _safe_link_or_copy(src, dst):
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def compute_fid_for_paths(src_paths, dst_paths):
    with tempfile.TemporaryDirectory(prefix="fid_src_full_") as tmp_src, tempfile.TemporaryDirectory(prefix="fid_dst_full_") as tmp_dst:
        for j, (src_p, dst_p) in enumerate(zip(src_paths, dst_paths)):
            _safe_link_or_copy(str(src_p), os.path.join(tmp_src, f"{j:08d}{src_p.suffix.lower()}"))
            _safe_link_or_copy(str(dst_p), os.path.join(tmp_dst, f"{j:08d}{dst_p.suffix.lower()}"))
        return float(fid.compute_fid(tmp_src, tmp_dst))


def bootstrap_fid_ci(src_paths, dst_paths, n_boot=100, ci_level=95.0, seed=0):
    if len(src_paths) != len(dst_paths):
        raise ValueError("src_paths and dst_paths must have the same length")
    n = len(src_paths)
    if n == 0:
        raise ValueError("No images available to bootstrap FID")

    rng = np.random.default_rng(seed)
    boot_fids = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        with tempfile.TemporaryDirectory(prefix="fid_src_") as tmp_src, tempfile.TemporaryDirectory(prefix="fid_dst_") as tmp_dst:
            for j, k in enumerate(idx):
                _safe_link_or_copy(str(src_paths[k]), os.path.join(tmp_src, f"{j:08d}{src_paths[k].suffix.lower()}"))
                _safe_link_or_copy(str(dst_paths[k]), os.path.join(tmp_dst, f"{j:08d}{dst_paths[k].suffix.lower()}"))
            boot_fids[i] = fid.compute_fid(tmp_src, tmp_dst)
    alpha = (100.0 - ci_level) / 2.0
    lo = np.percentile(boot_fids, alpha)
    hi = np.percentile(boot_fids, 100.0 - alpha)
    return float(lo), float(hi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate paired test set with PSNR, SSIM, LPIPS, and FID + confidence intervals")
    parser.add_argument('-s', '--src', type=str, required=True, help='Ground truth images directory')
    parser.add_argument('-d', '--dst', type=str, required=True, help='Generated images directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for LPIPS forward pass')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device used for LPIPS')
    parser.add_argument('--ci-level', type=float, default=95.0, help='Confidence interval level in percent')
    parser.add_argument('--bootstrap-samples', type=int, default=1000, help='Bootstrap samples for PSNR/SSIM/LPIPS CI')
    parser.add_argument('--fid-bootstrap-samples', type=int, default=100, help='Bootstrap samples for FID CI (slow)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for bootstrap sampling')

    args = parser.parse_args()

    src_map = list_images_by_stem(args.src)
    dst_map = list_images_by_stem(args.dst)
    common_stems = sorted(set(src_map.keys()) & set(dst_map.keys()))
    if not common_stems:
        raise ValueError("No common image stems between src and dst")

    missing_in_dst = sorted(set(src_map.keys()) - set(dst_map.keys()))
    missing_in_src = sorted(set(dst_map.keys()) - set(src_map.keys()))
    if missing_in_dst:
        print(f"Warning: {len(missing_in_dst)} src samples have no matching dst sample. Ignoring them.")
    if missing_in_src:
        print(f"Warning: {len(missing_in_src)} dst samples have no matching src sample. Ignoring them.")

    src_paths = [src_map[k] for k in common_stems]
    dst_paths = [dst_map[k] for k in common_stems]

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    psnr_vals = []
    ssim_vals = []
    lpips_vals = []

    with torch.no_grad():
        for i in range(0, len(common_stems), args.batch_size):
            batch_src_paths = src_paths[i:i + args.batch_size]
            batch_dst_paths = dst_paths[i:i + args.batch_size]

            src_np = [load_rgb_image(p) for p in batch_src_paths]
            dst_np = [load_rgb_image(p) for p in batch_dst_paths]

            for a, b in zip(src_np, dst_np):
                if a.shape != b.shape:
                    raise ValueError(f"Shape mismatch for a pair: {a.shape} vs {b.shape}")
                psnr_vals.append(psnr(a, b))
                ssim_vals.append(ssim_rgb(a, b))

            src_t = torch.stack([to_tensor_minus1_1(x) for x in src_np], dim=0).to(device)
            dst_t = torch.stack([to_tensor_minus1_1(x) for x in dst_np], dim=0).to(device)
            lp = lpips_model(src_t, dst_t).view(-1).cpu().numpy()
            lpips_vals.extend(lp.tolist())

    fid_score = compute_fid_for_paths(src_paths, dst_paths)

    psnr_mean, psnr_lo, psnr_hi = bootstrap_mean_ci(
        psnr_vals, n_boot=args.bootstrap_samples, ci_level=args.ci_level, seed=args.seed
    )
    ssim_mean, ssim_lo, ssim_hi = bootstrap_mean_ci(
        ssim_vals, n_boot=args.bootstrap_samples, ci_level=args.ci_level, seed=args.seed + 1
    )
    lpips_mean, lpips_lo, lpips_hi = bootstrap_mean_ci(
        lpips_vals, n_boot=args.bootstrap_samples, ci_level=args.ci_level, seed=args.seed + 2
    )
    fid_lo, fid_hi = bootstrap_fid_ci(
        src_paths, dst_paths, n_boot=args.fid_bootstrap_samples, ci_level=args.ci_level, seed=args.seed + 3
    )

    print(f"Matched test samples: {len(common_stems)}")
    print(f"PSNR  mean={psnr_mean:.6f}  CI[{args.ci_level:.1f}%]=[{psnr_lo:.6f}, {psnr_hi:.6f}]")
    print(f"SSIM  mean={ssim_mean:.6f}  CI[{args.ci_level:.1f}%]=[{ssim_lo:.6f}, {ssim_hi:.6f}]")
    print(f"LPIPS mean={lpips_mean:.6f} CI[{args.ci_level:.1f}%]=[{lpips_lo:.6f}, {lpips_hi:.6f}]")
    print(f"FID   value={fid_score:.6f} CI[{args.ci_level:.1f}%]=[{fid_lo:.6f}, {fid_hi:.6f}]")