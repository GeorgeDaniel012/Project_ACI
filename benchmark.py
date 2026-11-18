import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent


def load_adapters():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    print("Loading DeblurGANv2...")
    deblurgan_root = PROJECT_ROOT / "deblurganv2"
    sys.path.insert(0, str(deblurgan_root))
    os.chdir(deblurgan_root)
    from deblurgan_adapter import DeblurGANv2Adapter
    dgan = DeblurGANv2Adapter()

    print("Loading NAFNet...")
    nafnet_root = PROJECT_ROOT / "nafnet"
    sys.path.insert(0, str(nafnet_root))
    os.chdir(nafnet_root)
    from nafnet_adapter import NAFNetAdapter
    nafnet = NAFNetAdapter()

    print("Loading Restormer...")
    restormer_root = PROJECT_ROOT / "Restormer"
    sys.path.insert(0, str(restormer_root))
    os.chdir(restormer_root)
    from restormer_adapter import RestormerAdapter
    restormer = RestormerAdapter()

    os.chdir(PROJECT_ROOT)
    return {
        "DeblurGANv2": dgan,
        "NAFNet": nafnet,
        "Restormer": restormer,
    }


def collect_image_pairs(dataset_dir: Path, step: int = 20):
    blur_root = dataset_dir / "blur"
    sharp_root = dataset_dir / "sharp"

    if not blur_root.exists() or not sharp_root.exists():
        raise FileNotFoundError(
            f"Expected 'blur' and 'sharp' folders inside {dataset_dir}"
        )

    pairs = []
    scene_dirs = sorted([d for d in blur_root.iterdir() if d.is_dir()])
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        sharp_scene_dir = sharp_root / scene_name
        if not sharp_scene_dir.exists():
            continue

        blur_files = sorted(
            [f for f in scene_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
        )
        selected_files = blur_files[::step]
        for blur_file in selected_files:
            sharp_file = sharp_scene_dir / blur_file.name
            if sharp_file.exists():
                pairs.append((scene_name, blur_file.name, blur_file, sharp_file))

    return pairs


def evaluate_pair(blur_rgb: np.ndarray, sharp_rgb: np.ndarray, adapters: dict):
    results = {}

    # Baseline (Blurred vs Sharp)
    results["Blurred_PSNR"] = float(compute_psnr(sharp_rgb, blur_rgb, data_range=255))
    results["Blurred_SSIM"] = float(
        compute_ssim(sharp_rgb, blur_rgb, channel_axis=2, data_range=255)
    )

    restored_images = {}
    for model_name, adapter in adapters.items():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        restored_rgb = adapter.deblur(blur_rgb)

        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t0

        psnr_val = float(compute_psnr(sharp_rgb, restored_rgb, data_range=255))
        ssim_val = float(
            compute_ssim(sharp_rgb, restored_rgb, channel_axis=2, data_range=255)
        )

        results[f"{model_name}_PSNR"] = psnr_val
        results[f"{model_name}_SSIM"] = ssim_val
        results[f"{model_name}_Time_s"] = elapsed_s
        restored_images[model_name] = restored_rgb

    return results, restored_images


def save_sample_comparison(
    blur_rgb, sharp_rgb, restored_images, metrics, scene_name, filename, output_dir: Path
):
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"Visual & Quantitative Comparison — {scene_name}/{filename}", fontsize=14, fontweight="bold")

    panels = [
        (
            blur_rgb,
            f"Blurred (Input)\nPSNR: {metrics['Blurred_PSNR']:.2f} dB | SSIM: {metrics['Blurred_SSIM']:.4f}",
        ),
        (
            restored_images["DeblurGANv2"],
            f"DeblurGANv2\nPSNR: {metrics['DeblurGANv2_PSNR']:.2f} dB | SSIM: {metrics['DeblurGANv2_SSIM']:.4f}\nTime: {metrics['DeblurGANv2_Time_s']*1000:.0f} ms",
        ),
        (
            restored_images["NAFNet"],
            f"NAFNet\nPSNR: {metrics['NAFNet_PSNR']:.2f} dB | SSIM: {metrics['NAFNet_SSIM']:.4f}\nTime: {metrics['NAFNet_Time_s']*1000:.0f} ms",
        ),
        (
            restored_images["Restormer"],
            f"Restormer\nPSNR: {metrics['Restormer_PSNR']:.2f} dB | SSIM: {metrics['Restormer_SSIM']:.4f}\nTime: {metrics['Restormer_Time_s']*1000:.0f} ms",
        ),
        (
            sharp_rgb,
            "Ground Truth (Sharp)\nReference Image",
        ),
    ]

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "visual_comparison_sample.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_summary_and_charts(csv_path: Path, output_dir: Path):
    df = pd.read_csv(csv_path)
    if df.empty:
        print("No data in CSV to summarize.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    n_images = len(df)
    n_scenes = df["scene"].nunique()

    summary_rows = [
        {
            "Model": "Blurred (Baseline)",
            "Mean_PSNR_dB": df["Blurred_PSNR"].mean(),
            "Std_PSNR_dB": df["Blurred_PSNR"].std(),
            "Mean_SSIM": df["Blurred_SSIM"].mean(),
            "Std_SSIM": df["Blurred_SSIM"].std(),
            "Mean_Time_s": 0.0,
            "Mean_Time_ms": 0.0,
            "FPS": 0.0,
        }
    ]

    for m in ["DeblurGANv2", "NAFNet", "Restormer"]:
        mean_t = df[f"{m}_Time_s"].mean()
        summary_rows.append(
            {
                "Model": m,
                "Mean_PSNR_dB": df[f"{m}_PSNR"].mean(),
                "Std_PSNR_dB": df[f"{m}_PSNR"].std(),
                "Mean_SSIM": df[f"{m}_SSIM"].mean(),
                "Std_SSIM": df[f"{m}_SSIM"].std(),
                "Mean_Time_s": mean_t,
                "Mean_Time_ms": mean_t * 1000.0,
                "FPS": 1.0 / mean_t if mean_t > 0 else 0.0,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary_table.csv", index=False)

    # Per-scene breakdown
    scene_df = (
        df.groupby("scene")[
            [
                "Blurred_PSNR",
                "DeblurGANv2_PSNR",
                "NAFNet_PSNR",
                "Restormer_PSNR",
                "Blurred_SSIM",
                "DeblurGANv2_SSIM",
                "NAFNet_SSIM",
                "Restormer_SSIM",
            ]
        ]
        .mean()
        .reset_index()
    )
    scene_df.to_csv(output_dir / "per_scene_summary.csv", index=False)

    # Plot 1: PSNR Comparison
    colors = ["#8c8c8c", "#3498db", "#2ecc71", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(summary_df["Model"], summary_df["Mean_PSNR_dB"], color=colors, width=0.55)
    ax.set_ylabel("Average PSNR (dB) — Higher is Better", fontsize=11)
    ax.set_title(f"PSNR Comparison on GoPro Dataset ({n_images} images across {n_scenes} scenes)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(summary_df["Mean_PSNR_dB"]) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.2f} dB", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "psnr_comparison.png", dpi=200)
    plt.close(fig)

    # Plot 2: SSIM Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(summary_df["Model"], summary_df["Mean_SSIM"], color=colors, width=0.55)
    ax.set_ylabel("Average SSIM — Higher is Better (Max 1.0)", fontsize=11)
    ax.set_title(f"SSIM Comparison on GoPro Dataset ({n_images} images across {n_scenes} scenes)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.4f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "ssim_comparison.png", dpi=200)
    plt.close(fig)

    # Plot 3: Inference Time Comparison
    models_only = summary_df[summary_df["Model"] != "Blurred (Baseline)"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models_only["Model"], models_only["Mean_Time_ms"], color=colors[1:], width=0.5)
    ax.set_ylabel("Average Inference Time per Image (ms) — Lower is Faster", fontsize=11)
    ax.set_title("GPU Inference Time Comparison (1280×720 HD, RTX 3060)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(models_only["Mean_Time_ms"]) * 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + max(models_only["Mean_Time_ms"]) * 0.02,
            f"{h:.0f} ms ({h/1000:.2f} s)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(output_dir / "inference_time_comparison.png", dpi=200)
    plt.close(fig)

    # Plot 4: Speed vs Quality Tradeoff
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, row in models_only.iterrows():
        c = colors[idx]
        ax.scatter(row["Mean_Time_ms"], row["Mean_PSNR_dB"], s=180, color=c, label=row["Model"], zorder=5)
        ax.annotate(
            f"{row['Model']}\n({row['Mean_Time_ms']:.0f} ms, {row['Mean_PSNR_dB']:.2f} dB)",
            (row["Mean_Time_ms"], row["Mean_PSNR_dB"]),
            textcoords="offset points",
            xytext=(10, -5),
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xlabel("Average Inference Time per Image (ms) — Lower is Faster", fontsize=11)
    ax.set_ylabel("Average PSNR (dB) — Higher is Better", fontsize=11)
    ax.set_title("Speed vs. Quality Trade-off (GoPro Test Set)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.margins(x=0.25, y=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / "speed_vs_quality_tradeoff.png", dpi=200)
    plt.close(fig)

    # Plot 5: Visual Table PNG for Overall Summary (summary_table.png)
    baseline_psnr = summary_df.loc[summary_df["Model"] == "Blurred (Baseline)", "Mean_PSNR_dB"].values[0]
    baseline_ssim = summary_df.loc[summary_df["Model"] == "Blurred (Baseline)", "Mean_SSIM"].values[0]

    table_data = []
    for _, r in summary_df.iterrows():
        is_base = r["Model"] == "Blurred (Baseline)"
        gain_psnr = "—" if is_base else f"+{r['Mean_PSNR_dB'] - baseline_psnr:.2f} dB"
        gain_ssim = "—" if is_base else f"+{r['Mean_SSIM'] - baseline_ssim:.4f}"
        time_str = "—" if is_base else f"{r['Mean_Time_ms']:.0f} ms ({r['Mean_Time_s']:.2f} s)"
        fps_str = "—" if is_base else f"{r['FPS']:.2f} FPS"
        table_data.append([
            r["Model"],
            f"{r['Mean_PSNR_dB']:.2f} ± {r['Std_PSNR_dB']:.2f} dB",
            gain_psnr,
            f"{r['Mean_SSIM']:.4f} ± {r['Std_SSIM']:.4f}",
            gain_ssim,
            time_str,
            fps_str,
        ])

    col_labels = [
        "Model",
        "PSNR Mediu (dB) ↑",
        "Câștig PSNR ↑",
        "SSIM Mediu ↑",
        "Câștig SSIM ↑",
        "Timp / Imagine ↓",
        "Viteză (FPS) ↑",
    ]

    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")
    ax.set_title(
        f"Tabel Centralizator Benchmark — GoPro Test Set ({n_images} imagini din {n_scenes} scene, RTX 3060)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.9)

    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row_idx == 0:
            cell.set_facecolor("#1f4e78")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_idx == 1:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(style="italic")
        else:
            cell.set_facecolor("#ffffff" if row_idx % 2 == 0 else "#f9fbfd")
            if col_idx == 0:
                cell.set_text_props(fontweight="bold")
            # Highlight best PSNR/SSIM (Restormer, row 4) and best speed (DeblurGANv2, row 2)
            if row_idx == 4 and col_idx in (1, 2, 3, 4):
                cell.set_facecolor("#e2f0d9")
                cell.set_text_props(fontweight="bold", color="#274e13")
            if row_idx == 2 and col_idx in (5, 6):
                cell.set_facecolor("#d9eaf7")
                cell.set_text_props(fontweight="bold", color="#0b5394")

    plt.tight_layout()
    plt.savefig(output_dir / "summary_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Plot 6: Visual Table PNG for Per-Scene Breakdown (per_scene_table.png)
    scene_table_data = []
    for _, r in scene_df.iterrows():
        scene_table_data.append([
            r["scene"],
            f"{r['Blurred_PSNR']:.2f} dB",
            f"{r['DeblurGANv2_PSNR']:.2f} dB",
            f"{r['NAFNet_PSNR']:.2f} dB",
            f"{r['Restormer_PSNR']:.2f} dB",
            f"{r['Blurred_SSIM']:.4f}",
            f"{r['DeblurGANv2_SSIM']:.4f}",
            f"{r['NAFNet_SSIM']:.4f}",
            f"{r['Restormer_SSIM']:.4f}",
        ])
    # Append overall average row
    scene_table_data.append([
        "MEDIA GENERALĂ",
        f"{df['Blurred_PSNR'].mean():.2f} dB",
        f"{df['DeblurGANv2_PSNR'].mean():.2f} dB",
        f"{df['NAFNet_PSNR'].mean():.2f} dB",
        f"{df['Restormer_PSNR'].mean():.2f} dB",
        f"{df['Blurred_SSIM'].mean():.4f}",
        f"{df['DeblurGANv2_SSIM'].mean():.4f}",
        f"{df['NAFNet_SSIM'].mean():.4f}",
        f"{df['Restormer_SSIM'].mean():.4f}",
    ])

    scene_col_labels = [
        "Scenă Video (GoPro)",
        "Blur PSNR",
        "DeblurGANv2 PSNR",
        "NAFNet PSNR",
        "Restormer PSNR",
        "Blur SSIM",
        "DeblurGANv2 SSIM",
        "NAFNet SSIM",
        "Restormer SSIM",
    ]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.axis("off")
    ax.set_title(
        "Rezultate Medii PSNR și SSIM pe Fiecare Scenă din Dataset-ul GoPro",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    stbl = ax.table(
        cellText=scene_table_data,
        colLabels=scene_col_labels,
        loc="center",
        cellLoc="center",
    )
    stbl.auto_set_font_size(False)
    stbl.set_fontsize(9.5)
    stbl.scale(1.0, 1.65)

    last_row_idx = len(scene_table_data)
    for (row_idx, col_idx), cell in stbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row_idx == 0:
            cell.set_facecolor("#1f4e78")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_idx == last_row_idx:
            cell.set_facecolor("#d9e1f2")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#ffffff" if row_idx % 2 == 0 else "#f9fbfd")
            if col_idx == 0:
                cell.set_text_props(fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "per_scene_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Generate human-readable Markdown report (BENCHMARK_REPORT.md)
    md_lines = [
        f"# Raport Benchmark Deblurring — Dataset GoPro ({n_images} imagini, {n_scenes} scene)",
        "",
        "## 1. Tabel Centralizator (Medii Generale)",
        "",
        "| Model | PSNR Mediu (dB) ↑ | Câștig PSNR față de Blur ↑ | SSIM Mediu ↑ | Câștig SSIM față de Blur ↑ | Timp Inferență / Imagine ↓ | Viteză (FPS) ↑ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in table_data:
        md_lines.append(f"| **{row[0]}** | `{row[1]}` | **{row[2]}** | `{row[3]}` | **{row[4]}** | {row[5]} | {row[6]} |")

    md_lines.extend([
        "",
        "## 2. Rezultate Medii pe Fiecare Scenă (Per-Scene Breakdown)",
        "",
        "| Scenă Video | Blur PSNR | DeblurGANv2 PSNR | NAFNet PSNR | Restormer PSNR | Blur SSIM | DeblurGANv2 SSIM | NAFNet SSIM | Restormer SSIM |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for row in scene_table_data:
        md_lines.append(
            f"| **{row[0]}** | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} |"
        )

    (output_dir / "BENCHMARK_REPORT.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Print formatted table
    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY ({n_images} images across {n_scenes} scenes)")
    print("=" * 80)
    print(
        f"{'Model':<22} | {'PSNR (dB)':<18} | {'SSIM':<18} | {'Time / img':<15}"
    )
    print("-" * 80)
    for _, r in summary_df.iterrows():
        t_str = "-" if r["Mean_Time_s"] == 0 else f"{r['Mean_Time_ms']:.0f} ms ({r['Mean_Time_s']:.2f}s)"
        print(
            f"{r['Model']:<22} | {r['Mean_PSNR_dB']:6.2f} ± {r['Std_PSNR_dB']:4.2f} dB | {r['Mean_SSIM']:6.4f} ± {r['Std_SSIM']:6.4f} | {t_str:<15}"
        )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeblurGANv2, NAFNet, and Restormer on GoPro dataset")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=r"D:\GOPRO\GOPRO_3840FPS_AVG_3-21\test",
        help="Path to GoPro test directory containing 'blur' and 'sharp' folders",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=20,
        help="Sample every N-th frame per scene folder (default: 20 -> 110 images total; use 1 for all 2200 images)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on all 2200 images in the test set (equivalent to --step 1)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="benchmark_results.csv",
        help="CSV file to save per-image benchmark results (supports automatic resume)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_outputs",
        help="Directory to save summary tables and plots for presentation",
    )
    args = parser.parse_args()

    step = 1 if args.all else args.step
    dataset_dir = Path(args.dataset_dir)
    csv_path = PROJECT_ROOT / args.output_csv
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_image_pairs(dataset_dir, step=step)
    print(f"Found {len(pairs)} image pairs to benchmark (step={step}).")

    # Check already processed pairs for automatic resume
    processed = set()
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            for _, row in existing_df.iterrows():
                processed.add((str(row["scene"]), str(row["filename"])))
            print(f"Resuming: {len(processed)} images already in {csv_path.name}.")
        except Exception:
            pass

    remaining_pairs = [p for p in pairs if (p[0], p[1]) not in processed]
    if not remaining_pairs:
        print("All selected image pairs are already processed! Generating summary and charts...")
        generate_summary_and_charts(csv_path, output_dir)
        return

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    adapters = load_adapters()

    # Warmup GPU on a small dummy crop so first image timing is accurate
    print("Warming up GPU models...")
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    for adapter in adapters.values():
        _ = adapter.deblur(dummy)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    fieldnames = [
        "scene",
        "filename",
        "Blurred_PSNR",
        "Blurred_SSIM",
        "DeblurGANv2_PSNR",
        "DeblurGANv2_SSIM",
        "DeblurGANv2_Time_s",
        "NAFNet_PSNR",
        "NAFNet_SSIM",
        "NAFNet_Time_s",
        "Restormer_PSNR",
        "Restormer_SSIM",
        "Restormer_Time_s",
    ]

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    sample_saved = (output_dir / "visual_comparison_sample.png").exists()

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        for scene_name, filename, blur_path, sharp_path in tqdm(remaining_pairs, desc="Benchmarking"):
            blur_bgr = cv2.imread(str(blur_path))
            sharp_bgr = cv2.imread(str(sharp_path))
            if blur_bgr is None or sharp_bgr is None:
                continue

            blur_rgb = cv2.cvtColor(blur_bgr, cv2.COLOR_BGR2RGB)
            sharp_rgb = cv2.cvtColor(sharp_bgr, cv2.COLOR_BGR2RGB)

            metrics, restored_images = evaluate_pair(blur_rgb, sharp_rgb, adapters)

            if not sample_saved:
                save_sample_comparison(
                    blur_rgb, sharp_rgb, restored_images, metrics, scene_name, filename, output_dir
                )
                sample_saved = True

            row = {"scene": scene_name, "filename": filename, **metrics}
            writer.writerow(row)
            f.flush()

    generate_summary_and_charts(csv_path, output_dir)


if __name__ == "__main__":
    main()
