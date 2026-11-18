import os
import subprocess
import sys
import urllib.request
from pathlib import Path
import gdown

SCRIPT_DIR = Path(__file__).resolve().parent

def torch_checkpoints_dir() -> Path:
    try:
        import torch.hub
        return Path(torch.hub.get_dir()) / "checkpoints"
    except ImportError:
        torch_home = os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")
        return Path(torch_home) / "hub" / "checkpoints"

def progress(block_num, block_size, total_size):
    if total_size > 0:
        done = min(block_num * block_size, total_size)
        pct = done * 100 / total_size
        sys.stdout.write(f"\r  {pct:5.1f}%  ({done / 1e6:.1f} / {total_size / 1e6:.1f} MB)")
        sys.stdout.flush()

def download_http(url: str, dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    print()
    tmp.replace(dest)

def download_gdrive(gdown, file_id: str, dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".part")
    result = gdown.download(f"https://drive.google.com/uc?id={file_id}", str(tmp), quiet=False)
    if result is None or not tmp.exists():
        raise RuntimeError("gdown nu a putut descarca fisierul (limita de descarcari sau link privat?)")
    tmp.replace(dest)

def main():
    ok, failed = [], []
    ckpt_dir = torch_checkpoints_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dest = ckpt_dir / "inceptionresnetv2-520b38e4.pth"
    print(f"[1/4] {dest.name} -> {ckpt_dir}")
    if dest.exists():
        print("Sarim peste, fisierul exista deja.")
        ok.append(dest.name)
    else:
        try:
            download_http(
                "https://huggingface.co/jscanvic/mirror/resolve/main/inceptionresnetv2-520b38e4.pth",
                dest,
            )
            ok.append(dest.name)
        except Exception as e:
            print(f"  EROARE: {e}")
            failed.append(dest.name)

    gdrive_files = [
        ("deblurganv2", "", "fpn_inception.h5", "1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR"),
        ("nafnet", "experiments/pretrained_models", "NAFNet-GoPro-width64.pth", "1S0PVRbyTakYY9a82kujgZLbMihfNBLfC"),
        ("Restormer", "Motion_Deblurring/pretrained_models", "motion_deblurring.pth", "1pwcOhDS5Erzk8yfAbu7pXTud606SB4-L"),
    ]

    for i, (root, sub, name, file_id) in enumerate(gdrive_files, start=2):
        if gdown is None:
            failed.append(name)
            continue
        root_dir = SCRIPT_DIR / root
        target_dir = root_dir / sub if sub else root_dir
        dest = target_dir / name
        print(f"[{i}/4] {name} -> {target_dir}")

        if not root_dir.is_dir():
            print(f"  EROARE: folderul '{root}' nu exista in {SCRIPT_DIR}")
            failed.append(name)
            continue
        if dest.exists():
            print("Sarim peste, fisierul exista deja.")
            ok.append(name)
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            download_gdrive(gdown, file_id, dest)
            ok.append(name)
        except Exception as e:
            print(f"  EROARE: {e}")
            failed.append(name)

    print("=== Gata ===")
    for n in ok:
        print(f"  OK     {n}")
    for n in failed:
        print(f"  ESUAT  {n}")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
