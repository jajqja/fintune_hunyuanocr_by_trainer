import json
from pathlib import Path
from typing import Optional

def load_dataset(dataset_dir: str, prompt) -> list[dict]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    samples = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

    for label_file in ["labels.json", "labels.txt", "gt.txt", "annotation.txt"]:
        label_path = dataset_dir / label_file
        if label_path.exists():
            samples = _load_from_label_file(label_path, dataset_dir, image_exts, prompt)
            if samples:
                print(f"[Dataset] From: {dataset_dir}")
                print(f"\t- Found {len(samples)} samples from '{label_file}'")
                print(f"\t- Using prompt: {prompt}")
                return samples

    for img_path in sorted(dataset_dir.rglob("*")):
        if img_path.suffix.lower() not in image_exts:
            continue
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            gt = txt_path.read_text(encoding="utf-8").strip()
            samples.append({
                "image_path": str(img_path),
                "ground_truth": gt,
                "prompt": prompt,       
            })

    if not samples:
        raise ValueError(
            f"No valid samples found in '{dataset_dir}'.\n"
            "See README.md for instructions on preparing the dataset."
        )
    print(f"[Dataset] From: {dataset_dir}")
    print(f"\t- Found {len(samples)} samples (image+txt pairs)")
    print(f"\t- Using prompt: {prompt}")
    return samples


def _load_from_label_file(
    label_path: Path, 
    root: Path, 
    image_exts: set,
    prompt: Optional[str] = None,
) -> list[dict]:
    
    samples = []

    if label_path.suffix == ".json":
        with open(label_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            items = data.items()
        elif isinstance(data, list):
            items = [(d.get("file", d.get("filename", "")), d.get("text", d.get("label", "")))
                     for d in data]
        else:
            return []
        for fname, gt in items:
            img_path = _find_image(root, fname, image_exts)
            if img_path:
                samples.append({
                    "image_path": str(img_path),
                    "ground_truth": str(gt),
                    "prompt": prompt,
                })

    else:  
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    parts = line.split("\t", 1)
                elif "," in line and not line.startswith("/"):
                    parts = line.split(",", 1)  # csv
                else:
                    continue
                if len(parts) == 2:
                    fname, gt = parts
                    img_path = _find_image(root, fname.strip(), image_exts)
                    if img_path:
                        samples.append({
                            "image_path": str(img_path),
                            "ground_truth": gt.strip(),
                            "prompt": prompt,
                        })
    return samples


def _find_image(
    root: Path, 
    fname: str, 
    image_exts: set
) -> Optional[Path]:
    
    candidates = [
        root / fname,
        root / "images" / fname,
        root / Path(fname).name,
    ]
    if not Path(fname).suffix:
        for ext in image_exts:
            candidates.append(root / (fname + ext))
            candidates.append(root / "images" / (fname + ext))
    for p in candidates:
        if p.exists():
            return p
    return None
