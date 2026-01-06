#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess
from pathlib import Path
import yaml

def resolve_split_path(data_dict: dict, split: str) -> str:
    """
    Resolve train/val/test path from data.yaml.
    Supports:
      path: /root/dataset
      val: images/val
    or absolute paths.
    """
    root = data_dict.get("path", "")
    v = data_dict.get(split)
    if v is None:
        raise KeyError(f"'{split}' not found in data.yaml")

    def _resolve_one(p: str) -> Path:
        p = Path(p)
        if p.is_absolute():
            return p
        if root:
            return (Path(root) / p).resolve()
        return p.resolve()

    if isinstance(v, str):
        return str(_resolve_one(v))
    elif isinstance(v, (list, tuple)):
        # Some yolo datasets allow list paths; we run detect for each path
        return [str(_resolve_one(x)) for x in v]
    else:
        raise TypeError(f"Unsupported type for {split}: {type(v)}")


def run_detect(weights: str,
               source: str,
               img: int = 640,
               conf: float = 0.25,
               device: str = "0",
               project: str = "runs/detect",
               name: str = "head_tumor_val",
               save_txt: bool = True,
               save_conf: bool = True,
               exist_ok: bool = True):
    cmd = [
        sys.executable, "detect.py",
        "--weights", weights,
        "--conf", str(conf),
        "--img-size", str(img),
        "--source", source,
        "--project", project,
        "--name", name,
    ]
    if device:
        cmd += ["--device", device]
    if save_txt:
        cmd += ["--save-txt"]
    if save_conf:
        cmd += ["--save-conf"]
    if exist_ok:
        cmd += ["--exist-ok"]

    print("Running:\n", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)


def main():
    # =========================
    # 你在这里直接改参数即可
    # =========================
    DATA_YAML = "data/head_tumor_2d.yaml"
    WEIGHTS   = "/root/head_tumor_2D/head_tumor_2d/runs/train/headtumor_t1ce_flair_t2wi_LGG+GBM/weights/best.pt"          # 或者 runs/train/xxx/weights/best.pt
    SPLIT     = "val"               # "train" / "val" / "test"

    IMG_SIZE  = 512                 # 推理尺寸
    CONF_THRE = 0.5                # 置信度阈值
    DEVICE    = "0"                 # "0" / "0,1" / "cpu"

    PROJECT   = "runs/detect_LGG+GBM"
    RUN_NAME  = f"{Path(WEIGHTS).stem}_{SPLIT}"

    SAVE_TXT  = True                # 保存yolo格式预测txt
    SAVE_CONF = True                # txt里带conf
    EXIST_OK  = True                # 允许覆盖同名输出目录

    # =========================
    # 读取 data.yaml 并解析 split 路径
    # =========================
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data_dict = yaml.safe_load(f)

    src = resolve_split_path(data_dict, SPLIT)
    print(f"[INFO] Resolved {SPLIT} source:", src)

    # =========================
    # 批量推理（如果 src 是 list，就逐个跑）
    # =========================
    if isinstance(src, list):
        for i, s in enumerate(src):
            run_detect(
                weights=WEIGHTS,
                source=s,
                img=IMG_SIZE,
                conf=CONF_THRE,
                device=DEVICE,
                project=PROJECT,
                name=f"{RUN_NAME}_{i}",
                save_txt=SAVE_TXT,
                save_conf=SAVE_CONF,
                exist_ok=EXIST_OK
            )
    else:
        run_detect(
            weights=WEIGHTS,
            source=src,
            img=IMG_SIZE,
            conf=CONF_THRE,
            device=DEVICE,
            project=PROJECT,
            name=RUN_NAME,
            save_txt=SAVE_TXT,
            save_conf=SAVE_CONF,
            exist_ok=EXIST_OK
        )


if __name__ == "__main__":
    main()
