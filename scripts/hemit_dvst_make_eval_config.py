#!/usr/bin/env python3
"""Build D-VST eval.yaml listing all HEMIT test pairs for eval.py."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def collect_pairs(data_root: Path, split: str) -> list[tuple[Path, Path]]:
    inp_dir = data_root / split / "input"
    lab_dir = data_root / split / "label"
    if inp_dir.is_dir():
        pairs = []
        for pattern in ("*.tif", "*.tiff", "*.png", "*.TIF", "*.PNG"):
            for inp in sorted(inp_dir.glob(pattern)):
                lab = lab_dir / inp.name
                if lab.exists():
                    pairs.append((inp.resolve(), lab.resolve()))
        return pairs

    # D-VST folder layout: HE/slide/file, mIHC/slide/file
    he_root = data_root / "HE"
    mihc_root = data_root / "mIHC"
    pairs = []
    if he_root.is_dir():
        for slide in sorted(he_root.iterdir()):
            if not slide.is_dir():
                continue
            mdir = mihc_root / slide.name
            for inp in sorted(slide.iterdir()):
                lab = mdir / inp.name
                if lab.is_file():
                    pairs.append((inp.resolve(), lab.resolve()))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output", required=True)
    p.add_argument("--sample-size", type=int, default=512)
    p.add_argument("--video-length", type=int, default=1)
    p.add_argument("--reference-mode", choices=["paired_gt", "he_only"], default="paired_gt")
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--checkpoint", default=os.environ.get("DVST_EVAL_CKPT", ""))
    args = p.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    pairs = collect_pairs(data_root, args.split)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise SystemExit(f"No pairs under {data_root} split={args.split}")

    pose_images = []
    source_images = []
    target_images = []
    for he_path, mihc_path in pairs:
        pose_images.append(str(he_path))
        if args.reference_mode == "paired_gt":
            source_images.append(str(mihc_path))
        else:
            source_images.append(str(he_path))
        target_images.append(str(mihc_path))

    cfg = {
        "model_type": "dvst_modules",
        "motion_module": "",
        "pretrained_model_path": "./weights/dvst_pretrained",
        "pretrained_vae_path": "./weights/dvst_pretrained",
        "checkpoint_path": args.checkpoint or "./weights/dvst_pretrained/HE2mIHC.ckpt",
        "valid_seed": 42,
        "unet_additional_kwargs": {
            "cross_attention_dim": 1024,
            "unet_use_cross_frame_attention": False,
            "unet_use_temporal_attention": False,
            "use_motion_module": False,
            "motion_module_resolutions": [1, 2, 4, 8],
            "motion_module_mid_block": False,
            "motion_module_decoder_only": False,
            "motion_module_type": "Vanilla",
            "motion_module_kwargs": {
                "num_attention_heads": 8,
                "num_transformer_block": 1,
                "attention_block_types": ["Temporal_Self", "Temporal_Self"],
                "temporal_position_encoding": True,
                "temporal_position_encoding_max_len": 32,
                "temporal_attention_dim_div": 1,
            },
        },
        "noise_scheduler_kwargs": {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "steps_offset": 1,
            "clip_sample": False,
            "rescale_betas_zero_snr": True,
        },
        "infer_noise_scheduler_kwargs": {
            "algorithm_type": "dpmsolver++",
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "dynamic_thresholding_ratio": 0.995,
            "euler_at_final": False,
            "lower_order_final": True,
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "sample_max_value": 1.0,
            "solver_order": 2,
            "solver_type": "midpoint",
            "steps_offset": 0,
            "thresholding": False,
            "timestep_spacing": "linspace",
            "trained_betas": None,
            "use_karras_sigmas": False,
            "use_lu_lambdas": False,
            "variance_type": None,
        },
        "validation_data": {
            "pose_image": pose_images,
            "source_image": source_images,
            "target_image": target_images,
            "sample_size": [args.sample_size, args.sample_size],
            "video_length": args.video_length,
            "num_inference_steps": 25,
            "guidance_scale": [1.5, 1.5],
            "alpha": 1.0,
            "block_size": 32,
        },
        "context": {"context_frames": 4, "context_stride": 1, "context_overlap": 0},
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
