 #!/usr/bin/env python
"""
Export an MMDetection detector to ONNX.

Example
-------
python tools/export_onnx.py \
       configs/my_xray_hybrid.py \
       work_dirs/my_xray_hybrid/latest.pth \
       --output models/xray_detector.onnx \
       --device cuda \
       --opset 11 \
       --height 800 --width 800 \
       --simplify

Notes
-----
* Post–processing (NMS, score-threshold) is **NOT** included; the exported graph
  contains only the backbone / neck / head forward pass, which is the typical
  starting point for deployment toolchains such as TensorRT or ONNX Runtime.
* For a fully-featured export (with NMS), consider MMDeploy.
"""

import argparse
from pathlib import Path
import warnings

import torch
from mmdet.apis import init_detector

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MMDetection model to ONNX")
    parser.add_argument("config", help="MMDetection config file")
    parser.add_argument("checkpoint", help="Model checkpoint file")
    parser.add_argument(
        "-o", "--output", default="model.onnx", help="Output ONNX filename"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for model export",
    )
    parser.add_argument(
        "--opset", type=int, default=11, help="ONNX opset version (>=11 recommended)"
    )
    parser.add_argument(
        "--height", type=int, default=800, help="Input image height (pixels)"
    )
    parser.add_argument(
        "--width", type=int, default=800, help="Input image width (pixels)"
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="If set, try to run onnx-simplifier after export",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a single forward pass on the exported model for sanity check",
    )
    return parser.parse_args()


def export_onnx(args: argparse.Namespace) -> None:
    cfg_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = init_detector(str(cfg_path), str(ckpt_path), device=device)
    model.eval()

    # dummy input
    dummy = torch.randn(1, 3, args.height, args.width, device=device)

    print(f"[INFO] Exporting to {out_path} (opset={args.opset}) ...")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["dets", "labels"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "dets": {0: "batch"},
            "labels": {0: "batch"},
        },
    )
    print("[INFO] Raw ONNX graph saved.")

    # optional: simplify graph
    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify

            print("[INFO] Simplifying ONNX graph ...")
            model_onnx = onnx.load(str(out_path))
            model_simp, ok = simplify(model_onnx)
            if not ok:
                warnings.warn("ONNX simplification failed; using raw graph.")
            else:
                onnx.save(model_simp, str(out_path))
                print("[INFO] Simplified ONNX graph saved.")
        except ImportError:
            warnings.warn("onnx-simplifier not installed; skipping simplification.")

    # optional: quick verification
    if args.verify:
        try:
            import onnxruntime as ort

            print("[INFO] Verifying ONNX output via ONNX Runtime ...")
            sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
            ort_inputs = {"input": dummy.cpu().numpy()}
            _ = sess.run(None, ort_inputs)
            print("[INFO] Verification succeeded.")
        except Exception as e:
            warnings.warn(f"Verification failed: {e}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    cli_args = parse_args()
    export_onnx(cli_args)
    print("[DONE] Export procedure finished.")
