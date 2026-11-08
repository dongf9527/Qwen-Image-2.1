#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen-Image-2.1-Turbo 性能实测脚本（对应调研计划横切项 Q1/Q2）

在部署机（RTX 5080）上、激活 diffusers 环境后运行：
    python tools/bench_qwen_image.py --mode turbo --res 1024
    python tools/bench_qwen_image.py --mode turbo --res 2048
    python tools/bench_qwen_image.py --mode base  --res 1024   # 可选：Turbo vs Base 对照（Q2）

行为：预热 1 次（不计入统计）后正式计时 --runs 次，输出 JSON 结果到 stdout，
并写入 bench_result_<mode>_<res>.json（提交时随笔记一起入库或粘贴进笔记模板）。
"""
import argparse
import json
import platform
import subprocess
import time


def gpu_snapshot():
    """nvidia-smi 进程级显存快照（含 CUDA 上下文开销）。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "nvidia-smi 不可用"


def main():
    p = argparse.ArgumentParser(description="Qwen-Image-2.1 性能实测")
    p.add_argument("--mode", choices=["turbo", "base"], default="turbo",
                   help="turbo=Qwen-Image-2.1-Turbo(8步) / base=Qwen-Image-2.1(40步)")
    p.add_argument("--res", type=int, default=1024, help="正方形边长（32 的倍数）")
    p.add_argument("--runs", type=int, default=3, help="正式计时次数（另有 1 次预热）")
    p.add_argument("--prompt", default=(
        "A red panda sitting on a wooden bench in a quiet garden, "
        "soft morning light, detailed fur, natural photography."))
    args = p.parse_args()

    import torch
    from diffusers import QwenImage21Pipeline

    if not torch.cuda.is_available():
        raise SystemExit("未检测到可用 CUDA 设备，请在部署机上运行。")

    model_id = "Qwen/Qwen-Image-2.1-Turbo" if args.mode == "turbo" else "Qwen/Qwen-Image-2.1"

    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(model_id, dtype=torch.bfloat16)
    load_seconds = time.time() - t0
    pipe.enable_model_cpu_offload()  # 16GB 显存基线模式（对比 ComfyUI INT8 常驻请用 ComfyUI 实测）

    torch.cuda.reset_peak_memory_stats()

    # 预热 1 次（含首次 CUDA 内核编译），不计入统计
    pipe(prompt=args.prompt, width=args.res, height=args.res, use_kv_cache=True,
         generator=torch.Generator("cpu").manual_seed(42))

    run_seconds = []
    for i in range(args.runs):
        t1 = time.time()
        img = pipe(prompt=args.prompt, width=args.res, height=args.res, use_kv_cache=True,
                   generator=torch.Generator("cpu").manual_seed(42)).images[0]
        run_seconds.append(round(time.time() - t1, 2))
        img.save(f"bench_{args.mode}_{args.res}_run{i + 1}.png")

    result = {
        "model": model_id,
        "resolution": f"{args.res}x{args.res}",
        "engine": "diffusers + enable_model_cpu_offload",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "os": platform.platform(),
        "load_seconds": round(load_seconds, 2),
        "warmup": "1 次预热不计入",
        "run_seconds": run_seconds,
        "avg_seconds": round(sum(run_seconds) / len(run_seconds), 2),
        "vram_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "vram_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
        "nvidia_smi_at_end": gpu_snapshot(),
    }
    out_file = f"bench_result_{args.mode}_{args.res}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n结果已写入 {out_file}")


if __name__ == "__main__":
    main()
