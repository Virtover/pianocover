"""GPU self-test: run this first on a new machine (e.g. the RTX 5070).

  python scripts/check_env.py

Checks CUDA/sm_120 support, loads pretrained Transkun into the style-conditioned model,
runs one training step (forward + backward through the semi-CRF and FiLM hooks under bf16)
and one generation, and reports timings and peak VRAM. Uses random audio, so no data needed.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    print(f"torch {torch.__version__}, CUDA build {torch.version.cuda}")
    if not torch.cuda.is_available():
        sys.exit("CUDA not available: check the NVIDIA driver (>= 570 for RTX 50-series) and --gpus all")
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU {name}, compute capability {cap[0]}.{cap[1]}, {total:.1f} GiB")
    arch = f"sm_{cap[0]}{cap[1]}"
    if arch not in torch.cuda.get_arch_list():
        sys.exit(f"this PyTorch build has no kernels for {arch} ({torch.cuda.get_arch_list()}); "
                 "use a CUDA 12.8+ build of PyTorch 2.7+")
    torch.zeros(1, device=dev).add_(1)
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

    from pianocover import SAMPLE_RATE, events, style
    from pianocover.model import PianoCoverModel

    t0 = time.time()
    model = PianoCoverModel.from_pretrained_transkun().to(dev)
    print(f"model loaded in {time.time() - t0:.1f}s, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, "
          f"FiLM layers: {len(model.film)}")

    # one training step: 2 x 16 s random audio with a few target notes
    torch.cuda.reset_peak_memory_stats()
    model.train()
    audio = torch.randn(2, 16 * SAMPLE_RATE, 1, device=dev) * 0.05
    ev = np.array([[1.0, 2.0, 60, 80], [2.5, 3.0, 64, 70], [4.0, 6.0, -64, 127]])
    notes = [events.to_transkun_notes(ev) for _ in range(2)]
    st = torch.randn(2, style.DIM, device=dev)
    t0 = time.time()
    logp = model.log_prob(audio, notes, st)
    loss = -logp.sum() / 32.0
    loss.backward()
    torch.cuda.synchronize()
    film_grad = sum(float(p.grad.abs().sum()) for p in model.film.parameters() if p.grad is not None)
    print(f"train step: loss {float(loss):.2f}, {time.time() - t0:.2f}s, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB (batch 2 x 16 s)")
    if not np.isfinite(float(loss)):
        sys.exit("non-finite loss: try --no-amp in training")
    if film_grad == 0:
        sys.exit("FiLM layers received no gradient: style conditioning would not train")
    print("FiLM gradients OK")

    # generation on 30 s of audio
    model.eval()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        x = torch.randn(30 * SAMPLE_RATE, 1, device=dev) * 0.05
        t0 = time.time()
        out = model.generate(x, torch.zeros(style.DIM, device=dev))
        torch.cuda.synchronize()
    print(f"generate 30 s: {len(out)} events, {time.time() - t0:.2f}s, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
