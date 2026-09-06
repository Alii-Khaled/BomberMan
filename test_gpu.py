import multiprocessing as mp
import sys
import time

import torch

GPU_PROBE_TIMEOUT_S = 30


def gpu_probe(queue) -> None:
    try:
        torch.cuda.init()
        x = torch.randn(8, device="cuda")
        torch.cuda.synchronize()
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        cap = f"{props.major}.{props.minor}"
        vram = props.total_memory / 1024**3
        queue.put(("ok", name, cap, vram))
    except Exception as exc:
        queue.put(("error", repr(exc), "", 0.0))


def bench(device: str) -> None:
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    if device == "cuda":
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
    else:
        for _ in range(3):
            a @ b
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            c = a @ b
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 10 / 1000
    else:
        start = time.perf_counter()
        for _ in range(10):
            c = a @ b
        elapsed = (time.perf_counter() - start) / 10
    tflops = 2 * 4096**3 / elapsed / 1e12
    print(f"matmul 4096^3 fp32 on {device}: {elapsed * 1000:.0f} ms/iter (~{tflops:.1f} TFLOPS)")
    del a, b, c


def training_step(device: str) -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 6),
    ).to(device)
    opt = torch.optim.Adam(model.parameters())
    x = torch.randn(128, 256, device=device)
    target = torch.randint(0, 6, (128,), device=device)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(x), target)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    else:
        loss = torch.nn.functional.cross_entropy(model(x), target)
        loss.backward()
        opt.step()
        opt.zero_grad()
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"autograd training step (AMP={use_amp}) on {device} OK (loss={loss.item():.4f})")


def main() -> int:
    mp.set_start_method("spawn")
    print(f"torch: {torch.__version__}")
    print(f"cuda runtime: {torch.version.cuda} (available={torch.cuda.is_available()})")

    gpu_usable = False
    if torch.cuda.is_available():
        queue: mp.Queue = mp.Queue()
        proc = mp.Process(target=gpu_probe, args=(queue,))
        proc.start()
        proc.join(timeout=GPU_PROBE_TIMEOUT_S)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            print("GPU: detected but compute HANGS on first kernel submission")
        else:
            result = queue.get() if not queue.empty() else ("error", "no result", "", 0.0)
            if result[0] == "ok":
                print(f"GPU: {result[1]} (sm_{result[2]}, {result[3]:.1f} GiB)")
                gpu_usable = True
            else:
                print(f"GPU: probe failed: {result[1]}")

    device = "cuda" if gpu_usable else "cpu"
    if not gpu_usable:
        print(f"falling back to CPU ({torch.get_num_threads()} threads)")
        print(f"cpu: {torch.get_num_threads()} threads")

    bench(device)
    training_step(device)
    print(f"PASS (active device: {device})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
