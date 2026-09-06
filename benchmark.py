import time
import torch
import torchvision.models as models

COLAB_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"


def _cuda_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(device, tensor):
    """Forces synchronization between host and device to measure true latency."""
    if device.type != "cpu":
        torch.cuda.synchronize()


def bench_matmul(device, matrix_dim=4096, runs=10):
    a = torch.randn(matrix_dim, matrix_dim, device=device)
    b = torch.randn(matrix_dim, matrix_dim, device=device)

    # Warm-up iterations to load kernels
    for _ in range(3):
        c = torch.matmul(a, b)
        sync(device, c)

    start = time.perf_counter()
    for _ in range(runs):
        c = torch.matmul(a, b)
        sync(device, c)
    total_time = time.perf_counter() - start
    return (total_time / runs) * 1000


def bench_vision(device, batch_size=16, runs=10):
    model = models.resnet50(weights=None).to(device)
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    use_amp = device.type == "cuda"

    # Warm-up iterations
    with torch.no_grad():
        for _ in range(3):
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(x)
            else:
                out = model(x)
            sync(device, out)

        start = time.perf_counter()
        for _ in range(runs):
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(x)
            else:
                out = model(x)
            sync(device, out)
        total_time = time.perf_counter() - start

    avg_latency = (total_time / runs) * 1000
    images_per_sec = (batch_size * runs) / total_time
    return avg_latency, images_per_sec


def run_tests():
    cpu_device = torch.device("cpu")
    gpu_device = _cuda_device()
    gpu_name = torch.cuda.get_device_name(0) if gpu_device.type == "cuda" else "CPU-only (no CUDA)"

    print(f"torch {torch.__version__} | cuda={torch.version.cuda} | gpu={gpu_name}")
    print(f"{'Test':<28} | {'CPU':<15} | {f'CUDA ({gpu_name[:18]})':<22} | {'Speedup':<10}")
    print("-" * 83)

    # 1. Matrix Multiplication Benchmark
    cpu_mm = bench_matmul(cpu_device)
    if gpu_device.type == "cuda":
        gpu_mm = bench_matmul(gpu_device)
        mm_speedup = cpu_mm / gpu_mm
        print(f"{'4096x4096 MatMul (ms)':<28} | {cpu_mm:>10.2f} ms | {gpu_mm:>17.2f} ms | {mm_speedup:>7.2f}x")
    else:
        print(f"{'4096x4096 MatMul (ms)':<28} | {cpu_mm:>10.2f} ms | {'no CUDA':>17s} | {'-':>9s}")

    # 2. ResNet-50 Vision Benchmark (AMP on CUDA)
    cpu_res, cpu_fps = bench_vision(cpu_device)
    if gpu_device.type == "cuda":
        gpu_res, gpu_fps = bench_vision(gpu_device)
        res_speedup = cpu_res / gpu_res
        print(f"{'ResNet-50 Batch=16 (ms)':<28} | {cpu_res:>10.2f} ms | {gpu_res:>17.2f} ms | {res_speedup:>7.2f}x")
        print(f"{'ResNet-50 (images/sec)':<28} | {cpu_fps:>10.1f} img/s | {gpu_fps:>17.1f} img/s | {gpu_fps/cpu_fps:>7.2f}x")
    else:
        print(f"{'ResNet-50 Batch=16 (ms)':<28} | {cpu_res:>10.2f} ms | {'no CUDA':>17s} | {'-':>9s}")
        print(f"{'ResNet-50 (images/sec)':<28} | {cpu_fps:>10.1f} img/s | {'no CUDA':>17s} | {'-':>9s}")


if __name__ == "__main__":
    run_tests()
