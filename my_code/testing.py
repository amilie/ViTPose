import torch
import torch_directml

def main():
    dml = torch_directml.device(0)
    print(f"torch version: {torch.__version__}")
    print(f"DirectML device count: {torch_directml.device_count()}")
    print(f"Device: {dml}")
    # Quick smoke test: move a tensor to the GPU and do math
    x = torch.randn(3, 3).to(dml)
    y = torch.randn(3, 3).to(dml)
    z = x @ y
    print(f"Tensor device: {z.device}")
    print(f"Matrix multiply result shape: {z.shape}")
    print("DirectML GPU is working!")
    print(f"Device size: {dml.__sizeof__()}")

if __name__ == "__main__":
    main()
