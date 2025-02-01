import subprocess
import torch

def install_torch_geometric():
    try:
        import torch_geometric
    except ModuleNotFoundError:
        print("installing torch-geometric...")
        # Get PyTorch version and CUDA version
        TORCH = torch.__version__.split('+')[0]
        CUDA = 'cu' + torch.version.cuda.replace('.','')
        
        # Base URL for PyTorch Geometric wheels
        BASE_URL = f"https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html"
        
        # Packages to install
        packages = [
            "torch-scatter",
            "torch-sparse",
            "torch-cluster",
            "torch-spline-conv",
            "torch-geometric"
        ]
        
        # Install each package
        for package in packages[:-1]:  # All packages except torch-geometric
            cmd = f"pip install {package} -f {BASE_URL}"
            subprocess.check_call(cmd.split())
        
        # Install torch-geometric separately as it doesn't need the custom URL
        subprocess.check_call(["pip", "install", "torch-geometric"])

    try:
        import pytorch_lightning as pl
    except ModuleNotFoundError: 
        print("installing pytorch_lightning...")
        cmd = 'pip install --quiet pytorch-lightning>=1.4'
        subprocess.check_call(cmd.split())

    
if __name__ == "__main__":
    TORCH = torch.__version__.split('+')[0]
    CUDA = 'cu' + torch.version.cuda.replace('.','')
    print(f'pytorch version:{TORCH}, cuda version: {CUDA}')
    install_torch_geometric()