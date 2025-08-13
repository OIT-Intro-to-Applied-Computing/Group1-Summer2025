import torch, random, numpy as np, os

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"]=str(seed)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
