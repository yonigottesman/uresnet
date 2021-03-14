import torch as torch


class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")