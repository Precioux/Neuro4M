import os, json, torch
from typing import Optional

def save_config(save_dir: str, args_dict: dict):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

def save_state(save_path: str, epoch: int, model, optimizer, scaler, scheduler):
    state = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(state, save_path)

def load_state(load_path: str, model, optimizer=None, scaler=None, scheduler=None):
    state = torch.load(load_path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    start_epoch = int(state.get("epoch", 0)) + 1
    return start_epoch
