import os
import torch

_MIDAS_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "intel-isl_MiDaS_master")

_midas = None
_transform = None
downsampling = 1


def _ensure_midas():
    global _midas, _transform
    if _midas is not None:
        return _midas, _transform
    midas = torch.hub.load(_MIDAS_CACHE, "DPT_Hybrid", source="local", trust_repo=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    for param in midas.parameters():
        param.requires_grad = False
    midas_transforms = torch.hub.load(_MIDAS_CACHE, "transforms", source="local", trust_repo=True)
    _midas = midas
    _transform = midas_transforms.dpt_transform
    return _midas, _transform


# Backwards-compatible aliases (resolved on first estimate_depth call)


def estimate_depth(img, mode="test"):
    midas, _ = _ensure_midas()
    h, w = img.shape[1:3]
    norm_img = (img[None] - 0.5) / 0.5
    norm_img = torch.nn.functional.interpolate(
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)

    if mode == "test":
        with torch.no_grad():
            prediction = midas(norm_img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h // downsampling, w // downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    else:
        prediction = midas(norm_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h // downsampling, w // downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction
