import torch


def load_hubert(model_path: str, is_half: bool, device: torch.device):
    from fairseq import checkpoint_utils

    [model], _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        suffix="",
    )
    model.eval()
    return model.to(device).half() if is_half else model.to(device).float()


def load_net_g(model_path: str, is_half: bool, device: torch.device):
    from rvc.infer_pack.models import SynthesizerTrnMs256NSFsid

    cpt = torch.load(model_path, map_location="cpu")
    sampling_rate = cpt["config"][-1]
    net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half).to(device)
    net_g.eval()
    net_g.load_state_dict(cpt["weight"], strict=False)
    return (net_g.half() if is_half else net_g.float(), sampling_rate)
