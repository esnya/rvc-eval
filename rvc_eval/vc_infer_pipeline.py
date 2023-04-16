# Original Code From: w-okada, liujing04
# Modified by: esnya

import numpy as np
import parselmouth
import pyworld
import scipy.signal as signal
import torch
import torch.nn.functional as F

from rvc_eval.config import Config


class VC(object):
    def __init__(self, tgt_sr, device, is_half, x_pad):
        config = Config.get(is_half)
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.x_query = config.x_query
        self.x_pad = x_pad
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值
        self.device = device
        self.is_half = is_half

    def _pm(
        self,
        x: np.ndarray,
        p_len: int,
        time_step: float,
        f0_min: float,
        f0_max: float,
    ):
        f0 = (
            parselmouth.Sound(x, self.sr)
            .to_pitch_ac(
                time_step=time_step / 1000,
                voicing_threshold=0.6,
                pitch_floor=f0_min,
                pitch_ceiling=f0_max,
            )
            .selected_array["frequency"]
        )
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        return f0

    def _hervest(self, x: np.ndarray, f0_max: float):
        f0, t = pyworld.harvest(
            x.astype(np.double),
            fs=self.sr,
            f0_ceil=f0_max,
            frame_period=10,
        )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
        f0 = signal.medfilt(f0, 3)
        return f0

    def get_f0(self, x, p_len, f0_up_key, f0_method):
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0 = (
            self._pm(x, p_len, time_step, f0_min, f0_max)
            if f0_method == "pm"
            else self._hervest(x, f0_max)
        )
        f0 *= pow(2, f0_up_key / 12)

        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak  # 1-0

    @torch.no_grad()
    def vc(
        self,
        model,
        net_g,
        sid,
        audio0: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
    ):  # ,file_index,file_big_npy
        feats = audio0.half() if self.is_half else audio0.float()

        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False).to(self.device)

        logits = model.extract_features(
            source=feats.to(self.device),
            padding_mask=padding_mask,
            output_layer=9,
        )
        feats = model.final_proj(logits[0])

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        assert feats.shape[1] <= audio0.shape[0] // self.window
        p_len = feats.shape[1]

        audio1 = (
            net_g.infer(
                feats,
                torch.tensor([p_len], device=self.device, dtype=torch.long),
                pitch[:, :p_len],
                pitchf[:, :p_len],
                sid,
            )[0][0, 0]
        ).data

        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        f0_up_key,
        f0_method,
    ):
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")

        p_len = audio_pad.shape[0] // self.window

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = self.get_f0(audio_pad, p_len, f0_up_key, f0_method)
        pitch = torch.tensor(
            pitch[:p_len], device=self.device, dtype=torch.long
        ).unsqueeze(0)
        pitchf = torch.tensor(
            pitchf[:p_len],
            device=self.device,
            dtype=torch.float16 if self.is_half else torch.float32,
        ).unsqueeze(0)

        vc_output = self.vc(
            model,
            net_g,
            sid,
            torch.from_numpy(audio_pad),
            pitch,
            pitchf,
        )

        audio_output = (
            vc_output
            if self.t_pad_tgt == 0
            else vc_output[self.t_pad_tgt : -self.t_pad_tgt]
        )

        return audio_output
