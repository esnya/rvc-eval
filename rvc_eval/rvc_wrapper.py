# Original Code From: w-okada/voice-changer
# Modified by: esnya

import os
import sys
from dataclasses import asdict, dataclass, field
from typing import TypedDict

import numpy as np
import resampy
import torch
from fairseq import checkpoint_utils

sys.path.append(os.path.join(os.path.dirname(__file__), "../rvc/"))

from rvc.infer_pack.models import SynthesizerTrnMs256NSFsid
from rvc_eval.vc_infer_pipeline import VC


@dataclass
class RVCSettings:
    gpu: int = 0
    dstId: int = 0

    f0Detector: str = "dio"  # dio or harvest
    tran: int = 20
    noiceScale: float = 0.3
    predictF0: int = 0  # 0:False, 1:True
    silentThreshold: float = 0.00001
    extraConvertSize: int = 1024 * 32
    clusterInferRatio: float = 0.1

    model_file: str = ""
    config_file: str = ""

    indexRatio: float = 0
    rvcQuality: int = 0
    modelSamplingRate: int = 48000

    speakers: dict[str, int] = field(default_factory=lambda: {})

    # ↓mutableな物だけ列挙
    intData = [
        "gpu",
        "dstId",
        "tran",
        "predictF0",
        "extraConvertSize",
        "rvcQuality",
        "modelSamplingRate",
    ]
    floatData = ["noiceScale", "silentThreshold", "indexRatio"]
    strData = ["framework", "f0Detector"]


class RVCParams(TypedDict):
    hubert: str


class RVCWrapper:
    def __init__(self, params: RVCParams):
        self.settings = RVCSettings()
        self.net_g = None

        self.gpu_num = torch.cuda.device_count()
        self.prevVol = 0
        self.params = params
        print("RVC initialization: ", params)

    def loadModel(
        self,
        config: str,
        model_file: str,
        feature_file: str | None = None,
        index_file: str | None = None,
        is_half: bool = True,
    ):
        self.settings.config_file = config
        self.feature_file = feature_file
        self.index_file = index_file
        self.is_half = is_half

        try:
            hubert_path = self.params["hubert"]
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                [hubert_path],
                suffix="",
            )
            model = models[0]
            model.eval()
            if self.is_half:
                model = model.half()
            self.hubert_model = model

        except Exception as e:
            print("EXCEPTION during loading hubert model", e)

        self.settings.model_file = model_file

        # PyTorchモデル生成
        if model_file is not None:
            cpt = torch.load(model_file, map_location="cpu")
            self.settings.modelSamplingRate = cpt["config"][-1]
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=self.is_half)
            net_g.eval()
            net_g.load_state_dict(cpt["weight"], strict=False)
            if self.is_half:
                net_g = net_g.half()
            self.net_g = net_g

        return self.get_info()

    def get_info(self):
        data = asdict(self.settings)

        files = ["config_file", "model_file"]
        for f in files:
            if data[f] is not None and os.path.exists(data[f]):
                data[f] = os.path.basename(data[f])
            else:
                data[f] = ""

        return data

    def get_processing_sampling_rate(self):
        return self.settings.modelSamplingRate

    def generate_input(
        self,
        newData: np.ndarray,
        inputSize: int,
        crossfadeSize: int,
        solaSearchFrame: int = 0,
    ):
        newData = newData.astype(np.float32) / 32768.0

        if hasattr(self, "audio_buffer"):
            self.audio_buffer = np.concatenate(
                [self.audio_buffer, newData], 0
            )  # 過去のデータに連結
        else:
            self.audio_buffer = newData

        convertSize = (
            inputSize + crossfadeSize + solaSearchFrame + self.settings.extraConvertSize
        )

        if convertSize % 128 != 0:  # モデルの出力のホップサイズで切り捨てが発生するので補う。
            convertSize = convertSize + (128 - (convertSize % 128))

        self.audio_buffer = self.audio_buffer[-1 * convertSize :]  # 変換対象の部分だけ抽出

        crop = self.audio_buffer[
            -1 * (inputSize + crossfadeSize) : -1 * (crossfadeSize)
        ]  # 出力部分だけ切り出して音量を確認。(solaとの関係性について、現状は無考慮)
        rms = np.sqrt(np.square(crop).mean(axis=0))
        vol = max(rms, self.prevVol * 0.0)
        self.prevVol = vol

        return (self.audio_buffer, convertSize, vol)

    def inference(self, audio: np.ndarray, convertSize: int, vol: float):
        if not hasattr(self, "net_g") or self.net_g is None:
            print("[Voice Changer] No pyTorch session.")
            return np.zeros(1).astype(np.int16)

        if self.settings.gpu < 0 or self.gpu_num == 0:
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda", index=self.settings.gpu)

        self.hubert_model = self.hubert_model.to(dev)
        self.net_g = self.net_g.to(dev)

        audio = resampy.resample(audio, self.settings.modelSamplingRate, 16000)

        if vol < self.settings.silentThreshold:
            return np.zeros(convertSize).astype(np.int16)

        with torch.no_grad():
            repeat = 3 if self.is_half else 1
            repeat *= self.settings.rvcQuality  # 0 or 3
            vc = VC(self.settings.modelSamplingRate, dev, self.is_half, repeat)
            sid = 0
            times = [0, 0, 0]
            f0_up_key = self.settings.tran
            f0_method = "pm" if self.settings.f0Detector == "dio" else "harvest"
            file_index = self.index_file if self.index_file is not None else ""
            file_big_npy = self.feature_file if self.feature_file is not None else ""
            index_rate = self.settings.indexRatio
            if_f0 = 1
            f0_file = None

            audio_out = vc.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                times,
                f0_up_key,
                f0_method,
                file_index,
                file_big_npy,
                index_rate,
                if_f0,
                f0_file=f0_file,
            )
            result = audio_out * np.sqrt(vol)

        return result

    def __del__(self):
        del self.net_g
