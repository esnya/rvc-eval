import os
import sys
from argparse import ArgumentParser
from logging import getLogger

import pyaudio
import torch

from rvc_eval.model import load_hubert, load_net_g

sys.path.append(os.path.join(os.path.dirname(__file__), "../rvc/"))

logger = getLogger(__name__)


def main(args):
    import numpy as np

    from rvc_eval.vc_infer_pipeline import VC

    is_half = not args.float and args.device != "cpu"
    device = torch.device(args.device)

    hubert_model = load_hubert(args.hubert, is_half, device)
    net_g, sampling_ratio = load_net_g(args.model, is_half, device)

    repeat = 3 if is_half else 1
    repeat *= args.quality  # 0 or 3
    sid = 0
    f0_up_key = args.f0_up_key
    f0_method = args.f0_method
    vc = VC(sampling_ratio, device, is_half, repeat)

    pa = pyaudio.PyAudio()
    logger.info(
        "input_device: %s",
        pa.get_device_info_by_index(args.input_device_index)
        if args.input_device_index is not None
        else "Default",
    )
    logger.info(
        "output_device: %s",
        pa.get_device_info_by_index(args.output_device_index)
        if args.output_device_index is not None
        else "Default",
    )

    input_frame_rate = 16000
    frames_per_buffer = input_frame_rate * args.buffer_size // 1000

    input_stream = pa.open(
        rate=input_frame_rate,
        channels=1,
        format=pyaudio.paFloat32,
        input=True,
        input_device_index=args.input_device_index,
        frames_per_buffer=frames_per_buffer,
    )
    output_stream = pa.open(
        rate=sampling_ratio,
        channels=1,
        format=pyaudio.paFloat32,
        output=True,
        output_device_index=args.output_device_index,
    )
    input_stream.start_stream()

    try:
        while input_stream.is_active():
            audio_input = np.frombuffer(
                input_stream.read(frames_per_buffer), dtype=np.float32
            )
            logger.debug(
                "audio_input: %s, %s, %s, %s",
                audio_input.shape,
                audio_input.dtype,
                np.min(audio_input).item(),
                np.max(audio_input).item(),
            )

            audio_output = (
                vc.pipeline(
                    hubert_model,
                    net_g,
                    sid,
                    audio_input,
                    f0_up_key,
                    f0_method,
                )
                .cpu()
                .float()
                .numpy()
            )

            logger.debug(
                "audio_output: %s, %s, %s, %s",
                audio_output.shape,
                audio_output.dtype,
                np.min(audio_output).item() if audio_output.size > 0 else None,
                np.max(audio_output).item() if audio_output.size > 0 else None,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if np.isnan(audio_output).any():
                continue

            output_stream.write(audio_output.tobytes())

    finally:
        output_stream.close()
        input_stream.close()
        pa.terminate()


def list_audio_devices():
    pa = pyaudio.PyAudio()
    for i in range(pa.get_device_count()):
        print(pa.get_device_info_by_index(i))
    pa.terminate()


parser = ArgumentParser()
parser.add_argument("-l", "--log-level", type=str, default="WARNING")
parser.add_argument(
    "-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("--hubert", type=str, default="models/hubert_base.pt")
parser.add_argument("--float", action="store_true")
parser.add_argument("-i", "--input-device-index", type=int, default=None)
parser.add_argument("-o", "--output-device-index", type=int, default=None)
parser.add_argument("-q", "--quality", type=int, default=1)
parser.add_argument("-k", "--f0-up-key", type=int, default=0)
parser.add_argument("--f0-method", type=str, default="pm", choices=("pm", "harvest"))
parser.add_argument(
    "--buffer-size", type=int, default=1000, help="buffering size in ms"
)
parser.add_argument("--list-audio-devices", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    if args.list_audio_devices:
        list_audio_devices()
    else:
        main(args)
