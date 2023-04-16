from argparse import ArgumentParser
from logging import getLogger

import pyaudio


def main(args):
    import numpy as np

    from rvc_eval.rvc_wrapper import RVCParams, RVCWrapper

    is_half = not args.float

    rvc_wrapper = RVCWrapper(params=RVCParams(hubert=args.hubert))
    print(rvc_wrapper.loadModel("", args.model, is_half=is_half))

    pa = pyaudio.PyAudio()
    logger.info(
        "input_device: %s",
        pa.get_device_info_by_index(args.input_device_index),
    )
    input_stream = pa.open(
        rate=48000,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        input_device_index=args.input_device_index,
        frames_per_buffer=48000,
    )
    output_stream = pa.open(
        rate=48000,
        channels=1,
        format=pyaudio.paInt16,
        output=True,
        output_device_index=args.output_device_index,
    )
    input_stream.start_stream()

    try:
        while input_stream.is_active():
            audio_input = (
                np.frombuffer(input_stream.read(48000), dtype=np.int16).astype(
                    np.float32
                )
                / 32768.0
            )
            input_volume = max(np.max(audio_input).item(), -np.min(audio_input).item())
            print(
                "audio_input",
                audio_input.shape,
                audio_input.dtype,
                input_volume,
            )

            audio_output = rvc_wrapper.inference(
                audio_input, audio_input.size, input_volume
            )
            print(
                "audio_output",
                audio_output.shape,
                audio_output.dtype,
                np.max(audio_output) - np.min(audio_output),
            )

            output_stream.write(audio_output.astype(np.int16).tobytes())

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
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("--hubert", type=str, default="models/hubert_base.pt")
parser.add_argument("--float", action="store_true")
parser.add_argument("-i", "--input-device-index", type=int, default=None)
parser.add_argument("-o", "--output-device-index", type=int, default=None)
parser.add_argument("--list-audio-devices", action="store_true")

logger = getLogger(__name__)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.list_audio_devices:
        list_audio_devices()
    else:
        main(args)
