# RVC Eval Simplified

## Description

This project is a simplified implementation of the evaluation part of the [Retrieval-based Voice Conversion (RVC)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI) system. It aims to make it easy to use in any Python code by removing unnecessary components. As a sample, a command-line interface (CLI) is provided for real-time voice conversion.

## Features

- Simplified RVC evaluation
- Easy integration into any Python code
- Real-time voice conversion via CLI

## Project Name

RVC Eval Simplified (rvc-eval) is an appropriate name for this project as it highlights the simplified evaluation aspect of the original RVC system.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/esnya/rvc-eval.git
```

2. Initialize and update the RVC submodule in the `rvc` directory:
```bash
cd rvc-eval
git submodule update --init --recursive
```

3. Install dependencies using Pipenv:
```bash
pipenv install
```

4. Download the Hubert model (`hubert_base.pt`) from [Hugging Face](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) and place it in the `models` directory:

5. Activate the Pipenv environment:
```bash
pipenv shell
```


## Usage

1. To list available audio devices:
```bash
python cli.py --list-audio-devices
```

2. To run the voice conversion system (with the default `hubert_base.pt` model or specify a custom path with the `--hubert` option):
```
python cli.py -m path/to/your/model.pth --input-device-index 0 --output-device-index 1
```

## Dependencies and Requirements

- Python: Compatible version with PyTorch 2.0.0+cu118
- PyTorch: 2.0.0+cu118
- Pipenv is used for managing dependencies.

## Credits
- This project is based on the [Retrieval-based Voice Conversion](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI) system by liujing04.
- This project also refers to [Voice Changer](https://github.com/w-okada/voice-changer) by w-okada, and some parts of the code are based on it.

## License

This project is licensed under the MIT License, following the licenses of the [original RVC repository](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI) and the [Voice Changer repository](https://github.com/w-okada/voice-changer). See the [LICENSE](LICENSE) file for details.

