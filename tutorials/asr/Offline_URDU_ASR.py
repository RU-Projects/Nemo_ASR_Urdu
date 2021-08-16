#!/usr/bin/env python
# coding: utf-8

# # NeMo offline ASR

import os
import nemo.collections.asr as nemo_asr
from argparse import ArgumentParser
from nemo.utils import logging

parser = ArgumentParser()
parser.add_argument(
    "--asr_model", type=str, required=True, help="Pass: 'QuartzNet15x5Base-En'",
)
parser.add_argument("--audio_file", type=str, required=True, help="path to audio file")

args = parser.parse_args()


if args.asr_model.endswith('.nemo'):
    logging.info(f"Using local ASR model from {args.asr_model}")
    asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=args.asr_model)
else:
    logging.info(f"Using NGC cloud ASR model {args.asr_model}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.asr_model)

AUDIO_FILENAME = args.audio_file
# Convert our audio sample to text
files = [AUDIO_FILENAME]
transcript = asr_model.transcribe(paths2audio_files=files)[0]
print(f'Transcript: "{transcript}"')


