########################################################
# for flac to wav files 
########################################################


import os
import glob
import argparse
import librosa
import argparse
import json
import logging
import multiprocessing
import subprocess

from sox import Transformer
from tqdm import tqdm


parser = argparse.ArgumentParser(prog="Setup Urdu Transcripts")

parser.add_argument("--dir", "-d", type=str, default=None, help="Directory of dataset")

parser.add_argument("output", type=str, default=None, help="The output .tsv transcript file path")

args = parser.parse_args()

assert args.dir and args.output

args.dir = args.dir
args.output = args.output


entries = []
text_files = glob.glob(os.path.join(args.dir, "**", "*.txt"), recursive=True)

for text_file in tqdm(text_files, desc="[Loading]"):
    current_dir = os.path.dirname(text_file)
    with open(text_file, "r", encoding="utf-8") as txt:
        # lines = txt.read().splitlines()
        for line in txt:
            id, text = line[: line.index(" ")], line[line.index(" ") + 1 :]
            transcript_text = text.strip()

            # Convert FLAC file to WAV
            flac_file = os.path.join(current_dir, id + ".flac")
            wav_file = os.path.join(current_dir, id + ".wav")
            if not os.path.exists(wav_file):
                Transformer().build(flac_file, wav_file)
            # check duration
            duration = subprocess.check_output("soxi -D {0}".format(wav_file), shell=True)

            entry = {}
            entry['audio_filepath'] = os.path.abspath(wav_file)
            entry['duration'] = float(duration)
            entry['text'] = transcript_text
            entries.append(entry)

with open(args.output, "w") as out:
    for line in tqdm(entries, desc="[Writing]"):
        out.write(json.dumps(line)+ '\n')
