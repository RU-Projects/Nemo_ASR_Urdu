########################################################
# for wav files 
########################################################



import os
import glob
import argparse
import librosa
import unicodedata
import argparse
import json
import multiprocessing
import os

from sox import Transformer
from tqdm import tqdm


parser = argparse.ArgumentParser(prog="Setup Urdu Transcripts")

parser.add_argument("--dir", "-d", type=str, default=None, help="Directory of dataset")

parser.add_argument("output", type=str, default=None, help="The output .tsv transcript file path")

args = parser.parse_args()

assert args.dir and args.output

args.dir = args.dir
args.output = args.output

transcripts = []

text_files = glob.glob(os.path.join(args.dir, "**", "*.txt"), recursive=True)

for text_file in tqdm(text_files, desc="[Loading]"):
    print(text_file)
    current_dir = os.path.dirname(text_file)
    with open(text_file, "r", encoding="utf-8") as txt:
        lines = txt.read().splitlines()
    for line in lines:
        line = line.split(" ", maxsplit=1)
        audio_file = os.path.join(current_dir, line[0] + ".wav")
        print(audio_file)
        y, sr = librosa.load(audio_file, sr=None)
        duration = librosa.get_duration(y, sr)
        text = unicodedata.normalize("NFC", line[1].lower())
        
        entry = {}
        entry['audio_filepath'] = os.path.abspath(audio_file)
        entry['duration'] = float(duration)
        entry['text'] = text
        transcripts.append(entry)
       
with open(args.output, "w") as out:
    for line in tqdm(transcripts, desc="[Writing]"):
        out.write(json.dumps(line)+ '\n')
