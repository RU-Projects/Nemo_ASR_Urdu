import argparse
import fnmatch
import functools
import json
import logging
import multiprocessing
import os
import subprocess
import glob

from sox import Transformer
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Urdu Data')
parser.add_argument("--data_root", required=True, default=None, type=str)
parser.add_argument("--num_workers", default=4, type=int)
args = parser.parse_args()


def __process_transcript(file_path: str, dst_folder: str):
    """
    Converts flac files to wav from a given transcript, capturing the metadata.
    Args:
        file_path: path to a source transcript  with flac sources
        dst_folder: path where wav files will be stored
    Returns:
        a list of metadata entries for processed files.
    """
    entries = []
    root = os.path.dirname(file_path)
    with open(file_path, 'r', encoding="utf-8") as fin:
        for line in fin:
            id, text = line[: line.index(" ")], line[line.index(" ") + 1 :]
            transcript_text = text.strip()

            # Convert FLAC file to WAV
            flac_file = os.path.join(root, id + ".flac")
            wav_file = os.path.join(dst_folder, id + ".wav")
            if not os.path.exists(wav_file):
                Transformer().build(flac_file, wav_file)
            # check duration
            duration = subprocess.check_output("soxi -D {0}".format(wav_file), shell=True)

            entry = {}
            entry['audio_filepath'] = os.path.abspath(wav_file)
            entry['duration'] = float(duration)
            entry['text'] = transcript_text
            entries.append(entry)
    return entries


def __process_data(data_folder: str, dst_folder: str, manifest_file: str, num_workers: int):
    """
    Converts flac to wav and build manifests's json
    Args:
        data_folder: source with flac files
        dst_folder: where wav files will be stored
        manifest_file: where to store manifest
        num_workers: number of parallel workers processing files
    Returns:
    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    files = []
    entries = []

    for root, dirnames, filenames in os.walk(data_folder):
        for filename in fnmatch.filter(filenames, '*.txt'):
            files.append(os.path.join(root, filename))

    with multiprocessing.Pool(num_workers) as p:
        processing_func = functools.partial(__process_transcript, dst_folder=dst_folder)
        results = p.imap(processing_func, files)
        for result in tqdm(results, total=len(files)):
            entries.extend(result)

    with open(manifest_file, 'w') as fout:
        for m in entries:
            fout.write(json.dumps(m) + '\n')


def main():
    data_root = args.data_root
    num_workers = args.num_workers

    filepath = glob.glob(os.path.join(data_root), recursive=True)
    __process_data(
        os.path.join(data_root),
        os.path.join(data_root, "Urdu") + "-processed",
        os.path.join(data_root + ".json"),
        num_workers=num_workers,
        )
    logging.info('Done!')

if __name__ == "__main__":
    main()
