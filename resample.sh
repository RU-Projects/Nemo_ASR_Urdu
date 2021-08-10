#!/usr/bin/env bash
## 1 and 2. fie format conversion and resampling
for FILE in /home/hood/KK/MediaAnalysis/ASRdatasetDL/ASR_DL/dl_old/test_data/*; do
    filename="$(basename $FILE)"
    filename="${filename%.*}"
    echo $filename
    ffmpeg -i $FILE -ar 16000 -ac 1 "/home/hood/KK/MediaAnalysis/ASRdatasetDL/ASR_DL/dl_old/test_data/test_data_resample/$filename.wav"
done
## 3. print stats for review
for FILE in //home/hood/KK/MediaAnalysis/ASRdatasetDL/ASR_DL/dl_old/test_data/*; do
    echo "---------------------------------------------------"
    echo "NEW FILE"
    echo $FILE
    sox --i $FILE
done
#2nd solution
#audioconvert convert given_data test_dir --output-format .wav
# then downsample
