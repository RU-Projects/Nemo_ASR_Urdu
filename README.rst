
.. _main-readme:
**Nemo ASR URDU**
===============


Key Features
------------

* Speech processing
    * `Automatic Speech recognition (ASR) <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/intro.html>`_: Jasper, QuartzNet, CitriNet, Conformer
    * `NGC collection of pre-trained speech processing models. <https://ngc.nvidia.com/catalog/collections/nvidia:nemo_asr>`_

Built for speed, NeMo can utilize NVIDIA's Tensor Cores and scale out training to multiple GPUs and multiple nodes.

Requirements
------------

1) Python 3.6, 3.7 or 3.8
2) Pytorch 1.8.1 or above
3) NVIDIA GPU for training

Documentation
-------------

.. |main| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=main
  :alt: Documentation Status
  :scale: 100%
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/

.. |v1.0.2| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=v1.0.2
  :alt: Documentation Status
  :scale: 100%
  :target: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/v1.0.2/

.. |stable| image:: https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable
  :alt: Documentation Status
  :scale: 100%
  :target:  https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/

+---------+-------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Version | Status      | Description                                                                                                                              |
+=========+=============+==========================================================================================================================================+
| Latest  | |main|      | `Documentation of the latest (i.e. main) branch. <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/>`_                  |
+---------+-------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Stable  | |stable|    | `Documentation of the stable (i.e. most recent release) branch. <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/>`_ |
+---------+-------------+------------------------------------------------------------------------------------------------------------------------------------------+

Tutorials
---------
A great way to start with NeMo is by checking `one of our tutorials <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html>`_.

Getting help with NeMo
----------------------
FAQ can be found on NeMo's `Discussions board <https://github.com/NVIDIA/NeMo/discussions>`_. You are welcome to ask questions or start discussions there.


Installation
------------

Pip
~~~
Use this installation mode if you want the latest released version.

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython
    pip install nemo_toolkit['all']

Pip from source
~~~~~~~~~~~~~~~
Use this installation mode if you want the a version from particular GitHub branch (e.g main).

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    pip install Cython
    python -m pip install git+https://github.com/NVIDIA/NeMo.git@{BRANCH}#egg=nemo_toolkit[all]


From source
~~~~~~~~~~~
Use this installation mode if you are contributing to NeMo.

.. code-block:: bash

    apt-get update && apt-get install -y libsndfile1 ffmpeg
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    ./reinstall.sh

RNNT
~~~~
Note that RNNT requires numba to be installed from conda.

.. code-block:: bash

  conda remove numba
  pip uninstall numba
  conda install -c numba numba

Docker containers:
~~~~~~~~~~~~~~~~~~

If you chose to work with main branch, we recommend using NVIDIA's PyTorch container version 21.05-py3 and then installing from GitHub.

.. code-block:: bash

    docker run --gpus all -it --rm -v <nemo_github_folder>:/NeMo --shm-size=8g \
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \
    stack=67108864 --device=/dev/snd nvcr.io/nvidia/pytorch:21.05-py3

ASR (Automatic Speech Recognition)
----------------------------------

Data Preparation:
~~~~~~~~~~~~~~~~

**Recommendated Data Format**

  * Sample Rate = 16 kHz audio
  * Channel = Mono
  
**Dataset Format**: A folder containing all audio_files ``(.wav)`` with a '.txt' text file in format audio_filename(without_extension) with its transcription e.g
  - A folder name(urdu_dataset) contains 4 files i.e 3-audio_files 1-transcription_file  
      * 001.wav
      * 002.wav
      * 003.wav
      * dataset.txt 
     **dataset.txt format**:
      
.. code-block:: bash

        001 پیارے ابو جان میں آپ کو لینے آئی ہوں ہم اپنے ملک جائیں گے
        002 سفر کا بندوبست کیا کیا 
        003 ذہن ساتھ نہیں دے رہا تھا کہ یہ کیا ہو رہا ہے


Create Manifest file
~~~~~~~~~~~~~~~~~~~~~~

Each line of the manifest should be in the following format:

.. code-block:: bash

  {"audio_filepath": "/path/to/audio.wav", "text": "the transcription of the utterance", "duration": 23.147}


The audio_filepath field should provide an absolute path to the .wav file corresponding to the utterance. The text field should contain the full transcript for the utterance, and the duration field should reflect the duration of the utterance in seconds.

.wav file:

.. code-block:: bash

    python scripts/dataset_processing/get_urdu_data.py [--dir data_dir] output

.flac to .wav file:

.. code-block:: bash

    python scripts/dataset_processing/get_2_urdu.py [--dir data_dir] output
    or
    python scripts/dataset_processing/process_urdu_data.py  [--data_root data_dir]

Create Vocab file/Labels
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python scripts/tokenizers/process_asr_text_tokenizer.py [--manifest manifest_file] [--data_root output_dir]
   
  
Dataset Configurations:
~~~~~~~~~~~~~~~~~~~~~~~

Make modifications to the configuration file [examples/asr/conf/model/config.yaml]

* lables: &labels [vocab array(generated from Step#2)]
* [Optional] model.train_ds.manifest_filepath: [path_to_train_manifest.json]
* [Optional] model.validation_ds.manifest_filepath: [path_to_valid_manifest.json]
* [Optional] model.test_ds.manifest_filepath: [path_to_test_manifest.json]
* num_classes: [len(vocab)]
* [If required (for CUDA OOM)] (decrease batch_size respectively)
    * model.train_ds.batch_size: [32]
    * [model.validation_ds.batch_size: [32]
    * model.test_ds.batch_size: [32]
    

Tokenization Configurations:
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Some models utilize sub-word encoding via an external tokenizer instead of explicitly defining their vocabulary.

For such models, a tokenizer section is added to the model config. ASR models currently support two types of custom tokenizers:.

  - bpe
  - wpe

In order to build custom tokenizers, refer to the ASR_with_Subword_Tokenization notebook available in the ASR tutorials directory.

The following example sets up a SentencePiece Tokenizer at a path specified by the user:

.. code-block:: bash

    model:
      ...
      tokenizer:
        dir: "<path to the directory that contains the custom tokenizer files>"
        type: "bpe"  # can be "bpe" or "wpe"


Training & Testing
~~~~~~~~~~~~~~~~~~~
        
Fine-tuning Configurations:
~~~~~~~~~~~~~~~~~~~~~~~~~~

All ASR scripts support easy fine-tuning by partially/fully loading the pretrained weights from a checkpoint into the currently instantiated model. Pre-trained weights can be provided in multiple ways -

- Providing a path to a NeMo model ``(via init_from_nemo_model)``

.. code-block:: bash

    python examples/asr/script_to_<script_name>.py \
        --config-path=<path to dir of configs> \
        --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        trainer.gpus=-1 \
        trainer.max_epochs=50 \
        +init_from_nemo_model="<path to .nemo model file>"

- Providing a name of a pretrained NeMo model (which will be downloaded via the cloud) ``(via init_from_pretrained_model)``

.. code-block:: bash

    python examples/asr/script_to_<script_name>.py \
        --config-path=<path to dir of configs> \
        --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        trainer.gpus=-1 \
        trainer.max_epochs=50 \
        +init_from_pretrained_model="<name of pretrained checkpoint>"


- Providing a path to a Pytorch Lightning checkpoint file ``(via init_from_ptl_ckpt)``

.. code-block:: bash

    python examples/asr/script_to_<script_name>.py \
        --config-path=<path to dir of configs> \
        --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        trainer.gpus=-1 \
        trainer.max_epochs=50 \
        +init_from_ptl_ckpt="<name of pytorch lightning checkpoint>"


Training from Scratch:
~~~~~~~~~~~~~~~~~~~~~

run jupyter notebook `"ASR_Urdu_Train_from_scratch" <https://github.com/kkiyani/Nemo_ASR_Urdu/blob/main/tutorials/asr/ASR_Urdu_Train_from_scratch.ipynb>`_

or

Basic run (on CPU for 50 epochs):

.. code-block:: bash
  
    python examples/asr/speech_to_text.py \
        # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        trainer.gpus=0 \
        trainer.max_epochs=50
        

Transfer Learning:
~~~~~~~~~~~~~~~~~~

run jupyter notebook `"ASR_Urdu_Transfer_Learning" <https://github.com/kkiyani/Nemo_ASR_Urdu/blob/main/tutorials/asr/ASR_Urdu_Transfer_Learning.ipynb>`_

or

Basic run (on CPU for 50 epochs):

.. code-block:: bash
  
    python examples/asr/speech_to_text.py \
        # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
        model.train_ds.manifest_filepath="<path to manifest file>" \
        model.validation_ds.manifest_filepath="<path to manifest file>" \
        trainer.gpus=0 \
        trainer.max_epochs=50
        +init_from_nemo_model="<path to .nemo model file>"
        +init_from_pretrained_model="<name of pretrained checkpoint>"

Transcribing/Inference:
~~~~~~~~~~~~~~~~~~~~~~

To perform inference and transcribe a sample of speech after loading the model, use the transcribe() method:


    model.transcribe(paths2audio_files=[list of audio files], batch_size=BATCH_SIZE, logprobs=False)
    
.. code-block:: bash

  python tutorials/asr/Offline_URDU_ASR.py \
    --asr_model=<path to .nemo model file>" \
    --audio_file="<audio file path>"
    
    
.. code-block:: bash

  python examples/asr/transcribe_speech.py \
    model_path="<path to .nemo model file>" \
    pretrained_name="<name of pretrained checkpoint>" \
    audio_dir="<path to directory with audio files>" \
    dataset_manifest="<path to dataset JSON manifest file (in NeMo format)>" \
    output_filename=""
    
 
.. code-block:: bash

  python examples/asr/transcribe_urdu_speech.py \
    model_path="<path to .nemo model file>"\
    pretrained_name="<name of pretrained checkpoint>" \
    audio_dir="<file path>" \
    dataset_manifest="<path to dataset JSON manifest file (in NeMo format)>" \


**The audio files should be 16KHz monochannel wav files.**

Calcualte WER:
~~~~~~~~~~~~~~

.. code-block:: bash

  python examples/asr/speech_to_text_infer.py \
    --asr_model=<path to .nemo model file>" \
    --dataset="<path to evaluation data(manifest_file)>" \
    --dont_normalize_text \
       

Results:
--------

Transfer Learning 
~~~~~~~~~~~~~~~~~

QuartzNet - quartznet15x5
~~~~~~~~~~~~~~~~~~~~~~~~~

**Encoder**: QuartzNet15x5Base-En

  [PretrainedModelInfo(pretrained_model_name=QuartzNet15x5Base-En,
    description=QuartzNet15x5 model trained on six datasets: LibriSpeech, Mozilla Common Voice (validated clips from en_1488h_2019-12-10), WSJ, Fisher, Switchboard, and NSC Singapore English. It was trained with Apex/Amp optimization level O1 for 600 epochs. The model achieves a WER of 3.79% on LibriSpeech dev-clean, and a WER of 10.05% on dev-other. Please visit https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels for further details.,
    location=https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-En.nemo)]
 
**Decoder**: SEECS_Old

Example YAML Config
~~~~~~~~~~~~~~~~~~~

Go to `"quartznet_15x5dr.yml" <https://github.com/kkiyani/Nemo_ASR_Urdu/blob/main/examples/asr/conf/quartznet/quartznet_15x5.yaml>`_

**Summary**

- Train on: RTX 2080 Ti
- Training hours per epoch: ~1.5 h0urs

**Pretrained Model**, go to `"drive" <https://drive.google.com/drive/folders/1Tdqbsn6UvkuRFqWV0ujBYitH8ONqq5j8?usp=sharing>`_

**Error Rates**

+-----------------+-------+----------------+-----------------+-------+--------------------+
| **Train-Data**  |  SP   | **Test-Data**  | Test batch size | Epoch |      WER           |
+=================+=======+================+=================+=======+====================+
|  _SEECS_OLD_    | 78133 |   _SEECS_new_  |        4        |  12   |   0.33 (greedy)    |
+-----------------+-------+----------------+-----------------+-------+--------------------+
