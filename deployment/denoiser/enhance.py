import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import os
import sys
import torch
import torchaudio
from torch.utils.data import DataLoader
from demucs import DemucsStreamer,DemucsStreamer_RT

# this version is when running from command line
from audio import Audioset, find_audio_files
from pretrained import add_model_flags, get_model
from utils import LogProgress


import soundfile as sf
# this version is for running from within python
# from audio import Audioset, find_audio_files
# import pretrained
# from utils import LogProgress

import pathlib
FILE_PATH = str(pathlib.Path(__file__).parent.absolute())

logger = logging.getLogger(__name__)

# adding new section for args so we can run within fastapi and not need command line args
# Then also replaced all of the instances of args with the variable that was getting pulled from it
model_path = FILE_PATH + '/best.th'
noisy_dir = FILE_PATH
# new option for direct file version we are testing

file_location = None

out_dir = FILE_PATH + '/static/'

noisy_json = None
sample_rate = 16000
batch_size = 1
device = 'cpu'
num_workers = 10
dns48 = False
dns64 = False
master64 = False
dry = 0
streaming = True
verbose = 20


def add_flags(parser):
    """
    Add the flags for the argument parser that are related to model loading and evaluation"
    """
    add_model_flags(parser)
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--dry', type=float, default=0,
                        help='dry/wet knob coefficient. 0 is only input signal, 1 only denoised.')
    parser.add_argument('--sample_rate', default=16000, type=int, help='sample rate')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--streaming', action="store_true",
                        help="true streaming evaluation for Demucs")


parser = argparse.ArgumentParser(
        'denoiser.enhance',
        description="Speech enhancement using Demucs - Generate enhanced files")
add_flags(parser)
parser.add_argument("--out_dir", type=str, default="/Users/donkeyddddd/Documents/Rx_projects/git_projects/pulseaudio_speech_enhancement/deployment/wav",
                    help="directory putting enhanced wav files")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="more loggging")
# new arg for file location
parser.add_argument('-f', '--file_location', type=str, default="/Users/donkeyddddd/Documents/Rx_projects/git_projects/pulseaudio_speech_enhancement/deployment/wav/input_noise_car_talk.wav", help="file path for the .wav file")

group = parser.add_mutually_exclusive_group()
group.add_argument("--noisy_dir", type=str, default=None,
                   help="directory including noisy wav files")
group.add_argument("--noisy_json", type=str, default=None,
                   help="json file including noisy wav files")


def get_estimate(model, noisy, dry):
    torch.set_num_threads(1)
    with torch.no_grad():
        estimate = model(noisy)
        estimate = (1 - dry) * estimate + dry * noisy
    return estimate


def save_wavs(estimates, noisy_sigs, filenames, out_dir, sr=16_000):
    # Write result
    for estimate, noisy, filename in zip(estimates, noisy_sigs, filenames):
        filename = os.path.join(out_dir, os.path.basename(filename).rsplit(".", 1)[0])
        write(noisy, filename + "-noisy.wav", sr=sr)
        write(estimate, filename + "-enhanced.wav", sr=sr)


def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def get_dataset(noisy_dir):
    files = find_audio_files(noisy_dir)
    return Audioset(files, with_path=True, sample_rate=sample_rate)


def get_dataset_fast_api_version(file_location):
    files = []
    siginfo, _ = torchaudio.info(file_location)
    length = siginfo.length // siginfo.channels
    files.append((file_location, length))
    return Audioset(files, with_path=True, sample_rate=sample_rate)


# with file_location
# def enhance(model_path, noisy_dir, file_location, out_dir, noisy_json, sample_rate,
#             batch_size, device, num_workers, dns48, dns64, master64,
#             dry, streaming, verbose,
#             model=None, local_out_dir=None):
#     # Load model
#     if not model:
#         # Relies on get_model to load from path
#         # model = pretrained.get_model(args).to(device)
#         model = get_model(model_path).to(device)
#     model.eval()
#     if local_out_dir:
#         # Uses local_out_dir on call to enhance function
#         out_dir = local_out_dir
#     else:
#         out_dir = out_dir

#     # Fast API version using a file rather than a noisy_dir
#     dset = get_dataset_fast_api_version(file_location)

#     loader = DataLoader(dset, batch_size=1)
    
#     os.makedirs(out_dir, exist_ok=True)

#     with ProcessPoolExecutor(num_workers) as pool:
#         iterator = LogProgress(logger, loader, name="Generate enhanced files")
#         for data in iterator:
#             # Get batch data
#             noisy_signals, filenames = data
#             noisy_signals = noisy_signals.to(device)
            
#             # Forward
#             estimate = get_estimate(model, noisy_signals, dry)
#             save_wavs(estimate, noisy_signals, filenames, out_dir, sr=sample_rate)

def enhance(model_path, noisy_dir, file_location, out_dir, noisy_json, sample_rate,
            batch_size, device, num_workers, dns48, dns64, master64,
            dry, streaming, verbose,
            model=None, local_out_dir=None):
    # Load model
    if not model:
        # Relies on get_model to load from path
        # model = pretrained.get_model(args).to(device)
        model = get_model(model_path).to(device)
        streamer = DemucsStreamer(model, num_frames=1)
    model.eval()
    if local_out_dir:
        # Uses local_out_dir on call to enhance function
        out_dir = local_out_dir
    else:
        out_dir = out_dir

    # Fast API version using a file rather than a noisy_dir
    dset = get_dataset_fast_api_version(file_location)

    loader = DataLoader(dset, batch_size=1)
    
    os.makedirs(out_dir, exist_ok=True)

    with ProcessPoolExecutor(num_workers) as pool:
        iterator = LogProgress(logger, loader, name="Generate enhanced files")
        for data in iterator:
            # Get batch data
            noisy_signals, filenames = data
            noisy_signals = noisy_signals.to(device)
            res_rt = streamer.feed(noisy_signals.squeeze(0))
            
            # Forward
            # estimate = get_estimate(model, noisy_signals, dry)
            # save_wavs(estimate, noisy_signals, filenames, out_dir, sr=sample_rate)
            save_wavs(res_rt, noisy_signals, filenames, out_dir, sr=sample_rate)


class my_rt_enhance():
    def __init__(self):
        self.model = get_model(model_path).to(device)
        self.model.eval()
        self.streamer = DemucsStreamer(self.model, num_frames=1)

    def processBlock(self, audio):
        assert(audio.__len__()[1] == 480)






def ut():
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=verbose)
    logger.debug(args)

    # with file_location from args
    enhance(model_path, noisy_dir, args.file_location, out_dir, noisy_json, sample_rate,
            batch_size, device, num_workers, dns48, dns64, master64,
            dry, streaming, verbose, local_out_dir=out_dir)


def ut_my():
    import numpy as np
    import matplotlib.pyplot as plt

    FILE_WAV = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/pulseaudio_speech_enhancement/deployment/wav/input_noise_car_talk.wav"
    MODEL_PATH = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/pulseaudio_speech_enhancement/deployment/denoiser/best.th"
    FRAME_SIZE = 480

    # load wav
    dset = get_dataset_fast_api_version(FILE_WAV)
    loader = DataLoader(dset, batch_size=1)
    iterator = LogProgress(logger, loader, name="Generate enhanced files")
    for data in iterator:
        # Get all wav data
        audio, filenames = data#[1,160000]
    audio = audio.squeeze(0).to('cpu')
    audio_frame_num = audio.shape[1] // FRAME_SIZE

    # output_buffer
    output = np.zeros(audio_frame_num*FRAME_SIZE)

    # model
    model = get_model(MODEL_PATH).to(device)
    streamer = DemucsStreamer_RT(model)

    
    # loop processBlock for realtime stream mode
    for idx in range(audio_frame_num):
        # push tmp_in [1,480] data into EFX model
        # EFX.pushNewInputFrame(audio[:,idx*FRAME_SIZE:(idx+1)*FRAME_SIZE]) 
        output[idx*FRAME_SIZE:(idx+1)*FRAME_SIZE] = streamer.processBlock(audio[:,idx*FRAME_SIZE:(idx+1)*FRAME_SIZE])
        # get tmp_out [1,480] data from EFX model
        # output[idx*FRAME_SIZE:(idx+1)*FRAME_SIZE] = EFX.getNewOutputFrame()

    
    
    
    # save output
    sf.write("/Users/donkeyddddd/Documents/Rx_projects/git_projects/pulseaudio_speech_enhancement/deployment/wav/input_noise_car_talk_res_overlap256.wav",output,16000)
    

    plt.figure(1)
    plt.plot(output)
    plt.show()


    xxx = 1



# version with command line args
if __name__ == "__main__":
    # ut() #ok
    # shell = my_rt_enhance()
    ut_my()

