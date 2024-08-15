import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import logging
import os
import sys
import torchaudio
from torch.utils.data import DataLoader
from demucs import DemucsStreamer_RT

from audio import Audioset
from pretrained import add_model_flags, get_model
from utils import LogProgress

import soundfile as sf
from tqdm import *

logger = logging.getLogger(__name__)


def get_dataset_fast_api_version(file_location):
    files = []
    siginfo = torchaudio.info(file_location)
    length = siginfo.num_frames // siginfo.num_channels
    files.append((file_location, length))
    return Audioset(files, with_path=True, sample_rate=48000)



# def ut():
#     args = parser.parse_args()
#     logging.basicConfig(stream=sys.stderr, level=verbose)
#     logger.debug(args)

#     # with file_location from args
#     enhance(model_path, noisy_dir, args.file_location, out_dir, noisy_json, sample_rate,
#             batch_size, device, num_workers, dns48, dns64, master64,
#             dry, streaming, verbose, local_out_dir=out_dir)


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
        audio, filenames = data #audio:[1,160000] # audio samplerate is 48000
    audio = audio.squeeze(0).to('cpu') 
    audio_frame_num = audio.shape[1] // FRAME_SIZE

    # output_buffer
    output = np.zeros(audio_frame_num*FRAME_SIZE)

    # model
    model = get_model(MODEL_PATH).to('cpu')
    streamer = DemucsStreamer_RT(model)

    
    # loop processBlock for realtime stream mode
    for idx in tqdm(range(audio_frame_num)):
        output[idx*FRAME_SIZE:(idx+1)*FRAME_SIZE] = streamer.processBlock(audio[:,idx*FRAME_SIZE:(idx+1)*FRAME_SIZE])

    
    # save output
    sf.write("/Users/donkeyddddd/Documents/Rx_projects/git_projects/pulseaudio_speech_enhancement/deployment/wav/input_noise_car_talk_res_overlap256.wav",output,16000)
    

    # plt.figure(1)
    # plt.plot(output)
    # plt.show()

    print("process finish")

    xxx = 1



# version with command line args
if __name__ == "__main__":
    # ut() #ok
    # shell = my_rt_enhance()
    ut_my()

