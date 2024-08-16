import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import logging
import os
import sys
import torchaudio
from torch.utils.data import DataLoader
from demucs import Demucs, DemucsStreamer_RT

from audio import Audioset
from pretrained import add_model_flags, get_model
from utils import LogProgress

import soundfile as sf
from tqdm import *
import numpy as np
import torch

logger = logging.getLogger(__name__)



class RingBuffer:
    def __init__(self, element_count, element_size):
        self.size = element_count
        self.element_size = element_size
        self.buffer = np.zeros(element_count, dtype=np.float32)  # Assuming data type from C++ (float)
        self.write_index = 0
        self.read_index = 0

    def write(self, data):
        data_length = len(data)
        first_part_len = min(data_length, self.size - self.write_index)
        self.buffer[self.write_index:self.write_index + first_part_len] = data[:first_part_len]
        second_part_len = data_length - first_part_len
        if second_part_len > 0:
            self.buffer[:second_part_len] = data[first_part_len:first_part_len + second_part_len]
        self.write_index = (self.write_index + data_length) % self.size
        return data_length

    def read(self, element_count):
        first_part_len = min(element_count, self.size - self.read_index)
        data = self.buffer[self.read_index:self.read_index + first_part_len]
        second_part_len = element_count - first_part_len
        if second_part_len > 0:
            data = np.concatenate((data, self.buffer[:second_part_len]))
        self.read_index = (self.read_index + element_count) % self.size
        return data

    def move_read_pointer(self, element_count):
        self.read_index = (self.read_index + element_count) % self.size
        return element_count

    def available_to_read(self):
        return (self.write_index - self.read_index + self.size) % self.size

    def available_to_write(self):
        return self.size - self.available_to_read()

class AudioRingBuffer:
    def __init__(self, channels, max_frames):
        self.channels = channels
        self.max_frames = max_frames
        self.buffers = [RingBuffer(max_frames, 1) for _ in range(channels)]

    def write(self, data):
        if len(data) != self.channels:
            return  # Handle channel mismatch as in C++ code
        for i, channel_data in enumerate(data):
            self.buffers[i].write(channel_data)

    def read(self, frames):
        return [self.buffers[i].read(frames) for i in range(self.channels)]

    def available_to_read(self):
        return min(buffer.available_to_read() for buffer in self.buffers)

    def available_to_write(self):
        return min(buffer.available_to_write() for buffer in self.buffers)

def vorbis_window(n):
    indices = torch.arange(n, dtype=torch.float32) + 0.5
    n_double = torch.tensor(n, dtype=torch.float32)
    window = torch.sin((torch.pi / 2.0) * torch.pow(torch.sin(indices / n_double * torch.pi), 2.0))
    
    return window.unsqueeze(0)

def get_dataset_fast_api_version(file_location):
    files = []
    siginfo = torchaudio.info(file_location)
    length = siginfo.num_frames // siginfo.num_channels
    files.append((file_location, length))
    return Audioset(files, with_path=True, sample_rate=48000)


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


def ut_only_demucs():
    import numpy as np
    import matplotlib.pyplot as plt
    import soundfile as sf
    import torch as th

    

    FILE_WAV = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/pulseaudio_speech_enhancement/deployment/wav/input_noise_car_talk.wav"
    MODEL_PATH = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/pulseaudio_speech_enhancement/deployment/denoiser/best.th"
    SAVE_WAV_PATH = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/pulseaudio_speech_enhancement/deployment/wav/result_only_demucs_noise_car_talk.wav"
    FRAME_SIZE_480 = 480
    PROCESS_SIZE_661 = 661
    PROCESS_SIZE_960 = 960
    STEP_SIZE_256 = 256
    STEP_SIZE_480 = 480
    OUTPUT_FRAME_SIZE_256 = 256


    # load wav
    dset = get_dataset_fast_api_version(FILE_WAV)
    loader = DataLoader(dset, batch_size=1)
    iterator = LogProgress(logger, loader, name="Generate enhanced files")
    for data in iterator:
        # Get all wav data
        audio, filenames = data #audio:[1,160000] # audio samplerate is 48000
    audio = audio.squeeze(0).to('cpu') 
    audio_frame_num = audio.shape[1] // FRAME_SIZE_480

    # output_buffer
    output = np.zeros(audio_frame_num*FRAME_SIZE_480)


    demucs = get_model(MODEL_PATH).to('cpu')
    demucs.eval()
    
    for param in demucs.parameters():
        param.requirs_grad = False

    # res = demucs.forward(audio)
        
    # ===================== initial buffer ===============================
    in_buf_1024 = AudioRingBuffer(1,1024)
    out_buf_1024 = AudioRingBuffer(1,1024)
    out_buf_1024.write( th.as_tensor(np.zeros((1,480)), dtype=th.float32) )
    process_buf_960 = th.as_tensor( np.zeros((1,PROCESS_SIZE_960)), dtype=th.float32 )
    frame_960 = th.as_tensor( np.zeros((1,960)), dtype=th.float32 )
    output_tensor_480 = th.as_tensor( np.zeros((1,480)),dtype=th.float32 )


    # ===================== processBlock simulink start===================
    for idx in tqdm(range(audio_frame_num)):
        # self.counter+=1

        # push 480 new data into in_buf_960
        if in_buf_1024.available_to_write():
            in_buf_1024.write(audio[:,idx*FRAME_SIZE_480:(idx+1)*FRAME_SIZE_480])
        else:
            print("in_buf_1024 push error")

        
        # onnx_imlp = onnx.load
        # input_name = ort_sess.get_inputs()[0].name
        # output_name = ort_sess.get_outputs()[0].name
        # process if buffer_size>661
        while(in_buf_1024.available_to_read() >= STEP_SIZE_480):
            tensor_buffer_256 = th.as_tensor(in_buf_1024.read(STEP_SIZE_480),dtype=th.float32)
            
            
            # ============================
            # model inference once in here
            process_buf_960 = th.roll(process_buf_960, -STEP_SIZE_480, dims=-1)
            process_buf_960[:,-STEP_SIZE_480:] = tensor_buffer_256  

            # preprocess
            frame_960 = process_buf_960.clone()# * vorbis_win_960
            
            tmp_960 = demucs.forward(frame_960).squeeze(0)
            
            # overlap 
            output_tensor_480 = tmp_960[:,:STEP_SIZE_480]
            # ============================



            # push output into out_buf_960
            if out_buf_1024.available_to_write()>=STEP_SIZE_480:
                out_buf_1024.write( output_tensor_480.detach().numpy() )
                # out_buf_1024.write( output_tensor_256.detach().numpy() )
                
            else:
                print("out_buf_1024 push error")
                
        # pop 480 new data
        if(out_buf_1024.available_to_read() >= FRAME_SIZE_480):
            output[idx*FRAME_SIZE_480:(idx+1)*FRAME_SIZE_480] = out_buf_1024.read(FRAME_SIZE_480)[0]
        else:
            print("output buffer pop error")

    # ===================== processBlock simulink end===================

    sf.write(SAVE_WAV_PATH,output,16000)


    print("process done")
    xxx = 1




# version with command line args
if __name__ == "__main__":
    # ut() #ok
    # shell = my_rt_enhance()
    
    # ut_my() # ok

    ut_only_demucs()

