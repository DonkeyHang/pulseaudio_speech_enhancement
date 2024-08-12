import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math

import torch as th
from torch import nn
from torch.nn import functional as F
import numpy as np

from resample import downsample2, upsample2
from utils import capture_init

FRAME_SIZE_480 = 480
PROCESS_SIZE_661 = 661
STEP_SIZE = 256
OUTPUT_FRAME_SIZE = 256


class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class Demucs(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.

    """
    @capture_init
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 causal=True,
                 resample=4,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 rescale=0.1,
                 floor=1e-3):

        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        return std * x


def fast_conv(conv, x):
    """
    Faster convolution evaluation if either kernel size is 1
    or length of sequence is 1.
    """
    batch, chin, length = x.shape
    chout, chin, kernel = conv.weight.shape
    assert batch == 1
    if kernel == 1:
        x = x.view(chin, length)
        out = th.addmm(conv.bias.view(-1, 1),
                       conv.weight.view(chout, chin), x)
    elif length == kernel:
        x = x.view(chin * kernel, 1)
        out = th.addmm(conv.bias.view(-1, 1),
                       conv.weight.view(chout, chin * kernel), x)
    else:
        out = conv(x)
    return out.view(batch, chout, -1)


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


class DemucsStreamer_RT:
    def __init__(self, demucs,
                 dry=0,
                 num_frames=1,
                 resample_lookahead=64,
                 resample_buffer=256):
        device = next(iter(demucs.parameters())).device
        self.demucs = demucs
        self.lstm_state = None
        self.conv_state = None
        self.resample_buf_size_256 = 256# 256
        self.frame_length = demucs.valid_length(1)# 597
        self.total_length = 661
        self.stride_size = demucs.total_stride# 256
        self.resample_in = th.zeros(demucs.chin, self.resample_buf_size_256, device=device)# [1,256]
        self.resample_out = th.zeros(demucs.chin, self.resample_buf_size_256, device=device)# [1,256]

        self.frames = 0
        self.total_time = 0
        self.variance = 0
        self.pending = th.zeros(demucs.chin, 0, device=device)# [1,0]

        bias = demucs.decoder[0][2].bias# [384]
        weight = demucs.decoder[0][2].weight# [768,384,8]
        _, _, kernel = weight.shape# kernel:8
        self._bias = bias.view(-1, 1).repeat(1, kernel).view(-1, 1)# [3072,1]
        self._weight = weight.permute(1, 2, 0).contiguous()# [384,8,768]

        self.in_buf_1024 = AudioRingBuffer(1,1024)
        self.out_buf_1024 = AudioRingBuffer(1,1024)
        self.out_buf_1024.write( th.as_tensor(np.zeros((1,480)), dtype=th.float32) )
        self.process_buf_661 = th.as_tensor( np.zeros((1,PROCESS_SIZE_661)), dtype=th.float32 )
        self.tmp_out_480 = np.zeros(480)


    def processBlock(self, new_frame_480):
        # self.counter+=1

        # push 480 new data into in_buf_960
        if self.in_buf_1024.available_to_write():
            self.in_buf_1024.write(new_frame_480)
        else:
            print("in_buf_1024 push error")

        # process if buffer_size>661
        while(self.in_buf_1024.available_to_read() >= STEP_SIZE):
            #once process need 661 samples at least 
            self.process_buf_661[:,:-STEP_SIZE] = self.process_buf_661[:,STEP_SIZE:].clone()
            self.process_buf_661[:,-STEP_SIZE:] = th.as_tensor(self.in_buf_1024.read(STEP_SIZE),dtype=th.float32)

            # preprocess
            frame = self.process_buf_661#.unsqueeze(0)#[1,661]
            if self.demucs.normalize:
                mono = frame.mean(0)
                self.variance = (mono**2).mean()
                frame = frame / (self.demucs.floor + math.sqrt(self.variance))
            frame = th.cat([self.resample_in, frame], dim=-1)#[1,917]
            self.resample_in[:] = frame[:, self.stride_size - self.resample_buf_size_256:self.stride_size]#[1,256]

            if self.demucs.resample == 4:
                frame = upsample2(upsample2(frame))#[1,3668]
            elif self.demucs.resample == 2:
                frame = upsample2(frame)
            frame = frame[:, self.demucs.resample * self.resample_buf_size_256:]#[1,1024:]  # remove pre sampling buffer
            frame = frame[:, :self.demucs.resample * self.frame_length]#[1,:2388]  # remove extra samples after window
            
            # note infer
            out, extra = self._separate_frame(frame)#out:[1,1024] #extra:[1,1364]

            # post process
            padded_out = th.cat([self.resample_out, out, extra], 1)#[1,2644]
            self.resample_out[:] = out[:, -self.resample_buf_size_256:]#self.resample_out:[1,256]
            if self.demucs.resample == 4:
                out = downsample2(downsample2(padded_out))#[1,661]
            else:
                print("downresample error")
            out = out[:, self.resample_buf_size_256 // self.demucs.resample:]#[1,597]
            out = out[:, :self.stride_size]#[1,256]

            if self.demucs.normalize:
                out *= math.sqrt(self.variance)

            # push output into out_buf_960
            if self.out_buf_1024.available_to_write()>=STEP_SIZE:
                self.out_buf_1024.write( out.detach().numpy() )
            else:
                print("out_buf_1024 push error")
                
        # pop 480 new data
        if(self.out_buf_1024.available_to_read() >= FRAME_SIZE_480):
            self.tmp_out_480 = self.out_buf_1024.read(FRAME_SIZE_480)[0]
        else:
            print("output buffer pop error")

        return self.tmp_out_480



    def _separate_frame(self, frame):
        # demucs = self.demucs
        skips = []
        next_state = []
        first = self.conv_state is None
        stride = self.stride_size * self.demucs.resample
        x = frame[None]
        for idx, encode in enumerate(self.demucs.encoder):
            stride //= self.demucs.stride
            length = x.shape[2]
            if idx == self.demucs.depth - 1:
                # This is sligthly faster for the last conv
                x = fast_conv(encode[0], x)
                x = encode[1](x)
                x = fast_conv(encode[2], x)
                x = encode[3](x)
            else:
                if not first:
                    prev = self.conv_state.pop(0)
                    prev = prev[..., stride:]
                    tgt = (length - self.demucs.kernel_size) // self.demucs.stride + 1
                    missing = tgt - prev.shape[-1]
                    offset = length - self.demucs.kernel_size - self.demucs.stride * (missing - 1)
                    x = x[..., offset:]
                x = encode[1](encode[0](x))
                x = fast_conv(encode[2], x)
                x = encode[3](x)
                if not first:
                    x = th.cat([prev, x], -1)
                next_state.append(x)
            skips.append(x)

        x = x.permute(2, 0, 1)
        x, self.lstm_state = self.demucs.lstm(x, self.lstm_state)
        x = x.permute(1, 2, 0)
        # In the following, x contains only correct samples, i.e. the one
        # for which each time position is covered by two window of the upper layer.
        # extra contains extra samples to the right, and is used only as a
        # better padding for the online resampling.
        extra = None
        for idx, decode in enumerate(self.demucs.decoder):
            skip = skips.pop(-1)
            x += skip[..., :x.shape[-1]]
            x = fast_conv(decode[0], x)
            x = decode[1](x)

            if extra is not None:
                skip = skip[..., x.shape[-1]:]
                extra += skip[..., :extra.shape[-1]]
                extra = decode[2](decode[1](decode[0](extra)))
            x = decode[2](x)
            next_state.append(x[..., -self.demucs.stride:] - decode[2].bias.view(-1, 1))
            if extra is None:
                extra = x[..., -self.demucs.stride:]
            else:
                extra[..., :self.demucs.stride] += next_state[-1]
            x = x[..., :-self.demucs.stride]

            if not first:
                prev = self.conv_state.pop(0)
                x[..., :self.demucs.stride] += prev
            if idx != self.demucs.depth - 1:
                x = decode[3](x)
                extra = decode[3](extra)
        self.conv_state = next_state
        return x[0], extra[0]
    


# def test():
#     import argparse
#     import soundfile as sf
#     import torch
#     parser = argparse.ArgumentParser(
#         "denoiser.demucs",
#         description="Benchmark the streaming Demucs implementation, "
#                     "as well as checking the delta with the offline implementation.")
#     parser.add_argument("--depth", default=5, type=int)
#     parser.add_argument("--resample", default=4, type=int)
#     parser.add_argument("--hidden", default=48, type=int)
#     parser.add_argument("--sample_rate", default=16000, type=float)
#     parser.add_argument("--device", default="cpu")
#     parser.add_argument("-t", "--num_threads", type=int)
#     parser.add_argument("-f", "--num_frames", type=int, default=1)
#     args = parser.parse_args()
#     if args.num_threads:
#         th.set_num_threads(args.num_threads)
#     sr = args.sample_rate
#     sr_ms = sr / 1000
#     # x = th.randn(1, int(sr * 4)).to(args.device)
#     x,samplerate = sf.read("/Users/donkeyddddd/Documents/Rx_projects/git_projects/pulseaudio_speech_enhancement/deployment/wav/input_noise_car_talk.wav")
#     x = torch.as_tensor(x,dtype=torch.float32).squeeze(0).squeeze(0)
    
    
#     demucs = Demucs(depth=args.depth, hidden=args.hidden, resample=args.resample).to(args.device)
#     out = demucs(x[None])[0]

#     sf.write("/Users/donkeyddddd/Documents/Rx_projects/git_projects/pulseaudio_speech_enhancement/deployment/wav/output_noise_car_talk_all_data.wav",out.squeeze(0).detach().numpy(),48000)


#     streamer = DemucsStreamer(demucs, num_frames=args.num_frames)
#     out_rt = []
#     frame_size = streamer.total_length
#     with th.no_grad():
#         while x.shape[1] > 0:
#             out_rt.append(streamer.feed(x[:, :frame_size]))
#             x = x[:, frame_size:]
#             frame_size = streamer.demucs.total_stride
#     out_rt.append(streamer.flush())
#     out_rt = th.cat(out_rt, 1)
#     model_size = sum(p.numel() for p in demucs.parameters()) * 4 / 2**20
#     initial_lag = streamer.total_length / sr_ms
#     tpf = 1000 * streamer.time_per_frame
#     print(f"model size: {model_size:.1f}MB, ", end='')
#     print(f"delta batch/streaming: {th.norm(out - out_rt[:,:64000]) / th.norm(out):.2%}")
#     print(f"initial lag: {initial_lag:.1f}ms, ", end='')
#     print(f"stride: {streamer.stride * args.num_frames / sr_ms:.1f}ms")
#     print(f"time per frame: {tpf:.1f}ms, ", end='')
#     print(f"RTF: {((1000 * streamer.time_per_frame) / (streamer.stride / sr_ms)):.2f}")
#     print(f"Total lag with computation: {initial_lag + tpf:.1f}ms")


if __name__ == "__main__":
    # test()
    xxx = 1
