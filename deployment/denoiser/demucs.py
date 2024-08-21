import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math

import torch as th
from torch import nn
from torch.nn import functional as F
import numpy as np
import onnxruntime as ort

# from resample import downsample2, upsample2
import inspect

FRAME_SIZE_480 = 480
PROCESS_SIZE_661 = 661
STEP_SIZE = 256
OUTPUT_FRAME_SIZE = 256
MODEL_PATH = "/Users/donkeyddddd/Documents/Rx_projects/git_projects/pulseaudio_speech_enhancement/deployment/denoiser/best.th"


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
            self._rescale_module(reference=rescale)

    def _rescale_conv(self, conv, reference):
        std = conv.weight.std().detach()
        scale = (std / reference)**0.5
        conv.weight.data /= scale
        if conv.bias is not None:
            conv.bias.data /= scale

    def _rescale_module(self, reference):
        for sub in self.modules():
            if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
                self._rescale_conv(sub, reference)

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

    # ------------------ resample part --------------------
    def sinc(self, t):
        """sinc.

        :param t: the input tensor
        """
        t = t.to(th.float32)
        return th.where(t == 0, th.tensor(1., device=t.device, dtype=t.dtype), th.sin(t) / t)

    def kernel_upsample2(self, zeros=56):
        """kernel_upsample2.

        """
        win = th.hann_window(4 * zeros + 1, periodic=False)
        winodd = win[1::2]
        t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
        t *= math.pi
        kernel = (self.sinc(t) * winodd).view(1, 1, -1)
        return kernel

    def upsample2(self, x, zeros=56):
        """
        Upsampling the input by 2 using sinc interpolation.
        Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
        ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
        Vol. 9. IEEE, 1984.
        """
        *other, time = x.shape
        kernel = self.kernel_upsample2(zeros).to(x)
        out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
        y = th.stack([x, out], dim=-1)
        return y.view(*other, -1)

    def kernel_downsample2(self, zeros=56):
        """kernel_downsample2.

        """
        win = th.hann_window(4 * zeros + 1, periodic=False)
        winodd = win[1::2]
        t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
        t.mul_(math.pi)
        kernel = (self.sinc(t) * winodd).view(1, 1, -1)
        return kernel

    def downsample2(self, x, zeros=56):
        """
        Downsampling the input by 2 using sinc interpolation.
        Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
        ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
        Vol. 9. IEEE, 1984.
        """
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1))
        xeven = x[..., ::2]
        xodd = x[..., 1::2]
        *other, time = xodd.shape
        kernel = self.kernel_downsample2(zeros).to(x)
        out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(
            *other, time)
        return out.view(*other, -1).mul(0.5)



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
            x = self.upsample2(x)
        elif self.resample == 4:
            x = self.upsample2(x)
            x = self.upsample2(x)
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
            x = self.downsample2(x)
        elif self.resample == 4:
            x = self.downsample2(x)
            x = self.downsample2(x)

        x = x[..., :length]
        return std * x


class InferenceOnceImpl(nn.Module):
    '''
    this class means export onnx for inference once
    used demucs model to inference once and combine all of another opration, 
    such as upsample and downsample, fast_conv and so on
    '''

    def __init__(self):
        super().__init__()
        self.demucs = self._load_model()
        
        for param in self.demucs.parameters():
            param.requirs_grad = False

        self.lstm_state = None
        self.conv_state = None
        self.resample_buf_size_256 = 256# 256
        self.frame_length = self.demucs.valid_length(1)# 597
        self.total_length = 661
        self.stride_size = self.demucs.total_stride# 256
        self.resample_in = th.zeros(self.demucs.chin, self.resample_buf_size_256, device='cpu')# [1,256]
        self.resample_out = th.zeros(self.demucs.chin, self.resample_buf_size_256, device='cpu')# [1,256]
        self.process_buf_661 = th.as_tensor( np.zeros((1,PROCESS_SIZE_661)), dtype=th.float32 )

        self.variance = 0
        self.pending = th.zeros(self.demucs.chin, 0, device='cpu')# [1,0]

        bias = self.demucs.decoder[0][2].bias.detach()# [384]
        weight = self.demucs.decoder[0][2].weight.detach()# [768,384,8]
        _, _, kernel = weight.shape# kernel:8
        self._bias = bias.view(-1, 1).repeat(1, kernel).view(-1, 1)# [3072,1]
        self._weight = weight.permute(1, 2, 0).contiguous()# [384,8,768]

        self.frame_661 = th.as_tensor( np.zeros((1,661)), dtype=th.float32 )
        self.frame_917 = th.as_tensor( np.zeros((1,917)), dtype=th.float32 )
        self.frame_3668 = th.as_tensor( np.zeros((1,3668)), dtype=th.float32 )
        self.frame_2388 = th.as_tensor( np.zeros((1,2388)), dtype=th.float32 )
        self.out_1024 = th.as_tensor( np.zeros((1,1024)), dtype=th.float32 )
        self.extra_1364 = th.as_tensor( np.zeros((1,1364)), dtype=th.float32 )
        self.paddedout_2644 = th.as_tensor( np.zeros((1,2644)), dtype=th.float32 )
        self.output_tensor_256 = th.as_tensor( np.zeros((1,256)),dtype=th.float32 )

        self.hidden_state = th.as_tensor( np.zeros((2,1,768)), dtype=th.float32 )
        self.cell_state = th.as_tensor( np.zeros((2,1,768)), dtype=th.float32 )


    def forward(self, tensor_buffer_256):
        return self._processOnce(tensor_buffer_256)


    # ------------------ model infer part ----------------
    # push new 256 samples into ringbuffer and process once
    def _processOnce(self, tensor_buffer_256):
        # input:Tensor[1,256]

        #once process need 661 samples at least 
        self.process_buf_661 = th.roll(self.process_buf_661, -STEP_SIZE, dims=-1)
        self.process_buf_661[:,-STEP_SIZE:] = tensor_buffer_256  

        # preprocess
        self.frame_661 = self.process_buf_661.clone()#.unsqueeze(0)#[1,661]
        if self.demucs.normalize:
            mono = self.frame_661.mean(0)
            self.variance = (mono**2).mean()
            self.frame_661 = self.frame_661 / (self.demucs.floor + math.sqrt(self.variance))
        self.frame_917 = th.cat([self.resample_in, self.frame_661], dim=-1)#[1,917]
        self.resample_in = self.frame_917[:, self.stride_size - self.resample_buf_size_256:self.stride_size]#[1,256]

        if self.demucs.resample == 4:
            self.frame_3668 = self._upsample2(self._upsample2(self.frame_917))#[1,3668]
        else:
            print("upsample error")
        self.frame_2388 = self.frame_3668[:, (self.demucs.resample*self.resample_buf_size_256) : (self.demucs.resample*self.resample_buf_size_256 + self.demucs.resample*self.frame_length)]#[1,:2388]  # remove extra samples after window
        # return self.frame_2388
        # note infer
        self.out_1024, self.extra_1364 = self._separate_frame(self.frame_2388)#out:[1,1024] #extra:[1,1364]

        # post process
        self.paddedout_2644 = th.cat([self.resample_out, self.out_1024, self.extra_1364], 1)#[1,2644]
        self.resample_out[:] = self.out_1024[:, -self.resample_buf_size_256:]#self.resample_out:[1,256]
        if self.demucs.resample == 4:
            self.frame_661 = self._downsample2(self._downsample2(self.paddedout_2644))#[1,661]
        else:
            print("downresample error")
        self.output_tensor_256 = self.frame_661[:, (self.resample_buf_size_256 // self.demucs.resample) : (self.resample_buf_size_256 // self.demucs.resample + self.stride_size)]#[1,256]

        if self.demucs.normalize:
            self.output_tensor_256 *= math.sqrt(self.variance)

        #output: Tensor[1,256]
        return self.output_tensor_256.clone()

    def _fast_conv(self, conv, x):
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

    # infer once for demucs model 
    def _separate_frame(self, frame):
        # demucs = self.demucs
        skips = []
        next_state = []
        first = self.conv_state is None
        stride = self.stride_size * self.demucs.resample
        x = frame[None]
        
        #encoder
        for idx, encode in enumerate(self.demucs.encoder):
            stride //= self.demucs.stride
            length = x.shape[2]
            # print("length :",length)
            if idx == self.demucs.depth - 1:
                # This is sligthly faster for the last conv
                x = self._fast_conv(encode[0], x)
                x = encode[1](x)
                x = self._fast_conv(encode[2], x)
                x = encode[3](x).detach()
            else:
                if not first:
                    prev = self.conv_state.pop(0)
                    # prev = prev.detach()
                    prev = prev[..., stride:]
                    tgt = (length - self.demucs.kernel_size) // self.demucs.stride + 1
                    # print("tgt :",tgt)
                    missing = tgt - prev.shape[-1]
                    offset = length - self.demucs.kernel_size - self.demucs.stride * (missing - 1)
                    # print("offset is:",offset)
                    x = x[..., offset:]
                x = encode[1](encode[0](x)).detach()
                x = self._fast_conv(encode[2], x).detach()
                x = encode[3](x).detach()
                if not first:
                    x = th.cat([prev, x], -1).detach()
                next_state.append(x.detach())
            skips.append(x.detach())

        x = x.permute(2, 0, 1)
        # if(self.lstm_state==None):
        #     x, (new_hidden, new_cell) = self.demucs.lstm(x, (self.hidden_state,self.cell_state))
        #     self.hidden_state = new_hidden.detach()
        #     self.cell_state = new_cell.detach()
            # self.lstm_state==1
        # else:
        x, (new_hidden, new_cell) = self.demucs.lstm(x, (self.hidden_state,self.cell_state))
        self.hidden_state = new_hidden.detach()
        self.cell_state = new_cell.detach()
        x = x.permute(1, 2, 0).detach()
        # In the following, x contains only correct samples, i.e. the one
        # for which each time position is covered by two window of the upper layer.
        # extra contains extra samples to the right, and is used only as a
        # better padding for the online resampling.
        extra = None

        # decoder
        for idx, decode in enumerate(self.demucs.decoder):
            skip = skips.pop(-1)
            x += skip[..., :x.shape[-1]]
            x = self._fast_conv(decode[0], x)
            x = decode[1](x).detach()

            if extra is not None:
                skip = skip[..., x.shape[-1]:]
                extra += skip[..., :extra.shape[-1]]
                extra = decode[2](decode[1](decode[0](extra))).detach()
            x = decode[2](x).detach()
            next_state.append( (x[..., -self.demucs.stride:] - decode[2].bias.view(-1, 1)).detach() )
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


    # ------------------ resample part --------------------
    def _sinc(self, t):
        """sinc.

        :param t: the input tensor
        """
        t = t.to(th.float32)
        return th.where(t == 0, th.tensor(1., device=t.device, dtype=t.dtype), th.sin(t) / t)

    def _kernel_upsample2(self, zeros=56):
        """kernel_upsample2.

        """
        win = th.hann_window(4 * zeros + 1, periodic=False)
        winodd = win[1::2]
        t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
        t *= math.pi
        kernel = (self._sinc(t) * winodd).view(1, 1, -1)
        return kernel

    def _upsample2(self, x, zeros=56):
        """
        Upsampling the input by 2 using sinc interpolation.
        Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
        ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
        Vol. 9. IEEE, 1984.
        """
        *other, time = x.shape
        kernel = self._kernel_upsample2(zeros).to(x)
        out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
        y = th.stack([x, out], dim=-1)
        return y.view(*other, -1)
        

    def _kernel_downsample2(self, zeros=56):
        """kernel_downsample2.

        """
        win = th.hann_window(4 * zeros + 1, periodic=False)
        winodd = win[1::2]
        t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
        t.mul_(math.pi)
        kernel = (self._sinc(t) * winodd).view(1, 1, -1)
        return kernel

    def _downsample2(self, x, zeros=56):
        """
        Downsampling the input by 2 using sinc interpolation.
        Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
        ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
        Vol. 9. IEEE, 1984.
        """
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1))
        xeven = x[..., ::2]
        xodd = x[..., 1::2]
        *other, time = xodd.shape
        kernel = self._kernel_downsample2(zeros).to(x)
        out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(
            *other, time)
        return out.view(*other, -1).mul(0.5)


    # --------------- load model and coefs part -----------
    def _deserialize_model(self, package, strict=False):
        klass = package['class']
        if strict:
            model = klass(*package['args'], **package['kwargs'])
        else:
            sig = inspect.signature(klass)
            kw = package['kwargs']
            for key in list(kw):
                if key not in sig.parameters:
                    del kw[key]
            model = klass(*package['args'], **kw)
        model.load_state_dict(package['state'])
        return model.eval()
    
    def _get_model(self, model_path):
        if model_path:
            pkg = th.load(model_path)
            model = self._deserialize_model(pkg)
        return model.eval()
    
    def _load_model(self,model_path=MODEL_PATH):
        model = self._get_model(model_path).to('cpu')
        # model.eval()
        return model.eval()







# class RingBuffer:
#     def __init__(self, element_count, element_size):
#         self.size = element_count
#         self.element_size = element_size
#         self.buffer = np.zeros(element_count, dtype=np.float32)  # Assuming data type from C++ (float)
#         self.write_index = 0
#         self.read_index = 0

#     def write(self, data):
#         data_length = len(data)
#         first_part_len = min(data_length, self.size - self.write_index)
#         self.buffer[self.write_index:self.write_index + first_part_len] = data[:first_part_len]
#         second_part_len = data_length - first_part_len
#         if second_part_len > 0:
#             self.buffer[:second_part_len] = data[first_part_len:first_part_len + second_part_len]
#         self.write_index = (self.write_index + data_length) % self.size
#         return data_length

#     def read(self, element_count):
#         first_part_len = min(element_count, self.size - self.read_index)
#         data = self.buffer[self.read_index:self.read_index + first_part_len]
#         second_part_len = element_count - first_part_len
#         if second_part_len > 0:
#             data = np.concatenate((data, self.buffer[:second_part_len]))
#         self.read_index = (self.read_index + element_count) % self.size
#         return data

#     def move_read_pointer(self, element_count):
#         self.read_index = (self.read_index + element_count) % self.size
#         return element_count

#     def available_to_read(self):
#         return (self.write_index - self.read_index + self.size) % self.size

#     def available_to_write(self):
#         return self.size - self.available_to_read()


# class AudioRingBuffer:
#     def __init__(self, channels, max_frames):
#         self.channels = channels
#         self.max_frames = max_frames
#         self.buffers = [RingBuffer(max_frames, 1) for _ in range(channels)]

#     def write(self, data):
#         if len(data) != self.channels:
#             return  # Handle channel mismatch as in C++ code
#         for i, channel_data in enumerate(data):
#             self.buffers[i].write(channel_data)

#     def read(self, frames):
#         return [self.buffers[i].read(frames) for i in range(self.channels)]

#     def available_to_read(self):
#         return min(buffer.available_to_read() for buffer in self.buffers)

#     def available_to_write(self):
#         return min(buffer.available_to_write() for buffer in self.buffers)


# class DemucsStreamer_RT:
#     def __init__(self, demucs):
#         device = next(iter(demucs.parameters())).device
#         # self.demucs = demucs
#         self.impl = InferenceOnceImpl()

#         self.in_buf_1024 = AudioRingBuffer(1,1024)
#         self.out_buf_1024 = AudioRingBuffer(1,1024)
#         self.out_buf_1024.write( th.as_tensor(np.zeros((1,480)), dtype=th.float32) )
#         self.tmp_out_480 = np.zeros(480)
#         self.tmp_out_256 = np.zeros((1,256))

#         self.export_once_already = False

#         self.ort_sess = ort.InferenceSession("model.onnx")



#     def processBlock(self, new_frame_480):
#         # self.counter+=1

#         # push 480 new data into in_buf_960
#         if self.in_buf_1024.available_to_write():
#             self.in_buf_1024.write(new_frame_480)
#         else:
#             print("in_buf_1024 push error")

#         self.impl = self.impl.eval()
#         # self.onnx_imlp = onnx.load
#         # input_name = self.ort_sess.get_inputs()[0].name
#         # output_name = self.ort_sess.get_outputs()[0].name
#         # process if buffer_size>661
#         while(self.in_buf_1024.available_to_read() >= STEP_SIZE):
#             x = th.as_tensor(self.in_buf_1024.read(STEP_SIZE),dtype=th.float32)
            
#             # model inference once
#             self.tmp_out_256 = self.impl.forward(x)
#             # self.ort_res = self.ort_sess.run([output_name],{
#             #     input_name: x.cpu().numpy()
#             # })[0]

#             # print("diff :",self.tmp_out_256.detach().numpy() - self.ort_res)

#             # xxx = 1
#             # pass
            
#             # if(self.export_once_already==False):
#             #     th.onnx.export(
#             #         self.impl,
#             #         th.as_tensor(self.in_buf_1024.read(STEP_SIZE),dtype=th.float32),
#             #         "model.onnx"
#             #     )
#             #     # onnx_program.save("/Users/donkeyddddd/Documents/Rx_projects/git_projects/pulseaudio_speech_enhancement/deployment/wav/test.onnx")
#             #     self.export_once_already = True


#             # push output into out_buf_960
#             if self.out_buf_1024.available_to_write()>=STEP_SIZE:
#                 self.out_buf_1024.write( self.tmp_out_256.detach().numpy() )
#             else:
#                 print("out_buf_1024 push error")
                
#         # pop 480 new data
#         if(self.out_buf_1024.available_to_read() >= FRAME_SIZE_480):
#             self.tmp_out_480 = self.out_buf_1024.read(FRAME_SIZE_480)[0]
#         else:
#             print("output buffer pop error")

#         return self.tmp_out_480



if __name__ == "__main__":
    # test()
    xxx = 1
