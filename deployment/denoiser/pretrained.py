import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging

import torch.hub

# from .demucs import Demucs
# version for running from command line
from utils import deserialize_model

# version for running within python
# from utils import deserialize_model

logger = logging.getLogger(__name__)
# ROOT = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/"
# DNS_48_URL = ROOT + "dns48-11decc9d8e3f0998.th"
# DNS_64_URL = ROOT + "dns64-a7761ff99a7d5bb6.th"
# MASTER_64_URL = ROOT + "master64-8a5dfb4bb92753dd.th"


# def _demucs(pretrained, url, **kwargs):
#     model = Demucs(**kwargs)
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model


# def dns48(pretrained=True):
#     return _demucs(pretrained, DNS_48_URL, hidden=48)


# def dns64(pretrained=True):
#     return _demucs(pretrained, DNS_64_URL, hidden=64)


# def master64(pretrained=True):
#     return _demucs(pretrained, MASTER_64_URL, hidden=64)


def add_model_flags(parser):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-m", "--model_path", help="Path to local trained model.")
    group.add_argument("--dns48", action="store_true",
                       help="Use pre-trained real time H=48 model trained on DNS.")
    group.add_argument("--dns64", action="store_true",
                       help="Use pre-trained real time H=64 model trained on DNS.")
    group.add_argument("--master64", action="store_true",
                       help="Use pre-trained real time H=64 model trained on DNS and Valentini.")

# version with no args that we need to get to work
def get_model(model_path):
    """
    Load local model package or torchhub pre-trained model.
    """
    # print('args inside of get_model:', args)
    # print('\n')
    # print('args.model_path:', args.model_path)
    # print('model path', model_path)
    # section we need to get working
    if model_path:
        logger.info("Loading model from %s", model_path)
        pkg = torch.load(model_path)
        model = deserialize_model(pkg)
    logger.debug(model)
    return model

    # if args.model_path:
    #     logger.info("Loading model from %s", args.model_path)
    #     pkg = torch.load(args.model_path)
    #     model = deserialize_model(pkg)
        # if 'model' in pkg:
        #     if 'best_state' in pkg:
        #         pkg['model']['state'] = pkg['best_state']
        #     model = deserialize_model(pkg['model'])
        # else:
            #model = deserialize_model(pkg)
    # elif args.dns64:
    #     logger.info("Loading pre-trained real time H=64 model trained on DNS.")
    #     model = dns64()
    # elif args.master64:
    #     logger.info("Loading pre-trained real time H=64 model trained on DNS and Valentini.")
    #     model = master64()
    # else:
    #     logger.info("Loading pre-trained real time H=48 model trained on DNS.")
    #     model = dns48()
    
    # logger.debug(model)
    # return model


# version 2 with args that currently works, but we need the other version to work
# def get_model(args):
#     """
#     Load local model package or torchhub pre-trained model.
#     """
#     if args.model_path:
#         logger.info("Loading model from %s", args.model_path)
#         pkg = torch.load(args.model_path)
#         model = deserialize_model(pkg)
#     logger.debug(model)
#     return model
