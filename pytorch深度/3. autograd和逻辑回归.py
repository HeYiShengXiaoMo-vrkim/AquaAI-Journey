import torch
import matplotlib.pyplot as plt
import numpy

"""
1. autograd
~   torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False)
    功能 : 自动求导的张量,如loss
    tensors : 用来自动求导的张量
"""