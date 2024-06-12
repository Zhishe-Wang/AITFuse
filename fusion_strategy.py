import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import utils
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda"if torch.cuda.is_available()else"cpu")

EPSILON = 1e-5


# attention fusion strategy, average based on weight maps
def attention_fusion_weight(tensor1, tensor2, p_type):

    f_channel = row_vector_fusion(tensor1, tensor2, p_type)
    f_spatial = column_vector_fusion(tensor1, tensor2, p_type)

    tensor_f = (f_channel + f_spatial)

    return tensor_f

def add_fusion(tensor1,tensor2,p_type):
    x = tensor1 + tensor2
    return x

# select channel
def row_vector_fusion(tensor1, tensor2, p_type):

    shape = tensor1.size()

    # calculate row vector attention
    global_p1 = row_vector_attention(tensor1, p_type)
    global_p2 = row_vector_attention(tensor2, p_type)

    # get weight map
    global_p_w1 = torch.exp(global_p1) / (torch.exp(global_p1) + torch.exp(global_p2) + EPSILON)
    global_p_w2 = torch.exp(global_p2) / (torch.exp(global_p1) + torch.exp(global_p2) + EPSILON)

    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
    global_p_w1 = global_p_w1.to(device)
    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])
    global_p_w2 = global_p_w2.to(device)

    tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensor_f


def column_vector_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()
    # calculate column vector attention
    spatial1 = column_vector_attention(tensor1, spatial_type)
    spatial2 = column_vector_attention(tensor2, spatial_type)

    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w1 = spatial_w1.to(device)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.to(device)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f


# row vector_attention
def row_vector_attention(tensor, pooling_type="l1_mean"):
    # global pooling
    shape = tensor.size()

    c = shape[1]
    h = shape[2]
    w = shape[3]
    channel = torch.zeros(1, c, 1, 1)
    if pooling_type is"l1_mean":
        channel = torch.norm(tensor, p=1, dim=[2, 3], keepdim=True) / (h * w)
    elif pooling_type is"l2_mean":
        channel = torch.norm(tensor, p=2, dim=[2, 3], keepdim=True) / (h * w)
    elif pooling_type is "linf":
            # for i in range(c):
            #     tensor_1 = tensor[0,i,:,:]
            #     channel[0,i,0,0] = torch.max(tensor_1)
            ndarray = tensor.cpu().numpy()
            max = np.amax(ndarray,axis=(2,3))
            tensor = torch.from_numpy(max)
            channel = tensor.reshape(1,c,1,1)
            channel = channel.to(device)
    return channel


# # column vector attention
def column_vector_attention(tensor, spatial_type='l1_mean'):
    spatial = torch.zeros(1, 1, 1, 1)

    shape = tensor.size()
    c = shape[1]
    h = shape[2]
    w = shape[3]

    if spatial_type is 'l1_mean':
        spatial = torch.norm(tensor, p=1, dim=[1], keepdim=True) / c
    elif spatial_type is"l2_mean":
        spatial = torch.norm(tensor, p=2, dim=[1], keepdim=True) / c
    elif spatial_type is "linf":
        spatial, indices = tensor.max(dim=1, keepdim=True)
        spatial = spatial / c
        spatial = spatial.to(device)
    return spatial

