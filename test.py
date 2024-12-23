# test phase
import os
import torch
import torchvision.models as models
# from flops_counter import

from torchsummary import summary
from torchstat import stat
from torch.autograd import Variable
import utils
import numpy as np
import time
from net import net
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model(path):
    fuse_net = net()

    fuse_net.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in fuse_net.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(fuse_net._get_name(), para * type_size / 1000 / 1000))

    fuse_net.eval()
    fuse_net.cuda()
    return fuse_net


def _generate_fusion_image(model, vi, ir):

    ir0t, vi0t, ir1t, vi1t, ir2t, vi2t, ir3t, vi3t = model.encoder(ir, vi)

    fusion1 = model.fusion1(ir0t, vi0t)
    fusion2 = model.fusion2(ir1t, vi1t)
    fusion3 = model.fusion3(ir2t, vi2t)
    fusion4 = model.fusion4(ir3t, vi3t)

    outputs = model.decoder(fusion1, fusion2, fusion3, fusion4)

    return outputs


def run_demo(model, vi_path, ir_path, output_path_root, index):
    vi_img = utils.get_test_images(vi_path, height=None, width=None)
    ir_img = utils.get_test_images(ir_path, height=None, width=None)

    out = utils.get_image(vi_path, height=None, width=None)

    vi_img = vi_img.cuda()
    ir_img = ir_img.cuda()
    vi_img = Variable(vi_img, requires_grad=False)
    ir_img = Variable(ir_img, requires_grad=False)

    img_fusion = _generate_fusion_image(model, vi_img, ir_img)
    ############################ multi outputs ##############################################
    file_name = 'fusion_' + str(index) + '.png'
    output_path = output_path_root + file_name
    if torch.cuda.is_available():
        img = img_fusion.cpu().clamp(0, 255).numpy()
    else:
        img = img_fusion.clamp(0, 255).numpy()
    # img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = img * 255

    utils.save_images(output_path, img, out)
    print(output_path)


def main():
    #TNO
    # vi_path = "images/TNO/vi/"
    # ir_path = "images/TNO/ir/"
    # roadscene
    # vi_path = "images/Roadscene/visual/"
    # ir_path = "images/Roadscene/thermal/"
    # M3FD
    vi_path = "images/M3FD/Y/"
    ir_path = "images/M3FD_Fusion_L/ir/"

    # ir_path = "images\M3FD_Detection\ir/"
    # vi_path = "images\M3FD_Detection/Y/"

    output_path = './outputs/'

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    in_c = 1
    out_c = in_c
    model_path = "./models/fuse_Epoch_6.model"

    with torch.no_grad():

        model = load_model(model_path)

        start = time.time()


        for i in range(1000,1300):
            index = i + 1
            visible_path = vi_path + 'Y_' + str(index) + '.png'
            infrared_path = ir_path + str(index) + '.png'

            # visible_path = vi_path  + str(index) + '.png'
            # infrared_path = ir_path  + str(index) + '.png'


            run_demo(model, visible_path, infrared_path, output_path, index)
        end = time.time()
        print('time:', end - start, 'S')
        print('Done......')


if __name__ == "__main__":
    main()
