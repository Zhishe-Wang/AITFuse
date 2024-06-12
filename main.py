import os
import torch
from torch.autograd import Variable
# from net import FusionModule
# from net_new_spell import FusionModule
#from net_pool import FusionModule
import utils
from New_net import net
from args import args
import numpy as np
import cv2
import time

device = torch.device("cuda"if torch.cuda.is_available()else"cpu")

def load_model(path):
    RAF_model = net()

    RAF_model.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in RAF_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(RAF_model._get_name(), para * type_size / 1000 / 1000))
    # for name, param in RAF_model.named_parameters():
    #     print(name, param)

    RAF_model.eval()
    RAF_model.cuda()

    return RAF_model

def save_images(path, data, out):
    w, h = out.shape[0], out.shape[1]
    if data.shape[1] == 1:
        data = data.reshape([data.shape[2], data.shape[3]])
    ori = data[0:w, 0:h]
    cv2.imwrite(path, ori)

def run_demo(RAF_model, infrared_path, visible_path, output_path_root, index):
    is_testing = 1
    img_ir, h, w, c = utils.get_test_image(infrared_path)
    img_vi, h, w, c = utils.get_test_image(visible_path)

    out = utils.get_image(visible_path, height=None, width=None)
    if args.cuda:
        img_ir = img_ir.cuda()
        img_vi = img_vi.cuda()
    img_ir = Variable(img_ir, requires_grad=False)
    img_vi = Variable(img_vi, requires_grad=False)

    # en_ir = RAF_model.encoder(img_ir)
    # en_vi = RAF_model.encoder(img_vi)
    # concat = RAF_model.attention(en_ir,en_vi)
    # img_fusion_list = RAF_model.decoder(concat,is_testing)

    img_fusion_list = RAF_model.forward(img_ir,img_vi)

    ############################ multi outputs ##############################################
    file_name = 'fusion_' + str(index) + '.png'
    output_path = output_path_root + file_name
    if torch.cuda.is_available():
        img = img_fusion_list.cpu().clamp(0, 255).numpy()
    else:
        img = img_fusion_list.clamp(0, 255).numpy()
    img = img.astype('uint8')
    utils.save_images(output_path, img, out)
    # utils.save_images(output_path, img, out)
    print(output_path)




def main():
    # run demo
    test_path = "D:/IVs_images/"
    output_path = "output"
    model_name ='Final_RAF_Epoch_0.model'

    with torch.no_grad():
        # model_path = args.model_path
        # model_path = os.path.join(os.getcwd(), 'new train/消融实验/seperate', model_name)
        model_path = os.path.join(os.getcwd(), 'models_training', model_name)
        model = load_model(model_path)
        # start = time.time()

        if os.path.exists(output_path) is False:
            os.mkdir(output_path)
        output_path = output_path + '/'

        # TNO
        ir = utils.list_images("C:/Users/image fusion/Desktop/TNO/thermal")
        vis = utils.list_images("C:/Users/image fusion/Desktop/TNO/visual")

        # for a in range(10):
        for i in range(32):
            index = i  + 1
            infrared_path = ir[i]
            visible_path = vis[i]
            run_demo(model, infrared_path, visible_path, output_path, index)


    print('Done......')


if __name__ == '__main__':
    main()
