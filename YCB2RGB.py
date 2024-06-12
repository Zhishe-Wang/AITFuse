import os
import utils
import torch
from torch.autograd import Variable

def run_demo(Y_path,Cb_path,Cr_path,output_path_root, index):

    Y_img = utils.get_test_images(Y_path, height=None, width=None, mode='L')
    Y_img = Y_img.permute(0, 2, 3, 1)
    cb_img = utils.get_test_images(Cb_path, height=None, width=None, mode='L')
    cb_img = cb_img.permute(0, 2, 3, 1)
    cr_img = utils.get_test_images(Cr_path, height=None, width=None, mode='L')
    cr_img = cr_img.permute(0, 2, 3, 1)

    out = utils.get_image(Y_path, height=None, width=None)

    img = utils.YCbCr2RGB(Y_img,cb_img,cr_img)

    file_name = 'output_' + str(index) + '.png'
    output_path = output_path_root + file_name

    if torch.cuda.is_available():
        img = img.cpu().clamp(0, 255).numpy()
    else:
        img = img.clamp(0, 255).numpy()
    img = img * 255

    utils.imsave1(img, output_path, out)
    print(output_path)


def main():
    Y_path = "./images/MSRS/Ours/"
    Cb_path = "./images/MSRS/Cb/"
    Cr_path = "./images/MSRS/Cr/"
    # Cb_path = "./images/FMB/test/Cb/"
    # Cr_path = "./images/FMB/test/Cr/"
    # Cb_path = "C:/ZZQ/img/FMB/train/Cb/"
    # Cr_path = "C:/ZZQ/img/FMB/train/Cr/"

    # Y_path = "outputs_YCB/M3FD4200Y+IR/densefuse/"
    # Cb_path = "outputs_YCB/M3FD4200Cb/"
    # Cr_path = "outputs_YCB/M3FD4200Cr/"

    output_path = './outputs_RGB/'

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)


    with torch.no_grad():
        for i in range(1000, 1400):
            index = i + 1

            Y1_path = Y_path + 'Transformer' + str(index) + '.png'
            Cb1_path = Cb_path + 'Cb_' + str(index) + '.png'
            Cr1_path = Cr_path + 'Cr_' + str(index) + '.png'
            # Cb1_path = Cb_path + 'Cbs_' + str(index) + '.png'
            # Cr1_path = Cr_path + 'Crs_' + str(index) + '.png'
            run_demo(Y1_path,Cb1_path,Cr1_path,output_path, index)

    print('Done......')

if __name__ == "__main__":
    main()