import random
import numpy as np
import torch
from PIL import Image
from args import args
import imageio
import os
import cv2
from torchvision import transforms
from os import listdir
from os.path import join
from imageio import imsave

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images

def load_dataset(original_ir_path, original_vi_path, BATCH_SIZE, num_imgs=None):
    ir_path = list_images(original_ir_path)
    if num_imgs is None:
        num_imgs = len(ir_path)
    ir_path = ir_path[:num_imgs]
    # random
    random.shuffle(ir_path)
    vi_path = []
    for i in range(len(ir_path)):
        ir = ir_path[i]
        vis = ir.replace('ir', 'vi')
        vi_path.append(vis)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_ir_path = original_ir_path[:-mod]
    batches = int(len(original_ir_path) // BATCH_SIZE)
    return ir_path, vi_path, batches

def get_train_images_auto(paths, height=128, width=128, mode='L'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    images = images / 255
    return images

def get_image(path, height=128, width=128, mode='L'):
    global image
    if mode == 'L':
        image = cv2.imread(path, 0)
    elif mode == 'RGB':
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if height is not None and width is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image

def RGB2YCbCr(RGB_image):
    ## RGB_image [-1, 1]
    test_num1 = 16.0 / 255.0
    test_num2 = 128.0 / 255.0
    R = RGB_image[:, :, :, 0:1]
    G = RGB_image[:, :, :, 1:2]
    B = RGB_image[:, :, :, 2:3]
    Y = 0.257 * R + 0.564 * G + 0.098 * B + test_num1
    Cb = - 0.148 * R - 0.291 * G + 0.439 * B + test_num2
    Cr = 0.439 * R - 0.368 * G - 0.071 * B + test_num2
    return Y, Cb, Cr

def get_test_image(paths, height=None, width=None):
    global h, w, c
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = imageio.imread(path)
        if height is not None and width is not None:
            image = np.array(Image.fromarray(image).resize([height, width]))

        h = image.shape[0]
        w = image.shape[1]
        c = 1
        # image = (image - 127.5) / 127.5
        image = image / 255
        image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        images.append(image)
        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).float()
    return images, h, w, c

def get_test_images(paths, height=None, width=None, mode='L'):
    global image
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        w, h = image.shape[0], image.shape[1]
        w_s = 128 - w % 128
        h_s = 128 - h % 128
        image = cv2.copyMakeBorder(image, 0, w_s, 0, h_s, cv2.BORDER_CONSTANT,value=256)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    images = images/ 255
    return images

def YCbCr2RGB(Y, Cb, Cr, mode=0):
    ## Y, Cb, Cr :[-1, 1]
    test_num1 = 16.0 / 255.0
    test_num2 = 128.0 / 255.0
    R = 1.164 * (Y - test_num1) + 1.596 * (Cr - test_num2)
    G = 1.164 * (Y - test_num1) - 0.392 * (Cb - test_num2) - 0.813 * (Cr - test_num2)
    B = 1.164 * (Y - test_num1) + 2.017 * (Cb - test_num2)
    RGB_image = torch.concat([R, G, B], dim=-1)
    BGR_image = torch.concat([B, G, R], dim=-1)
    if mode == 1:
        return RGB_image
    else:
        return BGR_image

def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    # img_fusion = img_fusion * 127.5 + 127.5
    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    img_fusion = img_fusion.reshape([1,img_fusion.shape[0], img_fusion.shape[1]])#有些出来的是二维，需要变成3维
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    imageio.imwrite(output_path, img_fusion)

def save_image_scales(img_fusion, output_path):
    img_fusion = img_fusion.float()
    img_fusion = img_fusion.cpu().data[0].numpy()
    imageio.imwrite(output_path, img_fusion)

def make_floor(path1,path2):
    path = os.path.join(path1,path2)
    if os.path.exists(path) is False:
        os.makedirs(path)
    return path

def save_feat(index,C,ir_atten_feat,vi_atten_feat,result_path):
    ir_atten_feat = ir_atten_feat * 255
    vi_atten_feat = vi_atten_feat * 255

    ir_feat_path = make_floor(result_path, "ir_feat")
    index_irfeat_path = make_floor(ir_feat_path, str(index))

    vi_feat_path = make_floor(result_path, "vi_feat")
    index_vifeat_path = make_floor(vi_feat_path, str(index))

    for c in range(C):
        ir_temp = ir_atten_feat[:, c, :, :].squeeze()
        vi_temp = vi_atten_feat[:, c, :, :].squeeze()

        feat_ir = ir_temp.cpu().clamp(0, 255).data.numpy()
        feat_vi = vi_temp.cpu().clamp(0, 255).data.numpy()

        ir_feat_filenames = 'ir_feat_C' + str(c) + '.png'
        ir_atten_path = index_irfeat_path + '/' + ir_feat_filenames
        imsave(ir_atten_path, feat_ir)

        vi_feat_filenames = 'vi_feat_C' + str(c) + '.png'
        vi_atten_path = index_vifeat_path + '/' + vi_feat_filenames
        imsave(vi_atten_path, feat_vi)

def save_images(path, data, out):
    w, h = out.shape[0], out.shape[1]
    if data.shape[1] == 1:
        data = data.reshape([data.shape[2], data.shape[3]])
    ori = data[0:w, 0:h]
    # ori = ori *255
    cv2.imwrite(path, ori)

def imsave1(img, img_path,out):
    img = np.squeeze(img)
    w, h = out.shape[0], out.shape[1]
    print(img.shape)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    img = img[0:w, 0:h]
    cv2.imwrite(img_path, img)