import os
import time
from tqdm import tqdm, trange
import scipy.io as scio
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net import net
from args import args
import loss
from utils import make_floor

EPSILON = 1e-5
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    original_ir_path = args.data_ir_set
    original_vi_path = args.data_vi_set
    batch_size = args.batch_size
    train(original_ir_path, original_vi_path, batch_size)

def train(original_ir_path, original_vi_path, batch_size):

    models_save_path = make_floor(os.getcwd(), args.save_model_dir)
    print(models_save_path)
    model = net().cuda()
    g_content_criterion = loss.g_content_loss().cuda()
    if args.resume is not None:
        print('Resuming, initializing using weight from {}.'.format(args.resume))
        model.load_state_dict(torch.load(args.resume))

    optimizer = Adam(model.parameters(), args.lr)
    grad = loss.grad().cuda()

    if args.cuda:
        model.cuda()
        grad.cuda()

    tbar = trange(args.epochs) # 主要目的在于显示进度条
    print('Start training.....')

    count_loss = 0
    Loss_SSIM = []
    Loss_Texture =[]
    Loss_Indensity = []
    Loss_all = []

    for e in tbar:
        print('Epoch %d.....' % e)
        # torch.cuda.empty_cache()  # 释放显存
        image_set_ir, image_set_vi, batches = utils.load_dataset(original_ir_path, original_vi_path, batch_size)
        model.train()
        count = 0
        batches = int(len(image_set_ir) // batch_size) #主要是不确定他是否必须，先留一下
        for batch in range(batches):
            image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]#列表里取8张图
            image_paths_vi = image_set_vi[batch * batch_size:(batch * batch_size + batch_size)]
            ir = utils.get_train_images_auto(image_paths_ir, height=args.HEIGHT, width=args.WIDTH)
            vi = utils.get_train_images_auto(image_paths_vi, height=args.HEIGHT, width=args.WIDTH)
            count += 1
            optimizer.zero_grad()
            ir = Variable(ir, requires_grad=False)
            vi = Variable(vi, requires_grad=False)
            if args.cuda:
                ir = ir.cuda()
                vi = vi.cuda()
            # get fusion image

            ir0t, vi0t, ir1t, vi1t, ir2t, vi2t, ir3t, vi3t = model.encoder(ir,vi)

            fusion1 = model.fusion1(ir0t,vi0t)
            fusion2 = model.fusion2(ir1t, vi1t)
            fusion3 = model.fusion3(ir2t, vi2t)
            fusion4 = model.fusion4(ir3t, vi3t)

            outputs = model.decoder(fusion1,fusion2,fusion3,fusion4)

            img_ir = Variable(ir.data.clone(), requires_grad=False)
            img_vi = Variable(vi.data.clone(), requires_grad=False)

            SSIM_loss_value = 0.
            Texture_loss_value = 0.
            Intensity_loss_value = 0.
            all_Texture_loss =0.
            all_SSIM_loss = 0.
            all_intensity_loss = 0.
            all_total_loss = 0.

            total_loss,SSIM_loss,Texture_loss,Intensity_loss = g_content_criterion(img_ir,img_vi,outputs)

            all_SSIM_loss += SSIM_loss.item()
            all_Texture_loss += Texture_loss.item()
            all_intensity_loss += Intensity_loss.item()
            all_total_loss += total_loss.item()

            total_loss.backward()
            optimizer.step()

            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\t SSIM LOSS: {:.6f}\t Texture LOSS: {:.6f}\t Intensity LOSS: {:.6f}\t total: {:.6f}".format(
                    time.ctime(), e + 1, count, batches,
                                all_SSIM_loss / args.log_interval,
                                all_Texture_loss / args.log_interval,
                                all_intensity_loss / args.log_interval,
                                all_total_loss / args.log_interval
                )
                tbar.set_description(mesg)
                Loss_SSIM.append(all_SSIM_loss / args.log_interval)
                Loss_Texture.append(all_Texture_loss / args.log_interval)
                Loss_Indensity.append(all_intensity_loss / args.log_interval)
                Loss_all.append(all_total_loss / args.log_interval)
                count_loss = count_loss + 1

        if (e+1) % args.log_interval == 0:
            # save model
            model.eval()
            model.cuda()
            STfuse_model_filename = "fuse_Epoch_" + str(e) + ".model"
            STfuse_model_path = os.path.join(args.save_model_dir, STfuse_model_filename)
            torch.save(model.state_dict(), STfuse_model_path)

    # SSIM loss
    loss_data_SSIM = Loss_SSIM
    loss_filename_path = 'final_SSIM.mat'
    scio.savemat(loss_filename_path, {'final_loss_SSIM': loss_data_SSIM})

    # Indensity loss
    loss_data_Indensity = Loss_Indensity
    loss_filename_path = "final_Indensity.mat"
    scio.savemat(loss_filename_path, {'final_loss_Indensity': loss_data_Indensity})

    # Indensity loss
    loss_data_Texture = Loss_Texture
    loss_filename_path = "final_Texture.mat"
    scio.savemat(loss_filename_path, {'final_loss_Texture': loss_data_Texture})

    loss_data = Loss_all
    loss_filename_path = "final_all.mat"
    scio.savemat(loss_filename_path, {'final_loss_all': loss_data})

    # save model
    model.eval()
    model.cpu()
    save_model_filename = "final_epoch.model"
    torch.save(model.state_dict(), save_model_filename)

    print("\nDone, trained model saved at", save_model_filename)


if __name__ == "__main__":
    main()
