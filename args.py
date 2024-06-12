class args():
    # training args
    g1 = 0.5
    epochs = 8 # "number of training epochs, default is 2"
    batch_size = 8  # "batch size for training, default is 4"

    # URL:
    # data_ir_set = 'E:clip picture last 128/ir/'
    # data_vi_set = 'E:clip picture last 128/vi/'

    data_ir_set = 'E:\clip picture last re128 M3FD\ir'
    data_vi_set = 'E:\clip picture last re128 M3FD/vi'

    save_model_dir = "models"  # "path to folder where trained model will be saved."
    save_loss_dir = "models/loss"  # "path to folder where trained model will be saved."

    HEIGHT = 128 #用于get train image
    WIDTH = 128

    image_size = 512
    crop_stride = 64

    cuda = 1
    # log_interval = 1 # "number of images after which the training loss is logged, default is 10"
    # log_iter = 1
    resume = None

    ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']
    alpha = 0.5
    beta = 0.5
    gama = 1
    yita = 1
    deta = 1

    weight_SSIM = 1
    weight_Texture = 20
    weight_Intensity = 4

    lr = 1e-4  # "learning rate, default is 0.0001"
    lr_d = 1e-4
    # lr_light = 1e-4  # "learning rate, default is 0.001"
    log_interval = 1  # "number of images after which the training loss is logged, default is 500"

    trans_model_path = None
    is_para = False

    log_iter=1