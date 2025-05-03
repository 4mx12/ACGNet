
class DefaultConfig(object):


    data_root = r'./train/train'
    # data_root = '/home/yuzhang/experiment/hzf/DPDNN_PyTorch-master/data/RS_data'
    # label_root = './data/label_128'
    num_data = 20000
    crop_size = 128
    noise_level = 8
    batch_size = 16  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 1  # how many workers for loading data

    max_epoch = 50
    lr = 0.0005
    lr_decay = 0.5
    teacher_model_path = r'./checkpoints/model_201_sigma50.pth'
    load_model_path = r'./checkpoints/base_L%d.pth'%noise_level
    save_model_path = r'./checkpoints/base_L%d_no256_no_dega.pth'%noise_level

  #  load_model_path = '/home/yuzhang/experiment/hzf/DPDNN_PyTorch-master/DENOISE/checkpoints/DPDNN_denoischangenew_sigma%d.pth'% noise_level
   # save_model_path = '/home/yuzhang/experiment/hzf/DPDNN_PyTorch-master/DENOISE/checkpoints/DPDNN_denoischangenew_sigma%d.pth'% noise_level

opt = DefaultConfig()
#之前是加了BN层的


























