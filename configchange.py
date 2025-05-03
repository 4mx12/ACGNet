
class DefaultConfig(object):


    data_root = r'./train/train'

    num_data = 20000
    crop_size = 128
    noise_level = 1
    batch_size = 16  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 1  # how many workers for loading data

    max_epoch = 50
    lr = 0.0005
    lr_decay = 0.5

    load_model_path = r'./checkpoints/base_L%d.pth'%noise_level
    save_model_path = r'./checkpoints/base_L%d_no256_no_dega.pth'%noise_level



opt = DefaultConfig()



























