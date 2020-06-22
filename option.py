import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='TTSR')

### log setting
parser.add_argument('--save_dir', type=str, default='save_dir',
                    help='Directory to save log, arguments, models and images')
parser.add_argument('--reset', type=str2bool, default=False,
                    help='Delete save_dir to create a new one')
parser.add_argument('--log_file_name', type=str, default='TTSR.log',
                    help='Log file name')
parser.add_argument('--logger_name', type=str, default='TTSR',
                    help='Logger name')

### device setting
parser.add_argument('--cpu', type=str2bool, default=False,
                    help='Use CPU to run code')
parser.add_argument('--num_gpu', type=int, default=1,
                    help='The number of GPU used in training')

### dataset setting
parser.add_argument('--dataset', type=str, default='CUFED',
                    help='Which dataset to train and test')
parser.add_argument('--dataset_dir', type=str, default='/home/v-fuyang/Data/CUFED/',
                    help='Directory of dataset')

### dataloader setting
parser.add_argument('--num_workers', type=int, default=4,
                    help='The number of workers when loading data')

### model setting
parser.add_argument('--num_res_blocks', type=str, default='16+16+8+4',
                    help='The number of residual blocks in each stage')
parser.add_argument('--n_feats', type=int, default=64,
                    help='The number of channels in network')
parser.add_argument('--res_scale', type=float, default=1.,
                    help='Residual scale')

### loss setting
parser.add_argument('--GAN_type', type=str, default='WGAN_GP',
                    help='The type of GAN used in training')
parser.add_argument('--GAN_k', type=int, default=2,
                    help='Training discriminator k times when training generator once')
parser.add_argument('--tpl_use_S', type=str2bool, default=False,
                    help='Whether to multiply soft-attention map in transferal perceptual loss')
parser.add_argument('--tpl_type', type=str, default='l2',
                    help='Which loss type to calculate gram matrix difference in transferal perceptual loss [l1 / l2]')
parser.add_argument('--rec_w', type=float, default=1.,
                    help='The weight of reconstruction loss')
parser.add_argument('--per_w', type=float, default=0,
                    help='The weight of perceptual loss')
parser.add_argument('--tpl_w', type=float, default=0,
                    help='The weight of transferal perceptual loss')
parser.add_argument('--adv_w', type=float, default=0,
                    help='The weight of adversarial loss')

### optimizer setting
parser.add_argument('--beta1', type=float, default=0.9,
                    help='The beta1 in Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='The beta2 in Adam optimizer')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='The eps in Adam optimizer')
parser.add_argument('--lr_rate', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--lr_rate_dis', type=float, default=1e-4,
                    help='Learning rate of discriminator')
parser.add_argument('--lr_rate_lte', type=float, default=1e-5,
                    help='Learning rate of LTE')
parser.add_argument('--decay', type=float, default=999999,
                    help='Learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='Learning rate decay factor for step decay')

### training setting
parser.add_argument('--batch_size', type=int, default=9,
                    help='Training batch size')
parser.add_argument('--train_crop_size', type=int, default=40,
                    help='Training data crop size')
parser.add_argument('--num_init_epochs', type=int, default=2,
                    help='The number of init epochs which are trained with only reconstruction loss')
parser.add_argument('--num_epochs', type=int, default=1,
                    help='The number of training epochs')
parser.add_argument('--print_every', type=int, default=1,
                    help='Print period')
parser.add_argument('--save_every', type=int, default=999999,
                    help='Save period')
parser.add_argument('--val_every', type=int, default=999999,
                    help='Validation period')

### evaluate / test / finetune setting
parser.add_argument('--eval', type=str2bool, default=False,
                    help='Evaluation mode')
parser.add_argument('--eval_save_results', type=str2bool, default=False,
                    help='Save each image during evaluation')
parser.add_argument('--model_path', type=str, default=None,
                    help='The path of model to evaluation')
parser.add_argument('--test', type=str2bool, default=False,
                    help='Test mode')
parser.add_argument('--lr_path', type=str, default='./test/demo/lr/lr.png',
                    help='The path of input lr image when testing')
parser.add_argument('--ref_path', type=str, default='./test/demo/ref/ref.png',
                    help='The path of ref image when testing')

args = parser.parse_args()
