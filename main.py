import tensorflow as tf
import argparse
from VPAD import *
from ops import *
from utils import *

def parse_args():
    
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    desc = "Tensorflow implementation of VPAD"
    parser = argparse.ArgumentParser(description = desc)
    
    parser.add_argument('--is_training', default = True, type = boolean_string, help = 'Choose Training|Testing')
    parser.add_argument('--only_critic', default = False, type = boolean_string, help = 'Choose training mode')

    parser.add_argument('--dataset', type = str, default = 'cifar10',
                        choices = ['fashion_mnist', 'cifar10', 'cifar100', 'catdog'], help = 'dataset')
    parser.add_argument('--loss', type=str, default='hinge', choices = ['hinge', 'wasserstein'])
    parser.add_argument('--save_dir', type = str, default = './experiments', help = 'directory for saving results')

    parser.add_argument('--n_z', type=int, default = 128, help = 'Dimension of latent variables')
    parser.add_argument('--inlier_cls', type = int, default = 1, help = 'The class number of normal images[0~10, 0~10, 0~20, 0~1, 0~1]')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'Batch size, shall be a multiple of 4')
    parser.add_argument('--depth', type = int, default = 16, help = 'Hyperparameter(depth) for WRN')
    parser.add_argument('--widen_factor', type = int, default = 8, help = 'Hyperparameter(widen_factor) for WRN')
    parser.add_argument('--training_step', type = int, default = 100, help = 'The number of training_step(X1000)')
    parser.add_argument('--gpu', type=int, default = 0, help = 'GPU Device for tensor operation')

    parser.add_argument('--momentum', type = float, default = 0.9 , help = 'mementum(batch normalization)')
    parser.add_argument('--epsilon', type = float, default = 2e-5, help = 'epsilon(batch normalization)')
    parser.add_argument('--lr', type = float, default = 0.0002  , help = 'Learning rate')
    parser.add_argument('--beta1', type = float, default = 0.0, help = 'Hyperparameter for AdamOptimier')
    parser.add_argument('--beta2', type = float, default = 0.9, help = 'Hyperparameter for AdamOptimier')
    
    return parser.parse_args()

def main():
    args = parse_args()
    if args is None:
        exit()
    print_saver("-"*100, args.save_dir)
    print_arg(args, args.save_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    VPAD_net = VPAD(args, args.inlier_cls)
    VPAD_net.define_dataset()
    VPAD_net.define_placeholder()
    VPAD_net.define_network()
    VPAD_net.define_flow()
    VPAD_net.define_loss()
    VPAD_net.define_optim()
    VPAD_net.define_summary_session()
    VPAD_net.define_dict()
    
    if args.is_training:
        VPAD_net.train()
        print_saver("Training finished!", args.save_dir)
    else:
        VPAD_net.test()
        print_saver("Testing finished!", args.save_dir)

if __name__ =="__main__":
    main()

