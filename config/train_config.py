import argparse
import os.path as osp

root_dir = '/home/oguz/workspace/projects/arcface-pfe-torch'
lfw_dir = osp.join(root_dir, 'data/lfw')
casia_dir = osp.join(root_dir, 'data/casia')
cp_dir = '/home/oguz/workspace/projects/arcface-pfe-torch/checkpoints'


def training_args():
    parser = argparse.ArgumentParser(description='PyTorch metricface')

    # -- env
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0])
    parser.add_argument('--workers', type=int, default=12)

    # -- model
    parser.add_argument('--in_size', type=tuple, default=(112, 112))  # FIXED
    parser.add_argument('--backbone', type=str, default='mobilenet')
    parser.add_argument('--in_feats', type=int, default=512)
    parser.add_argument('--classnum', type=int, default=10574)  # CASIA (10574)

    # fine-tuning
    parser.add_argument('--resume', type=str,
                        default='')  # checkpoint
    parser.add_argument('--fine_tuning', type=bool, default=False)  # just fine-tuning
    parser.add_argument('--freeze_backbone', type=bool, default=True)

    # -- optimizer
    parser.add_argument('--start_epoch', type=int, default=1)  #
    parser.add_argument('--end_epoch', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_face_pb', type=int, default=4)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--lr_adjust', type=list, default=[2, 3, 4])
    parser.add_argument('--gamma', type=float, default=0.1)  # FIXED
    parser.add_argument('--weight_decay', type=float, default=5e-4)  # FIXED

    # -- dataset
    parser.add_argument('--casia_dir', type=str, default=casia_dir)
    parser.add_argument('--lfw_dir', type=str, default=osp.join(lfw_dir, 'align_112_112'))
    parser.add_argument('--train_file', type=str, default=osp.join(root_dir, 'data/list_casia_mtcnncaffe_aligned_nooverlap.txt'))
    parser.add_argument('--pairs_file', type=str, default=osp.join(lfw_dir, 'anno_file/pairs.txt'))
    parser.add_argument('--try_times', type=int, default=5)

    # -- verification
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--thresh_iv', type=float, default=0.005)

    # -- save or print
    parser.add_argument('--is_debug', type=str, default=False)
    parser.add_argument('--save_to', type=str, default=osp.join(cp_dir, 'pfe'))
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=1)

    args = parser.parse_args()

    return args