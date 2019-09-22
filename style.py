#coding=utf-8

#TRIAN STYLE_MODEL
#checkpoint路径+学习风格
#style.py —checkpoint-dir ./model/ —style ./style/wave.jpg
import sys, os, pdb
from src.optimize import optimize
#命令行解析
from argparse import ArgumentParser
from src.utils import save_img, get_img, exists, list_files

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2 #让最后图像看起来更真一些

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_ITERATIONS = 2000
VGG_PATH = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH = 'data/train2014'
BATCH_SIZE = 4
DEVICE = '/gpu:0'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in')

    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--slow', dest='slow', action='store_true',
                        help='gatys\' approach (for debugging, not supported)',
                        default=False)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    return parser

def check_opts(opts):
    exists(opts.checkpoint_dir, "checkpoint dir not found!")
    exists(opts.style, "style path not found!")
    exists(opts.train_path, "train path not found!")
    exists(opts.vgg_path, "vgg network data not found!")
    assert opts.epochs > 0 and opts.batch_size > 0 and opts.checkpoint_iterations > 0 \
    and os.path.exists(opts.vgg_path) and opts.content_weight >= 0 and opts.style_weight >= 0 \
    and opts.tv_weight >= 0 and opts.learning_rate >= 0

def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options) #check参数有没有问题

    style_target = get_img(options.style)
    content_targets = list_files(options.train_path)
    #指定参数
    kwargs = {
        "epochs":options.epochs,
        "print_iterations":options.checkpoint_iterations,
        "batch_size": options.batch_size,
        "checkpoint_dir": os.path.join(options.checkpoint_dir, 'fns.ckpt'),
        "learning_rate": options.learning_rate
    }

    args = [
        content_targets,
        style_target,
        options.content_weight,
        options.style_weight,
        options.tv_weight,
        options.vgg_path
    ]

    #注意for循环, i = iterations
    for preds, losses, i, epoch in optimize(*args, **kwargs):
        #提出来单个
        #need more than 3 values to unpack
        style_loss, content_loss, tv_loss, loss = losses

        print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i ,loss))
        to_print = (style_loss, content_loss, tv_loss)
        print('style: %s, content: %s, tv: %s' % to_print)

if __name__ == '__main__':
    main()


