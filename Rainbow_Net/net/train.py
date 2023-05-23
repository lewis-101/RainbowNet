import argparse, os, torch, json
from RainNet import RainNet


"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-c', '--config', default='configs/places2.json', type=str, required=False)
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed'],
                        help='The name of dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=28, help='The size of input image')
    parser.add_argument('--gpu_mode', type=bool, default=True)

    # return check_args(parser.parse_args())
    return parser.parse_args()

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args









if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    # loading configs
    config = json.load(open(args.config))
    config['distributed'] = True
    config['world_size'] = 1
    config['global_rank'] = 0
    trainer = RainNet(config, args)
    # trainer._train_epoch()
    trainer.train()





