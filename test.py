import argparse
from mmengine.config import Config
from mmengine.runner import Runner
import os
from datasets.metrics import TrackAccuracy
from mmengine.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('--config', default='configs/voxel/kitti/car.py', help='train config file path')#20241109，配置文件
    # parser.add_argument('--config', default='configs/voxel/kitti/ped.py', help='train config file path')
    # parser.add_argument('--resume_from', default=None, help='path to the checkpoint file')
    # parser.add_argument(
    #     '--load_from', default='pretrained/voxel/kitti/car.pth',
    #     help='train config file path')
    parser.add_argument(
        '--load_from', default='/home/ly/P2P-main/output/epoch_221.pth',#读取与训练模型
        help='train config file path')
    parser.add_argument(
        '--launcher',#作业启动其
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)#分布式训练所需参数
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

args = parse_args()
cfg = Config.fromfile(args.config)

metric = TrackAccuracy()
evaluator = Evaluator(metric)

runner = Runner(model=cfg.model,
                load_from=args.load_from,
                default_hooks=cfg.default_hooks,
                env_cfg=cfg.env_cfg,
                #work_dir='./work_dir',
                work_dir='/home/ly/P2P-main/outputtest',
                test_dataloader=cfg.test_dataloader,
                test_evaluator=evaluator,
                test_cfg=cfg.test_cfg,
                launcher=args.launcher,
                cfg=dict())

runner.test()
