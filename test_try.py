import argparse
from mmengine.config import Config
from mmengine.runner import Runner
import os
from datasets.metrics import TrackAccuracy
from mmengine.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description='Test a 3D detector')
    parser.add_argument('--config', default='configs/voxel/kitti/ped.py', help='test config file path')
    parser.add_argument('--resume_from', default=None, help='path to the checkpoint file')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--phase', default='test', choices=['train', 'val', 'test'], help='execution phase')
    # 添加 --data-root 参数
    parser.add_argument('--data-root', default=None, help='root directory of the dataset')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

args = parse_args()
cfg = Config.fromfile(args.config)

# 更新配置文件中的数据集根路径
if args.data_root is not None:
    cfg.dataset_root = args.data_root

metric = TrackAccuracy()
evaluator = Evaluator(metric)

load_from = args.resume_from if args.phase == 'test' and args.resume_from else None

runner = Runner(model=cfg.model,
                load_from=load_from,
                default_hooks=cfg.default_hooks,
                env_cfg=cfg.env_cfg,
                work_dir='/home/ly/P2P-main/outputtest',
                test_dataloader=cfg.test_dataloader,
                test_evaluator=evaluator,
                test_cfg=cfg.test_cfg,
                launcher=args.launcher,
                cfg=dict())

if args.phase == 'test':
    runner.test()