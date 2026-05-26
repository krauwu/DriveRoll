"""
Dynamic sampling evaluation script - reuses UniAD test pipeline

Usage:
    # 2Hz sampling test
    python run_dynamic_eval.py ./adzoo/uniad/configs/stage2_e2e/base_e2e_12hz.py ./ckpts/uniad_base_e2e.pth --fps 2

    # Test by scene
    python run_dynamic_eval.py ./adzoo/uniad/configs/stage2_e2e/base_e2e_12hz.py ./ckpts/uniad_base_e2e.pth --scene-idx 0
"""

import argparse
import torch
import os
import warnings
import pickle
import time
import os.path as osp
from torch.nn.parallel.distributed import DistributedDataParallel
from mmcv.utils import get_dist_info, init_dist, wrap_fp16_model, set_random_seed, Config, load_checkpoint
from mmcv.fileio.io import dump
from mmcv.datasets import build_dataloader, replace_ImageToTensor
from mmcv.models import build_model
from mmcv.datasets.nuscenes_e2e_dataset import NuScenesE2EDataset
from adzoo.uniad.test_utils import custom_multi_gpu_test

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Dynamic sampling evaluation')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Model checkpoint path')
    parser.add_argument('--pkl-path', type=str, default=None,
                        help='PKL file path')
    parser.add_argument('--fps', type=int, default=12, choices=[2, 6, 12],
                        help='Target FPS')
    parser.add_argument('--frame-start', type=int, default=0,
                        help='Start frame index')
    parser.add_argument('--frame-end', type=int, default=None,
                        help='End frame index')
    parser.add_argument('--scene-idx', type=int, default=None,
                        help='Scene index')
    parser.add_argument('--out', default='output/results_dynamic.pkl')
    parser.add_argument('--eval', type=str, nargs='+', default=['bbox'])
    parser.add_argument('--launcher', choices=['none', 'pytorch'],
                        default='pytorch', help='Launch method')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    return args


def sample_infos(infos, fps=12, frame_start=0, frame_end=None, scene_idx=None):
    """Sample infos"""
    if scene_idx is not None:
        unique_scenes = list(set(info['scene_token'] for info in infos))
        unique_scenes.sort()
        if scene_idx >= len(unique_scenes):
            scene_idx = 0
        target_scene = unique_scenes[scene_idx]
        infos = [info for info in infos if info['scene_token'] == target_scene]
        print(f"Scene filtering: {len(infos)} frames")

    if fps != 12:
        infos = infos[::(12 // fps)]
        print(f"FPS sampling: {len(infos)} frames")

    if frame_end is None:
        frame_end = len(infos)
    infos = infos[frame_start:frame_end]
    print(f"Frame window: {len(infos)} frames")

    return infos


def build_dataset_from_cfg(cfg, infos, metadata):
    """Build dataset from config using provided infos"""
    dataset_cfg = cfg.data.test
    data_root = dataset_cfg.get('data_root', cfg.data_root)

    return NuScenesE2EDataset(
        ann_file=dataset_cfg.ann_file,
        pipeline=dataset_cfg.pipeline,
        data_root=data_root,
        test_mode=True,
        infos=infos,
        metadata=metadata,
        queue_length=dataset_cfg.get('queue_length', 4),
        bev_size=dataset_cfg.get('bev_size', (200, 200)),
        patch_size=dataset_cfg.get('patch_size', (102.4, 102.4)),
        canvas_size=dataset_cfg.get('canvas_size', (200, 200)),
        overlap_test=dataset_cfg.get('overlap_test', False),
        predict_steps=dataset_cfg.get('predict_steps', 12),
        planning_steps=dataset_cfg.get('planning_steps', 6),
        past_steps=dataset_cfg.get('past_steps', 4),
        fut_steps=dataset_cfg.get('fut_steps', 4),
        use_nonlinear_optimizer=dataset_cfg.get('use_nonlinear_optimizer', False),
        lane_ann_file=dataset_cfg.get('lane_ann_file', None),
        eval_mod=dataset_cfg.get('eval_mod', None),
        file_client_args=dataset_cfg.get('file_client_args', dict(backend='disk')),
        load_interval=1,  # sampling already done in sample_infos
        with_velocity=dataset_cfg.get('with_velocity', True),
        modality=dataset_cfg.get('modality', None),
        box_type_3d=dataset_cfg.get('box_type_3d', 'LiDAR'),
        filter_empty_gt=dataset_cfg.get('filter_empty_gt', True),
        use_valid_flag=dataset_cfg.get('use_valid_flag', False),
    )


def main():
    args = parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    # Initialize distributed
    distributed = args.launcher != 'none'
    if distributed:
        torch.backends.cudnn.benchmark = True
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()

    # Load pkl and sample
    pkl_path = args.pkl_path or cfg.data.test.ann_file
    print(f"Loaded pkl: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    infos = sample_infos(
        data['infos'], args.fps, args.frame_start, args.frame_end, args.scene_idx)
    metadata = data.get('metadata', {'version': 'interp_12Hz_trainval'})

    # Build dataset (using provided infos)
    dataset = build_dataset_from_cfg(cfg, infos, metadata)
    print(f"Dataset: {len(dataset)} samples")

    # Dataloader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # Model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # Distributed
    model = DistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
    )

    # Test
    outputs = custom_multi_gpu_test(model, data_loader, None, False)

    # Save and evaluate
    if not distributed or rank == 0:
        print(f'\nSave results to {args.out}')
        dump(outputs, args.out)

        kwargs = {'jsonfile_prefix': osp.join('test', f'dynamic_fps{args.fps}', time.ctime().replace(' ', '_'))}
        eval_kwargs = cfg.get('evaluation', {}).copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()