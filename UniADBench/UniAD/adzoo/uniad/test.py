import argparse
import torch
import os
import warnings
from torch.nn.parallel.distributed import DistributedDataParallel
from mmcv.utils import get_dist_info, init_dist, wrap_fp16_model, set_random_seed, Config, DictAction, load_checkpoint
from mmcv.fileio.io import dump
from mmcv.datasets import build_dataset, build_dataloader, replace_ImageToTensor
from mmcv.models import build_model, fuse_conv_bn
import time
import os.path as osp
from adzoo.uniad.test_utils import custom_multi_gpu_test, custom_single_gpu_test


warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='output/results.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        torch.backends.cudnn.benchmark = True
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()

    set_random_seed(args.seed, deterministic=args.deterministic)

    # Dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset,
                                    samples_per_gpu=samples_per_gpu,
                                    workers_per_gpu=cfg.data.workers_per_gpu,
                                    dist=distributed,
                                    shuffle=False,
                                    nonshuffler_sampler=cfg.data.nonshuffler_sampler,
                                    )

    # Model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    
    # Add classese info
    if 'CLASSES' in checkpoint.get('meta', {}): # for det
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):  # for seg
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE

    if not distributed:
        assert False #TODO(yzj)
        # model = MMDataParallel(model, device_ids=[0])
        # outputs = custom_single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = DistributedDataParallel(model.cuda(),
                                        device_ids=[torch.cuda.current_device()],
                                        broadcast_buffers=False,
                                        )
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            dump(outputs, args.out)
        kwargs = {}
        # jsonfile_prefix placed under test/{config_name}/{timestamp} in same directory as --out
        config_name = args.config.split('/')[-1].split('.')[-2]
        timestamp = time.ctime().replace(' ', '_').replace(':', '_')
        if args.out:
            _out_dir = osp.dirname(osp.dirname(args.out))  # parent directory of results
            kwargs['jsonfile_prefix'] = osp.join(_out_dir, 'test', config_name, timestamp)
        else:
            kwargs['jsonfile_prefix'] = osp.join('test', config_name, timestamp)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            eval_results = dataset.evaluate(outputs, **eval_kwargs)
            print(eval_results)
            # Save complete evaluation results to json
            import json as _json
            _save_dict = {}
            for _k, _v in eval_results.items():
                if hasattr(_v, 'item'):
                    _v = _v.item()
                try:
                    _json.dumps(_v)
                    _save_dict[_k] = _v
                except (TypeError, ValueError):
                    _save_dict[_k] = str(_v)
            _eval_json_path = osp.join(osp.dirname(kwargs['jsonfile_prefix']), 'eval_results.json')
            with open(_eval_json_path, 'w') as _f:
                _json.dump(_save_dict, _f, indent=2)
            print(f"Eval results saved to {_eval_json_path}")


if __name__ == '__main__':
    main()
