import os
import os.path as osp
import pickle
import shutil
import tempfile
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mmcv.models.dense_heads.occ_head_plugin import IntersectionOverUnion, PanopticMetric
from mmcv.models.dense_heads.planning_head_plugin import PlanningMetric
from mmcv.utils import ProgressBar, mkdir_or_exist, get_dist_info
from mmcv.fileio.io import load, dump
import numpy as np
import pycocotools.mask as mask_util


# Comment out interpolation function - use official code
# def interpolate_traj(traj, num_points):
#     """
#     Linear interpolation of trajectory
#     Args:
#         traj: (B, T, 2) or (B, T, 3) trajectory tensor
#         num_points: number of points after interpolation
#     Returns:
#         interpolated trajectory (B, num_points, 2) or (B, num_points, 3)
#     """
#     if traj.shape[1] == num_points:
#         return traj
#
#     # traj: (B, T, C) -> (B, C, T) for interpolation
#     B, T, C = traj.shape
#     traj_transposed = traj.permute(0, 2, 1)  # (B, C, T)
#
#     # Use linear interpolation
#     traj_interp = F.interpolate(traj_transposed, size=num_points, mode='linear', align_corners=True)
#
#     # (B, C, num_points) -> (B, num_points, C)
#     traj_interp = traj_interp.permute(0, 2, 1)
#
#     return traj_interp

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()


    enable_plot_mode = os.environ.get('ENABLE_PLOT_MODE', None)
    print(f"ENABLE_PLOT_MODE = {enable_plot_mode}")
    # Occ eval init
    eval_occ = hasattr(model.module, 'with_occ_head') \
                and model.module.with_occ_head
    if eval_occ:
        # 30mx30m, 100mx100m at 50cm resolution
        EVALUATION_RANGES = {'30x30': (70, 130),
                            '100x100': (0, 200)}
        n_classes = 2
        iou_metrics = {}
        for key in EVALUATION_RANGES.keys():
            iou_metrics[key] = IntersectionOverUnion(n_classes).cuda()
        panoptic_metrics = {}
        for key in EVALUATION_RANGES.keys():
            panoptic_metrics[key] = PanopticMetric(n_classes=n_classes, temporally_consistent=True).cuda()
    
    # Plan eval init
    eval_planning =  hasattr(model.module, 'with_planning_head') \
                      and model.module.with_planning_head

    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()

    if eval_planning:
        planning_metrics = PlanningMetric().cuda()
    if rank == 0:
        prog_bar = ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    num_occ = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(data, return_loss=False, rescale=True)

            # EVAL planning
            if eval_planning:
                # TODO: Wrap below into a func
                segmentation = result[0]['planning']['planning_gt']['segmentation']
                sdc_planning = result[0]['planning']['planning_gt']['sdc_planning']
                sdc_planning_mask = result[0]['planning']['planning_gt']['sdc_planning_mask']
                pred_sdc_traj = result[0]['planning']['result_planning']['sdc_traj']
                result[0]['planning_traj'] = result[0]['planning']['result_planning']['sdc_traj']
                result[0]['planning_traj_gt'] = result[0]['planning']['planning_gt']['sdc_planning']
                result[0]['command'] = result[0]['planning']['planning_gt']['command']

                # Official evaluation logic
                planning_metrics(pred_sdc_traj[:, :6, :2], sdc_planning[0][0,:, :6, :2], sdc_planning_mask[0][0,:, :6, :2], segmentation[0][:, [1,2,3,4,5,6]])

            # Eval Occ
            if eval_occ:
                occ_has_invalid_frame = data['gt_occ_has_invalid_frame'][0]
                occ_to_eval = not occ_has_invalid_frame.item()
                if occ_to_eval and 'occ' in result[0].keys():
                    num_occ += 1
                    for key, grid in EVALUATION_RANGES.items():
                        limits = slice(grid[0], grid[1])
                        iou_metrics[key](result[0]['occ']['seg_out'][..., limits, limits].contiguous(),
                                        result[0]['occ']['seg_gt'][..., limits, limits].contiguous())
                        panoptic_metrics[key](result[0]['occ']['ins_seg_out'][..., limits, limits].contiguous().detach(),
                                                result[0]['occ']['ins_seg_gt'][..., limits, limits].contiguous())


            # enable_plot_mode = os.environ.get('ENABLE_PLOT_MODE', None)
            # print(f"[DEBUG] ENABLE_PLOT_MODE = {enable_plot_mode}")
            # print(f"[DEBUG] result[0] keys = {list(result[0].keys())}")
            # if 'occ' in result[0]:
            #     print(f"[DEBUG] result[0]['occ'] keys = {list(result[0]['occ'].keys())}")
            # if 'pts_bbox' in result[0]:
            #     print(f"[DEBUG] result[0]['pts_bbox'] keys = {list(result[0]['pts_bbox'].keys())}")
                        # Pop out unnecessary occ results, avoid appending it to cpu when collect_results_cpu

            # Pop out unnecessary occ results, avoid appending it to cpu when collect_results_cpu
            if os.environ.get('ENABLE_PLOT_MODE', None) is None:

                result[0].pop('occ', None)
                result[0].pop('planning', None)
            else:
                for k in ['seg_gt', 'ins_seg_gt', 'pred_ins_sigmoid', 'seg_out', 'ins_seg_out']:
                    if k in result[0]['occ']:
                        result[0]['occ'][k] = result[0]['occ'][k].detach().cpu()
                for k in ['bbox', 'segm', 'labels', 'panoptic', 'drivable', 'score_list', 'lane', 'lane_score', 'stuff_score_list']:
                    if k in result[0]['pts_bbox'] and isinstance(result[0]['pts_bbox'][k], torch.Tensor):
                        result[0]['pts_bbox'][k] = result[0]['pts_bbox'][k].detach().cpu()

            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
                
        # break

    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None

    if eval_planning:
        planning_results = planning_metrics.compute()
        planning_metrics.reset()

    ret_results = dict()
    ret_results['bbox_results'] = bbox_results
    if eval_occ:
        occ_results = {}
        for key, grid in EVALUATION_RANGES.items():
            panoptic_scores = panoptic_metrics[key].compute()
            for panoptic_key, value in panoptic_scores.items():
                occ_results[f'{panoptic_key}'] = occ_results.get(f'{panoptic_key}', []) + [100 * value[1].item()]
            panoptic_metrics[key].reset()

            iou_scores = iou_metrics[key].compute()
            occ_results['iou'] = occ_results.get('iou', []) + [100 * iou_scores[1].item()]
            iou_metrics[key].reset()

        occ_results['num_occ'] = num_occ  # count on one gpu
        occ_results['ratio_occ'] = num_occ / len(dataset)  # count on one gpu, but reflect the relative ratio
        ret_results['occ_results_computed'] = occ_results
    if eval_planning:
        ret_results['planning_results_computed'] = planning_results

    if mask_results is not None:
        ret_results['mask_results'] = mask_results
    return ret_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)

def custom_single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(data, result, out_dir=out_dir)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results