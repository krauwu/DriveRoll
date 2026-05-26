import argparse
import dwm.common
import json
import os
import torch
import time
from tqdm import tqdm


def customize_text(clip_text, preview_config):

    # text
    if preview_config["text"] is not None:
        text_config = preview_config["text"]

        if text_config["type"] == "add":
            new_clip_text = \
                [
                    [
                        [
                            text_config["prompt"] + k
                            for k in j
                        ]
                        for j in i
                    ]
                    for i in clip_text
                ]

        elif text_config["type"] == "replace":
            new_clip_text = \
                [
                    [
                        [
                            text_config["prompt"]
                            for k in j
                        ]
                        for j in i
                    ]
                    for i in clip_text
                ]

        elif text_config["type"] == "template":
            time = text_config["time"]
            weather = text_config["weather"]
            new_clip_text = \
                [
                    [
                        [
                            text_config["template"][time][weather][idx][0]
                            for idx, k in enumerate(j)
                        ]
                        for j in i
                    ]
                    for i in clip_text
                ]

        else:
            raise NotImplementedError(
                f"{text_config['type']}has not been implemented yet.")

        return new_clip_text

    else:

        return clip_text


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to finetune a stable diffusion model to the "
        "driving dataset.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config to load the train model and dataset.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path to save checkpoint files.")
    parser.add_argument(
        "-pc", "--preview-config-path", default=None, type=str,
        help="The config for preview setting")
    parser.add_argument(
        "-eic", "--export-item-config", default=False, type=bool,
        help="The flag to export the item config as JSON")
    parser.add_argument(
        "-n", "--num-samples", default=None, type=int,
        help="The number of samples to preview (default: all)")
    parser.add_argument(
        "-mcp", "--model-checkpoint-path", default=None, type=str,
        help="The full path to the model checkpoint file (e.g., /path/to/checkpoints/2000.pth)")
    parser.add_argument(
        "-r", "--resume-from", default=None, type=int,
        help="The checkpoint step to resume from output_path/checkpoints/ (e.g., 2000)")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    # import debugpy
    # debugpy.listen(("0.0.0.0", 9870))
    # print("[debugpy] listening on, waiting for VS Code to attach...")
    # debugpy.wait_for_client()        
    # print("attached")
    
    # bad = set(range(3, 151, 3))
    bad = []

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.preview_config_path is not None:
        with open(args.preview_config_path, "r", encoding="utf-8") as f:
            preview_config = json.load(f)
    else:
        preview_config = None

    # set distributed training (if enabled), log, random number generator, and
    # load the checkpoint (if required).
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(config["device"], local_rank)
        if config["device"] == "cuda":
            torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(backend=config["ddp_backend"])
    else:
        device = torch.device(config["device"])

    # setup the global state
    if "global_state" in config:
        for key, value in config["global_state"].items():
            dwm.common.global_state[key] = \
                dwm.common.create_instance_from_config(value)

    should_log = (ddp and local_rank == 0) or not ddp
    should_save = not torch.distributed.is_initialized() or \
        torch.distributed.get_rank() == 0

    # load the pipeline including the models
    pipeline_kwargs = dict(
        output_path=args.output_path, config=config, device=device,
        resume_from=args.resume_from)
    # 如果指定了完整的 checkpoint 路径，使用 model_checkpoint_path 参数
    if args.model_checkpoint_path is not None:
        pipeline_kwargs["model_checkpoint_path"] = args.model_checkpoint_path
        # 如果同时指定了 resume_from，清除它以避免冲突
        if args.resume_from is not None:
            pipeline_kwargs["resume_from"] = None
    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], **pipeline_kwargs)
    if should_log:
        print("The pipeline is loaded.")

    validation_dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    good = [k for k in range(len(validation_dataset)) if k not in bad]

    # 应用 num-samples 限制
    if args.num_samples is not None:
        good = good[:args.num_samples]
        if should_log:
            print(f"Limiting to {args.num_samples} samples")

    # good = [3]
    # preview_dataloader = torch.utils.data\
    #     .DataLoader(
    #         torch.utils.data.Subset(validation_dataset, good),
    #         **dwm.common.instantiate_config(config["preview_dataloader"])) if \
    #     "preview_dataloader" in config else None

    if ddp:
        # 不截断数据，让 DistributedSampler 自动处理不均匀分配
        subset = torch.utils.data.Subset(validation_dataset, good)
    else:
        subset = torch.utils.data.Subset(validation_dataset, good)
        
    sampler = None
    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            subset, shuffle=False, drop_last=False
        )
        sampler.set_epoch(0)

    preview_dataloader = torch.utils.data.DataLoader(
        subset,
        sampler=sampler,
        **dwm.common.instantiate_config(config["preview_dataloader"])
    )

    # process_group = torch.distributed.group.WORLD if ddp else None

    # preview_datasampler = VariableVideoBatchSampler(
    #     validation_dataset,
    #     config["mix_config"],
    #     num_replicas=(process_group.size() if ddp else 1),
    #     rank=(process_group.rank() if ddp else 0),
    #     shuffle=False,                # preview 建议关闭 shuffle 方便对齐验证
    #     seed=config["generator_seed"]
    # )

    # dl_cfg = config["preview_dataloader"]
    # for k in ["batch_size", "shuffle", "sampler", "drop_last"]:
    #     dl_cfg.pop(k, None)

    preview_dataloader = torch.utils.data.DataLoader(
        subset,
        sampler=sampler,
        **dwm.common.instantiate_config(config["preview_dataloader"])
    )

    if should_log:
        print("The validation dataset is loaded with {} items.".format(
            len(validation_dataset)))

    export_batch_except = ["vae_images"]
    output_path = args.output_path
    
    rank = torch.distributed.get_rank() if ddp else 0
    world = torch.distributed.get_world_size() if ddp else 1

    total_samples = len(preview_dataloader)
    if should_log:
        print(f"----------------- 开始推理: 共 {total_samples} 个样本 -----------------")

    # 初始化时间统计
    start_time = time.time()
    sample_times = []

    # 使用 tqdm 进度条（只在主进程显示）
    pbar = tqdm(preview_dataloader, total=total_samples, desc="生成进度",
                disable=not should_log, unit="样本")

    for i, batch in enumerate(pbar):
        sample_start = time.time()

        if ddp and sampler is not None:
            sampler.set_epoch(i)
        global_step = i * world + rank

        if preview_config is not None:
            new_clip_text = customize_text(batch["clip_text"], preview_config)
            batch["clip_text"] = new_clip_text

        pipeline.preview_pipeline(batch, output_path, global_step)

        if args.export_item_config:
            with open(
                os.path.join(
                    output_path, "preview",
                    "{}.json".format(global_step)),
                "w", encoding="utf-8"
            ) as f:
                json.dump({
                    k: v.tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                    if k not in export_batch_except
                }, f, indent=4)

        # 计算当前样本耗时
        sample_time = time.time() - sample_start
        sample_times.append(sample_time)

        # 更新进度条信息
        if should_log:
            avg_time = sum(sample_times) / len(sample_times)
            remaining_samples = total_samples - len(sample_times)
            remaining_time = remaining_samples * avg_time
            pbar.set_postfix({
                '当前耗时': f'{sample_time:.1f}s',
                '平均耗时': f'{avg_time:.1f}s',
                '预计剩余': f'{remaining_time:.0f}s'
            })

    # 总耗时统计
    total_time = time.time() - start_time

    if should_log:
        print("\n" + "=" * 60)
        print("推理完成统计")
        print("=" * 60)
        print(f"总样本数: {len(sample_times)}")
        print(f"总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
        if sample_times:
            avg_time = sum(sample_times) / len(sample_times)
            print(f"平均每样本耗时: {avg_time:.1f}s")
            print(f"最快: {min(sample_times):.1f}s, 最慢: {max(sample_times):.1f}s")
        print("=" * 60)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
