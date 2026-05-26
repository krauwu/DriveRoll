from __future__ import annotations

import traceback

import gradio as gr

from tools.roll.data_agent import NuscRollingDataAgent
from tools.roll.rolling_gen_agent import RollingGenAgent


CSS = """
#app-title {
    margin-bottom: 8px;
}

.ctrl-btn button,
.top-btn button {
    width: 100% !important;
    min-height: 64px !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    border: 2px solid #3b82f6 !important;
    background: #2563eb !important;
    color: white !important;
    box-shadow: none !important;
}

.ctrl-btn button:hover,
.top-btn button:hover {
    background: #1d4ed8 !important;
    color: white !important;
}

#text-box textarea,
#pending-box textarea {
    font-size: 15px !important;
    line-height: 1.5 !important;
}

.img-panel {
    min-height: 320px !important;
}

video {
    width: 100% !important;
    max-height: 760px !important;
    background: black !important;
}
"""

RULE_TEXT = """
### Generation rules
- Click the same directional control twice to commit generation.
- The first click records a pending action and previews the future bbox + HD-map condition.
- If another direction is clicked before confirmation, the latest direction is used.
- The left video panel shows GT history followed by generated frames.
"""


def is_empty_text_value(text_value):
    if text_value is None:
        return True

    if isinstance(text_value, str):
        return text_value.strip() == ""

    if isinstance(text_value, list):
        i = 0
        while i < len(text_value):
            if str(text_value[i]).strip() != "":
                return False
            i += 1
        return True

    return str(text_value).strip() == ""


def resolve_text_override(data_agent: NuscRollingDataAgent | None, text_condition: str):
    if data_agent is None:
        return None

    value = data_agent.parse_text_condition_for_model(text_condition)
    if is_empty_text_value(value):
        return None

    return value


def safe_build_progress_video(data_agent: NuscRollingDataAgent | None):
    if data_agent is None or not data_agent.is_initialized:
        return None

    try:
        return data_agent.build_progress_video()
    except Exception:
        traceback.print_exc()
        return None


def safe_build_condition_preview(
    data_agent: NuscRollingDataAgent | None,
    action_name: str,
    text_condition: str,
):
    if data_agent is None or not data_agent.is_initialized:
        return None

    try:
        text_override = resolve_text_override(data_agent, text_condition)
        return data_agent.build_condition_preview(
            command=action_name,
            text_override=text_override,
        )
    except Exception:
        traceback.print_exc()
        return None


def build_initialized_payload(data_agent: NuscRollingDataAgent | None):
    if data_agent is None or not data_agent.is_initialized:
        return None, None, ""

    latest_frame_idx = data_agent.get_latest_progress_frame_idx()
    if latest_frame_idx is None:
        return None, None, ""

    latest_image = None
    try:
        latest_image, _ = data_agent.build_history_detail(latest_frame_idx)
    except Exception:
        traceback.print_exc()
        try:
            latest_image = data_agent.build_main_image(latest_frame_idx)
        except Exception:
            traceback.print_exc()
            latest_image = None

    next_text = ""
    try:
        next_text = data_agent.get_current_text_condition()
    except Exception:
        traceback.print_exc()
        next_text = ""

    video_path = safe_build_progress_video(data_agent)
    return video_path, latest_image, next_text


def format_pending_action_text(pending_action):
    if pending_action is None:
        return "None"

    action = pending_action.get("action", "")
    if action == "":
        return "None"

    return f"Pending: {action}"


def load_cfg(cfg_path: str):
    if not cfg_path.strip():
        return (
            None,
            None,
            gr.update(choices=[], value=None),
            None,
            None,
            "",
            "None",
            None,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    try:
        data_agent = NuscRollingDataAgent(cfg_path)
        gen_agent = RollingGenAgent(cfg_path)

        choices = data_agent.get_segment_choices()
        first_seg = choices[0][1] if len(choices) > 0 else None

        default_text = ""
        if first_seg is not None:
            default_text = data_agent.get_segment_default_text(first_seg)

        return (
            data_agent,
            gen_agent,
            gr.update(choices=choices, value=first_seg),
            None,
            None,
            default_text,
            "None",
            None,
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    except Exception:
        traceback.print_exc()
        return (
            None,
            None,
            gr.update(choices=[], value=None),
            None,
            None,
            "",
            "None",
            None,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )


def on_segment_change(segment_id: int, data_agent: NuscRollingDataAgent):
    if data_agent is None or segment_id is None:
        return (
            None,
            None,
            "",
            "None",
            None,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    try:
        seg_id = int(segment_id)
        default_text = data_agent.get_segment_default_text(seg_id)

        return (
            None,
            None,
            default_text,
            "None",
            None,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    except Exception:
        traceback.print_exc()
        return (
            None,
            None,
            "",
            "None",
            None,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )


def on_init(
    segment_id: int,
    text_condition: str,
    data_agent: NuscRollingDataAgent,
    gen_agent: RollingGenAgent,
):
    if data_agent is None or gen_agent is None:
        return (
            None,
            None,
            text_condition,
            "None",
            None,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    if segment_id is None:
        return (
            None,
            None,
            text_condition,
            "None",
            None,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    try:
        gen_agent.reset()
        data_agent.select_segment(int(segment_id))

        video_path, latest_image, next_text = build_initialized_payload(data_agent)

        current_text = text_condition
        if not str(current_text).strip():
            current_text = next_text

        video_update = gr.update()
        if video_path is not None:
            video_update = video_path

        image_update = gr.update()
        if latest_image is not None:
            image_update = latest_image

        return (
            video_update,
            image_update,
            current_text,
            "None",
            None,
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    except Exception:
        traceback.print_exc()
        return (
            None,
            None,
            text_condition,
            "None",
            None,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )


def on_action_click(
    action_name: str,
    text_condition: str,
    pending_action,
    data_agent: NuscRollingDataAgent,
    gen_agent: RollingGenAgent,
):
    if data_agent is None or gen_agent is None:
        return (
            None,
            gr.update(),
            gr.update(),
            text_condition,
            "None",
        )

    if not data_agent.is_initialized:
        return (
            None,
            gr.update(),
            gr.update(),
            text_condition,
            "None",
        )

    if not data_agent.can_continue_generation():
        return (
            None,
            gr.update(),
            gr.update(),
            text_condition,
            "End of segment",
        )

    current_text_key = "" if text_condition is None else str(text_condition)
    same_pending = (
        pending_action is not None
        and pending_action.get("action") == action_name
        and pending_action.get("text_condition") == current_text_key
    )

    text_override = resolve_text_override(data_agent, text_condition)

    try:
        infos_all, step_meta = data_agent.build_infos_all(
            command=action_name,
            text_override=text_override,
        )

        if not same_pending:
            preview_image = None
            try:
                preview_image, _ = data_agent.render_condition_preview_from_infos(infos_all)
            except Exception:
                traceback.print_exc()
                preview_image = None

            new_pending = {
                "action": action_name,
                "text_condition": current_text_key,
            }

            image_update = gr.update()
            if preview_image is not None:
                image_update = preview_image

            return (
                new_pending,
                gr.update(),
                image_update,
                text_condition,
                f"{format_pending_action_text(new_pending)} (condition previewed)",
            )

        cond_pack, hist_cam_list = data_agent.build_cond_from_infos(infos_all)

        next_views, new_state = gen_agent.generate_next_views(
            cond_pack=cond_pack,
            hist_cam_list=hist_cam_list,
            gen_state=data_agent.gen_state,
        )

        if next_views is None:
            return (
                None,
                gr.update(),
                gr.update(),
                text_condition,
                "None",
            )

        data_agent.gen_state = new_state

        target_rel_idx = min(data_agent.history_len, len(infos_all) - 1)
        target_clip_text = infos_all[target_rel_idx].get("clip_text", None)

        data_agent.commit_generated_frame(
            next_views=next_views,
            command=action_name,
            commit_ego_transform=step_meta["commit_ego_transform"],
            clip_text_value=target_clip_text,
        )

        video_path, latest_image, next_text = build_initialized_payload(data_agent)

        next_text_value = text_condition
        if not str(text_condition).strip():
            next_text_value = next_text

        video_update = gr.update()
        if video_path is not None:
            video_update = video_path

        image_update = gr.update()
        if latest_image is not None:
            image_update = latest_image

        return (
            None,
            video_update,
            image_update,
            next_text_value,
            "None",
        )

    except StopIteration:
        return (
            None,
            gr.update(),
            gr.update(),
            text_condition,
            "End of segment",
        )
    except Exception:
        traceback.print_exc()
        return (
            None,
            gr.update(),
            gr.update(),
            text_condition,
            "None",
        )


def on_forward_click(
    text_condition: str,
    pending_action,
    data_agent: NuscRollingDataAgent,
    gen_agent: RollingGenAgent,
):
    return on_action_click(
        action_name="forward",
        text_condition=text_condition,
        pending_action=pending_action,
        data_agent=data_agent,
        gen_agent=gen_agent,
    )


def on_slow_click(
    text_condition: str,
    pending_action,
    data_agent: NuscRollingDataAgent,
    gen_agent: RollingGenAgent,
):
    return on_action_click(
        action_name="slow",
        text_condition=text_condition,
        pending_action=pending_action,
        data_agent=data_agent,
        gen_agent=gen_agent,
    )


def on_left_click(
    text_condition: str,
    pending_action,
    data_agent: NuscRollingDataAgent,
    gen_agent: RollingGenAgent,
):
    return on_action_click(
        action_name="left",
        text_condition=text_condition,
        pending_action=pending_action,
        data_agent=data_agent,
        gen_agent=gen_agent,
    )


def on_right_click(
    text_condition: str,
    pending_action,
    data_agent: NuscRollingDataAgent,
    gen_agent: RollingGenAgent,
):
    return on_action_click(
        action_name="right",
        text_condition=text_condition,
        pending_action=pending_action,
        data_agent=data_agent,
        gen_agent=gen_agent,
    )


with gr.Blocks(title="EasyTry Rolling Demo", css=CSS) as demo:
    gr.Markdown("## EasyTry Rolling Demo", elem_id="app-title")

    data_agent_state = gr.State(None)
    gen_agent_state = gr.State(None)
    pending_action_state = gr.State(None)

    with gr.Row():
        cfg_path = gr.Textbox(
            label="Config path",
            value="rolling_demo.json",
            scale=8,
        )
        load_btn = gr.Button("Load config", elem_classes=["top-btn"], scale=2)

    with gr.Row():
        segment_dropdown = gr.Dropdown(
            label="segment-id",
            choices=[],
            value=None,
            scale=8,
        )
        init_btn = gr.Button("Initialize", elem_classes=["top-btn"], scale=2, interactive=False)

    with gr.Row():
        with gr.Column(scale=6):
            progress_video = gr.Video(
                label="GT history + generated video",
                height=760,
                interactive=False,
                autoplay=False,
                loop=False,
            )

            text_condition_box = gr.Textbox(
                label="Text condition (one line per view; leave empty to use dataset defaults)",
                lines=8,
                placeholder="Enter or edit the text condition",
                elem_id="text-box",
            )

        with gr.Column(scale=6):
            latest_detail_image = gr.Image(
                label="Latest frame detail / pending condition preview",
                type="pil",
                elem_classes=["img-panel"],
                height=760,
            )

            pending_box = gr.Textbox(
                label="Pending action",
                value="None",
                lines=1,
                interactive=False,
                elem_id="pending-box",
            )

            gr.Markdown("### Controls")

            with gr.Row():
                gr.Markdown("")
                forward_btn = gr.Button("↑ Forward / W", elem_classes=["ctrl-btn"], interactive=False)
                gr.Markdown("")

            with gr.Row():
                left_btn = gr.Button("← Left / A", elem_classes=["ctrl-btn"], interactive=False)
                slow_btn = gr.Button("↓ Slow / S", elem_classes=["ctrl-btn"], interactive=False)
                right_btn = gr.Button("→ Right / D", elem_classes=["ctrl-btn"], interactive=False)

            gr.Markdown(RULE_TEXT)

    load_btn.click(
        fn=load_cfg,
        inputs=[cfg_path],
        outputs=[
            data_agent_state,
            gen_agent_state,
            segment_dropdown,
            progress_video,
            latest_detail_image,
            text_condition_box,
            pending_box,
            pending_action_state,
            init_btn,
            forward_btn,
            slow_btn,
            left_btn,
            right_btn,
        ],
    )

    segment_dropdown.change(
        fn=on_segment_change,
        inputs=[segment_dropdown, data_agent_state],
        outputs=[
            progress_video,
            latest_detail_image,
            text_condition_box,
            pending_box,
            pending_action_state,
            forward_btn,
            slow_btn,
            left_btn,
            right_btn,
        ],
    )

    init_btn.click(
        fn=on_init,
        inputs=[segment_dropdown, text_condition_box, data_agent_state, gen_agent_state],
        outputs=[
            progress_video,
            latest_detail_image,
            text_condition_box,
            pending_box,
            pending_action_state,
            forward_btn,
            slow_btn,
            left_btn,
            right_btn,
        ],
    )

    forward_btn.click(
        fn=on_forward_click,
        inputs=[
            text_condition_box,
            pending_action_state,
            data_agent_state,
            gen_agent_state,
        ],
        outputs=[
            pending_action_state,
            progress_video,
            latest_detail_image,
            text_condition_box,
            pending_box,
        ],
    )

    slow_btn.click(
        fn=on_slow_click,
        inputs=[
            text_condition_box,
            pending_action_state,
            data_agent_state,
            gen_agent_state,
        ],
        outputs=[
            pending_action_state,
            progress_video,
            latest_detail_image,
            text_condition_box,
            pending_box,
        ],
    )

    left_btn.click(
        fn=on_left_click,
        inputs=[
            text_condition_box,
            pending_action_state,
            data_agent_state,
            gen_agent_state,
        ],
        outputs=[
            pending_action_state,
            progress_video,
            latest_detail_image,
            text_condition_box,
            pending_box,
        ],
    )

    right_btn.click(
        fn=on_right_click,
        inputs=[
            text_condition_box,
            pending_action_state,
            data_agent_state,
            gen_agent_state,
        ],
        outputs=[
            pending_action_state,
            progress_video,
            latest_detail_image,
            text_condition_box,
            pending_box,
        ],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        root_path="https://ai-notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-e754cc6f-3141-4d5c-af33-9182b2086005/user-17e177ba-08cc-4b43-adf3-a391730a32e5/vscode/fdfbf8e2-4059-4a2b-91cf-4b05dce96f44/a496df80-2179-4888-aa2c-d693c96dca47/proxy/7860",
    )