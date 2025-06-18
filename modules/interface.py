import gradio as gr
import time
import datetime
import random
import json
import os
import shutil
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import base64
import io
import functools

from modules.version import APP_VERSION, APP_VERSION_DISPLAY

import subprocess
import itertools
import re
from collections import defaultdict
import imageio
import imageio.plugins.ffmpeg
import ffmpeg
from diffusers_helper.utils import generate_timestamp

from modules.video_queue import JobStatus, Job, JobType
from modules.prompt_handler import get_section_boundaries, get_quick_prompts, parse_timestamped_prompt
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.bucket_tools import find_nearest_bucket
from modules.pipelines.metadata_utils import create_metadata
from modules import DUMMY_LORA_NAME # Import the constant

from modules.toolbox_app import tb_create_video_toolbox_ui, tb_get_formatted_toolbar_stats
from modules.xy_plot_ui import create_xy_plot_ui, xy_plot_process

# Define the dummy LoRA name as a constant

def create_interface(
    process_fn,
    monitor_fn,
    end_process_fn,
    update_queue_status_fn,
    load_lora_file_fn,
    job_queue,
    settings,
    default_prompt: str = '[1s: The person waves hello] [3s: The person jumps up and down] [5s: The person does a dance]',
    lora_names: list = [],
    lora_values: list = []
):
    """
    Create the Gradio interface for the video generation application

    Args:
        process_fn: Function to process a new job
        monitor_fn: Function to monitor an existing job
        end_process_fn: Function to cancel the current job
        update_queue_status_fn: Function to update the queue status display
        default_prompt: Default prompt text
        lora_names: List of loaded LoRA names

    Returns:
        Gradio Blocks interface
    """
    def is_video_model(model_type_value):
        return model_type_value in ["Video", "Video with Endframe", "Video F1"]

    # Get section boundaries and quick prompts
    section_boundaries = get_section_boundaries()
    quick_prompts = get_quick_prompts()

    # --- Function to update queue stats (Moved earlier to resolve UnboundLocalError) ---
    def update_stats(*args): # Accept any arguments and ignore them
        # Get queue status data
        queue_status_data = update_queue_status_fn()
        
        # Get queue statistics for the toolbar display
        jobs = job_queue.get_all_jobs()
        
        # Count jobs by status
        pending_count = 0
        running_count = 0
        completed_count = 0
        
        for job in jobs:
            if hasattr(job, 'status'):
                status = str(job.status)
                if status == "JobStatus.PENDING":
                    pending_count += 1
                elif status == "JobStatus.RUNNING":
                    running_count += 1
                elif status == "JobStatus.COMPLETED":
                    completed_count += 1
        
        # Format the queue stats display text
        queue_stats_text = f"<p style='margin:0;color:white;' class='toolbar-text'>佇列: {pending_count} | 運行中: {running_count} | 已完成: {completed_count}</p>"
        
        return queue_status_data, queue_stats_text

    # --- Preset System Functions ---
    PRESET_FILE = os.path.join(".framepack", "generation_presets.json")

    def load_presets(model_type):
        if not os.path.exists(PRESET_FILE):
            return []
        with open(PRESET_FILE, 'r') as f:
            data = json.load(f)
        return list(data.get(model_type, {}).keys())

    # Create the interface
    css = make_progress_bar_css()
    css += """
    .short-import-box, .short-import-box > div {
        min-height: 40px !important;
        height: 40px !important;
    }
    /* Image container styling - more aggressive approach */
    .contain-image, .contain-image > div, .contain-image > div > img {
        object-fit: contain !important;
    }

    #non-mirrored-video {
        transform: scaleX(-1) !important;
    }
    
    /* Target all images in the contain-image class and its children */
    .contain-image img,
    .contain-image > div > img,
    .contain-image * img {
        object-fit: contain !important;
        width: 100% !important;
        height: 60vh !important;
        max-height: 100% !important;
        max-width: 100% !important;
    }
    
    /* Additional selectors to override Gradio defaults */
    .gradio-container img,
    .gradio-container .svelte-1b5oq5x,
    .gradio-container [data-testid="image"] img {
        object-fit: contain !important;
    }
    
    /* Toolbar styling */
    #fixed-toolbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        z-index: 1000;
        background: #333;
        color: #fff;
        padding: 0px 10px; /* Reduced top/bottom padding */
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Responsive toolbar title */
    .toolbar-title {
        font-size: 1.4rem;
        margin: 0;
        color: white;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Toolbar Patreon link */
    .toolbar-patreon {
        margin: 0 0 0 20px;
        color: white;
        font-size: 0.9rem;
        white-space: nowrap;
        display: inline-block;
    }
    .toolbar-patreon a {
        color: white;
        text-decoration: none;
    }
    .toolbar-patreon a:hover {
        text-decoration: underline;
    }

    /* Toolbar Version number */
    .toolbar-version {
        margin: 0 15px; /* Space around version */
        color: white;
        font-size: 0.8rem;
        white-space: nowrap;
        display: inline-block;
    }
    
    /* Responsive design for screens */
    @media (max-width: 1147px) {
        .toolbar-patreon, .toolbar-version { /* Hide both on smaller screens */
            display: none;
        }
        .footer-patreon, .footer-version { /* Show both in footer on smaller screens */
            display: inline-block !important; /* Ensure they are shown */
        }
        #fixed-toolbar {
            gap: 4px !important; /* Reduce gap for screens <= 1024px */
        }
        #fixed-toolbar > div:first-child { /* Target the first gr.Column (Title) */
            min-width: fit-content !important; /* Override Python-set min-width */
            flex-shrink: 0 !important; /* Prevent title column from shrinking too much */
        }
    }
    
    @media (min-width: 1148px) {
        .footer-patreon, .footer-version { /* Hide both in footer on larger screens */
            display: none !important;
        }
    }
    
    @media (max-width: 768px) {
        .toolbar-title {
            font-size: 1.1rem;
            max-width: 150px;
        }
        #fixed-toolbar {
            padding: 3px 6px;
            gap: 4px;
        }
        .toolbar-text {
            font-size: 0.75rem;
        }
    }
    
    @media (max-width: 510px) {
        #toolbar-ram-col, #toolbar-vram-col, #toolbar-gpu-col {
            display: none !important;
        }
    }

    @media (max-width: 480px) {
        .toolbar-title {
            font-size: 1rem;
            max-width: 120px;
        }
        #fixed-toolbar {
            padding: 2px 4px;
            gap: 2px;
        }
        .toolbar-text {
            font-size: 0.7rem;
        }
    }
    
    /* Button styling */
    #toolbar-add-to-queue-btn button {
        font-size: 14px !important;
        padding: 4px 16px !important;
        height: 32px !important;
        min-width: 80px !important;
    }
    .narrow-button {
        min-width: 40px !important;
        width: 40px !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .gr-button-primary {
        color: white;
    }
    
    /* Layout adjustments */
    body, .gradio-container {
        padding-top: 42px !important; /* Adjusted for new toolbar height (36px - 10px) */
    }
    
    @media (max-width: 848px) {
        body, .gradio-container {
            padding-top: 48px !important;
        }
    }
    
    @media (max-width: 768px) {
        body, .gradio-container {
            padding-top: 22px !important; /* Adjusted for new toolbar height (32px - 10px) */
        }
    }
    
    @media (max-width: 480px) {
        body, .gradio-container {
            padding-top: 18px !important; /* Adjusted for new toolbar height (28px - 10px) */
        }
    }
    
    /* control sizing for tb_input_video_component */    
    .video-size video {
        max-height: 60vh;
        min-height: 300px !important;
        object-fit: contain;
    }

    /* hide the gr.Video source selection bar for tb_input_video_component */
    #toolbox-video-player .source-selection {
        display: none !important;
    }

    """

    # Get the theme from settings
    current_theme = settings.get("gradio_theme", "default") # Use default if not found
    block = gr.Blocks(css=css, title="FramePack Studio", theme=current_theme).queue()

    with block:
        with gr.Row(elem_id="fixed-toolbar"):
            with gr.Column(scale=0, min_width=400): # Title/Version/Patreon
                gr.HTML(f"""
                <div style="display: flex; align-items: center;">
                    <h1 class='toolbar-title'>FP Studio</h1>
                    <p class='toolbar-version'>{APP_VERSION_DISPLAY}</p>
                    <p class='toolbar-patreon'><a href='https://patreon.com/Colinu' target='_blank'>在 Patreon 上支持</a></p>
                </div>
                """)
            # REMOVED: refresh_stats_btn - Toolbar refresh button is no longer needed
            # with gr.Column(scale=0, min_width=40):
            #     refresh_stats_btn = gr.Button("⟳", elem_id="refresh-stats-btn", elem_classes="narrow-button")  
            with gr.Column(scale=1, min_width=180): # Queue Stats
                queue_stats_display = gr.Markdown("<p style='margin:0;color:white;' class='toolbar-text'>佇列: 0 | 運行中: 0 | 已完成: 0</p>")
                
            # --- System Stats Display - Single gr.Textbox per stat ---
            with gr.Column(scale=0, min_width=173, elem_id="toolbar-ram-col"): # RAM Column
                toolbar_ram_display_component = gr.Textbox(
                    value="RAM: N/A",
                    interactive=False,
                    lines=1,
                    max_lines=1,
                    show_label=False,
                    container=False,
                    elem_id="toolbar-ram-stat",
                    elem_classes="toolbar-stat-textbox"
                )
            with gr.Column(scale=0, min_width=138, elem_id="toolbar-vram-col"): # VRAM Column
                toolbar_vram_display_component = gr.Textbox(
                    value="VRAM: N/A",
                    interactive=False,
                    lines=1,
                    max_lines=1,
                    show_label=False,
                    container=False,
                    elem_id="toolbar-vram-stat",
                    elem_classes="toolbar-stat-textbox"
                    # Visibility controlled by tb_get_formatted_toolbar_stats
                )
            with gr.Column(scale=0, min_width=130, elem_id="toolbar-gpu-col"): # GPU Column
                toolbar_gpu_display_component = gr.Textbox(
                    value="GPU: N/A",
                    interactive=False,
                    lines=1,
                    max_lines=1,
                    show_label=False,
                    container=False,
                    elem_id="toolbar-gpu-stat",
                    elem_classes="toolbar-stat-textbox"
                    # Visibility controlled by tb_get_formatted_toolbar_stats
                )
            # --- End of System Stats Display ---
            
            # Removed old version_display column
            # --- End of Toolbar ---
            
        # Essential to capture main_tabs_component for later use by send_to_toolbox_btn
        with gr.Tabs(elem_id="main_tabs") as main_tabs_component:
            with gr.Tab("生成", id="generate_tab"): # "Generate" -> "生成"
                with gr.Row():
                    with gr.Column(scale=2):
                        model_type = gr.Radio(
                            choices=[("原始", "Original"), ("帶結束幀的原始", "Original with Endframe"), ("F1", "F1"), ("影片", "Video"), ("帶結束幀的影片", "Video with Endframe"), ("影片 F1", "Video F1")],
                            value="Original",
                            label="生成類型" # "Generation Type" -> "生成類型"
                        )
                        with gr.Accordion("原始預設", open=False, visible=True) as preset_accordion: # "Original Presets" -> "原始預設"
                            with gr.Row():
                                preset_dropdown = gr.Dropdown(label="選擇預設", choices=load_presets("Original"), interactive=True, scale=2) # "Select Preset" -> "選擇預設"
                                delete_preset_button = gr.Button("刪除", variant="stop", scale=1) # "Delete" -> "刪除"
                            with gr.Row():
                                preset_name_textbox = gr.Textbox(label="預設名稱", placeholder="輸入您的預設名稱", scale=2) # "Preset Name" -> "預設名稱", "Enter a name for your preset" -> "輸入您的預設名稱"
                                save_preset_button = gr.Button("儲存", variant="primary", scale=1) # "Save" -> "儲存"
                            with gr.Row(visible=False) as confirm_delete_row:
                                gr.Markdown("### 您確定要刪除此預設嗎？") # "Are you sure you want to delete this preset?" -> "您確定要刪除此預設嗎？"
                                confirm_delete_yes_btn = gr.Button("是，刪除", variant="stop") # "Yes, Delete" -> "是，刪除"
                                confirm_delete_no_btn = gr.Button("否，返回") # "No, Go Back" -> "否，返回"
                        with gr.Accordion("基本參數", open=True, visible=True) as basic_parameters_accordion: # "Basic Parameters" -> "基本參數"
                            with gr.Group():
                                total_second_length = gr.Slider(label="影片長度（秒）", minimum=1, maximum=120, value=6, step=0.1) # "Video Length (Seconds)" -> "影片長度（秒）"
                                with gr.Row("解析度"): # "Resolution" -> "解析度"
                                    resolutionW = gr.Slider(
                                        label="寬度", minimum=128, maximum=768, value=640, step=32, 
                                        info="將使用最接近的有效寬度。" # "Nearest valid width will be used." -> "將使用最接近的有效寬度。"
                                    )
                                    resolutionH = gr.Slider(
                                        label="高度", minimum=128, maximum=768, value=640, step=32, 
                                        info="將使用最接近的有效高度。" # "Nearest valid height will be used." -> "將使用最接近的有效高度。"
                                    )
                                resolution_text = gr.Markdown(value="<div style='text-align:right; padding:5px 15px 5px 5px;'>所選解析度桶：640 x 640</div>", label="", show_label=False) # "Selected bucket for resolution:" -> "所選解析度桶："

                        # --- START OF REFACTORED XY PLOT SECTION ---
                        xy_plot_components = create_xy_plot_ui(
                            lora_names=lora_names,
                            default_prompt=default_prompt,
                            DUMMY_LORA_NAME=DUMMY_LORA_NAME,
                        )
                        xy_group = xy_plot_components["group"]
                        xy_plot_status = xy_plot_components["status"]
                        xy_plot_output = xy_plot_components["output"]
                        # --- END OF REFACTORED XY PLOT SECTION ---

                        with gr.Group(visible=True) as standard_generation_group:    # Default visibility: True because "Original" model is not "Video"
                            with gr.Group(visible=True) as image_input_group: # This group now only contains the start frame image
                                with gr.Row():
                                    with gr.Column(scale=1): # Start Frame Image Column
                                        input_image = gr.Image(
                                            sources='upload',
                                            type="numpy",
                                            label="起始幀 (可選)", # "Start Frame (optional)" -> "起始幀 (可選)"
                                            elem_classes="contain-image",
                                            image_mode="RGB",
                                            show_download_button=False,
                                            show_label=True, # Keep label for clarity
                                            container=True
                                        )
                            
                            with gr.Group(visible=False) as video_input_group:
                                input_video = gr.Video(
                                    sources='upload',
                                    label="影片輸入", # "Video Input" -> "影片輸入"
                                    height=420,
                                    show_label=True
                                )
                                combine_with_source = gr.Checkbox(
                                    label="與來源影片結合", # "Combine with source video" -> "與來源影片結合"
                                    value=True,
                                    info="如果勾選，來源影片將與生成的影片結合", # "If checked, the source video will be combined with the generated video" -> "如果勾選，來源影片將與生成的影片結合"
                                    interactive=True
                                )
                                num_cleaned_frames = gr.Slider(label="上下文幀數（影片依賴度）", minimum=2, maximum=10, value=5, step=1, interactive=True, info="昂貴。保留更多影片細節。如果遇到記憶體問題或動作過於受限（跳接、忽略提示、靜止）則減少。") # "Number of Context Frames (Adherence to Video)" -> "上下文幀數（影片依賴度）", info translated

                            # End Frame Image Input
                            # Initial visibility is False, controlled by update_input_visibility
                            with gr.Column(scale=1, visible=False) as end_frame_group_original:
                                end_frame_image_original = gr.Image(
                                    sources='upload',
                                    type="numpy",
                                    label="結束幀 (可選)", # "End Frame (Optional)" -> "結束幀 (可選)"
                                    elem_classes="contain-image",
                                    image_mode="RGB",
                                    show_download_button=False,
                                    show_label=True,
                                    container=True
                                )
                            
                            # End Frame Influence slider
                            # Initial visibility is False, controlled by update_input_visibility
                            with gr.Group(visible=False) as end_frame_slider_group:
                                end_frame_strength_original = gr.Slider(
                                    label="結束幀影響力", # "End Frame Influence" -> "結束幀影響力"
                                    minimum=0.05,
                                    maximum=1.0,
                                    value=1.0,
                                    step=0.05,
                                    info="控制結束幀對生成影片的引導強度。1.0 為完全影響。" # "Controls how strongly the end frame guides the generation. 1.0 is full influence." -> "控制結束幀對生成影片的引導強度。1.0 為完全影響。"
                                )

                            
                            prompt = gr.Textbox(label="提示詞", value=default_prompt) # "Prompt" -> "提示詞"

                            with gr.Accordion("提示詞參數", open=False): # "Prompt Parameters" -> "提示詞參數"
                                n_prompt = gr.Textbox(label="負向提示詞", value="", visible=True) # "Negative Prompt" -> "負向提示詞"

                                blend_sections = gr.Slider(
                                    minimum=0, maximum=10, value=4, step=1,
                                    label="提示詞之間混合的區塊數量" # "Number of sections to blend between prompts" -> "提示詞之間混合的區塊數量"
                                )
                            with gr.Accordion("生成參數", open=True): # "Generation Parameters" -> "生成參數"
                                with gr.Row():
                                    steps = gr.Slider(label="步數", minimum=1, maximum=100, value=25, step=1) # "Steps" -> "步數"
                                def on_input_image_change(img):
                                    if img is not None:
                                        return gr.update(info="將使用最接近的有效桶大小。高度將自動調整。"), gr.update(visible=False) # Info translated
                                    else:
                                        return gr.update(info="將使用最接近的有效寬度。"), gr.update(visible=True) # Info translated
                                input_image.change(fn=on_input_image_change, inputs=[input_image], outputs=[resolutionW, resolutionH])
                                def on_resolution_change(img, resolutionW, resolutionH):
                                    out_bucket_resH, out_bucket_resW = [640, 640]
                                    if img is not None:
                                        H, W, _ = img.shape
                                        out_bucket_resH, out_bucket_resW = find_nearest_bucket(H, W, resolution=resolutionW)
                                    else:
                                        out_bucket_resH, out_bucket_resW = find_nearest_bucket(resolutionH, resolutionW, (resolutionW+resolutionH)/2) # if resolutionW > resolutionH else resolutionH
                                    return gr.update(value=f"<div style='text-align:right; padding:5px 15px 5px 5px;'>所選解析度桶：{out_bucket_resW} x {out_bucket_resH}</div>") # Value translated
                                resolutionW.change(fn=on_resolution_change, inputs=[input_image, resolutionW, resolutionH], outputs=[resolution_text], show_progress="hidden")
                                resolutionH.change(fn=on_resolution_change, inputs=[input_image, resolutionW, resolutionH], outputs=[resolution_text], show_progress="hidden")
                                
                                with gr.Row():
                                    seed = gr.Number(label="種子", value=2500, precision=0) # "Seed" -> "種子"
                                    randomize_seed = gr.Checkbox(label="隨機化", value=True, info="為每個任務生成新的隨機種子") # "Randomize" -> "隨機化", info translated
                                with gr.Accordion("LoRA 模型", open=False): # "LoRAs" -> "LoRA 模型"
                                    with gr.Row():
                                        lora_selector = gr.Dropdown(
                                            choices=lora_names,
                                            label="選擇要載入的 LoRA 模型", # "Select LoRAs to Load" -> "選擇要載入的 LoRA 模型"
                                            multiselect=True,
                                            value=[],
                                            info="選擇一個或多個 LoRA 模型用於此任務" # "Select one or more LoRAs to use for this job" -> "選擇一個或多個 LoRA 模型用於此任務"
                                        )
                                        lora_names_states = gr.State(lora_names)
                                        lora_sliders = {}
                                        for lora in lora_names:
                                            lora_sliders[lora] = gr.Slider(
                                                minimum=0.0, maximum=2.0, value=1.0, step=0.01,
                                                label=f"{lora} 權重", visible=False, interactive=True # "{lora} Weight" -> "{lora} 權重"
                                            )
                                with gr.Accordion("潛在圖像選項", open=False): # "Latent Image Options" -> "潛在圖像選項"
                                    latent_type = gr.Dropdown(
                                        ["Black", "White", "Noise", "Green Screen"], label="潛在圖像", value="Black", info="如果未提供圖像，將作為起始點使用" # "Latent Image" -> "潛在圖像", info translated
                                    )
                                with gr.Accordion("進階參數", open=False): # "Advanced Parameters" -> "進階參數"
                                    latent_window_size = gr.Slider(label="潛在視窗大小", minimum=1, maximum=33, value=9, step=1, visible=True, info='風險自負，非常實驗性') # "Latent Window Size" -> "潛在視窗大小", info translated
                                    cfg = gr.Slider(label="CFG 比例", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False) # "CFG Scale" -> "CFG 比例"
                                    gs = gr.Slider(label="蒸餾 CFG 比例", minimum=1.0, maximum=32.0, value=10.0, step=0.01) # "Distilled CFG Scale" -> "蒸餾 CFG 比例"
                                    rs = gr.Slider(label="CFG 重新縮放", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False) # "CFG Re-Scale" -> "CFG 重新縮放"
                                    with gr.Row("TeaCache"):
                                        use_teacache = gr.Checkbox(label='使用 TeaCache', value=True, info='速度更快，但通常會讓手和手指稍微變差。') # "Use TeaCache" -> "使用 TeaCache", info translated
                                        teacache_num_steps = gr.Slider(label="TeaCache 步數", minimum=1, maximum=50, step=1, value=25, visible=True, info='在快取中保留多少中間區塊') # "TeaCache steps" -> "TeaCache 步數", info translated
                                        teacache_rel_l1_thresh = gr.Slider(label="TeaCache rel_l1_thresh", minimum=0.01, maximum=1.0, step=0.01, value=0.15, visible=True, info='相對 L1 閾值') # "TeaCache rel_l1_thresh" -> "TeaCache rel_l1_thresh", info translated
                                        use_teacache.change(lambda enabled: (gr.update(visible=enabled), gr.update(visible=enabled)), inputs=use_teacache, outputs=[teacache_num_steps, teacache_rel_l1_thresh])
                                with gr.Row("中繼資料"): # "Metadata" -> "中繼資料"
                                    json_upload = gr.File(
                                        label="上傳中繼資料 JSON (可選)", # "Upload Metadata JSON (optional)" -> "上傳中繼資料 JSON (可選)"
                                        file_types=[".json"],
                                        type="filepath",
                                        height=140,
                                    )

                    with gr.Column():
                        preview_image = gr.Image(
                            label="下一個潛在", # "Next Latents" -> "下一個潛在"
                            height=150, 
                            visible=True, 
                            type="numpy", 
                            interactive=False,
                            elem_classes="contain-image",
                            image_mode="RGB"
                        )
                        result_video = gr.Video(label="已完成幀", autoplay=True, show_share_button=False, height=256, loop=True) # "Finished Frames" -> "已完成幀"
                        progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                        progress_bar = gr.HTML('', elem_classes='no-generating-animation')
                        with gr.Row():
                            current_job_id = gr.Textbox(label="當前任務 ID", value="", visible=True, interactive=True) # "Current Job ID" -> "當前任務 ID"
                            start_button = gr.Button(value="加入佇列", variant="primary", elem_id="toolbar-add-to-queue-btn") # "Add to Queue" -> "加入佇列"
                            xy_plot_process_btn = gr.Button("提交", visible=False) # "Submit" -> "提交"
                            video_input_required_message = gr.Markdown(
                                "<p style='color: red; text-align: center;'>需要輸入影片</p>", visible=False # "Input video required" -> "需要輸入影片"
                            )
                            end_button = gr.Button(value="取消當前任務", interactive=True, visible=False) # "Cancel Current Job" -> "取消當前任務"

            with gr.Tab("佇列"): # "Queue" -> "佇列"
                with gr.Row():
                    with gr.Column():
                        with gr.Row() as queue_controls_row:
                            refresh_button = gr.Button("重新整理佇列") # "Refresh Queue" -> "重新整理佇列"
                            load_queue_button = gr.Button("繼續佇列") # "Resume Queue" -> "繼續佇列"
                            queue_export_button = gr.Button("匯出佇列") # "Export Queue" -> "匯出佇列"
                            clear_complete_button = gr.Button("清除已完成任務", variant="secondary") # "Clear Completed Jobs" -> "清除已完成任務"
                            clear_queue_button = gr.Button("取消排隊任務", variant="stop") # "Cancel Queued Jobs" -> "取消排隊任務"
                        with gr.Row():
                            import_queue_file = gr.File(
                                label="匯入佇列", # "Import Queue" -> "匯入佇列"
                                file_types=[".json", ".zip"],
                                type="filepath",
                                visible=True,
                                elem_classes="short-import-box"
                            )
                        
                        with gr.Row(visible=False) as confirm_cancel_row:
                            gr.Markdown("### 您確定要取消所有待處理的任務嗎？") # "Are you sure you want to cancel all pending jobs?" -> "您確定要取消所有待處理的任務嗎？"
                            confirm_cancel_yes_btn = gr.Button("是，取消所有", variant="stop") # "Yes, Cancel All" -> "是，取消所有"
                            confirm_cancel_no_btn = gr.Button("否，返回") # "No, Go Back" -> "否，返回"

                        with gr.Row():
                            queue_status = gr.DataFrame(
                                headers=["任務 ID", "類型", "狀態", "建立時間", "開始時間", "完成時間", "耗時", "預覽"], # Headers translated
                                datatype=["str", "str", "str", "str", "str", "str", "str", "html"], 
                                label="任務佇列" # "Job Queue" -> "任務佇列"
                            )

                        with gr.Accordion("佇列說明文件", open=False): # "Queue Documentation" -> "佇列說明文件"
                            gr.Markdown("""
                            ## 佇列分頁指南
                            
                            此分頁用於管理您的生成任務。
                            
                            - **重新整理佇列**: 更新任務列表。
                            - **取消佇列**: 停止所有待處理的任務。
                            - **清除已完成**: 從列表中移除已完成、失敗或已取消的任務。
                            - **載入佇列**: 從預設的 `queue.json` 載入任務。
                            - **匯出佇列**: 將當前任務列表及其圖像儲存為 zip 檔案。
                            - **匯入佇列**: 從 `.json` 或 `.zip` 檔案載入佇列。
                            """) # All markdown content translated
                        
                        # --- Event Handlers for Queue Tab ---

                        # Function to clear all jobs in the queue
                        def clear_all_jobs():
                            try:
                                cancelled_count = job_queue.clear_queue()
                                print(f"Cleared {cancelled_count} jobs from the queue") # Logging strings are kept in English for console output
                                return update_stats()
                            except Exception as e:
                                import traceback
                                print(f"Error in clear_all_jobs: {e}")
                                traceback.print_exc()
                                return [], ""

                        # Function to clear completed and cancelled jobs
                        def clear_completed_jobs():
                            try:
                                removed_count = job_queue.clear_completed_jobs()
                                print(f"Removed {removed_count} completed/cancelled jobs from the queue")
                                return update_stats()
                            except Exception as e:
                                import traceback
                                print(f"Error in clear_completed_jobs: {e}")
                                traceback.print_exc()
                                return [], ""

                        # Function to load queue from queue.json
                        def load_queue_from_json():
                            try:
                                loaded_count = job_queue.load_queue_from_json()
                                print(f"Loaded {loaded_count} jobs from queue.json")
                                return update_stats()
                            except Exception as e:
                                import traceback
                                print(f"Error loading queue from JSON: {e}")
                                traceback.print_exc()
                                return [], ""

                        # Function to import queue from a custom JSON file
                        def import_queue_from_file(file_path):
                            if not file_path:
                                return update_stats()
                            try:
                                loaded_count = job_queue.load_queue_from_json(file_path)
                                print(f"Loaded {loaded_count} jobs from {file_path}")
                                return update_stats()
                            except Exception as e:
                                import traceback
                                print(f"Error importing queue from file: {e}")
                                traceback.print_exc()
                                return [], ""

                        # Function to export queue to a zip file
                        def export_queue_to_zip():
                            try:
                                zip_path = job_queue.export_queue_to_zip()
                                if zip_path and os.path.exists(zip_path):
                                    print(f"Queue exported to {zip_path}")
                                else:
                                    print("Failed to export queue to zip")
                                return update_stats()
                            except Exception as e:
                                import traceback
                                print(f"Error exporting queue to zip: {e}")
                                traceback.print_exc()
                                return [], ""

                        # --- Connect Buttons ---
                        refresh_button.click(fn=update_stats, inputs=[], outputs=[queue_status, queue_stats_display])
                        
                        # Confirmation logic for Cancel Queue
                        def show_cancel_confirmation():
                            return gr.update(visible=False), gr.update(visible=True)

                        def hide_cancel_confirmation():
                            return gr.update(visible=True), gr.update(visible=False)

                        def confirmed_clear_all_jobs():
                            qs_data, qs_text = clear_all_jobs()
                            return qs_data, qs_text, gr.update(visible=True), gr.update(visible=False)

                        clear_queue_button.click(fn=show_cancel_confirmation, inputs=None, outputs=[queue_controls_row, confirm_cancel_row])
                        confirm_cancel_no_btn.click(fn=hide_cancel_confirmation, inputs=None, outputs=[queue_controls_row, confirm_cancel_row])
                        confirm_cancel_yes_btn.click(fn=confirmed_clear_all_jobs, inputs=None, outputs=[queue_status, queue_stats_display, queue_controls_row, confirm_cancel_row])

                        clear_complete_button.click(fn=clear_completed_jobs, inputs=[], outputs=[queue_status, queue_stats_display])
                        queue_export_button.click(fn=export_queue_to_zip, inputs=[], outputs=[queue_status, queue_stats_display])

                        # Create a container for thumbnails (kept for potential future use, though not displayed in DataFrame)
                        with gr.Row():
                            thumbnail_container = gr.Column()
                            thumbnail_container.elem_classes = ["thumbnail-container"]

                        # Add CSS for thumbnails
                        
            with gr.Tab("輸出", id="outputs_tab"): # "Outputs" -> "輸出"
                outputDirectory_video = settings.get("output_dir", settings.default_settings['output_dir'])
                outputDirectory_metadata = settings.get("metadata_dir", settings.default_settings['metadata_dir'])
                def get_gallery_items():
                    items = []
                    for f in os.listdir(outputDirectory_metadata):
                        if f.endswith(".png"):
                            prefix = os.path.splitext(f)[0]
                            latest_video = get_latest_video_version(prefix)
                            if latest_video:
                                video_path = os.path.join(outputDirectory_video, latest_video)
                                mtime = os.path.getmtime(video_path)
                                preview_path = os.path.join(outputDirectory_metadata, f)
                                items.append((preview_path, prefix, mtime))
                    items.sort(key=lambda x: x[2], reverse=True)
                    return [(i[0], i[1]) for i in items]
                def get_latest_video_version(prefix):
                    max_number = -1
                    selected_file = None
                    for f in os.listdir(outputDirectory_video):
                        if f.startswith(prefix + "_") and f.endswith(".mp4"):
                            # Skip files that include "combined" in their name
                            if "combined" in f:
                                continue
                            try:
                                num = int(f.replace(prefix + "_", '').replace(".mp4", ''))
                                if num > max_number:
                                    max_number = num
                                    selected_file = f
                            except ValueError:
                                # Ignore files that do not have a valid number in their name
                                continue
                    return selected_file
                # load_video_and_info_from_prefix now also returns button visibility
                def load_video_and_info_from_prefix(prefix):
                    video_file = get_latest_video_version(prefix)
                    json_path = os.path.join(outputDirectory_metadata, prefix) + ".json"
                    
                    if not video_file or not os.path.exists(os.path.join(outputDirectory_video, video_file)) or not os.path.exists(json_path):
                        # If video or info not found, button should be hidden
                        return None, "影片或 JSON 未找到。", gr.update(visible=False) # Translated "Video or JSON not found."

                    video_path = os.path.join(outputDirectory_video, video_file)
                    info_content = {"description": "no info"}
                    if os.path.exists(json_path):
                        with open(json_path, "r", encoding="utf-8") as f:
                            info_content = json.load(f)
                    # If video and info found, button should be visible
                    return video_path, json.dumps(info_content, indent=2, ensure_ascii=False), gr.update(visible=True)

                gallery_items_state = gr.State(get_gallery_items())
                selected_original_video_path_state = gr.State(None) # Holds the ORIGINAL, UNPROCESSED path
                with gr.Row():
                    with gr.Column(scale=2):
                        thumbs = gr.Gallery(
                            # value=[i[0] for i in get_gallery_items()],
                            columns=[4],
                            allow_preview=False,
                            object_fit="cover",
                            height="auto"
                        )
                        refresh_button = gr.Button("更新") # "Update" -> "更新"
                    with gr.Column(scale=5):
                        video_out = gr.Video(sources=[], autoplay=True, loop=True, visible=False)
                    with gr.Column(scale=1):
                        info_out = gr.Textbox(label="生成資訊", visible=False) # "Generation info" -> "生成資訊"
                        send_to_toolbox_btn = gr.Button("➡️ 送至後處理", visible=False) # "➡️ Send to Post-processing" -> "➡️ 送至後處理"
                    def refresh_gallery():
                        new_items = get_gallery_items()
                        return gr.update(value=[i[0] for i in new_items]), new_items
                    refresh_button.click(fn=refresh_gallery, outputs=[thumbs, gallery_items_state])
                    
                    # MODIFIED: on_select now handles visibility of the new button
                    def on_select(evt: gr.SelectData, gallery_items):
                        if evt.index is None or not gallery_items or evt.index >= len(gallery_items):
                            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), None

                        prefix = gallery_items[evt.index][1]
                        # original_video_path is e.g., "outputs/my_actual_video.mp4"
                        original_video_path, info_string, button_visibility_update = load_video_and_info_from_prefix(prefix)

                        # Determine visibility for video and info based on whether video_path was found
                        video_out_update = gr.update(value=original_video_path, visible=bool(original_video_path))
                        info_out_update = gr.update(value=info_string, visible=bool(original_video_path))

                        # IMPORTANT: Store the ORIGINAL, UNPROCESSED path in the gr.State
                        return video_out_update, info_out_update, button_visibility_update, original_video_path

                    thumbs.select(
                        fn=on_select,
                        inputs=[gallery_items_state],
                        outputs=[video_out, info_out, send_to_toolbox_btn, selected_original_video_path_state] # Output original path to State
                    )
            with gr.Tab("後處理", id="toolbox_tab"): # "Post-processing" -> "後處理"
                # Call the function from toolbox_app.py to build the Toolbox UI
                # The toolbox_ui_layout (e.g., a gr.Column) is automatically placed here.                
                toolbox_ui_layout, tb_target_video_input = tb_create_video_toolbox_ui()
                
            with gr.Tab("設定"): # "Settings" -> "設定"
                with gr.Row():
                    with gr.Column():
                        save_metadata = gr.Checkbox(
                            label="儲存中繼資料", # "Save Metadata" -> "儲存中繼資料"
                            info="儲存為 JSON 檔案", # "Save to JSON file" -> "儲存為 JSON 檔案"
                            value=settings.get("save_metadata", 6),
                        )
                        gpu_memory_preservation = gr.Slider(
                            label="記憶體緩衝區以保持穩定性 (VRAM GB)", # "Memory Buffer for Stability (VRAM GB)" -> "記憶體緩衝區以保持穩定性 (VRAM GB)"
                            minimum=1,
                            maximum=128,
                            step=0.1,
                            value=settings.get("gpu_memory_preservation", 6),
                            info="如果看到電腦凍結、生成停滯或採樣步驟非常慢（每次嘗試 1G），請增加保留。否則較小的緩衝區速度更快。某些模型和 LoRA 比其他模型需要更多緩衝區。（5.5 - 8.5 是常見範圍）" # Info translated
                        )
                        mp4_crf = gr.Slider(
                            label="MP4 壓縮率", # "MP4 Compression" -> "MP4 壓縮率"
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=settings.get("mp4_crf", 16),
                            info="數值越低表示品質越好。0 為無壓縮。如果輸出為黑色，請更改為 16。" # Info translated
                        )
                        clean_up_videos = gr.Checkbox(
                            label="清理影片檔案", # "Clean up video files" -> "清理影片檔案"
                            value=settings.get("clean_up_videos", True),
                            info="如果勾選，生成後只會保留最終影片。" # Info translated
                        )
                        cleanup_temp_folder = gr.Checkbox(
                            label="生成後清理暫存資料夾", # "Clean up temp folder after generation" -> "生成後清理暫存資料夾"
                            visible=False,
                            value=settings.get("cleanup_temp_folder", True),
                            info="如果勾選，每次生成後將清理暫存檔案。" # Info translated
                        )
                        
                        # gr.Markdown("---")
                        # gr.Markdown("### Startup Settings")
                        gr.Markdown("")  
                        # Initial values for startup preset dropdown
                        # Ensure settings and load_presets are available in this scope
                        initial_startup_model_val = settings.get("startup_model_type", "None")
                        initial_startup_presets_choices_val = []
                        initial_startup_preset_value_val = None

                        if initial_startup_model_val and initial_startup_model_val != "None":
                            # load_presets is defined further down in create_interface
                            initial_startup_presets_choices_val = load_presets(initial_startup_model_val)
                            saved_preset_for_initial_model_val = settings.get("startup_preset_name")
                            if saved_preset_for_initial_model_val in initial_startup_presets_choices_val:
                                initial_startup_preset_value_val = saved_preset_for_initial_model_val
                        
                        startup_model_type_dropdown = gr.Dropdown(
                            label="啟動模型類型", # "Startup Model Type" -> "啟動模型類型"
                            choices=["無"] + [choice[0] for choice in model_type.choices if choice[0] != "XY Plot"], # "None" -> "無"
                            value=initial_startup_model_val,
                            info="選擇在啟動時載入的模型類型。'無' 以禁用。" # Info translated, "None" -> "無"
                        )
                        startup_preset_name_dropdown = gr.Dropdown(
                            label="啟動預設", # "Startup Preset" -> "啟動預設"
                            choices=initial_startup_presets_choices_val,
                            value=initial_startup_preset_value_val,
                            info="為啟動模型選擇一個預設。當啟動模型類型改變時更新。", # Info translated
                            interactive=True # Must be interactive to be updated by another component
                        )

                        with gr.Accordion("系統提示詞", open=False): # "System Prompt" -> "系統提示詞"
                            with gr.Row(equal_height=True): # New Row to contain checkbox and reset button
                                override_system_prompt = gr.Checkbox(
                                    label="覆寫系統提示詞", # "Override System Prompt" -> "覆寫系統提示詞"
                                    value=settings.get("override_system_prompt", False),
                                    info="如果勾選，將使用下面的系統提示詞模板而不是預設模板。", # Info translated
                                    scale=1 # Give checkbox some scale
                                )
                                reset_system_prompt_btn = gr.Button(
                                    "重置", # "Reset" -> "重置"
                                    scale=0
                                )
                            system_prompt_template = gr.Textbox(
                                label="系統提示詞模板", # "System Prompt Template" -> "系統提示詞模板"
                                value=settings.get("system_prompt_template", "{\"template\": \"<|start_header_id|>system<|end_header_id|>\\n\\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{}<|eot_id|>\", \"crop_start\": 95}"),
                                lines=10,
                                info="用於影片生成的系統提示詞模板。必須是有效的 JSON 或 Python 字典字串，包含 'template' 和 'crop_start' 鍵。範例：{\"template\": \"您的模板在此\", \"crop_start\": 95}" # Info translated
                            )
                            # The reset_system_prompt_btn is now defined above within the Row

                        # --- Settings Tab Event Handlers ---

                        output_dir = gr.Textbox(
                            label="輸出目錄", # "Output Directory" -> "輸出目錄"
                            value=settings.get("output_dir"),
                            placeholder="儲存生成影片的路徑" # Placeholder translated
                        )
                        metadata_dir = gr.Textbox(
                            label="中繼資料目錄", # "Metadata Directory" -> "中繼資料目錄"
                            value=settings.get("metadata_dir"),
                            placeholder="儲存中繼資料檔案的路徑" # Placeholder translated
                        )
                        lora_dir = gr.Textbox(
                            label="LoRA 目錄", # "LoRA Directory" -> "LoRA 目錄"
                            value=settings.get("lora_dir"),
                            placeholder="LoRA 模型路徑" # Placeholder translated
                        )
                        gradio_temp_dir = gr.Textbox(label="Gradio 暫存目錄", value=settings.get("gradio_temp_dir")) # "Gradio Temporary Directory" -> "Gradio 暫存目錄"
                        auto_save = gr.Checkbox(
                            label="自動儲存設定", # "Auto-save settings" -> "自動儲存設定"
                            value=settings.get("auto_save_settings", True)
                        )
                        # Add Gradio Theme Dropdown
                        gradio_themes = ["default", "base", "soft", "glass", "mono", "origin", "citrus", "monochrome", "ocean", "NoCrypt/miku", "earneleh/paris", "gstaff/xkcd"]
                        theme_dropdown = gr.Dropdown(
                            label="主題", # "Theme" -> "主題"
                            choices=gradio_themes,
                            value=settings.get("gradio_theme", "default"),
                            info="選擇 Gradio UI 主題。需要重新啟動。" # Info translated
                        )
                        save_btn = gr.Button("儲存設定") # "Save Settings" -> "儲存設定"
                        cleanup_btn = gr.Button("清理暫存檔案") # "Clean Up Temporary Files" -> "清理暫存檔案"
                        status = gr.HTML("")
                        cleanup_output = gr.Textbox(label="清理狀態", interactive=False) # "Cleanup Status" -> "清理狀態"

                        def save_settings(save_metadata, gpu_memory_preservation, mp4_crf, clean_up_videos, cleanup_temp_folder, override_system_prompt_value, system_prompt_template_value, output_dir, metadata_dir, lora_dir, gradio_temp_dir, auto_save, selected_theme, startup_model_type_val, startup_preset_name_val):
                            """Handles the manual 'Save Settings' button click."""
                            # This function is for the manual save button.
                            # It collects all current UI values and saves them.
                            # The auto-save logic is handled by individual .change() and .blur() handlers
                            # calling settings.set().

                            # First, update the settings object with all current values from the UI
                            try:
                                # Save the system prompt template as is, without trying to parse it
                                # The hunyuan.py file will handle parsing it when needed
                                processed_template = system_prompt_template_value
                                
                                settings.save_settings(
                                    save_metadata=save_metadata,
                                    gpu_memory_preservation=gpu_memory_preservation,
                                    mp4_crf=mp4_crf,
                                    clean_up_videos=clean_up_videos,
                                    cleanup_temp_folder=cleanup_temp_folder,
                                    override_system_prompt=override_system_prompt_value,
                                    system_prompt_template=processed_template,
                                    output_dir=output_dir,
                                    metadata_dir=metadata_dir,
                                    lora_dir=lora_dir,
                                    gradio_temp_dir=gradio_temp_dir,
                                    auto_save_settings=auto_save,
                                    gradio_theme=selected_theme,
                                    startup_model_type=startup_model_type_val,
                                    startup_preset_name=startup_preset_name_val
                                )
                                # settings.save_settings() is called inside settings.save_settings if auto_save is true,
                                # but for the manual button, we ensure it saves regardless of the auto_save flag's previous state.
                                # The call above to settings.save_settings already handles writing to disk.
                                return "<p style='color:green;'>設定已成功儲存！主題變更需要重新啟動。</p>" # Translated
                            except Exception as e:
                                return f"<p style='color:red;'>儲存設定時發生錯誤：{str(e)}</p>" # Translated

                        def handle_individual_setting_change(key, value, setting_name_for_ui):
                            """Called by .change() and .submit() events of individual setting components."""
                            if key == "auto_save_settings":
                                # For the "auto_save_settings" checkbox itself:
                                # 1. Update its value directly in the settings object in memory.
                                #    This bypasses the conditional save logic within settings.set() for this specific action.
                                settings.settings[key] = value
                                # 2. Force a save of all settings to disk. This will be correct because either:
                                #    - auto_save_settings is turning True: so all changes already in memory need to be saved now.
                                #    - auto_save_settings turning False from True: prior changes already saved so only auto_save_settings will be saved.
                                settings.save_settings()
                                # 3. Provide feedback.
                                if value is True:
                                    return f"<p style='color:green;'>'{setting_name_for_ui}' 設定已開啟並儲存。</p>" # Translated
                                else:
                                    return f"<p style='color:green;'>'{setting_name_for_ui}' 設定已關閉並儲存。</p>" # Translated
                            else:
                                # For all other settings:
                                # Let settings.set() handle the auto-save logic based on the current "auto_save_settings" value.
                                settings.set(key, value) # settings.set() will call save_settings() if auto_save is True
                                if settings.get("auto_save_settings"): # Check the current state of auto_save
                                    return f"<p style='color:blue;'>'{setting_name_for_ui}' 設定已自動儲存。</p>" # Translated
                                else:
                                    return f"<p style='color:gray;'>'{setting_name_for_ui}' 設定已變更（自動儲存已關閉，請點擊「儲存設定」）。</p>" # Translated

                        save_btn.click(
                            fn=save_settings,
                            inputs=[save_metadata, gpu_memory_preservation, mp4_crf, clean_up_videos, cleanup_temp_folder, override_system_prompt, system_prompt_template, output_dir, metadata_dir, lora_dir, gradio_temp_dir, auto_save, theme_dropdown, startup_model_type_dropdown, startup_preset_name_dropdown],
                            outputs=[status]
                        )

                        def reset_system_prompt_template_value():
                            return settings.default_settings["system_prompt_template"], False

                        reset_system_prompt_btn.click(
                            fn=reset_system_prompt_template_value,
                            outputs=[system_prompt_template, override_system_prompt]
                        ).then( # Trigger auto-save for the reset values if auto-save is on
                            lambda val_template, val_override: handle_individual_setting_change("system_prompt_template", val_template, "系統提示詞模板") or handle_individual_setting_change("override_system_prompt", val_override, "覆寫系統提示詞"), # Translated
                            inputs=[system_prompt_template, override_system_prompt], outputs=[status])

                        def cleanup_temp_files():
                            """Clean up temporary files and folders in the Gradio temp directory"""
                            temp_dir = settings.get("gradio_temp_dir")
                            if not temp_dir or not os.path.exists(temp_dir):
                                return "找不到暫存目錄或目錄不存在。" # Translated "No temporary directory found or directory does not exist."
                            
                            try:
                                # Get all items in the temp directory
                                items = os.listdir(temp_dir)
                                removed_count = 0
                                print(f"Finding items in {temp_dir}")
                                for item in items:
                                    item_path = os.path.join(temp_dir, item)
                                    try:
                                        if os.path.isfile(item_path) or os.path.islink(item_path):
                                            print(f"Removing {item_path}")
                                            os.remove(item_path)
                                            removed_count += 1
                                        elif os.path.isdir(item_path):
                                            print(f"Removing directory {item_path}")
                                            shutil.rmtree(item_path)
                                            removed_count += 1
                                    except Exception as e:
                                        print(f"Error removing {item_path}: {e}")
                                
                                return f"已清理 {removed_count} 個暫存檔案/資料夾。" # Translated "Cleaned up {removed_count} temporary files/folders."
                            except Exception as e:
                                return f"清理暫存檔案時發生錯誤：{str(e)}" # Translated "Error cleaning up temporary files: {str(e)}"

                        # Add .change handlers for auto-saving individual settings
                        save_metadata.change(lambda v: handle_individual_setting_change("save_metadata", v, "儲存中繼資料"), inputs=[save_metadata], outputs=[status]) # Translated
                        gpu_memory_preservation.change(lambda v: handle_individual_setting_change("gpu_memory_preservation", v, "GPU 記憶體保留"), inputs=[gpu_memory_preservation], outputs=[status]) # Translated
                        mp4_crf.change(lambda v: handle_individual_setting_change("mp4_crf", v, "MP4 壓縮"), inputs=[mp4_crf], outputs=[status]) # Translated
                        clean_up_videos.change(lambda v: handle_individual_setting_change("clean_up_videos", v, "清理影片"), inputs=[clean_up_videos], outputs=[status]) # Translated

                        # This setting is not visible in the UI, but still handle it in case it's re-added to the UI
                        cleanup_temp_folder.change(lambda v: handle_individual_setting_change("cleanup_temp_folder", v, "清理暫存資料夾"), inputs=[cleanup_temp_folder], outputs=[status]) # Translated

                        override_system_prompt.change(lambda v: handle_individual_setting_change("override_system_prompt", v, "覆寫系統提示詞"), inputs=[override_system_prompt], outputs=[status]) # Translated
                        # Using .blur for text changes so they are processed after the user finishes, not on every keystroke
                        system_prompt_template.blur(lambda v: handle_individual_setting_change("system_prompt_template", v, "系統提示詞模板"), inputs=[system_prompt_template], outputs=[status]) # Translated
                        # reset_system_prompt_btn # is handled separately above, on click
                        
                        # Using .blur for text changes so they are processed after the user finishes, not on every keystroke
                        output_dir.blur(lambda v: handle_individual_setting_change("output_dir", v, "輸出目錄"), inputs=[output_dir], outputs=[status]) # Translated
                        metadata_dir.blur(lambda v: handle_individual_setting_change("metadata_dir", v, "中繼資料目錄"), inputs=[metadata_dir], outputs=[status]) # Translated
                        lora_dir.blur(lambda v: handle_individual_setting_change("lora_dir", v, "LoRA 目錄"), inputs=[lora_dir], outputs=[status]) # Translated
                        gradio_temp_dir.blur(lambda v: handle_individual_setting_change("gradio_temp_dir", v, "Gradio 暫存目錄"), inputs=[gradio_temp_dir], outputs=[status]) # Translated
                        
                        auto_save.change(lambda v: handle_individual_setting_change("auto_save_settings", v, "自動儲存設定"), inputs=[auto_save], outputs=[status]) # Translated
                        theme_dropdown.change(lambda v: handle_individual_setting_change("gradio_theme", v, "主題"), inputs=[theme_dropdown], outputs=[status]) # Translated

                        # Event handlers for startup settings
                        def update_startup_preset_dropdown_choices(selected_startup_model_type_from_ui):
                            if not selected_startup_model_type_from_ui or selected_startup_model_type_from_ui == "None":
                                return gr.update(choices=[], value=None)

                            loaded_presets_for_model = load_presets(selected_startup_model_type_from_ui)
                            
                            # Get the preset name that was saved for the *previous* model type
                            current_saved_startup_preset = settings.get("startup_preset_name")

                            # Default to None
                            value_to_select = None
                            # If the previously saved preset name exists for the new model, select it
                            if current_saved_startup_preset and current_saved_startup_preset in loaded_presets_for_model:
                                value_to_select = current_saved_startup_preset
                            
                            return gr.update(choices=loaded_presets_for_model, value=value_to_select)

                        startup_model_type_dropdown.change(
                            fn=lambda v: handle_individual_setting_change("startup_model_type", v, "啟動模型類型"), # Translated
                            inputs=[startup_model_type_dropdown], outputs=[status]
                        ).then( # Chain the update to the preset dropdown
                            fn=update_startup_preset_dropdown_choices, inputs=[startup_model_type_dropdown], outputs=[startup_preset_name_dropdown])
                        startup_preset_name_dropdown.change(lambda v: handle_individual_setting_change("startup_preset_name", v, "啟動預設名稱"), inputs=[startup_preset_name_dropdown], outputs=[status]) # Translated

        # --- Event Handlers and Connections (Now correctly indented) ---

        # --- Connect Monitoring ---
        # Auto-check for current job on page load and job change
        def check_for_current_job():
            # This function will be called when the interface loads
            # It will check if there's a current job in the queue and update the UI
            with job_queue.lock:
                current_job = job_queue.current_job
                if current_job:
                    # Return all the necessary information to update the preview windows
                    job_id = current_job.id
                    result = current_job.result
                    preview = current_job.progress_data.get('preview') if current_job.progress_data else None
                    desc = current_job.progress_data.get('desc', '') if current_job.progress_data else ''
                    html = current_job.progress_data.get('html', '') if current_job.progress_data else ''
                    
                    # Also trigger the monitor_job function to start monitoring this job
                    print(f"Auto-check found current job {job_id}, triggering monitor_job")
                    return job_id, result, preview, desc, html
                return None, None, None, '', ''
                
        # Auto-check for current job on page load and handle handoff between jobs.
        def check_for_current_job_and_monitor():
            # This function is now the key to the handoff.
            # It finds the current job and returns its ID, which will trigger the monitor.
            job_id, result, preview, desc, html = check_for_current_job()
            # We also need to get fresh stats at the same time.
            queue_status_data, queue_stats_text = update_stats()
            # Return everything needed to update the UI atomically.
            return job_id, result, preview, desc, html, queue_status_data, queue_stats_text

        # Connect the main process function (wrapper for adding to queue)
        def process_with_queue_update(model_type_arg, *args):
            # Call update_stats to get both queue_status_data and queue_stats_text
            queue_status_data, queue_stats_text = update_stats() # MODIFIED

            # Extract all arguments (ensure order matches inputs lists)
            # The order here MUST match the order in the `ips` list.
            # RT_BORG: Global settings gpu_memory_preservation, mp4_crf, save_metadata removed from direct args.
            (input_image_arg,
             input_video_arg,
             end_frame_image_original_arg,
             end_frame_strength_original_arg,
             prompt_text_arg,
             n_prompt_arg,
             seed_arg, # the seed value
             randomize_seed_arg, # the boolean value of the checkbox
             total_second_length_arg,
             latent_window_size_arg,
             steps_arg,
             cfg_arg, 
             gs_arg,
             rs_arg,
             use_teacache_arg,
             teacache_num_steps_arg,
             teacache_rel_l1_thresh_arg,
             blend_sections_arg,
             latent_type_arg,
             clean_up_videos_arg, # UI checkbox from Generate tab
             selected_loras_arg,
             resolutionW_arg, resolutionH_arg,
             combine_with_source_arg, 
             num_cleaned_frames_arg,
             lora_names_states_arg,    # This is from lora_names_states (gr.State)
             *lora_slider_values_tuple # Remaining args are LoRA slider values
            ) = args
            # DO NOT parse the prompt here. Parsing happens once in the worker.

            # Determine the model type to send to the backend
            backend_model_type = model_type_arg # model_type_arg is the UI selection
            if model_type_arg == "Video with Endframe":
                backend_model_type = "Video" # The backend "Video" model_type handles with and without endframe

            # Use the appropriate input based on model type
            is_ui_video_model = is_video_model(model_type_arg)
            input_data = input_video_arg if is_ui_video_model else input_image_arg

            # Define actual end_frame params to pass to backend
            actual_end_frame_image_for_backend = None
            actual_end_frame_strength_for_backend = 1.0  # Default strength

            if model_type_arg == "Original with Endframe" or model_type_arg == "F1 with Endframe" or model_type_arg == "Video with Endframe":
                actual_end_frame_image_for_backend = end_frame_image_original_arg
                actual_end_frame_strength_for_backend = end_frame_strength_original_arg

            # Get the input video path for Video model
            input_image_path = None
            if is_ui_video_model and input_video_arg is not None:
                # For Video models, input_video contains the path to the video file
                input_image_path = input_video_arg

            # Use the current seed value as is for this job
            # Call the process function with all arguments
            # Pass the backend_model_type and the ORIGINAL prompt_text string to the backend process function
            result = process_fn(backend_model_type, input_data, actual_end_frame_image_for_backend, actual_end_frame_strength_for_backend,
                                 prompt_text_arg, n_prompt_arg, seed_arg, total_second_length_arg,
                                 latent_window_size_arg, steps_arg, cfg_arg, gs_arg, rs_arg,
                                 use_teacache_arg, teacache_num_steps_arg, teacache_rel_l1_thresh_arg,
                                 blend_sections_arg, latent_type_arg, clean_up_videos_arg, # clean_up_videos_arg is from UI
                                 selected_loras_arg, resolutionW_arg, resolutionH_arg, 
                                 input_image_path, 
                                 combine_with_source_arg,
                                 num_cleaned_frames_arg,
                                 lora_names_states_arg,
                                 *lora_slider_values_tuple
                                )
            # If randomize_seed is checked, generate a new random seed for the next job
            new_seed_value = None
            if randomize_seed_arg:
                new_seed_value = random.randint(0, 21474)
                print(f"Generated new seed for next job: {new_seed_value}")

            # Create the button update for start_button WITHOUT interactive=True.
            # The interactivity will be set by update_start_button_state later in the chain.
            start_button_update_after_add = gr.update(value="加入佇列") # Translated
            
            # If a job ID was created, automatically start monitoring it and update queue
            if result and result[1]:  # Check if job_id exists in results
                job_id = result[1]
                # queue_status_data = update_queue_status_fn() # OLD: update_stats now called earlier
                # Call update_stats again AFTER the job is added to get the freshest stats
                queue_status_data, queue_stats_text = update_stats()


                # Add the new seed value to the results if randomize is checked
                if new_seed_value is not None:
                    # Use result[6] directly for end_button to preserve its value. Add gr.update() for video_input_required_message.
                    return [result[0], job_id, result[2], result[3], result[4], start_button_update_after_add, result[6], queue_status_data, queue_stats_text, new_seed_value, gr.update()]
                else:
                    # Use result[6] directly for end_button to preserve its value. Add gr.update() for video_input_required_message.
                    return [result[0], job_id, result[2], result[3], result[4], start_button_update_after_add, result[6], queue_status_data, queue_stats_text, gr.update(), gr.update()]

            # If no job ID was created, still return the new seed if randomize is checked
            # Also, ensure we return the latest stats even if no job was created (e.g., error during param validation)
            queue_status_data, queue_stats_text = update_stats()
            if new_seed_value is not None:
                # Make sure to preserve the end_button update from result[6]
                return [result[0], result[1], result[2], result[3], result[4], start_button_update_after_add, result[6], queue_status_data, queue_stats_text, new_seed_value, gr.update()]
            else:
                # Make sure to preserve the end_button update from result[6]
                return [result[0], result[1], result[2], result[3], result[4], start_button_update_after_add, result[6], queue_status_data, queue_stats_text, gr.update(), gr.update()]

        # Custom end process function that ensures the queue is updated and changes button text
        def end_process_with_update():
            _ = end_process_fn() # Call the original end_process_fn
            # Now, get fresh stats for both queue table and toolbar
            queue_status_data, queue_stats_text = update_stats()
            
            # Don't try to get the new job ID immediately after cancellation
            # The monitor_job function will handle the transition to the next job
            
            # Change the cancel button text to "Cancelling..." and make it non-interactive
            # This ensures the button stays in this state until the job is fully cancelled
            return queue_status_data, queue_stats_text, gr.update(value="取消中...", interactive=False), gr.update(value=None) # Translated "Cancelling..."

        # MODIFIED handle_send_video_to_toolbox:
        def handle_send_video_to_toolbox(original_path_from_state): # Input is now the original path from gr.State
            print(f"Sending selected Outputs' video to Post-processing: {original_path_from_state}")

            if original_path_from_state and isinstance(original_path_from_state, str) and os.path.exists(original_path_from_state):
                # tb_target_video_input will now process the ORIGINAL path (e.g., "outputs/my_actual_video.mp4").
                return gr.update(value=original_path_from_state), gr.update(selected="toolbox_tab")
            else:
                print(f"No valid video path (from State) found to send. Path: {original_path_from_state}")
                return gr.update(), gr.update()

        send_to_toolbox_btn.click(
            fn=handle_send_video_to_toolbox,
            inputs=[selected_original_video_path_state], # INPUT IS NOW THE gr.State holding the ORIGINAL path
            outputs=[
                tb_target_video_input, # This is tb_input_video_component from toolbox_app.py
                main_tabs_component
            ]
        )
        
        # --- Inputs Lists ---
        # --- Inputs for all models ---
        ips = [
            input_image,                 # Corresponds to input_image_arg
            input_video,                 # Corresponds to input_video_arg
            end_frame_image_original,    # Corresponds to end_frame_image_original_arg
            end_frame_strength_original, # Corresponds to end_frame_strength_original_arg
            prompt,                      # Corresponds to prompt_text_arg
            n_prompt,                    # Corresponds to n_prompt_arg
            seed,                        # Corresponds to seed_arg
            randomize_seed,              # Corresponds to randomize_seed_arg
            total_second_length,         # Corresponds to total_second_length_arg
            latent_window_size,          # Corresponds to latent_window_size_arg
            steps,                       # Corresponds to steps_arg
            cfg,                         # Corresponds to cfg_arg
            gs,                          # Corresponds to gs_arg
            rs,                          # Corresponds to rs_arg
            use_teacache,                # Corresponds to use_teacache_arg
            teacache_num_steps,          # Corresponds to teacache_num_steps_arg
            teacache_rel_l1_thresh,      # Corresponds to teacache_rel_l1_thresh_arg
            blend_sections,              # Corresponds to blend_sections_arg
            latent_type,                 # Corresponds to latent_type_arg
            clean_up_videos,             # Corresponds to clean_up_videos_arg (UI checkbox)
            lora_selector,               # Corresponds to selected_loras_arg
            resolutionW,                 # Corresponds to resolutionW_arg
            resolutionH,                 # Corresponds to resolutionH_arg
            combine_with_source,         # Corresponds to combine_with_source_arg
            num_cleaned_frames,          # Corresponds to num_cleaned_frames_arg
            lora_names_states            # Corresponds to lora_names_states_arg
        ]
        # Add LoRA sliders to the input list
        ips.extend([lora_sliders[lora] for lora in lora_names])


        # --- Connect Buttons ---
        def handle_start_button(selected_model, *args):
            # For other model types, use the regular process function
            return process_with_queue_update(selected_model, *args)
                
        # Validation ensures the start button is only enabled when appropriate
        def update_start_button_state(*args):
            """
            Validation fails if a video model is selected and no input video is provided.
            Updates the start button interactivity and validation message visibility.
            Handles variable inputs from different Gradio event chains.
            """
            # The required values are the last two arguments provided by the Gradio event
            if len(args) >= 2:
                selected_model = args[-2]
                input_video_value = args[-1]
            else:
                # Fallback or error handling if not enough arguments are received
                # This might happen if the event is triggered in an unexpected way
                print(f"Warning: update_start_button_state received {len(args)} args, expected at least 2.")
                # Default to a safe state (button disabled)
                return gr.Button(value="錯誤", interactive=False), gr.update(visible=True) # Translated "Error"

            video_provided = input_video_value is not None
            
            if is_video_model(selected_model) and not video_provided:
                # Video model selected, but no video provided
                return gr.Button(value="缺少影片", interactive=False), gr.update(visible=True) # Translated "Missing Video"
            else:
                # Either not a video model, or video model selected and video provided
                return gr.update(value="加入佇列", interactive=True), gr.update(visible=False) # Translated "Add to Queue"
        # Function to update button state before processing
        def update_button_before_processing(selected_model, *args):
            # First update the button to show "Adding..." and disable it
            # Also return current stats so they don't get blanked out during the "Adding..." phase
            qs_data, qs_text = update_stats()
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(value="加入中...", interactive=False), gr.update(), qs_data, qs_text, gr.update(), gr.update() # Translated "Adding..."
        
        # Connect the start button to first update its state
        start_button.click(
            fn=update_button_before_processing,
            inputs=[model_type] + ips,
            outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button, queue_status, queue_stats_display, seed, video_input_required_message]
        ).then(
            # Then process the job
            fn=handle_start_button,
            inputs=[model_type] + ips,
            outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button, queue_status, queue_stats_display, seed, video_input_required_message] # Added video_input_required_message
        ).then( # Ensure validation is re-checked after job processing completes
            fn=update_start_button_state,
            inputs=[model_type, input_video], # Current values of model_type and input_video
            outputs=[start_button, video_input_required_message]
        )

        # --- START OF REFACTORED XY PLOT EVENT WIRING ---
        # Get the process button from the created components
        xy_plot_process_btn = xy_plot_components["process_btn"]
        
        # Prepare the process function with its static dependencies (job_queue, settings)
        fn_xy_process_with_deps = functools.partial(xy_plot_process, job_queue, settings)
        
        # Construct the full list of inputs for the click handler in the correct order
        c = xy_plot_components
        xy_plot_input_components = [
            c["model_type"], c["input_image"], c["end_frame_image_original"],
            c["end_frame_strength_original"], c["latent_type"], c["prompt"], 
            c["blend_sections"], c["steps"], c["total_second_length"], 
            resolutionW, resolutionH, # The components from the main UI
            c["seed"], c["randomize_seed"], c["use_teacache"], 
            c["teacache_num_steps"], c["teacache_rel_l1_thresh"], 
            c["latent_window_size"], c["cfg"], c["gs"], c["rs"], 
            c["gpu_memory_preservation"], c["mp4_crf"], 
            c["axis_x_switch"], c["axis_x_value_text"], c["axis_x_value_dropdown"], 
            c["axis_y_switch"], c["axis_y_value_text"], c["axis_y_value_dropdown"], 
            c["axis_z_switch"], c["axis_z_value_text"], c["axis_z_value_dropdown"],
            c["lora_selector"]
        ]
        # LoRA sliders are in a dictionary, so we add their values to the list
        xy_plot_input_components.extend(c["lora_sliders"].values())

        # Wire the click handler for the XY Plot button
        xy_plot_process_btn.click(
            fn=fn_xy_process_with_deps, 
            inputs=xy_plot_input_components, 
            outputs=[xy_plot_status, xy_plot_output]
        ).then(
            fn=update_stats,
            inputs=None, 
            outputs=[queue_status, queue_stats_display]
        ).then(
            fn=check_for_current_job,
            inputs=None, 
            outputs=[current_job_id, result_video, preview_image, progress_desc, progress_bar]
        )
        # --- END OF REFACTORED XY PLOT EVENT WIRING ---


        # MODIFIED: on_model_type_change to handle new "XY Plot" option
        def on_model_type_change(selected_model):
            is_xy_plot = selected_model == "XY Plot"
            is_ui_video_model_flag = is_video_model(selected_model)
            shows_end_frame = selected_model in ["Original with Endframe", "Video with Endframe"]

            return (
                gr.update(visible=not is_xy_plot),   # standard_generation_group
                gr.update(visible=is_xy_plot),       # xy_group
                gr.update(visible=not is_xy_plot and not is_ui_video_model_flag),   # image_input_group
                gr.update(visible=not is_xy_plot and is_ui_video_model_flag),       # video_input_group
                gr.update(visible=not is_xy_plot and shows_end_frame),      # end_frame_group_original
                gr.update(visible=not is_xy_plot and shows_end_frame),      # end_frame_slider_group
                gr.update(visible=not is_xy_plot),   # start_button
                gr.update(visible=is_xy_plot)        # xy_plot_process_btn
            )

        # Model change listener
        model_type.change(
            fn=on_model_type_change,
            inputs=model_type,
            outputs=[
                standard_generation_group, 
                xy_group,
                image_input_group,
                video_input_group,
                end_frame_group_original,
                end_frame_slider_group,
                start_button,
                xy_plot_process_btn # This is the button returned from the dictionary
            ]
        ).then( # Also trigger validation after model type changes
            fn=update_start_button_state,
            inputs=[model_type, input_video],
            outputs=[start_button, video_input_required_message]
        )
        
        # Connect input_video change to the validation function
        input_video.change(
            fn=update_start_button_state,
            inputs=[model_type, input_video],
            outputs=[start_button, video_input_required_message]
        )
        # Also trigger validation when video is cleared
        input_video.clear(
            fn=update_start_button_state,
            inputs=[model_type, input_video],
            outputs=[start_button, video_input_required_message]
        )

        
        # Auto-monitor the current job when job_id changes
        current_job_id.change(
            fn=monitor_fn,
            inputs=[current_job_id],
            outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button]
        ).then(
            fn=update_stats, # When a monitor finishes, always update the stats.
            inputs=None,
            outputs=[queue_status, queue_stats_display]
        ).then( # re-validate button state
            fn=update_start_button_state,
            inputs=[model_type, input_video],
            outputs=[start_button, video_input_required_message]
        )

        cleanup_btn.click(
            fn=cleanup_temp_files,
            outputs=[cleanup_output]
        )
        
        # The "end_button" (Cancel Job) is the trigger for the next job's monitor.
        # When a job is cancelled, we check for the next one.
        end_button.click(
            fn=end_process_with_update,
            outputs=[queue_status, queue_stats_display, end_button, current_job_id]
        ).then(
            fn=check_for_current_job_and_monitor,
            inputs=[],
            outputs=[current_job_id, result_video, preview_image, progress_desc, progress_bar, queue_status, queue_stats_display]
        )
        
        load_queue_button.click(
            fn=load_queue_from_json,
            inputs=[],
            outputs=[queue_status, queue_stats_display]
        ).then( # ADD THIS .then() CLAUSE
            fn=check_for_current_job,
            inputs=[],
            outputs=[current_job_id, result_video, preview_image, progress_desc, progress_bar]
        )
        
        import_queue_file.change(
            fn=import_queue_from_file,
            inputs=[import_queue_file],
            outputs=[queue_status, queue_stats_display]
        ).then( # ADD THIS .then() CLAUSE
            fn=check_for_current_job,
            inputs=[],
            outputs=[current_job_id, result_video, preview_image, progress_desc, progress_bar]
        )

                        
        # --- Connect Queue Refresh ---
        # The update_stats function is now defined much earlier.
        
        # REMOVED: refresh_stats_btn.click - Toolbar refresh button is no longer needed
        # refresh_stats_btn.click(
        #     fn=update_stats,
        #     inputs=None,
        #     outputs=[queue_status, queue_stats_display]
        # )

        # Set up auto-refresh for queue status
        # Instead of using a timer with 'every' parameter, we'll use the queue refresh button
        # and rely on manual refreshes. The user can click the refresh button in the toolbar
        # to update the stats.

        # --- Connect LoRA UI ---
        # Function to update slider visibility based on selection
        def update_lora_sliders(selected_loras):
            updates = []
            # Suppress dummy LoRA from workaround for the single lora bug.
            # Filter out the dummy LoRA for display purposes in the dropdown
            actual_selected_loras_for_display = [lora for lora in selected_loras if lora != DUMMY_LORA_NAME]
            updates.append(gr.update(value=actual_selected_loras_for_display)) # First update is for the dropdown itself

            # Need to handle potential missing keys if lora_names changes dynamically
            # lora_names is from the create_interface scope
            for lora_name_key in lora_names: # Iterate using lora_names to maintain order
                    if lora_name_key == DUMMY_LORA_NAME: # Check for dummy LoRA
                        updates.append(gr.update(visible=False))
                    else:
                        # Visibility of sliders should be based on actual_selected_loras_for_display
                        updates.append(gr.update(visible=(lora_name_key in actual_selected_loras_for_display)))
            return updates # This list will be correctly ordered

        # Connect the dropdown to the sliders
        lora_selector.change(
            fn=update_lora_sliders,
            inputs=[lora_selector],
            outputs=[lora_selector] + [lora_sliders[lora] for lora in lora_names if lora in lora_sliders]
        )

        def apply_preset(preset_name, model_type):
            if not preset_name:
                # Create a list of empty updates matching the number of components
                return [gr.update()] * len(ui_components)

            with open(PRESET_FILE, 'r') as f:
                data = json.load(f)
            preset = data.get(model_type, {}).get(preset_name, {})

            # Initialize updates for all components
            updates = {key: gr.update() for key in ui_components.keys()}

            # Update components based on the preset
            for key, value in preset.items():
                if key in updates:
                    updates[key] = gr.update(value=value)

            # Handle LoRA sliders specifically
            if 'lora_values' in preset and isinstance(preset['lora_values'], dict):
                lora_values_dict = preset['lora_values']
                for lora_name, lora_value in lora_values_dict.items():
                    if lora_name in updates:
                        updates[lora_name] = gr.update(value=lora_value)
            
            # Convert the dictionary of updates to a list in the correct order
            return [updates[key] for key in ui_components.keys()]

        def save_preset(preset_name, model_type, *args):
            if not preset_name:
                return gr.update()

            # Ensure the directory exists
            os.makedirs(os.path.dirname(PRESET_FILE), exist_ok=True)

            if not os.path.exists(PRESET_FILE):
                with open(PRESET_FILE, 'w') as f:
                    json.dump({}, f)

            with open(PRESET_FILE, 'r') as f:
                data = json.load(f)

            if model_type not in data:
                data[model_type] = {}

            keys = list(ui_components.keys())
            
            # Create a dictionary from the passed arguments
            args_dict = {keys[i]: args[i] for i in range(len(keys))}

            # Build the preset data from the arguments dictionary
            preset_data = {key: args_dict[key] for key in ui_components.keys() if key not in lora_sliders}

            # Handle LoRA values separately
            selected_loras = args_dict.get("lora_selector", [])
            lora_values = {}
            for lora_name in selected_loras:
                if lora_name in args_dict:
                    lora_values[lora_name] = args_dict[lora_name]
            
            preset_data['lora_values'] = lora_values
            
            # Remove individual lora sliders from the top-level preset data
            for lora_name in lora_sliders:
                if lora_name in preset_data:
                    del preset_data[lora_name]

            data[model_type][preset_name] = preset_data

            with open(PRESET_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            
            return gr.update(choices=load_presets(model_type), value=preset_name)

        def delete_preset(preset_name, model_type):
            if not preset_name:
                return gr.update(), gr.update(visible=True), gr.update(visible=False)
                
            with open(PRESET_FILE, 'r') as f:
                data = json.load(f)

            if model_type in data and preset_name in data[model_type]:
                del data[model_type][preset_name]

            with open(PRESET_FILE, 'w') as f:
                json.dump(data, f, indent=2)

            return gr.update(choices=load_presets(model_type), value=None), gr.update(visible=True), gr.update(visible=False)

        # --- Connect Preset UI ---
        # Without this refresh, if you define a new preset for the Startup Model Type, and then try to select it in settings, it won't show up.
        def refresh_settings_tab_startup_presets_if_needed(generate_tab_model_type_value, settings_tab_startup_model_type_value):
            # generate_tab_model_type_value is the model for which a preset was just saved
            # settings_tab_startup_model_type_value is the current selection in the startup model dropdown on settings tab
            if generate_tab_model_type_value == settings_tab_startup_model_type_value and settings_tab_startup_model_type_value != "None":
                return update_startup_preset_dropdown_choices(settings_tab_startup_model_type_value)
            return gr.update()

        ui_components = {
            "steps": steps, "total_second_length": total_second_length, "resolutionW": resolutionW,
            "resolutionH": resolutionH, "seed": seed, "randomize_seed": randomize_seed,
            "use_teacache": use_teacache, "teacache_num_steps": teacache_num_steps,
            "teacache_rel_l1_thresh": teacache_rel_l1_thresh, "latent_window_size": latent_window_size,
            "gs": gs, "combine_with_source": combine_with_source, "lora_selector": lora_selector, **lora_sliders
        }
        
        model_type.change(
            fn=lambda mt: (gr.update(choices=load_presets(mt)), gr.update(label=f"{mt} 預設")), # Translated
            inputs=[model_type],
            outputs=[preset_dropdown, preset_accordion]
        )
        
        preset_dropdown.select(
            fn=apply_preset,
            inputs=[preset_dropdown, model_type],
            outputs=list(ui_components.values())
        ).then(
            lambda name: name,
            inputs=[preset_dropdown],
            outputs=[preset_name_textbox]
        )

        save_preset_button.click(
            fn=save_preset,
            inputs=[preset_name_textbox, model_type, *list(ui_components.values())],
            outputs=[preset_dropdown] # preset_dropdown is on Generate tab
        ).then(
            fn=refresh_settings_tab_startup_presets_if_needed,
            inputs=[model_type, startup_model_type_dropdown], # model_type (Generate tab), startup_model_type_dropdown (Settings tab)
            outputs=[startup_preset_name_dropdown] # startup_preset_name_dropdown (Settings tab)
        )
        
        def show_delete_confirmation():
            return gr.update(visible=False), gr.update(visible=True)

        def hide_delete_confirmation():
            return gr.update(visible=True), gr.update(visible=False)

        delete_preset_button.click(
            fn=show_delete_confirmation,
            outputs=[save_preset_button, confirm_delete_row]
        )
        
        confirm_delete_no_btn.click(
            fn=hide_delete_confirmation,
            outputs=[save_preset_button, confirm_delete_row]
        )

        confirm_delete_yes_btn.click(
            fn=delete_preset,
            inputs=[preset_dropdown, model_type],
            outputs=[preset_dropdown, save_preset_button, confirm_delete_row]
        )

        # --- Definition of apply_startup_settings (AFTER ui_components and apply_preset are defined) ---
        # This function needs access to `settings`, `model_type` (Generate tab Radio),
        # `preset_dropdown` (Generate tab Dropdown), `preset_name_textbox` (Generate tab Textbox),
        # `ui_components` (dict of all other UI elements), `load_presets`, and `apply_preset`.
        # All these are available in the scope of `create_interface`.
        def apply_startup_settings():
            startup_model_val = settings.get("startup_model_type", "None")
            startup_preset_val = settings.get("startup_preset_name", None)

            # Default updates (no change)
            model_type_update = gr.update()
            preset_dropdown_update = gr.update()
            preset_name_textbox_update = gr.update()
            
            # ui_components is now defined
            ui_components_updates_list = [gr.update() for _ in ui_components]  

            if startup_model_val and startup_model_val != "None":
                model_type_update = gr.update(value=startup_model_val)
                
                presets_for_startup_model = load_presets(startup_model_val) # load_presets is defined earlier
                preset_dropdown_update = gr.update(choices=presets_for_startup_model)
                preset_name_textbox_update = gr.update(value="")

                if startup_preset_val and startup_preset_val in presets_for_startup_model:
                    preset_dropdown_update = gr.update(choices=presets_for_startup_model, value=startup_preset_val)
                    preset_name_textbox_update = gr.update(value=startup_preset_val)
                    
                    # apply_preset is now defined
                    ui_components_updates_list = apply_preset(startup_preset_val, startup_model_val)  
            
            return tuple([model_type_update, preset_dropdown_update, preset_name_textbox_update] + ui_components_updates_list)


        # --- Auto-refresh for Toolbar System Stats Monitor (Timer) ---
        main_toolbar_system_stats_timer = gr.Timer(2, active=True)  
        
        main_toolbar_system_stats_timer.tick(
            fn=tb_get_formatted_toolbar_stats, # Function imported from toolbox_app.py
            inputs=None,  
            outputs=[ # Target the Textbox components
                toolbar_ram_display_component,
                toolbar_vram_display_component,
                toolbar_gpu_display_component  
            ]
        )
        
        # --- Connect Metadata Loading ---
        # Function to load metadata from JSON file
        def load_metadata_from_json(json_path):
            # Define the total number of output components to handle errors gracefully
            num_outputs = 17 + len(lora_sliders)

            if not json_path:
                # Return empty updates for all components if no file is provided
                return [gr.update()] * num_outputs

            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)

                # Extract values from metadata with defaults
                prompt_val = metadata.get('prompt')
                n_prompt_val = metadata.get('negative_prompt')
                seed_val = metadata.get('seed')
                steps_val = metadata.get('steps')
                total_second_length_val = metadata.get('total_second_length')
                end_frame_strength_val = metadata.get('end_frame_strength')
                model_type_val = metadata.get('model_type')
                lora_weights = metadata.get('loras', {})
                latent_window_size_val = metadata.get('latent_window_size')
                resolutionW_val = metadata.get('resolutionW')
                resolutionH_val = metadata.get('resolutionH')
                blend_sections_val = metadata.get('blend_sections')
                use_teacache_val = metadata.get('use_teacache')
                teacache_num_steps_val = metadata.get('teacache_num_steps')
                teacache_rel_l1_thresh_val = metadata.get('teacache_rel_l1_thresh')
                latent_type_val = metadata.get('latent_type')
                combine_with_source_val = metadata.get('combine_with_source')
                
                # Get the names of the selected LoRAs from the metadata
                selected_lora_names = list(lora_weights.keys())

                print(f"Loaded metadata from JSON: {json_path}")
                print(f"Model Type: {model_type_val}, Prompt: {prompt_val}, Seed: {seed_val}, LoRAs: {selected_lora_names}")

                # Create a list of UI updates
                updates = [
                    gr.update(value=prompt_val) if prompt_val is not None else gr.update(),
                    gr.update(value=n_prompt_val) if n_prompt_val is not None else gr.update(),
                    gr.update(value=seed_val) if seed_val is not None else gr.update(),
                    gr.update(value=steps_val) if steps_val is not None else gr.update(),
                    gr.update(value=total_second_length_val) if total_second_length_val is not None else gr.update(),
                    gr.update(value=end_frame_strength_val) if end_frame_strength_val is not None else gr.update(),
                    gr.update(value=model_type_val) if model_type_val else gr.update(),
                    gr.update(value=selected_lora_names) if selected_lora_names else gr.update(),
                    gr.update(value=latent_window_size_val) if latent_window_size_val is not None else gr.update(),
                    gr.update(value=resolutionW_val) if resolutionW_val is not None else gr.update(),
                    gr.update(value=resolutionH_val) if resolutionH_val is not None else gr.update(),
                    gr.update(value=blend_sections_val) if blend_sections_val is not None else gr.update(),
                    gr.update(value=use_teacache_val) if use_teacache_val is not None else gr.update(),
                    gr.update(value=teacache_num_steps_val) if teacache_num_steps_val is not None else gr.update(),
                    gr.update(value=teacache_rel_l1_thresh_val) if teacache_rel_l1_thresh_val is not None else gr.update(),
                    gr.update(value=latent_type_val) if latent_type_val else gr.update(),
                    gr.update(value=combine_with_source_val) if combine_with_source_val else gr.update(),
                ]

                # Update LoRA sliders based on loaded weights
                for lora in lora_names:
                    if lora in lora_weights:
                        updates.append(gr.update(value=lora_weights[lora], visible=True))
                    else:
                        # Hide sliders for LoRAs not in the metadata
                        updates.append(gr.update(visible=False))

                return updates

            except Exception as e:
                print(f"Error loading metadata: {e}")
                import traceback
                traceback.print_exc()
                # Return empty updates for all components on error
                return [gr.update()] * num_outputs


        # Connect JSON metadata loader for Original tab
        json_upload.change(
            fn=load_metadata_from_json,
            inputs=[json_upload],
            outputs=[
                prompt,
                n_prompt,
                seed,
                steps,
                total_second_length,
                end_frame_strength_original,
                model_type,
                lora_selector,
                latent_window_size,
                resolutionW,
                resolutionH,
                blend_sections,
                use_teacache,
                teacache_num_steps,
                teacache_rel_l1_thresh,
                latent_type,
                combine_with_source
            ] + [lora_sliders[lora] for lora in lora_names]
        )


        # --- Helper Functions (defined within create_interface scope if needed by handlers) ---
        # Function to get queue statistics
        def get_queue_stats():
            try:
                # Get all jobs from the queue
                jobs = job_queue.get_all_jobs()

                # Count jobs by status
                status_counts = {
                    "QUEUED": 0,
                    "RUNNING": 0,
                    "COMPLETED": 0,
                    "FAILED": 0,
                    "CANCELLED": 0
                }

                for job in jobs:
                    if hasattr(job, 'status'):
                        status = str(job.status) # Use str() for safety
                        if status == "JobStatus.PENDING":
                            status_counts["QUEUED"] += 1
                        elif status == "JobStatus.RUNNING":
                            status_counts["RUNNING"] += 1
                        elif status == "JobStatus.COMPLETED":
                            status_counts["COMPLETED"] += 1
                        elif status == "JobStatus.FAILED":
                            status_counts["FAILED"] += 1
                        elif status == "JobStatus.CANCELLED":
                            status_counts["CANCELLED"] += 1

                # Format the display text
                stats_text = f"佇列: {status_counts['QUEUED']} | 運行中: {status_counts['RUNNING']} | 已完成: {status_counts['COMPLETED']} | 失敗: {status_counts['FAILED']} | 已取消: {status_counts['CANCELLED']}" # Translated

                return f"<p style='margin:0;color:white;'>{stats_text}</p>"

            except Exception as e:
                print(f"Error getting queue stats: {e}")
                return "<p style='margin:0;color:white;'>載入佇列狀態時發生錯誤</p>" # Translated "Error loading queue stats"

        # Add footer with social links
        with gr.Row(elem_id="footer"):
            with gr.Column(scale=1):
                gr.HTML(f"""
                <div style="text-align: center; padding: 20px; color: #666;">
                    <div style="margin-top: 10px;">
                        <span class="footer-version" style="margin: 0 10px; color: #666;">{APP_VERSION_DISPLAY}</span>
                        <a href="https://patreon.com/Colinu" target="_blank" style="margin: 0 10px; color: #666; text-decoration: none;" class="footer-patreon">
                            <i class="fab fa-patreon"></i>在 Patreon 上支持
                        </a>
                        <a href="https://discord.gg/MtuM7gFJ3V" target="_blank" style="margin: 0 10px; color: #666; text-decoration: none;">
                            <i class="fab fa-discord"></i> Discord
                        </a>
                        <a href="https://github.com/colinurbs/FramePack-Studio" target="_blank" style="margin: 0 10px; color: #666; text-decoration: none;">
                            <i class="fab fa-github"></i> GitHub
                        </a>
                    </div>
                </div>
                """)

        # Add CSS for footer

        # gr.HTML("""
        #     <script>
        #     (function() {
        #         "use strict";
        #         console.log("Stat Bar Script: Initializing");

        #         const statConfig = {
        #             ram: { selector: '#toolbar-ram-stat', regex: /\((\d+)%\)/, valueIndex: 1, isRawPercentage: true },
        #             vram: { selector: '#toolbar-vram-stat', regex: /VRAM: (\d+\.?\d+)\/(\d+\.?\d+)GB/, usedIndex: 1, totalIndex: 2, isRawPercentage: false },
        #             gpu: { selector: '#toolbar-gpu-stat', regex: /GPU: \d+°C (\d+)%/, valueIndex: 1, isRawPercentage: true }
        #         };

        #         function setBarPercentage(statElement, percentage) {
        #             if (!statElement) {
        #                 console.warn("Stat Bar Script: setBarPercentage called with no element.");
        #                 return;
        #             }
        #             let bar = statElement.querySelector('.stat-bar');
        #             if (!bar) {
        #                 console.log("Stat Bar Script: Creating .stat-bar for", statElement.id);
        #                 bar = document.createElement('div');
        #                 bar.className = 'stat-bar';
        #                 statElement.insertBefore(bar, statElement.firstChild);
        #             }
        #             const clampedPercentage = Math.min(100, Math.max(0, parseFloat(percentage)));
        #             statElement.style.setProperty('--stat-percentage', clampedPercentage + '%');
        #             // console.log("Stat Bar Script: Updated", statElement.id, "to", clampedPercentage + "%");
        #         }

        #         function updateSingleStatVisual(key, config) {
        #             try {
        #                 const container = document.querySelector(config.selector);
        #                 if (!container) {
        #                     // console.warn("Stat Bar Script: Container not found for", key, config.selector);
        #                     return false; // Element not ready
        #                 }
        #                 const textarea = container.querySelector('textarea');
        #                 if (!textarea) {
        #                     // console.warn("Stat Bar Script: Textarea not found for", key);
        #                     return false; // Element not ready
        #                 }

        #                 const textValue = textarea.value;
        #                 if (textValue === "RAM: N/A" || textValue === "VRAM: N/A" || textValue === "GPU: N/A") {
        #                     setBarPercentage(container, 0); // Set to 0 if N/A
        #                     return true;
        #                 }

        #                 const match = textValue.match(config.regex);
        #                 if (match) {
        #                     let percentage = 0;
        #                     if (config.isRawPercentage) {
        #                         percentage = parseInt(match[config.valueIndex]);
        #                     } else { // VRAM case
        #                         const used = parseFloat(match[config.usedIndex]);
        #                         const total = parseFloat(match[config.totalIndex]);
        #                         percentage = (total > 0) ? (used / total) * 100 : 0;
        #                     }
        #                     setBarPercentage(container, percentage);
        #                 } else {
        #                     // console.warn("Stat Bar Script: Regex mismatch for", key, "-", textValue);
        #                     setBarPercentage(container, 0); // Default to 0 on mismatch after initial load
        #                 }
        #                 return true; // Processed or N/A
        #             } catch (error) {
        #                 console.error("Stat Bar Script: Error updating visual for", key, error);
        #                 return true; // Assume processed to avoid retry loops on error
        #             }
        #         }
                
        #         function updateAllStatVisuals() {
        #             let allReady = true;
        #             for (const key in statConfig) {
        #                 if (!updateSingleStatVisual(key, statConfig[key])) {
        #                     allReady = false;
        #                 }
        #             }
        #             return allReady;
        #         }

        #         function initStatBars() {
        #             console.log("Stat Bar Script: initStatBars called");
        #             if (updateAllStatVisuals()) {
        #                 console.log("Stat Bar Script: All stats initialized. Setting up MutationObserver.");
        #                 setupMutationObservers();
        #             } else {
        #                 console.log("Stat Bar Script: Elements not ready, retrying init in 250ms.");
        #                 setTimeout(initStatBars, 250); // Retry if not all elements were ready
        #             }
        #         }

        #         function setupMutationObservers() {
        #             const observer = new MutationObserver((mutationsList) => {
        #                 // Use a Set to avoid redundant updates if multiple mutations point to the same stat
        #                 const changedStats = new Set();

        #                 for (const mutation of mutationsList) {
        #                     let targetElement = mutation.target;
        #                     // Traverse up to find the .toolbar-stat-textbox parent if mutation is deep
        #                     while(targetElement && !targetElement.matches('.toolbar-stat-textbox')) {
        #                         targetElement = targetElement.parentElement;
        #                     }

        #                     if (targetElement && targetElement.matches('.toolbar-stat-textbox')) {
        #                         for (const key in statConfig) {
        #                             if (targetElement.id === statConfig[key].selector.substring(1)) {
        #                                 changedStats.add(key);
        #                                 break;
        #                             }
        #                         }
        #                     }
        #                 }
        #                 if (changedStats.size > 0) {
        #                     // console.log("Stat Bar Script: MutationObserver detected changes for:", Array.from(changedStats));
        #                     changedStats.forEach(key => updateSingleStatVisual(key, statConfig[key]));
        #                 }
        #             });

        #             for (const key in statConfig) {
        #                 const container = document.querySelector(statConfig[key].selector);
        #                 if (container) {
        #                     // Observe the container for changes to its children (like textarea value)
        #                     // and the textarea itself if it exists.
        #                     observer.observe(container, { childList: true, subtree: true, characterData: true });
        #                     console.log("Stat Bar Script: Observer attached to", container.id);
        #                 } else {
        #                     console.warn("Stat Bar Script: Could not attach observer, container not found for", key);
        #                 }
        #             }
        #         }

        #         // More robust DOM ready check
        #         if (document.readyState === "complete" || (document.readyState !== "loading" && !document.documentElement.doScroll)) {
        #             console.log("Stat Bar Script: DOM already ready.");
        #             initStatBars();
        #         } else {
        #             document.addEventListener("DOMContentLoaded", () => {
        #                 console.log("Stat Bar Script: DOMContentLoaded event.");
        #                 initStatBars();
        #             });
        #         }
        #         // Fallback for Gradio's dynamic loading, if DOMContentLoaded isn't enough
        #         window.addEventListener('gradio.rendered', () => {
        #             console.log('Stat Bar Script: Gradio rendered event detected.');
        #             initStatBars();
        #         });

        #     })();
        #     </script>
        # """)

        # Connect the auto-check function to the interface load event
        block.load(
            fn=check_for_current_job_and_monitor, # Use the new combined function
            inputs=[],
            outputs=[current_job_id, result_video, preview_image, progress_desc, progress_bar, queue_status, queue_stats_display]
        ).then(
            fn=apply_startup_settings, # apply_startup_settings is now defined
            inputs=None,
            outputs=[model_type, preset_dropdown, preset_name_textbox] + list(ui_components.values()) # ui_components is now defined
        ).then(
            fn=update_start_button_state, # Ensure button state is correct after startup settings
            inputs=[model_type, input_video],  
            outputs=[start_button, video_input_required_message]
        )
        
        return block

# --- Top-level Helper Functions (Used by Gradio callbacks, must be defined outside create_interface) ---

def format_queue_status(jobs):
    """Format job data for display in the queue status table"""
    rows = []
    for job in jobs:
        created = time.strftime('%H:%M:%S', time.localtime(job.created_at)) if job.created_at else ""
        started = time.strftime('%H:%M:%S', time.localtime(job.started_at)) if job.started_at else ""
        completed = time.strftime('%H:%M:%S', time.localtime(job.completed_at)) if job.completed_at else ""

        # Calculate elapsed time
        elapsed_time = ""
        if job.started_at:
            if job.completed_at:
                start_datetime = datetime.datetime.fromtimestamp(job.started_at)
                complete_datetime = datetime.datetime.fromtimestamp(job.completed_at)
                elapsed_seconds = (complete_datetime - start_datetime).total_seconds()
                elapsed_time = f"{elapsed_seconds:.2f}秒" # Translated "s" -> "秒"
            else:
                # For running jobs, calculate elapsed time from now
                start_datetime = datetime.datetime.fromtimestamp(job.started_at)
                current_datetime = datetime.datetime.now()
                elapsed_seconds = (current_datetime - start_datetime).total_seconds()
                elapsed_time = f"{elapsed_seconds:.2f}秒 (運行中)" # Translated "s (running)" -> "秒 (運行中)"

        # Get generation type from job data
        # Note: Original English values are "Original", "Original with Endframe", etc.
        # We need to map these to their Chinese display names for the table.
        generation_type_display = {
            "Original": "原始",
            "Original with Endframe": "帶結束幀的原始",
            "F1": "F1",
            "Video": "影片",
            "Video with Endframe": "帶結束幀的影片",
            "Video F1": "影片 F1",
            # Add other types if they exist, otherwise default to the original string
        }.get(getattr(job, 'generation_type', 'Original'), getattr(job, 'generation_type', 'Original'))


        # Get thumbnail from job data and format it as HTML for display
        thumbnail = getattr(job, 'thumbnail', None)
        thumbnail_html = f'<img src="{thumbnail}" width="64" height="64" style="object-fit: contain;">' if thumbnail else ""

        rows.append([
            job.id[:6] + '...',
            generation_type_display, # Use translated display name
            {
                "JobStatus.PENDING": "待處理",
                "JobStatus.RUNNING": "運行中",
                "JobStatus.COMPLETED": "已完成",
                "JobStatus.FAILED": "失敗",
                "JobStatus.CANCELLED": "已取消"
            }.get(str(job.status), str(job.status)), # Translate JobStatus
            created,
            started,
            completed,
            elapsed_time,
            thumbnail_html  # Add formatted thumbnail HTML to row data
        ])
    return rows

# Create the queue status update function (wrapper around format_queue_status)
def update_queue_status_with_thumbnails(): # Function name is now slightly misleading, but keep for now to avoid breaking clicks
    # This function is likely called by the refresh button and potentially the timer
    # It needs access to the job_queue object
    # Assuming job_queue is accessible globally or passed appropriately
    # For now, let's assume it's globally accessible as defined in studio.py
    # If not, this needs adjustment based on how job_queue is managed.
    try:
        # Need access to the global job_queue instance from studio.py
        # This might require restructuring or passing job_queue differently.
        # For now, assuming it's accessible (this might fail if run standalone)
        from __main__ import job_queue # Attempt to import from main script scope

        jobs = job_queue.get_all_jobs()
        for job in jobs:
            if job.status == JobStatus.PENDING:
                job.queue_position = job_queue.get_queue_position(job.id)

        if job_queue.current_job:
            job_queue.current_job.status = JobStatus.RUNNING

        return format_queue_status(jobs)
    except ImportError:
        print("Error: Could not import job_queue. Queue status update might fail.")
        return [] # Return empty list on error
    except Exception as e:
        print(f"Error updating queue status: {e}")
        return []
