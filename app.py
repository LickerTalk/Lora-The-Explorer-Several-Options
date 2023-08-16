import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from huggingface_hub import hf_hub_download
import lora
from time import sleep
import copy
import json

with open("sdxl_loras.json", "r") as file:
    sdxl_loras = [
        (
            item["image"],
            item["title"],
            item["repo"],
            item["trigger_word"],
            item["weights"],
            item["is_compatible"],
        )
        for item in json.load(file)
    ]

saved_names = [
    hf_hub_download(repo_id, filename) for _, _, repo_id, _, filename, _ in sdxl_loras
]

device = "cuda" #replace this to `mps` if on a MacOS Silicon

def update_selection(selected_state: gr.SelectData):
    lora_repo = sdxl_loras[selected_state.index][2]
    instance_prompt = sdxl_loras[selected_state.index][3]
    updated_text = f"### Selected: [{lora_repo}](https://huggingface.co/{lora_repo})"
    return updated_text, instance_prompt, selected_state


vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
).to("cpu")
original_pipe = copy.deepcopy(pipe)
pipe.to(device)

last_lora = ""
last_merged = False


def run_lora(prompt, negative, weight, selected_state):
    global last_lora, last_merged, pipe
    if not selected_state:
        raise gr.Error("You must select a LoRA")
    repo_name = sdxl_loras[selected_state.index][2]
    weight_name = sdxl_loras[selected_state.index][4]
    full_path_lora = saved_names[selected_state.index]
    cross_attention_kwargs = None
    if last_lora != repo_name:
        if last_merged:
            pipe = copy.deepcopy(original_pipe)
            pipe.to(device)
        else:
            pipe.unload_lora_weights()
        is_compatible = sdxl_loras[selected_state.index][5]
        if is_compatible:
            pipe.load_lora_weights(full_path_lora)
            cross_attention_kwargs = {"scale": weight}
        else:
            for weights_file in [full_path_lora]:
                if ";" in weights_file:
                    weights_file, multiplier = weights_file.split(";")
                    multiplier = float(weight)
                else:
                    multiplier = 1.0
                
                multiplier = torch.tensor([multiplier], dtype=torch.float16, device=device)
                lora_model, weights_sd = lora.create_network_from_weights(
                    multiplier,
                    full_path_lora,
                    pipe.vae,
                    pipe.text_encoder,
                    pipe.unet,
                    for_inference=True,
                )
                #lora_model.to(device)
                lora_model.apply_to(pipe.text_encoder, pipe.unet) #is apply too all you need?
                #lora_model.to(device)
                #lora_model.merge_to(
                #    pipe.text_encoder, pipe.unet, weights_sd, torch.float16, "cuda"
                #)
                pipe.to(device)
            last_merged = True

    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=20,
        guidance_scale=7.5,
        cross_attention_kwargs=cross_attention_kwargs,
    ).images[0]
    last_lora = repo_name
    return image


css = """
#title{text-align: center;margin-bottom: 0.5em}
#title h1{font-size: 3em}
#prompt textarea{width: calc(100% - 160px);border-top-right-radius: 0px;border-bottom-right-radius: 0px;}
#run_button{position:absolute;margin-top: 38px;right: 0;margin-right: 0.8em;border-bottom-left-radius: 0px;
    border-top-left-radius: 0px;}
#gallery{display:flex}
#gallery .grid-wrap{min-height: 100%;}
"""

with gr.Blocks(css=css) as demo:
    title = gr.Markdown("# LoRA the Explorer 🔎", elem_id="title")
    with gr.Row():
        gallery = gr.Gallery(
            value=[(a, b) for a, b, _, _, _, _ in sdxl_loras],
            label="SDXL LoRA Gallery",
            allow_preview=False,
            columns=3,
            elem_id="gallery",
        )
        with gr.Column():
            prompt_title = gr.Markdown(
                value="### Click on a LoRA in the gallery to select it", visible=True
            )
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", elem_id="prompt")
                button = gr.Button("Run", elem_id="run_button")
            result = gr.Image(interactive=False, label="result")
            with gr.Accordion("Advanced options", open=False):
                negative = gr.Textbox(label="Negative Prompt")
                weight = gr.Slider(0, 1, value=1, step=0.1, label="LoRA weight")
    with gr.Column():
        gr.Markdown("Use it with:")
        with gr.Row():
            with gr.Accordion("🧨 diffusers", open=False):
                gr.Markdown("")
            with gr.Accordion("ComfyUI", open=False):
                gr.Markdown("")
            with gr.Accordion("Invoke AI", open=False):
                gr.Markdown("")
            with gr.Accordion("SD.Next (AUTO1111 fork)", open=False):
                gr.Markdown("")
    selected_state = gr.State()
    gallery.select(
        update_selection,
        outputs=[prompt_title, prompt, selected_state],
        queue=False,
        show_progress=False,
    )
    prompt.submit(
        fn=run_lora, inputs=[prompt, negative, weight, selected_state], outputs=result
    )
    button.click(
        fn=run_lora, inputs=[prompt, negative, weight, selected_state], outputs=result
    )


demo.launch()
