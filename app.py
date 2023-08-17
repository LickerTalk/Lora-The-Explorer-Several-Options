import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from huggingface_hub import hf_hub_download
from share_btn import community_icon_html, loading_icon_html, share_js
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

device = "cuda"  # replace this to `mps` if on a MacOS Silicon

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


def update_selection(selected_state: gr.SelectData):
    lora_repo = sdxl_loras[selected_state.index][2]
    instance_prompt = sdxl_loras[selected_state.index][3]
    weight_name = sdxl_loras[selected_state.index][4]
    updated_text = f"### Selected: [{lora_repo}](https://huggingface.co/{lora_repo})"
    use_with_diffusers = f'''
                    ## Using [`{lora_repo}`](https://huggingface.co/{lora_repo})
                    
                    ## Use it with diffusers: 

                    ```python
                    from diffusers import StableDiffusionXLPipeline
                    import torch

                    model_path = "stabilityai/stable-diffusion-xl-base-1.0"
                    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
                    pipe.to("cuda")
                    pipe.load_lora_weights("{lora_repo}", weight_name={weight_name})
                        

                    prompt = "{instance_prompt}..." 
                    lora_weight = 0.5
                    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, cross_attention_kwargs={{"scale":lora_weight}}).images[0]
                    image.save("image.png")
                    ```
                    """
    use_with_uis = f"""
    ## Use it with Comfy UI, Invoke AI, SD.Next, AUTO1111: 

    ### Download the `*.safetensors` weights of [here](https://huggingface.co/{lora_repo}/resolve/main/{weight_name})
    
    - [ComfyUI guide](https://comfyanonymous.github.io/ComfyUI_examples/lora/)
    - [Invoke AI guide](https://invoke-ai.github.io/InvokeAI/features/CONCEPTS/?h=lora#using-loras)
    - [SD.Next guide](https://github.com/vladmandic/automatic)
    - [AUTOMATIC1111 guide](https://stable-diffusion-art.com/lora/)
    '''
    return (
        updated_text,
        instance_prompt,
        selected_state,
        use_with_diffusers,
        use_with_uis,
    )


def run_lora(prompt, negative, lora_scale, selected_state):
    global last_lora, last_merged, pipe

    if negative == "":
        negative = None

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
            cross_attention_kwargs = {"scale": lora_scale}
        else:
            for weights_file in [full_path_lora]:
                if ";" in weights_file:
                    weights_file, multiplier = weights_file.split(";")
                    multiplier = float(multiplier)
                else:
                    multiplier = lora_scale

                lora_model, weights_sd = lora.create_network_from_weights(
                    multiplier,
                    full_path_lora,
                    pipe.vae,
                    pipe.text_encoder,
                    pipe.unet,
                    for_inference=True,
                )
                lora_model.merge_to(
                    pipe.text_encoder, pipe.unet, weights_sd, torch.float16, "cuda"
                )
            last_merged = True

    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=768,
        height=768,
        num_inference_steps=20,
        guidance_scale=7.5,
        cross_attention_kwargs=cross_attention_kwargs,
    ).images[0]
    last_lora = repo_name
    return image, gr.update(visible=True)


with gr.Blocks(css="custom.css") as demo:
    title = gr.HTML(
        """<h1><img src="https://i.imgur.com/vT48NAO.png" alt="LoRA"> LoRA the Explorer</h1>""",
        elem_id="title",
    )
    selected_state = gr.State()
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
                value="### Click on a LoRA in the gallery to select it",
                visible=True,
                elem_id="selected_lora",
            )
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", elem_id="prompt")
                button = gr.Button("Run", elem_id="run_button")
            result = gr.Image(
                interactive=False, label="Generated Image", elem_id="result-image"
            )
            with gr.Group(elem_id="share-btn-container", visible=False) as share_group:
                community_icon = gr.HTML(community_icon_html)
                loading_icon = gr.HTML(loading_icon_html)
                share_button = gr.Button("Share to community", elem_id="share-btn")
            with gr.Accordion("Advanced options", open=False):
                negative = gr.Textbox(label="Negative Prompt")
                weight = gr.Slider(0, 10, value=1, step=0.1, label="LoRA weight")

    with gr.Column(elem_id="extra_info"):
        with gr.Accordion(
            "Use it with: 🧨  diffusers, ComfyUI, Invoke AI, SD.Next, AUTO1111",
            open=False,
            elem_id="accordion",
        ):
            with gr.Row():
                use_diffusers = gr.Markdown("""## Select a LoRA first 🤗""")
                use_uis = gr.Markdown()
        with gr.Accordion("Submit a LoRA! 📥", open=False):
            submit_title = gr.Markdown(
                "### Streamlined submission coming soon! Until then [suggest your LoRA in the community tab](https://huggingface.co/spaces/multimodalart/LoraTheExplorer/discussions) 🤗"
            )
            with gr.Box(elem_id="soon"):
                submit_source = gr.Radio(
                    ["Hugging Face", "CivitAI"],
                    label="LoRA source",
                    value="Hugging Face",
                )
                with gr.Row():
                    submit_source_hf = gr.Textbox(
                        label="Hugging Face Model Repo",
                        info="In the format `username/model_id`",
                    )
                    submit_safetensors_hf = gr.Textbox(
                        label="Safetensors filename",
                        info="The filename `*.safetensors` in the model repo",
                    )
                with gr.Row():
                    submit_trigger_word_hf = gr.Textbox(label="Trigger word")
                    submit_image = gr.Image(
                        label="Example image (optional if the repo already contains images)"
                    )
                submit_button = gr.Button("Submit!")
                submit_disclaimer = gr.Markdown(
                    "This is a curated gallery by me, [apolinário (multimodal.art)](https://twitter.com/multimodalart). I'll try to include as many cool LoRAs as they are submitted! You can [duplicate this Space](https://huggingface.co/spaces/multimodalart/LoraTheExplorer?duplicate=true) to use it privately, and add your own LoRAs by editing `sdxl_loras.json` in the Files tab of your private space."
                )

    gallery.select(
        update_selection,
        outputs=[prompt_title, prompt, selected_state, use_diffusers, use_uis],
        queue=False,
        show_progress=False,
    )
    prompt.submit(
        fn=run_lora,
        inputs=[prompt, negative, weight, selected_state],
        outputs=[result, share_group],
    )
    button.click(
        fn=run_lora,
        inputs=[prompt, negative, weight, selected_state],
        outputs=[result, share_group],
    )
    share_button.click(None, [], [], _js=share_js)

demo.launch()
