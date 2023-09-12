import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from share_btn import community_icon_html, loading_icon_html, share_js
from cog_sdxl_dataset_and_utils import TokenEmbeddingsHandler
import lora
from time import sleep
import copy
import json
import gc

with open("sdxl_loras.json", "r") as file:
    data = json.load(file)
    sdxl_loras = [
        {
            "image": item["image"],
            "title": item["title"],
            "repo": item["repo"],
            "trigger_word": item["trigger_word"],
            "weights": item["weights"],
            "is_compatible": item["is_compatible"],
            "is_pivotal": item.get("is_pivotal", False),
            "text_embedding_weights": item.get("text_embedding_weights", None),
            "is_nc": item.get("is_nc", False)
        }
        for item in data
    ]

device = "cuda" 

for item in sdxl_loras:
    saved_name = hf_hub_download(item["repo"], item["weights"])
    
    if not saved_name.endswith('.safetensors'):
        state_dict = torch.load(saved_name)
    else:
        state_dict = load_file(saved_name)
        
    item["saved_name"] = saved_name
    item["state_dict"] = state_dict #{k: v.to(device=device, dtype=torch.float16) for k, v in state_dict.items() if torch.is_tensor(v)}

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
)
original_pipe = copy.deepcopy(pipe)
pipe.to(device)

last_lora = ""
last_merged = False
last_fused = False
def update_selection(selected_state: gr.SelectData):
    lora_repo = sdxl_loras[selected_state.index]["repo"]
    instance_prompt = sdxl_loras[selected_state.index]["trigger_word"]
    new_placeholder = "Type a prompt. This LoRA applies for all prompts, no need for a trigger word" if instance_prompt == "" else "Type a prompt to use your selected LoRA"
    weight_name = sdxl_loras[selected_state.index]["weights"]
    updated_text = f"### Selected: [{lora_repo}](https://huggingface.co/{lora_repo}) ✨ {'(non-commercial LoRA, `cc-by-nc`)' if sdxl_loras[selected_state.index]['is_nc'] else '' }"
    is_compatible = sdxl_loras[selected_state.index]["is_compatible"]
    is_pivotal = sdxl_loras[selected_state.index]["is_pivotal"]
    
    use_with_diffusers = f'''
    ## Using [`{lora_repo}`](https://huggingface.co/{lora_repo})
                        
    ## Use it with diffusers:
    '''
    if is_compatible:
        use_with_diffusers += f'''
        from diffusers import StableDiffusionXLPipeline
        import torch
    
        model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.to("cuda")
        pipe.load_lora_weights("{lora_repo}", weight_name="{weight_name}")
    
        prompt = "{instance_prompt}..."
        lora_scale= 0.9
        image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, cross_attention_kwargs={{"scale": lora_scale}}).images[0]
        image.save("image.png")
        '''
    elif not is_pivotal:
        use_with_diffusers += "This LoRA is not compatible with diffusers natively yet. But you can still use it on diffusers with `bmaltais/kohya_ss` LoRA class, check out this [Google Colab](https://colab.research.google.com/drive/14aEJsKdEQ9_kyfsiV6JDok799kxPul0j )"
    else:
        use_with_diffusers += f"This LoRA is not compatible with diffusers natively yet. But you can still use it on diffusers with sdxl-cog `TokenEmbeddingsHandler` class, check out the [model repo](https://huggingface.co/{lora_repo}#inference-with-🧨-diffusers)"
    use_with_uis = f'''
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
        gr.update(placeholder=new_placeholder),
        selected_state,
        use_with_diffusers,
        use_with_uis,
    )


def check_selected(selected_state):
    if not selected_state:
        raise gr.Error("You must select a LoRA")

def merge_incompatible_lora(full_path_lora, lora_scale):
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
                del weights_sd
                del lora_model
                gc.collect()

def run_lora(prompt, negative, lora_scale, selected_state, progress=gr.Progress(track_tqdm=True)):
    global last_lora, last_merged, pipe

    if negative == "":
        negative = None

    if not selected_state:
        raise gr.Error("You must select a LoRA")
    repo_name = sdxl_loras[selected_state.index]["repo"]
    weight_name = sdxl_loras[selected_state.index]["weights"]
    full_path_lora = sdxl_loras[selected_state.index]["saved_name"]
    loaded_state_dict = sdxl_loras[selected_state.index]["state_dict"]
    cross_attention_kwargs = None
    if last_lora != repo_name:
        if last_merged:
            del pipe
            gc.collect()
            pipe = copy.deepcopy(original_pipe)
            pipe.to(device)
        elif(last_fused):
            pipe.unload_lora_weights()
            pipe.unfuse_lora()
        is_compatible = sdxl_loras[selected_state.index]["is_compatible"]
        if is_compatible:
            pipe.load_lora_weights(loaded_state_dict)
            pipe.fuse_lora(lora_scale)
            last_fused = True
        else:
            is_pivotal = sdxl_loras[selected_state.index]["is_pivotal"]
            if(is_pivotal):
                pipe.load_lora_weights(loaded_state_dict)
                pipe.fuse_lora(lora_scale)
                last_fused = True
                
                #Add the textual inversion embeddings from pivotal tuning models
                text_embedding_name = sdxl_loras[selected_state.index]["text_embedding_weights"]
                text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
                tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
                embedding_path = hf_hub_download(repo_id=repo_name, filename=text_embedding_name, repo_type="model")
                embhandler = TokenEmbeddingsHandler(text_encoders, tokenizers)
                embhandler.load_embeddings(embedding_path)
                
            else:
                merge_incompatible_lora(full_path_lora, lora_scale)
                last_merged = True
                last_fused=False

    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=768,
        height=768,
        num_inference_steps=20,
        guidance_scale=7.5,
    ).images[0]
    last_lora = repo_name
    gc.collect()
    return image, gr.update(visible=True)


with gr.Blocks(css="custom.css") as demo:
    title = gr.HTML(
        """<h1><img src="https://i.imgur.com/vT48NAO.png" alt="LoRA"> LoRA the Explorer</h1>""",
        elem_id="title",
    )
    selected_state = gr.State()
    with gr.Row():
        gallery = gr.Gallery(
            value=[(item["image"], item["title"]) for item in sdxl_loras],
            label="SDXL LoRA Gallery",
            allow_preview=False,
            columns=3,
            elem_id="gallery",
            show_share_button=False
        )
        with gr.Column():
            prompt_title = gr.Markdown(
                value="### Click on a LoRA in the gallery to select it",
                visible=True,
                elem_id="selected_lora",
            )
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", show_label=False, lines=1, max_lines=1, placeholder="Type a prompt after selecting a LoRA", elem_id="prompt")
                button = gr.Button("Run", elem_id="run_button")
            with gr.Group(elem_id="share-btn-container", visible=False) as share_group:
                community_icon = gr.HTML(community_icon_html)
                loading_icon = gr.HTML(loading_icon_html)
                share_button = gr.Button("Share to community", elem_id="share-btn")
            result = gr.Image(
                interactive=False, label="Generated Image", elem_id="result-image"
            )
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
        outputs=[prompt_title, prompt, prompt, selected_state, use_diffusers, use_uis],
        queue=False,
        show_progress=False,
    )
    prompt.submit(
        fn=check_selected,
        inputs=[selected_state],
        queue=False,
        show_progress=False
    ).success(
        fn=run_lora,
        inputs=[prompt, negative, weight, selected_state],
        outputs=[result, share_group],
    )
    button.click(
        fn=check_selected,
        inputs=[selected_state],
        queue=False,
        show_progress=False
    ).success(
        fn=run_lora,
        inputs=[prompt, negative, weight, selected_state],
        outputs=[result, share_group],
    )
    share_button.click(None, [], [], _js=share_js)

demo.queue(max_size=20)
demo.launch()