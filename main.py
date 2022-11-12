import argparse
import gc

import gradio as gr
import torch

from pipeline_stable_diffusion_ait import StableDiffusionAITPipeline


def run(prompt, width, height, num_inference_steps=50, guidance_scale=7.5):
    torch.cuda.empty_cache()
    gc.collect()
    with torch.autocast("cuda"):
        image = pipe(prompt, height=int(height), width=int(width), num_inference_steps=num_inference_steps,
                     guidance_scale=guidance_scale).images[0]
        return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--token', help='Your huggingface token')

    args = parser.parse_args()
    pipe = StableDiffusionAITPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=args.token,
    ).to("cuda")
    demo = gr.Interface(
        fn=run,
        inputs=[
            gr.Textbox(),
            gr.Slider(step=64, minimum=512, maximum=1024, label="Width", value=512),
            gr.Slider(step=64, minimum=512, maximum=1024, label="Height", value=512)
        ],
        outputs=gr.Image(),
    )
    demo.launch()
