import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "./models/stable-diffusion-2-1-gridattn"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a winter landscape"
default_prompt = "masterpiece, photographic, intricate detail,Ultra Detailed hyperrealistic real photo, 8k, hdr, high quality,"
negative_prompt = "obese, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, mutation, missing limb, ugly, floating limbs, mutated hands, extra fingers, disconnected limbs, disconnected limbs, mutated, disgusting, blurry, amputation, low quality, long neck,"

num_inference_steps=50
width,height = 768, 768
guidance_scale = 9
image = pipe(prompt=prompt+", "+default_prompt,
             negative_prompt=negative_prompt,
             height=height,width=width,
             num_inference_steps=num_inference_steps,
             guidance_scale=guidance_scale,
             ).images[0]

image.save("image.png")
print("complete")
