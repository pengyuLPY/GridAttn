from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

repo_id = "stabilityai/stable-diffusion-2"
pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "High quality photo of an astronaut riding a horse in space"
prompt = "The Luffy in one-piece are laughing and introducing himself"
prompt = "A sweet beautiful 16 year old girl with sweet face, dark shoulder length hair and bangs sitting in a chair. in style of Balthus and Mark Ryden. She is wearing a Coco Fennell dress with peter pan collar. Volumetric lighting, atmospheric, photographic, intricate detail, enchanting, enigmatic, mysterious"
prompt = "young ginger blonde woman 20 years old in a 19th century dress that looks like young Virginia Gardner and young Ramona Gorai,historical romance genre"
prompt = "photograph wide image, goth, skull, darkness, moonlight, cross, chains, music,guita"
prompt = "historical romance genre, a young ginger blonde woman 20 years old that looks like a blend of young Virginia Gardner and young Romola Garson, in a beautiful regency era dress realistic, hyperrealistic"
prompt = "a succulent wet apple"

prompt = "a single cut Pikachu introduces himself friendly. The background is the galaxy space, a style of cartoon, intricate detail"
prompt = "Picture a snug living room with a blazing fireplace, flames leaping and dancing inside. Beside the fireplace, a basket filled with soft and fuzzy blankets. On a plush sofa, a couple is snuggled up, wrapped in blankets, sipping hot chocolate, and wearing satisfied, contented smiles. The whole scene exudes warmth, comfort, and coziness, enveloping you in a feeling of true relaxation and warmth."
prompt = "Transport yourself into a scene so realistic, you can almost feel the warmth radiating from the blazing fireplace. As you gaze upon the flickering flames inside the fireplace, you notice the burning embers glowing bright red-orange. The basket beside the fireplace is filled with fluffy blankets, each with a unique texture, and you can almost feel their softness. The plush sofa and its blankets appear so inviting, you can feel yourself sinking into its comfort. The couple's smiles look so genuine, it's as if you're there with them, experiencing their joy and warmth firsthand."
prompt = "Generate a serene and peaceful natural scene to take you away from the bustling crowds. The image is of a tranquil lake, its surface as calm as a mirror reflecting the surrounding mountains. You can see a row of tall trees by the lake, with sunlight filtering through their branches and leaves, and the grass rustling in the gentle breeze. In the middle of the lake, a small boat can be seen slowly drifting with an elderly fisherman fishing. This image allows you to take a deep breath of fresh air, relax your body and mind, and escape from the noise."
prompt = "In the center of the picture is a stream in the forest. The water in the river is clear and fish swim freely. On the river bank, there is a row of flowers swaying gently, while butterflies play among them. In the distance, there is a forest shaded by green trees. The sunlight shines through the trees and reflects on the stream, creating a beautiful light and shadow. In the lower left corner of the picture, a little boy is sitting on the grass, enjoying the beautiful scenery."

height, width = 720, 1080

image = pipe(prompt, guidance_scale=9, num_inference_steps=50, height=height, width=width).images[0]
image.save("/home/Rhossolas.Lee/JK.png")

print('complete')
