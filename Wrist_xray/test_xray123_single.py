import os

import torch
import torchvision
import torchvision.transforms as transforms
from accelerate import Accelerator
from diffusers.utils import load_image
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline

# model_path = "./models_zero123/xray123-hard-200000-autosaved/"
# model_path = "/work/XRAYDIFF/share/svdrr-wrist/xray123-simple-model-wrist"
model_path = "/work/XRAYDIFF/eitahtnk/zero123-hf/svdrr-wrist/xray123-simple-model-wrist"
accelerator = Accelerator()

pipe = Zero1to3StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
)

# pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_tiling()
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")


image_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((256, 256)),  # 256, 256
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


step = 45
query_poses = [[0, i, 0] for i in range(-90, 95, 45) if i != 0]
# query_poses = [[0, i, 0] for i in range(-90, 91, step) if i != 0]

# input_image = load_image("real_xray.jpg")
input_image = load_image(
    "/work/XRAYDIFF/eitahtnk/zero123-hf/CR_data/###_20210719_HAND__1（小）.png"
)
path = "###_20210719_HAND__1（小）"
# log_dir = f"outputs/xray123_test_out/{path}/{step}"
log_dir = f"/work/XRAYDIFF/eitahtnk/zero123-hf/inference/generated_image/fine-tuned/{path}/{step}"
os.makedirs(log_dir, exist_ok=True)

# resize image to 256x256
input_image = input_image.resize((256, 256))
# save image as 0.png
input_image.save(os.path.join(log_dir, f"0.png"))

for idx, query_pose in enumerate(query_poses):
    with torch.autocast("cuda"):
        image = pipe(
            input_imgs=input_image,
            prompt_imgs=input_image,
            poses=query_pose,
            height=256,
            width=256,
            guidance_scale=3.0,
            num_images_per_prompt=1,
            num_inference_steps=50,
        ).images[0]

    # save imgs

    image.save(os.path.join(log_dir, f"{query_pose[1]}.png"))
