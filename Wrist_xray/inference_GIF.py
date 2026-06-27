# svdrr-wrist/test_xray123_single.py をもとに GIF で出力するように変更

import os

import torch
import torchvision
import torchvision.transforms as transforms
from accelerate import Accelerator
from diffusers.utils import load_image
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline

from PIL import Image  # 追加

# ------------------------------------------------------------
# 設定
# ------------------------------------------------------------
model_path = "/work/XRAYDIFF/share/svdrr-wrist/xray123-simple-model-wrist"
accelerator = Accelerator()

step = 10  # 回転の刻み幅（度数法）
angles = list(range(-90, 91, step))  # -90, -80, ..., 0, ..., 80, 90（19枚）

input_image_path = "/work/XRAYDIFF/eitahtnk/zero123-hf/CR_data/###_20210707_HAND__1（小）.png"
path = "###_20210707_HAND"
log_dir = f"/work/XRAYDIFF/eitahtnk/zero123-hf/inference/generated_image/GIF/{path}/{step}"
os.makedirs(log_dir, exist_ok=True)

gif_name = f"rotate_-90_to_90_step{step}.gif"
gif_path = os.path.join(log_dir, gif_name)

# ------------------------------------------------------------
# パイプライン準備
# ------------------------------------------------------------
pipe = Zero1to3StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
)

pipe.enable_vae_tiling()
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")

# （使わないけど元コードに合わせて残しておく）
image_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# ------------------------------------------------------------
# 入力画像ロード
# ------------------------------------------------------------
input_image = load_image(input_image_path)
input_image = input_image.resize((256, 256))
input_image.save(os.path.join(log_dir, "input_0deg.png"))

# ------------------------------------------------------------
# 推論してフレーム生成
# ------------------------------------------------------------
frames = []
failed = []

for a in angles:
    try:
        with torch.autocast("cuda"):
            result = pipe(
                input_imgs=input_image,
                prompt_imgs=input_image,
                poses=[0, a, 0],
                height=256,
                width=256,
                guidance_scale=3.0,
                num_images_per_prompt=1,
                num_inference_steps=50,
            )

        if result is None or getattr(result, "images", None) is None or len(result.images) == 0:
            raise RuntimeError(f"pipe returned no images at angle {a}")

        out = result.images[0]  # PIL.Image

        # 必要なら各フレームPNGも保存（ファイル増えるので必要なときだけON）
        # out.save(os.path.join(log_dir, f"{a:+04d}.png"))

        # GIF用に溜める（RGBで保持するのが安定）
        frames.append(out.convert("RGB"))

        print(f"[OK] angle={a:+d}  frames={len(frames)}/{len(angles)}")

    except Exception as e:
        failed.append((a, repr(e)))
        print(f"[WARN] angle {a} failed: {e}")

# ------------------------------------------------------------
# GIF保存
# ------------------------------------------------------------
if len(frames) == 0:
    raise RuntimeError(f"No frames generated. failed={failed}")

frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=240,  # 速さ（ms）
    loop=1,       # 0=無限ループ
    optimize=True,
)

print(f"Saved GIF: {gif_path}")

if failed:
    print("[INFO] Some angles failed:")
    for a, err in failed:
        print(f"  angle {a}: {err}")
