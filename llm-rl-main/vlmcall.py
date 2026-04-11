"""Quick test: call VLM via LiteLLM with 20 images at once."""

import litellm
import base64
import time
import os
import glob

os.environ["NVIDIA_NIM_API_KEY"] = "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"

# Collect up to 20 image files from current directory
image_extensions = ("*.png", "*.jpg", "*.jpeg")
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(ext))
image_files = sorted(image_files)[:20]

if not image_files:
    print("ERROR: No image files found in the current directory.")
    exit(1)

print(f"Found {len(image_files)} image(s): {image_files}")

# Build content list: one text prompt + all images
content = [
    {"type": "text", "text": f"I am sending you {len(image_files)} images. Describe each image in one sentence."},
]

for img_path in image_files:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # Detect mime type
    ext = os.path.splitext(img_path)[1].lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    })

messages = [{"role": "user", "content": content}]

model = "nvidia_nim/google/gemma-3-27b-it"
print(f"Calling {model} with {len(image_files)} images ...")
t0 = time.time()
try:
    resp = litellm.completion(model=model, messages=messages, temperature=0.7)
    elapsed = time.time() - t0
    print(f"Response ({elapsed:.1f}s):\n{resp.choices[0].message.content}")
except Exception as e:
    print(f"ERROR: {e}")