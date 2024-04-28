# %%
import os

os.environ["HTTPS_PROXY"] = "http://10.254.21.78:10809/"
os.environ["HF_DATASETS_CACHE"] = "$HOME/datasets/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "$HOME/models/huggingface"

# %%
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel
from pytorch_classification.models.hf_clip import HFCLIPClassifier


# %%
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

classifer = HFCLIPClassifier(model, processor)

# %%
classifer.set_classification_task(["cat", "dog"])

# %%
classifer.zeroshot_weights
# %%

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# %%
inputs = processor(images=[image], return_tensors="pt")
classifer(inputs["pixel_values"])

# %%
