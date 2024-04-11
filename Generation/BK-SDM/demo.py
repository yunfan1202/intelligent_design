import torch
from diffusers import StableDiffusionPipeline

# pipe = StableDiffusionPipeline.from_pretrained("checkpoints/bk-sdm-small", torch_dtype=torch.float16)  # 大概3.5G左右显存消耗，对应“pytorch_model.fp16”这样的模型
pipe = StableDiffusionPipeline.from_pretrained("checkpoints/bk-sdm-small")   # 大概4G左右显存消耗

pipe = pipe.to("cuda")

# prompt = "a golden vase with different flowers"
# image = pipe(prompt).images[0]
# image.save("example1.png")

for i in range(5):
    prompt = "a tropical bird sitting on a branch of a tree"
    image = pipe(prompt).images[0]
    image.save("example_"+str(i)+".png")


# https://sofar-sogood.tistory.com/entry/transformers-%EC%97%90%EB%9F%AC-RuntimeError-Failed-to-import-transformerspipelines-because-of-the-following-error