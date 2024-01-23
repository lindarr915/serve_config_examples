from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import Response
import torch

from ray import serve


app = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle) -> None:
        self.handle = diffusion_model_handle

    @app.get(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        image = await self.handle.generate.remote(prompt, img_size=img_size)

        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        return Response(content=file_stream.getvalue(), media_type="image/png")


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 0, "max_replicas": 20},

)
class StableDiffusionV2:
    def __init__(self):
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline, DiffusionPipeline
        import subprocess


        model_id = "stabilityai/stable-diffusion-2"

        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )

        media = "-ebs" # "-nvme"

        result = subprocess.run(f"s5cmd cp s3://ml-loading-test/v2-1_768-ema-pruned.safetensors ./model{media}/", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        self.pipe = StableDiffusionPipeline.from_single_file(
            f"./model{media}/v2-1_768-ema-pruned.safetensors",
            # "./model-s3mount/v2-1_768-ema-pruned.safetensors",
            scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
        )

        # self.pipe = StableDiffusionPipeline.from_pretrained(
        #     model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
        # )

        self.pipe = self.pipe.to("cuda")

    def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        with torch.autocast("cuda"):
            image = self.pipe(prompt, height=img_size, width=img_size).images[0]
            # image = self.pipe(prompt).images[0]
            return image

entrypoint = APIIngress.bind(StableDiffusionV2.bind())

# if __name__ == "__main__":
#     serve.run(entrypoint)