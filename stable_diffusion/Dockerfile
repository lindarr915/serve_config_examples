# File name: Dockerfile
FROM rayproject/ray-ml:2.9.0-py310

RUN pip install diffusers==0.25.1 transformers==4.36.2 accelerate==0.26.1 omegaconf==2.3.0
WORKDIR /serve_app
COPY stable_diffusion.py /serve_app/stable_diffusion.py
