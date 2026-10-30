# The scene route's GPU steps on the scene VM (wsroute/vm.py setup): WorldStereo's image (torch 2.7.1 cu128,
# pytorch3d and MapAnything built for Blackwell) plus gsplat compiled for the 3090 (8.6), the L40S (8.9) and the
# RTX PRO 6000 (12.0), what the splat, conflict and next-best-view steps import, and 3DGRUT's checkout for the
# USDZ export (gs_usd_light.py: its transcoding path, no CUDA extension). Runs the steps of SIM2CHIP_SCENE_PYTHON,
# SIM2CHIP_MAPANYTHING_PYTHON and SIM2CHIP_GSPLAT_PYTHON through vm.py's route-python wrapper. Built and pushed
# by build.sh route; FROM is the WorldStereo tag build.sh computes.
ARG BASE
FROM ${BASE}

ARG GRUT_COMMIT=e6f4552
ENV TORCH_CUDA_ARCH_LIST="8.6 8.9 12.0"
RUN pip install -q --no-build-isolation gsplat==1.5.3 -c /opt/keep.txt
RUN pip install -q lpips==0.1.4 plyfile==1.1.3 jaxtyping rich usd-core nvidia-ncore==19.8.3 simplejpeg tensorboard \
        imageio-ffmpeg av==17.1.0 numpy==1.26.4 -c /opt/keep.txt
RUN git clone -q https://github.com/nv-tlabs/3dgrut.git /opt/3dgrut && git -C /opt/3dgrut checkout -q $GRUT_COMMIT
# The Fixer step starts Fixer's own container through the host's Docker socket (route_python.sh mounts it).
RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz \
    | tar -xz -C /usr/local/bin --strip-components=1 docker/docker

RUN python -c "import gsplat, lpips, plyfile, pxr, pytorch3d.renderer, numpy; from gsplat.strategy import MCMCStrategy; \
assert numpy.__version__ == '1.26.4'; import sys; sys.path.insert(0, '/opt/3dgrut'); \
from threedgrut.export.usd.nurec.exporter import NuRecExporter; print('imports ok')"
