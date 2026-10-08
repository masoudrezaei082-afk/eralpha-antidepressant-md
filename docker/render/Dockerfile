# syntax=docker/dockerfile:1.7
# GPU image for the trajectory animations (analysis/anim/render.py): Blender 4.5 LTS as a Python module (bpy, with
# its Cycles CUDA/OptiX kernels) + Molecular Nodes, run on a Salad GPU node. Built by
# .github/workflows/build-render.yml and pushed to the eralpha-antidepressant-md package under a render-<sha> tag. No data and no
# credentials: inputs come from Drive anim-inputs/ (rclone, drive.file scope; token as a Salad environment secret)
# and the MP4s go back to Drive anim-out/.
FROM python:3.11-slim-bookworm

# bpy's runtime libraries (X11 client libs and GL even when rendering headless), fonts, ffmpeg for encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
      libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libgl1 libegl1 libgomp1 \
      fonts-liberation ffmpeg ca-certificates unzip \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "bpy==4.5.4" "molecularnodes==4.5.11" "MDAnalysis==2.10.0" matplotlib pillow

# rclone, checked against the release checksum (as in docker/analysis/Dockerfile)
ARG RCLONE_REL=1.75.1
ARG RCLONE_ZIP_SHA256=982b5aa772841168f8e380f139e9e787b2a105403e32b94da8676a0e1c0a13ab
RUN python - <<PY
import hashlib, io, urllib.request, zipfile
url = "https://github.com/rclone/rclone/releases/download/v${RCLONE_REL}/rclone-v${RCLONE_REL}-linux-amd64.zip"
data = urllib.request.urlopen(url, timeout=120).read()
assert hashlib.sha256(data).hexdigest() == "${RCLONE_ZIP_SHA256}", "rclone checksum mismatch"
z = zipfile.ZipFile(io.BytesIO(data))
name = next(n for n in z.namelist() if n.endswith("/rclone"))
open("/usr/local/bin/rclone", "wb").write(z.read(name))
PY
RUN chmod 755 /usr/local/bin/rclone

COPY analysis/anim/render.py /opt/eralpha/render.py
COPY --chmod=755 docker/render/run_render.sh /opt/eralpha/run_render.sh
# graphics: libnvoptix (OptiX) is mounted by the NVIDIA runtime only with this capability
ENV NVIDIA_DRIVER_CAPABILITIES=all \
    MPLBACKEND=Agg
# os._exit: bpy 4.5 deadlocks in interpreter finalisation (see render.py), so a plain exit would hang the build
RUN timeout 300 python -c "import os, bpy, molecularnodes; print('bpy', bpy.app.version_string, flush=True); os._exit(0)" \
 && rclone version | head -n 1
WORKDIR /tmp
CMD ["/opt/eralpha/run_render.sh"]
