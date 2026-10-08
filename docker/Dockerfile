# syntax=docker/dockerfile:1.7
# GPU production image: GROMACS (CUDA, single precision, thread-MPI) + the run scripts + Salad's job queue worker.
# Built by .github/workflows/build-and-push.yml and pushed to ghcr.io/<owner>/eralpha-antidepressant-md. Contains no data and
# no credentials: inputs and checkpoints live on Google Drive (rclone, drive.file scope), and the rclone token is a
# Salad environment secret.
#
#   SM 89  = Ada (RTX 4070 Ti / 4070 Ti Super on Salad, L4 on Colab): the production target
#   SM 120 = consumer Blackwell (RTX 5070 Ti-5090): fallback if 4070 Ti capacity is short
# CUDA 12.9 rather than 13.x keeps Salad hosts with pre-580 drivers usable: the image ships SASS for both
# targets, so CUDA minor-version compatibility (driver >= 525) is enough and no PTX JIT is needed.
ARG CUDA_VERSION=12.9.1
ARG UBUNTU=ubuntu24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS build
ARG GMX_VERSION=2026.4
ARG GMX_MD5=fbe7d5071dee17e3f448502b1fd30d55
ARG CUDA_SM="89;120"
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates curl python3-pip unzip \
 && pip3 install --no-cache-dir --break-system-packages "cmake>=3.28,<5" \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN curl -fsSL -o gmx.tgz https://ftp.gromacs.org/gromacs/gromacs-${GMX_VERSION}.tar.gz \
 && echo "${GMX_MD5}  gmx.tgz" | md5sum -c - \
 && tar xzf gmx.tgz && rm gmx.tgz
# AVX2_256 runs on every x86 desktop CPU from the last decade (Salad hosts are mixed Intel/AMD); with update,
# bonded, PME and non-bonded all on the GPU the CPU SIMD level barely affects ns/day.
# Makefiles, not Ninja: GROMACS refuses to build its own FFTW under the Ninja generator.
RUN cmake -S gromacs-${GMX_VERSION} -B build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/gromacs \
      -DGMX_GPU=CUDA -DGMX_CUDA_TARGET_SM="${CUDA_SM}" -DGMX_CUDA_TARGET_COMPUTE="" \
      -DGMX_SIMD=AVX2_256 -DGMX_BUILD_OWN_FFTW=ON -DGMX_MPI=OFF -DGMX_THREAD_MPI=ON -DGMX_OPENMP=ON \
      -DGMX_HWLOC=OFF -DGMX_INSTALL_NBNXM_BENCH=OFF -DBUILD_TESTING=OFF -DGMX_DEVELOPER_BUILD=OFF \
 && cmake --build build -j"$(nproc)" \
 && cmake --install build \
 && rm -rf build
# rclone moves inputs, checkpoints and trajectory parts to and from Google Drive (see run_replica.sh).
ARG RCLONE_VERSION=1.75.1
ARG RCLONE_SHA256=982b5aa772841168f8e380f139e9e787b2a105403e32b94da8676a0e1c0a13ab
RUN curl -fsSL -o rclone.zip https://github.com/rclone/rclone/releases/download/v${RCLONE_VERSION}/rclone-v${RCLONE_VERSION}-linux-amd64.zip \
 && echo "${RCLONE_SHA256}  rclone.zip" | sha256sum -c - \
 && unzip -j rclone.zip "*/rclone" -d /usr/local/bin && rm rclone.zip
# Salad's job queue worker: takes one job at a time from the container group's queue and POSTs it to job_server.py.
ARG QUEUE_WORKER_VERSION=0.7.0
ARG QUEUE_WORKER_SHA256=074a329cf6462e77fc7b72100f59d8a690831456d9420186a834a8f30634c9e4
RUN curl -fsSL -o worker.tgz https://github.com/SaladTechnologies/salad-cloud-job-queue-worker/releases/download/v${QUEUE_WORKER_VERSION}/salad-http-job-queue-worker_x86_64.tar.gz \
 && echo "${QUEUE_WORKER_SHA256}  worker.tgz" | sha256sum -c - \
 && tar xzf worker.tgz -C /usr/local/bin salad-http-job-queue-worker && rm worker.tgz \
 && chmod 755 /usr/local/bin/salad-http-job-queue-worker

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU}
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 ca-certificates python3 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/gromacs /opt/gromacs
COPY --from=build /usr/local/bin/rclone /usr/local/bin/salad-http-job-queue-worker /usr/local/bin/
COPY mdp/ /opt/eralpha/mdp/
COPY --chmod=755 docker/run_replica.sh docker/benchmark.sh docker/job_server.py docker/queue_entry.sh /opt/eralpha/bin/
ENV PATH=/opt/eralpha/bin:/opt/gromacs/bin:$PATH \
    GMXLIB=/opt/gromacs/share/gromacs/top \
    ERALPHA_MDP=/opt/eralpha/mdp \
    SALAD_LOG_LEVEL=info
# CUDA graphs and GPU halo/PME settings are chosen per run by run_replica.sh (see benchmark.sh results).
WORKDIR /work
RUN gmx --version | grep -E "GROMACS version|GPU support|SIMD instructions" && rclone version | head -n 1
# The queue worker hands each job to job_server.py, which runs run_replica.sh; that does its own Drive transfers.
CMD ["/opt/eralpha/bin/queue_entry.sh"]
