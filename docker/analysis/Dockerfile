# syntax=docker/dockerfile:1.7
# CPU image for the MM/GBSA stage of the post-MD analysis (analysis/mdana), run on Salad CPU-only nodes.
# The environment is the one the Colab notebook uses: docker/tools-env.lock.txt + gmx_MMPBSA 1.7.0, so a job gives
# the same numbers wherever it runs. Built by .github/workflows/build-analysis.yml and pushed to the eralpha-antidepressant-md
# package under an ana-<sha> tag. No data and no credentials: inputs and results live on Google Drive (rclone,
# drive.file scope) and the rclone token is a Salad environment secret.
FROM mambaorg/micromamba:2.3.2-ubuntu24.04

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/tools-env.lock.txt /tmp/env.lock.txt
RUN micromamba install -y -n base --file /tmp/env.lock.txt \
 && micromamba clean --all --yes
# No env activation (as on Colab: env/bin on PATH only): the GROMACS package's activation script fails under the
# micromamba build shell (GMXRC.bash globs a completion file that is not there).
ENV PATH=/opt/conda/bin:$PATH
RUN /opt/conda/bin/pip install --no-cache-dir "gmx_MMPBSA==1.7.0"

# rclone, checked against the release checksum (same version as the production image). Not named RCLONE_*_VERSION:
# rclone reads every RCLONE_<FLAG> variable as a flag, and build ARGs are in the environment of RUN.
ARG RCLONE_REL=1.75.1
ARG RCLONE_ZIP_SHA256=982b5aa772841168f8e380f139e9e787b2a105403e32b94da8676a0e1c0a13ab
RUN /opt/conda/bin/python - <<PY
import hashlib, io, urllib.request, zipfile
url = "https://github.com/rclone/rclone/releases/download/v${RCLONE_REL}/rclone-v${RCLONE_REL}-linux-amd64.zip"
data = urllib.request.urlopen(url, timeout=120).read()
assert hashlib.sha256(data).hexdigest() == "${RCLONE_ZIP_SHA256}", "rclone checksum mismatch"
z = zipfile.ZipFile(io.BytesIO(data))
name = next(n for n in z.namelist() if n.endswith("/rclone"))
open("/opt/conda/bin/rclone", "wb").write(z.read(name))
PY
RUN chmod 755 /opt/conda/bin/rclone

COPY --chown=$MAMBA_USER:$MAMBA_USER analysis/mdana /opt/eralpha/md-workflow/analysis/mdana
COPY --chown=$MAMBA_USER:$MAMBA_USER analysis/mmpbsa_gb.in analysis/mmpbsa_pb.in /opt/eralpha/md-workflow/analysis/
COPY --chmod=755 docker/analysis/run_mmgbsa.sh /opt/eralpha/bin/
ENV PATH=/opt/eralpha/bin:$PATH \
    PYTHONPATH=/opt/eralpha/md-workflow/analysis \
    AMBERHOME=/opt/conda \
    MPLBACKEND=Agg \
    MDANA_DRIVE=/tmp/drive \
    MDANA_WORK=/tmp/work \
    OMPI_MCA_btl_vader_single_copy_mechanism=none \
    PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
RUN gmx --version | grep -E "GROMACS version" && rclone version | head -n 1 \
 && python -c "import mdana.mmgbsa, GMXMMPBSA; print('mdana + gmx_MMPBSA import ok')"
WORKDIR /tmp
ENTRYPOINT []
CMD ["/opt/eralpha/bin/run_mmgbsa.sh"]
