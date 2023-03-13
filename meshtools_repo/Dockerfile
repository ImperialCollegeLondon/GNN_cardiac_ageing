# based on https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.0.3/ubuntu2004/runtime/Dockerfile
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.0.3/ubuntu2004/base/Dockerfile

FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

ENV NVARCH x86_64
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.0 brand=tesla,driver>=418,driver<419"
ENV NV_CUDA_CUDART_VERSION 11.0.221-1
ENV NV_CUDA_COMPAT_PACKAGE cuda-compat-11-0

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH}/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 11.0.3

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-0=${NV_CUDA_CUDART_VERSION} \
    ${NV_CUDA_COMPAT_PACKAGE} \
    && ln -s cuda-11.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NV_CUDA_LIB_VERSION 11.0.3-1
ENV NV_NVTX_VERSION 11.0.167-1
ENV NV_LIBNPP_VERSION 11.1.0.245-1
ENV NV_LIBNPP_PACKAGE libnpp-11-0=${NV_LIBNPP_VERSION}
ENV NV_LIBCUSPARSE_VERSION 11.1.1.245-1
ENV NV_LIBCUBLAS_PACKAGE_NAME libcublas-11-0
ENV NV_LIBCUBLAS_VERSION 11.2.0.252-1
ENV NV_LIBCUBLAS_PACKAGE ${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}
ENV NV_LIBNCCL_PACKAGE_NAME libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION 2.13.4-1
ENV NCCL_VERSION 2.13.4-1
ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda11.0
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-11-0=${NV_CUDA_LIB_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    cuda-nvtx-11-0=${NV_NVTX_VERSION} \
    libcusparse-11-0=${NV_LIBCUSPARSE_VERSION} \
    ${NV_LIBCUBLAS_PACKAGE} \
    ${NV_LIBNCCL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*
# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME}

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install libgl-dev curl -y

RUN curl -L https://github.com/conda-forge/miniforge/releases/download/4.13.0-1/Mambaforge-4.13.0-1-Linux-x86_64.sh -o /tmp/mambaforge.sh
RUN chmod 770 /tmp/mambaforge.sh
RUN /tmp/mambaforge.sh -b -p /opt/mambaforge
ENV PATH=/opt/mambaforge/bin:${PATH}
RUN mamba init

RUN mamba install pytorch=*=*cuda11* pyg \
    pytorch-lightning ipython pyvista ipdb torchvision torchaudio \
    jupyter statsmodels catboost shap sdv r-circlize optuna mlflow  r-corrplot r-networkd3 r-superheat joypy lifelines scikit-survival captum xlrd openpyxl -c pyg -c pytorch -y
RUN pip install open3d -U

RUN mamba install rpy2 r-tidyverse r-lme4 r-emmeans r-iswr r-ggridges r-dplyr r-jtools r-broom r-irkernel \
r-dplyr r-ggplot2 r-ggthemes r-tidyr r-broom r-purrr r-plyr r-tibble r-systemfit r-ggpmisc \
r-rlang r-car r-magrittr r-minpack.lm r-scales \
r-ggdendro r-gridextra r-shiny r-miniui r-matching r-mass \
r-bitops r-rcurl r-rcppprogress -y

RUN Rscript -e 'install.packages(c("forestmangr", "MatchIt"), repos="http://cran.us.r-project.org")'

RUN mamba install r-cowplot r-ggextra -y

RUN pip install vaecompare

#RUN mamba update pytorch=*=*cuda11* pyg -c pyg -c pytorch -y

COPY . /tmp/meshtools

RUN cd /tmp/meshtools && pip install . --no-deps && python -c 'import meshtools'
