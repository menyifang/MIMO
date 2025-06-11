FROM modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.3.1-tf2.16.1-1.21.0

WORKDIR /app
ENV LD_LIBRARY_PATH /usr/local/lib:/usr/lib/:/usr/local/python-3/lib

# hamer, hmr2,  ProPainter, sam_automask, segment-anything-2-main, setup.sh, third-party, tools, vitpose_model.py, infer.py, setup.py
COPY demo_occ.py /app/demo_occ.py
COPY depth_anything_v2 /app/depth_anything_v2
COPY hamer /app/hamer
COPY hmr2 /app/hmr2
COPY ProPainter /app/ProPainter
COPY sam_automask /app/sam_automask
COPY segment-anything-2-main /app/segment-anything-2-main
COPY chumpy /app/chumpy
COPY setup.sh /app/setup.sh
COPY third-party /app/third-party
COPY tools /app/tools
COPY vitpose_model.py /app/vitpose_model.py
COPY infer.py /app/infer.py
COPY setup.py /app/setup.py

COPY demo.py /app/demo.py
COPY run.sh /app/run.sh

# ubuntu
RUN apt install -y iputils-ping && \
    apt install -y libjemalloc-dev && \
    chmod +x /usr/lib/x86_64-linux-gnu/libjemalloc.so && \
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so

RUN curl https://gosspublic.alicdn.com/ossutil/install.sh | bash

RUN pip3 install flask
RUN bash /app/setup.sh


