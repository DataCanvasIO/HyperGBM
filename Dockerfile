From python:3.8-buster

ARG PIP_PKGS="hypergbm[all] pandas<1.4"
ARG PIP_OPTS="--disable-pip-version-check --no-cache-dir"
#ARG PIP_OPTS="--disable-pip-version-check --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple/"

RUN pip install $PIP_OPTS $PIP_PKGS\
    && mkdir -p /opt/datacanvas \
    && cp -r /usr/local/lib/python3.8/site-packages/hypergbm/examples /opt/datacanvas/ \
    && echo "#!/bin/bash\njupyter lab --notebook-dir=/opt/datacanvas --ip=0.0.0.0 --port=\$NotebookPort --no-browser --allow-root --NotebookApp.token=\$NotebookToken" > /entrypoint.sh \
    && chmod +x /entrypoint.sh \
    && rm -rf /tmp/*

EXPOSE 8888

ENV NotebookToken="" \
    NotebookPort=8888

CMD ["/entrypoint.sh"]

# docker run --rm --name hypergbm -p 8830:8888 -e NotebookToken=your-token  datacanvas/hypergbm
