
# install mmcv-full 1.7.0
cd mmcv
pip install -r requirements/optional.txt
nvcc --version
gcc --version
pip install -e . -v
cd ..

pip install -e sam_automask
pip install -e chumpy
pip install -e .[all]
pip install -v -e third-party/ViTPose
pip install -U trimesh
pip install onnxruntime-gpu==1.18.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install pyrender==0.1.45
pip install loguru==0.7.3
pip install flask
