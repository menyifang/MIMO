
# install pytorch230+cuda121
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# install tensorflow
pip install tensorflow==2.16.1

# install xformers
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

# install diffusers
pip install diffusers==0.24.0

pip install -r requirements.txt
