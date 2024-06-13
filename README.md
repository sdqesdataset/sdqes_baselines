## Install

create environment
~~~
conda create -n streaming_query python=3.10 -y
conda activate streaming_query
~~~

install requirements
~~~
pip install -r requirements.txt
~~~

clone the required repositories
~~~
git clone https://github.com/openai/CLIP.git
git clone https://github.com/facebookresearch/LaViLa.git
git clone https://github.com/showlab/EgoVLP.git                  
~~~

download the model weights
~~~
# for EgoVLP
https://drive.usercontent.google.com/download?id=1-cP3Gcg0NGDcMZalgJ_615BQdbFIbcj7&export=download&authuser=0&confirm=t&uuid=cdd022c6-5409-4a70-8930-d82ca1669d6c&at=APZUnTXbTHrGakBpwOWsCD5u3c6I%3A1715711309453
# TODO: rename the file to `model.pth` and place it in the `EgoVLP` directory
#mv model.pth EgoVLP/

# image encoder for egoVLP
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
mv jx_vit_base_p16_224-80ecf9dd.pth EgoVLP/

# lavila
wget https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth
wget https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth
# TODO: move into the `LaViLa` directory
~~~

**Important**: need to modify line 48 in EgoVLP/model/model.py to hardcode the path to the `jx_vit_base_p16_224-80ecf9dd.pth` file.


### Optional: pre-compile QRNN cuda kernel

This step is optional but recommended faster model loading (otherwise it'll compile on the first run).
To compile the QRNN cuda kernel, run the following command:

```bash
export CUDA_HOME=/usr/local/cuda-11.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

python setup.py build_ext --inplace
```