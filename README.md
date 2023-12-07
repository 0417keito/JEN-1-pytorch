# JEN-1-pytorch
Unofficial implementation JEN-1: Text-Guided Universal Music Generation with Omnidirectional Diffusion Models(https://arxiv.org/abs/2308.04729)

![JEN-1](https://github.com/0417keito/JEN-1-pytorch/blob/main/JEN1.png)

## README

## 📖 Quick Index
* [💻 Installation](#-installation)
* [🐍Usage](#-method)
* [🧠TODO](#-todo)
* [🚀Demo](#-demo)
* [🙏Appreciation](#-appreciation)
* [⭐️Show Your Support](#-show_your_support)
* [🙆Welcome Contributions](#-welcom_contributions)

## 💻 Installation
```commandline
git clone https://github.com/0417keito/JEN-1-pytorch.git
cd JEN-1-pytorch
pip install -r requirements.txt
```

## 🐍Usage
### Sampling
```python
import torch
from generation import Jen1

ckpt_path =  'your ckpt path'
jen1 = Jen1(ckpt_path)

prompt = 'a beautiful song'
samples = jen1.generate(prompt)
```

### Training
```commandline
torchrun train.py
```

### Dataset format
Json format. the name of the Json file must be the same as the target music file.
```json
{"prompt": "a beautiful song"}
```
```python
How should the data_dir be created?

'''
dataset_dir
├── audios
|    ├── music1.wav
|    ├── music2.wav
|    .......
|    ├── music{n}.wav
|
├── metadata
|   ├── music1.json
|   ├── music2.json
|   ......
|   ├── music{n}.json
|
'''
```

### About config
please see [config.py](https://github.com/0417keito/JEN-1-pytorch/blob/main/utils/config.py) and [conditioner_config.py](https://github.com/0417keito/JEN-1-pytorch/blob/main/utils/conditioner_config.py)

## 🧠TODO
- [ ] Extension to [JEN-1-Composer](https://arxiv.org/abs/2310.19180) 
- [ ] Extension to music generation with singing voice
- [ ] Adaptation of Consistency Model
- [ ] In the paper, Diffusion Autoencoder was used, but I did not have much computing resources, so I used Encodec instead. So, if I can afford it, I will implement Diffusion Autoencoder.

## 🚀Demo
coming soon !

## 🙏Appreciation
[Dr Adam Fils](https://github.com/adamfils) - for support and brought this to my attention.

## ⭐️Show Your Support

If you find this repo interesting and useful, give us a ⭐️ on GitHub! It encourages us to keep improving the model and adding exciting features.
Please inform us of any deficiencies by issue.

## 🙆Welcome Contributions
Contributions are always welcome.
