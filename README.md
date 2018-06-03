## Dependency

- Python3.5
- Numpy
- Pandas
- PyTorch
- cv2
- scikit-learn
- py3nvml and nvidia-ml-py3
- tqdm

## Dataset
- FashionAI Global Challenge - Key Points Detection of Apparel [Dataset](https://tianchi.aliyun.com/competition/information.htm?spm=5176.11165261.5678.2.34b72ec5iFguTn&raceId=231648&_lang=en_US)

## Usage

### Train

To train a model for each clothing category

```bash stage2/autorun.sh```

It actually runs ```stage2/trainval.py``` five times for five clothing types.

Data preprocessing is performed in ```stage2/data_generator.py``` which is called during the training

Two networks are used during this challenge, which are ```stage2/cascaded_pyramid_network.py``` and ```stage2v9/cascaded_pyramid_network_v9.py```. They are both re-implementation of [Cascaded Pyramid Network for Multi-Person Pose Estimation](https://arxiv.org/abs/1711.07319). The final score result from ensemble learning. The two networks share the same architecture with different backbones.

### Test

To test and generate result, go the folder

```cd kpdetector```

run ```python3 predict.py``` five times with corresponding configuration (clothing type and model used)

run ```python3 concatenate_results.py``` to merge all results in a .csv file for submission.

## Experiments
- Replace ResNet50 by ResNet152 as backbone network (-0.5%)
- Increase input resolution from 256x256 to 512x512 (-2.5%)
- Gaussian blur on predicted heatmap (-0.5%)
- Reduce rotaton angle from 40 degree to 30 for data augmentation (-0.6%)
- Use ```(x+2, y+2)``` where ```(x, y)``` is max value coordinate (-0.4%)
- Use 1/4 offset from coordinate of the max value to the one of second max value (-0.2%)
- Flip left to right for data augmentation (-0.2%)

## External Data

Pre-trained [ResNet152](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) and [SENet154](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py)  are used.