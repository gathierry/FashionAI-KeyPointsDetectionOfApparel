## Dependency

- Python3.5
- Numpy
- Pandas
- PyTorch
- cv2
- scikit-learn
- py3nvml and nvidia-ml-py3
- tqdm

## Usage

### Train

To train a model for each clothing category

```bash stage2/autorun.sh```

It actually runs ```stage2/trainval.py``` five times for five clothing types.

Data preprocessing is performed in ```stage2/data_generator.py``` which is called during the training

Two networks are used during this challenge, which are ```stage2/cascaded_pyramid_network.py``` and ```stage2v9/cascaded_pyramid_network_v9.py``` . The final score result from ensemble learning. The two networks share the same architecture with different backbones.

### Test

To test and generate result, go the folder

```cd kpdetector```

run ```python3 predict.py``` five times with corresponding configuration (clothing type and model used)

run ```python3 concatenate_results.py``` to merge all results in a .csv file for submission.

## External Data

Pre-trained [ResNet152](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) and [SENet154](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py)  are used.