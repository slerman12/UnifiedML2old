import torchvision
from torchvision.transforms import transforms

IMAGE_DATASETS = [
    'LSUN', 'LSUNClass',
    'ImageFolder', 'DatasetFolder', 'FakeData',
    'CocoCaptions', 'CocoDetection',
    'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST', 'QMNIST',
    'MNIST', 'KMNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
    'Omniglot', 'SBU', 'Flickr8k', 'Flickr30k',
    'VOCSegmentation', 'VOCDetection', 'Cityscapes', 'ImageNet',
    'Caltech101', 'Caltech256', 'CelebA', 'WIDERFace', 'SBDataset',
    'VisionDataset', 'USPS', 'Kinetics400', 'HMDB51', 'UCF101',
    'Places365'
]

out = ""
for task in IMAGE_DATASETS:
    f = open(f"./{task.lower()}.yaml", "w")
    f.write(r"""defaults:
  - 500K
  - _self_

suite: classify
frame_stack: null
action_repeat: null
nstep: 1
evaluate_per_steps: 20000
evaluate_episodes: 1000
update_per_steps: 1
seed_steps: 20000
explore_steps: 20000
task_name: {}""".format(task))
    f.close()
    out += ' "' + task.lower() + '"'
print(out)
