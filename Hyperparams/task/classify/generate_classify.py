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
  - _self_

suite: classify
stddev_schedule: 'linear(1.0,0.1,100000)'
frame_stack: 1
task_name: {}""".format(task))
    f.close()
    out += ' "' + task.lower() + '"'
print(out)
