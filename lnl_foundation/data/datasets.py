from torchvision import datasets, transforms


DATASET_META = {
    "cifar10": {"class": datasets.CIFAR10, "num_classes": 10},
    "cifar100": {"class": datasets.CIFAR100, "num_classes": 100},
}


def load_cifar_base(dataset, data_root, train=True, transform=None, download=True):
    return DATASET_META[dataset]["class"](
        root=str(data_root), train=train, transform=transform, download=download,
    )


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
