from torchvision import transforms

def get_cifar10_transform():
    img_size = 32
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
    return transform

def get_celeba_transform():
    img_size = 64
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
    return transform

def get_color_mnist_transform():
    img_size = 32
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))])
    return transform

def get_mnist_fmnist_transform():
    img_size = 32
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    return transform


DATASET_DICT = {
    'celeba': get_celeba_transform,
    'cifar10': get_cifar10_transform,
    'color_mnist': get_color_mnist_transform,
    'mnist_fmnist': get_mnist_fmnist_transform,
}

def get_transform(dataset_name):
    transform = DATASET_DICT[dataset_name]()
    return transform