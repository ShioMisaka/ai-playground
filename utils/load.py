from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(path : str, is_train: bool):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = datasets.MNIST(path, is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)