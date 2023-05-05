from torch.utils.data import Dataset


class TransformDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        transform=None,
        target_transform=None,
    ):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y
