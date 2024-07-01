from typing import TYPE_CHECKING, Union, cast

import torch

if TYPE_CHECKING:
    import torch.utils.data
    from transformers import CLIPProcessor

    import datasets


class CLIPDataset(torch.utils.data.Dataset):
    """
    This class is designed to work with datasets where each item is a dictionary
    containing an 'image' and a 'label'. It preprocesses images using a specified
    processor before they are fed into a CLIP model.
    """

    def __init__(self, dataset: Union["datasets.Dataset", "torch.utils.data.Dataset"], processor: "CLIPProcessor"):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Retrieves an item by its index, preprocesses the image, and returns the
        preprocessed image tensor along with its label.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[Tensor, int]: A tuple containing the preprocessed image tensor and
                                the label of the item.
        """
        item = self.dataset[idx]
        image, label = item["image"], cast(int, item["label"])
        image = cast(torch.Tensor, self.processor(images=[image], return_tensors="pt")["pixel_values"][0])
        return image, label
