import torch
from torch.utils.data import Dataset


class StoryDataset(Dataset):

    def __init__(self, hf_dataset, transform=None):

        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        images = sample["images"]

        processed_images = []

        for img in images:

            if self.transform:
                img = self.transform(img)

            processed_images.append(img)

        images_tensor = torch.stack(processed_images)

        return {
            "images": images_tensor,
            "story": sample["story"]
        }