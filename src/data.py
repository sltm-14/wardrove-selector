import os
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms

from torch.utils.data import DataLoader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
targets_dir = f"{BASE_DIR}/../data/datasets/paramaggarwal/fashion-product-images-small/versions/1/images"


# Fixed list, same order every time, so train/val/test instances all
# agree on which integer maps to which articleType
categories = ["Tshirts","Shirts","Tops","Jeans","Sweatshirts","Sweaters",
      "Jackets","Dresses","Skirts","Shorts","Track Pants",
      "Casual Shoes","Sports Shoes","Formal Shoes"]

class wardroveDataset(Dataset):
    def __init__(self, dataframe, target_imgs_dir, transform):
        # DataFrame slice for this split (train_df, val_df, or test_df)
        self.dataframe = dataframe
        # Folder holding the actual .jpg files, shared across all splits
        self.target_imgs_dir = target_imgs_dir
        # torchvision transforms.Compose pipeline, different for train vs eval
        self.transform = transform

        # Translates articleType strings to the integer ids CrossEntropyLoss expects
        self.label_to_val = {label: val for val, label in enumerate(categories)}

    def __len__(self):
        # Tells the DataLoader how many samples exist, so it knows the valid idx range
        return len(self.dataframe)

    def __getitem__(self, idx):
        # idx is a position (0 to len-1)
        row = self.dataframe.iloc[idx]
        # Build the path to this row's target image using its real id
        path = f'{self.target_imgs_dir}/{row["id"]}.jpg'
        # convert("RGB") normalizes channels in case a file is grayscale or has alpha
        image = Image.open(path).convert("RGB")
        # Applies resize/augmentation/ToTensor, returns a tensor ready for the model
        image = self.transform(image)
        # String label ("Shirts") converted to the integer the loss function expects
        label_val = self.label_to_val[row["articleType"]]

        return image, label_val


# Augmentation only for train: flip + color jitter, applied randomly each time
# an image is loaded, so the model sees slightly different versions across epochs
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])


# No augmentation for val/test: evaluation needs to be deterministic and reflect
# real, unaltered data, not artificially varied inputs
eval_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


def get_dataloaders(batch_size=32):

    df = pd.read_csv(f"{BASE_DIR}/../data/clean_data/clean_df.csv")

    # Two-step split: 80% train, then split the remaining 20% into val/test (10%/10%)
    # stratify keeps class proportions consistent across splits, important given the class imbalance
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["articleType"]
    )

    # stratify here uses temp_df's own labels, not df's, since we're splitting temp_df now
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["articleType"]
    )

    # One Dataset instance per split, each with its own dataframe and transform
    train_dataset = wardroveDataset(train_df, targets_dir, train_transform)
    val_dataset = wardroveDataset(val_df, targets_dir, eval_transform)
    test_dataset = wardroveDataset(test_df, targets_dir, eval_transform)

    # shuffle=True only for train, so batch order varies each epoch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # shuffle=False for val/test, order doesn't matter and stays predictable
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader