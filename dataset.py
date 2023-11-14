import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms


class CatDog(Dataset):
    def __init__(self, root, transform=None, train=False, all_images=False, valid_split=0.1):
        list_images = os.listdir(root)
        if all_images:
            self.images = list_images
        else:
            x_train, x_val = train_test_split(list_images, test_size=valid_split, random_state=42)
            self.images = x_train if train else x_val
        self.root = root
        self.transform = transform
        self.class_to_idx = {"dog": 1, "cat": 0}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file = self.images[index]
        img = Image.open(os.path.join(self.root, file))

        if self.transform is not None:
            img = self.transform(img)
        if "dog" in file:
            label = self.class_to_idx["dog"]
        elif "cat" in file:
            label = self.class_to_idx["cat"]
        else:
            label = -1

        return img, label


def create_dataloader(root, img_size=448, batch_size=16, num_workers=4):
    # Transform dataset
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create dataset
    train_dataset = CatDog(root, transform=train_transform, train=True)
    val_dataset = CatDog(root, transform=val_transform, train=False)

    # Dataloader
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              )

    val_loader = DataLoader(val_dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers
                            )

    return train_loader, val_loader

