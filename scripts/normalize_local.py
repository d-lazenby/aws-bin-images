import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import tqdm

import os
import time
import json


def get_data_loader(data_dir, batch_size=64):
    transform = transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.15),
            transforms.RandomAdjustSharpness(2, p=0.2),
            transforms.RandomAutocontrast(p=0.1),
            transforms.RandomApply([
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)
            ], p=0.15),
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomRotation((90, 90)),
                    transforms.RandomRotation((-90, -90)),
                ])
            ], p=0.1),
            transforms.ToTensor()
    ])

    print("Making ImageFolder...")
    image_data = torchvision.datasets.ImageFolder(root=data_dir,
                                                  transform=transform
                                                  )

    data_loader = DataLoader(image_data,
                             batch_size=batch_size,
                             num_workers=1)

    return data_loader


def batch_mean_and_std(data_loader, max_images=10000):
    num_images = 0
    count = 0
    mean = torch.empty(3)
    std = torch.empty(3)
    print("Calculating mean and std of images...")
    for images, _ in tqdm.tqdm(data_loader):
        batch, _, height, width = images.shape
        num_pixels = batch * height * width
        agg = torch.sum(images, dim=[0, 2, 3])
        agg_of_squares = torch.sum(images ** 2,
                                   dim=[0, 2, 3])
        mean = (count * mean + agg) / (count + num_pixels)
        std = (count * std + agg_of_squares) / (count + num_pixels)
        count += num_pixels
        num_images += len(images)
        if num_images > max_images:
            print(f"{max_images} images processed okay...")
            break

    mean_final, std_final = mean, torch.sqrt(std - mean ** 2)
    print(f"Normalization complete; {num_images} processed")
    print(f"Mean: {mean_final};\nStd: {std_final}")
    return mean_final, std_final


def main():
    tick = time.time()
    data_dir = "binImages/train"

    data_loader = get_data_loader(data_dir)
    mean, std = batch_mean_and_std(data_loader)
    normalization = {"Mean": str(mean), "Std": str(std)}

    tock = time.time()
    time_to_run = tock - tick
    print(f"Program took {time_to_run} to process dataset at {data_dir}.")

    # Make local directory and store mean and std
    path = 'normalization'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created")
        with open(os.path.join(path, 'imp3.json'), 'w') as f:
            json.dump(normalization, f)
            print(f"Mean and std written to {os.path.join(path, 'imp3.json')}")
    else:
        with open(os.path.join(path, 'imp3.json'), 'w') as f:
            json.dump(normalization, f)
            print(f"Mean and std written to {os.path.join(path, 'imp3.json')}")


if __name__ == "__main__":
    main()
