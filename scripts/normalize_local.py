import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import tqdm
from typing import Tuple

import os
import time
import logging
import datetime

log_path = "./transform_output"
if not os.path.exists(log_path):
    os.makedirs(log_path)

filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logging.basicConfig(filename=os.path.join(log_path, f'{filename}.log'),
                    level=logging.DEBUG,
                    format='%(asctime)s — %(message)s')


def get_data_loader(data_dir: str, batch_size: int = 64) -> DataLoader:
    transform = transforms.Compose([
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
    logging.info(f"{transform}")

    return data_loader


def batch_mean_and_std(data_loader: DataLoader, max_images: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
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


def main() -> None:
    tick = time.time()
    data_dir = "data/train"
    logging.info(f"For data at {data_dir} with transform:")
    data_loader = get_data_loader(data_dir)
    mean, std = batch_mean_and_std(data_loader)
    logging.info(f"Standardization metrics are Mean: {mean} and Std: {std}")

    tock = time.time()
    time_to_run = tock - tick
    print(f"Program took {time_to_run} to process dataset at {data_dir}.")


if __name__ == "__main__":
    main()
