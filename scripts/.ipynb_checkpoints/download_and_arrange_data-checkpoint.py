import os
import json
import boto3
import tqdm
import numpy as np
from typing import List, Tuple, Dict


def get_split_indexes(class_label: str, data_json: Dict[str, List[str]]) -> Tuple[List[int], List[int], List[int]]:
    """
    Gets the split indexes for training, validation and test sets.

    Args:
        class_label: The key in the data JSON representing the class label/number of objects.
        data_json: A dictionary containing lists of file paths.

    Returns:
        A tuple of three lists representing the indexes for the validation,
        testing and training sets.
    """
    len_val = len(data_json[class_label])

    # Take first 20% as valid, next 20% as test, rest as train. Just store max, min idx.
    num_val = int(np.ceil(0.2 * len_val))

    valid_idx = [0, int(np.ceil(0.2 * len_val))]
    test_idx = [valid_idx[-1] + 1, valid_idx[-1] + num_val + 1]
    train_idx = [test_idx[-1] + 1, len_val - 1]

    return valid_idx, test_idx, train_idx


def download_and_arrange_data() -> None:
    """
    Downloads and arranges images into train, valid, test folders, and subdirectories
    for images of 1–5 objects
    """
    s3_client = boto3.client('s3')

    with open('file_list.json', 'r') as file:
        data: Dict[str, List[str]] = json.load(file)

    for num_obj, num_images in data.items():
        print(f"Downloading Images with {num_obj} objects")
        # Get train, valid, test indexes for images with num_obj objects
        idx = get_split_indexes(num_obj, data)

        for i, sub_dir in enumerate(('valid', 'test', 'train')):
            directory = os.path.join('binImages', sub_dir, num_obj)
            if not os.path.exists(directory):
                os.makedirs(directory)

            print(f"Downloading images for {sub_dir} set")
            for file_path in tqdm.tqdm(num_images[idx[i][0]:idx[i][-1]]):
                file_name = os.path.basename(file_path).split('.')[0] + '.jpg'
                s3_client.download_file('aft-vbi-pds',
                                        os.path.join('bin-images', file_name),
                                        os.path.join(directory, file_name))


if __name__ == "__main__":
    download_and_arrange_data()
