import itertools
from typing import List, Tuple

import albumentations as A
import h5py
import numpy as np
import pandas as pd
import skimage.io
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def filter_persons_by_idx(file_names: List[str], keep_person_idxs: List[int]) -> List[int]:
    """
    Only keep idx that match the person ids in `keep_person_idxs`.

    :param file_names: list of all file names
    :param keep_person_idxs: list of person ids to keep
    :return: list of valid idxs that match `keep_person_idxs`
    """
    idx_per_person = [[] for _ in range(15)]
    if keep_person_idxs is not None:
        keep_person_idxs = [f'p{person_idx:02d}/' for person_idx in set(keep_person_idxs)]
        for idx, file_name in enumerate(file_names):
            if any(keep_person_idx in file_name for keep_person_idx in keep_person_idxs):  # is a valid person_idx ?
                person_idx = int(file_name.split('/')[-3][1:])
                idx_per_person[person_idx].append(idx)
    else:
        for idx, file_name in enumerate(file_names):
            person_idx = int(file_name.split('/')[-3][1:])
            idx_per_person[person_idx].append(idx)

    return list(itertools.chain(*idx_per_person))  # flatten list


def remove_error_data(data_path: str, file_names: List[str]) -> List[int]:
    """
    Remove erroneous data, where the gaze point is not in the screen.

    :param data_path: path to the dataset including the `not_on_screen.csv` file
    :param file_names: list of all file names
    :return: list of idxs of valid data
    """
    valid_idxs = []

    df = pd.read_csv(f'{data_path}/not_on_screen.csv')
    error_file_names = set([error_file_name[:-8] for error_file_name in df['file_name'].tolist()])
    file_names = [file_name[:-4] for file_name in file_names]
    for idx, file_name in enumerate(file_names):
        if file_name not in error_file_names:
            valid_idxs.append(idx)

    return valid_idxs


class MPIIFaceGaze(Dataset):
    """
    MPIIFaceGaze dataset with offline preprocessing (= already preprocessed)
    """

    def __init__(self, data_path: str, file_name: str, keep_person_idxs: List[int], transform=None, train: bool = False, force_flip: bool = False, use_erroneous_data: bool = False):
        if keep_person_idxs is not None:
            assert len(keep_person_idxs) > 0
            assert max(keep_person_idxs) <= 14  # last person id = 14
            assert min(keep_person_idxs) >= 0  # first person id = 0

        self.data_path = data_path
        self.hdf5_file_name = f'{data_path}/{file_name}'
        self.h5_file = None

        self.transform = transform
        self.train = train
        self.force_flip = force_flip

        with h5py.File(self.hdf5_file_name, 'r') as f:
            file_names = [file_name.decode('utf-8') for file_name in f['file_name_base']]

        if not train:
            by_person_idx = filter_persons_by_idx(file_names, keep_person_idxs)
            self.idx2ValidIdx = by_person_idx
        else:
            by_person_idx = filter_persons_by_idx(file_names, keep_person_idxs)
            non_error_idx = file_names if use_erroneous_data else remove_error_data(data_path, file_names)
            self.idx2ValidIdx = list(set(by_person_idx) & set(non_error_idx))

    def __len__(self) -> int:
        return len(self.idx2ValidIdx) * 2 if self.train else len(self.idx2ValidIdx)

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

    def open_hdf5(self):
        self.h5_file = h5py.File(self.hdf5_file_name, 'r')

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.h5_file is None:
            self.open_hdf5()

        augmented_person = idx >= len(self.idx2ValidIdx)
        if augmented_person:
            idx -= len(self.idx2ValidIdx)  # fix idx

        idx = self.idx2ValidIdx[idx]

        file_name = self.h5_file['file_name_base'][idx].decode('utf-8')
        gaze_location = self.h5_file['gaze_location'][idx]
        screen_size = self.h5_file['screen_size'][idx]

        person_idx = int(file_name.split('/')[-3][1:])

        left_eye_image = skimage.io.imread(f"{self.data_path}/{file_name}-left_eye.png")
        left_eye_image = np.flip(left_eye_image, axis=1)
        right_eye_image = skimage.io.imread(f"{self.data_path}/{file_name}-right_eye.png")
        full_face_image = skimage.io.imread(f"{self.data_path}/{file_name}-full_face.png")
        gaze_pitch = np.array(self.h5_file['gaze_pitch'][idx])
        gaze_yaw = np.array(self.h5_file['gaze_yaw'][idx])

        if augmented_person or self.force_flip:
            person_idx += 15  # fix person_idx
            left_eye_image = np.flip(left_eye_image, axis=1)
            right_eye_image = np.flip(right_eye_image, axis=1)
            full_face_image = np.flip(full_face_image, axis=1)
            gaze_yaw *= -1

        if self.transform:
            left_eye_image = self.transform(image=left_eye_image)["image"]
            right_eye_image = self.transform(image=right_eye_image)["image"]
            full_face_image = self.transform(image=full_face_image)["image"]

        return {
            'file_name': file_name,
            'gaze_location': gaze_location,
            'screen_size': screen_size,

            'person_idx': person_idx,

            'left_eye_image': left_eye_image,
            'right_eye_image': right_eye_image,
            'full_face_image': full_face_image,

            'gaze_pitch': gaze_pitch,
            'gaze_yaw': gaze_yaw,
        }


def get_dataloaders(path_to_data: str, validate_on_person: int, test_on_person: int, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, valid and test dataset.
    The train dataset includes all persons except `validate_on_person` and `test_on_person`.

    :param path_to_data: path to dataset
    :param validate_on_person: person id to validate on during training
    :param test_on_person: person id to test on after training
    :param batch_size: batch size
    :return: train, valid and test dataset
    """
    transform = {
        'train': A.Compose([
            A.ShiftScaleRotate(p=1.0, shift_limit=0.2, scale_limit=0.1, rotate_limit=10),
            A.Normalize(),
            ToTensorV2()
        ]),
        'valid': A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
    }

    train_on_persons = list(range(0, 15))
    if validate_on_person in train_on_persons:
        train_on_persons.remove(validate_on_person)
    if test_on_person in train_on_persons:
        train_on_persons.remove(test_on_person)
    print('train on persons', train_on_persons)
    print('valid on person', validate_on_person)
    print('test on person', test_on_person)

    dataset_train = MPIIFaceGaze(path_to_data, 'data.h5', keep_person_idxs=train_on_persons, transform=transform['train'], train=True)
    print('len(dataset_train)', len(dataset_train))
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

    dataset_valid = MPIIFaceGaze(path_to_data, 'data.h5', keep_person_idxs=[validate_on_person], transform=transform['valid'])
    print('len(dataset_train)', len(dataset_valid))
    valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)

    dataset_test = MPIIFaceGaze(path_to_data, 'data.h5', keep_person_idxs=[test_on_person], transform=transform['valid'], use_erroneous_data=True)
    print('len(dataset_train)', len(dataset_test))
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, valid_dataloader, test_dataloader
