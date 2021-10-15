from enum import Enum
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from PIL import Image
import io


class PitchYaw(Enum):
    PITCH = 'pitch'
    YAW = 'yaw'


def pitchyaw_to_3d_vector(pitchyaw: torch.Tensor) -> torch.Tensor:
    """
    2D pitch and yaw value to a 3D vector

    :param pitchyaw: 2D gaze value in pitch and yaw
    :return: 3D vector
    """
    return torch.stack([
        -torch.cos(pitchyaw[:, 0]) * torch.sin(pitchyaw[:, 1]),
        -torch.sin(pitchyaw[:, 0]),
        -torch.cos(pitchyaw[:, 0]) * torch.cos(pitchyaw[:, 1])
    ], dim=1)


def calc_angle_error(labels: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """
    Calculate the angle between `labels` and `outputs` in degrees.

    :param labels: ground truth gaze vectors
    :param outputs: predicted gaze vectors
    :return: Mean angle in degrees.
    """
    labels = pitchyaw_to_3d_vector(labels)
    labels_norm = labels / torch.linalg.norm(labels, axis=1).reshape((-1, 1))

    outputs = pitchyaw_to_3d_vector(outputs)
    outputs_norm = outputs / torch.linalg.norm(outputs, axis=1).reshape((-1, 1))

    angles = F.cosine_similarity(outputs_norm, labels_norm, dim=1)
    angles = torch.clip(angles, -1.0, 1.0)  # fix NaN values for 1.0 < angles < -1.0

    rad = torch.arccos(angles)
    return torch.rad2deg(rad).mean()


def plot_prediction_vs_ground_truth(labels, outputs, axis: PitchYaw):
    """
    Create a plot between the predictions and the ground truth values.

    :param labels: ground truth values
    :param outputs: predicted values
    :param axis: weather pitch or yaw
    :return: scatter plot of predictions and the ground truth values
    """

    labels = torch.rad2deg(labels)
    outputs = torch.rad2deg(outputs)

    if axis == PitchYaw.PITCH:
        plt.scatter(labels[:, :1].cpu().detach().numpy().reshape(-1), outputs[:, :1].cpu().detach().numpy().reshape(-1))
    else:
        plt.scatter(labels[:, 1:].cpu().detach().numpy().reshape(-1), outputs[:, 1:].cpu().detach().numpy().reshape(-1))
    plt.plot([-30, 30], [-30, 30], color='#ff7f0e')
    plt.xlabel('ground truth (degrees)')
    plt.ylabel('prediction (degrees')
    plt.title(axis.value)
    if axis == PitchYaw.PITCH:
        plt.xlim((-30, 5))
        plt.ylim((-30, 5))
    else:
        plt.xlim((-30, 30))
        plt.ylim((-30, 30))

    return plt.gcf()


def plot_to_image(fig) -> torch.Tensor:
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.

    :param fig: matplotlib figure
    :return: plot for torchvision
    """

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    image = Image.open(buf).convert("RGB")
    image = torchvision.transforms.ToTensor()(image)
    return image


def log_figure(loggers: List, tag: str, figure, global_step: int) -> None:
    """
    Log figure as image. Only works for `TensorBoardLogger`.

    :param loggers:
    :param tag:
    :param figure:
    :param global_step:
    :return:
    """

    if isinstance(loggers, list):
        for logger in loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(tag, plot_to_image(figure), global_step, dataformats="CHW")
    elif isinstance(loggers, TensorBoardLogger):
        loggers.experiment.add_image(tag, plot_to_image(figure), global_step, dataformats="CHW")


def get_random_idx(k: int, size: int) -> np.ndarray:
    """
    Get `k` random values of a list of size `size`.

    :param k: number or random values
    :param size: total number of values
    :return: list of `k` random values
    """
    return (np.random.rand(k) * size).astype(int)


def get_each_of_one_grid_idx(k: int, gaze_locations: np.ndarray, screen_sizes: np.ndarray) -> np.ndarray:
    """
    Get `k` random values of each of the $\sqrt{k}\times\sqrt{k}$ grid.

    :param k: number or random values
    :param gaze_locations: list of the position on the screen in pixels for each gaze value
    :param screen_sizes: list of the screen sizes in pixels for each gaze value
    :return: list of `k` random values
    """
    grids = int(np.sqrt(k))  # get grid size from k

    grid_width = screen_sizes[0][0] / grids
    height_width = screen_sizes[0][1] / grids

    gaze_locations = np.asarray(gaze_locations)

    valid_random_idx = []

    for width_range in range(grids):
        filter_width = (grid_width * width_range < gaze_locations[:, :1]) & (gaze_locations[:, :1] < grid_width * (width_range + 1))
        for height_range in range(grids):
            filter_height = (height_width * height_range < gaze_locations[:, 1:]) & (gaze_locations[:, 1:] < height_width * (height_range + 1))
            complete_filter = filter_width & filter_height
            complete_filter = complete_filter.reshape(-1)
            if sum(complete_filter) > 0:
                true_idxs = np.argwhere(complete_filter)
                random_idx = (np.random.rand(1) * len(true_idxs)).astype(int).item()
                valid_random_idx.append(true_idxs[random_idx].item())

    if len(valid_random_idx) != k:
        # fill missing calibration samples
        missing_k = k - len(valid_random_idx)
        missing_idxs = (np.random.rand(missing_k) * len(gaze_locations)).astype(int)
        for missing_idx in missing_idxs:
            valid_random_idx.append(missing_idx.item())

    return valid_random_idx
