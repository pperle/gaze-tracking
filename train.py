from argparse import ArgumentParser
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT

from dataset.mpii_face_gaze_dataset import get_dataloaders
from model import FinalModel
from utils import calc_angle_error, PitchYaw, plot_prediction_vs_ground_truth, log_figure, get_random_idx, get_each_of_one_grid_idx


class Model(FinalModel):
    def __init__(self, learning_rate: float = 0.001, weight_decay: float = 0., k=None, adjust_slope: bool = False, grid_calibration_samples: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.k = [9, 128] if k is None else k
        self.adjust_slope = adjust_slope
        self.grid_calibration_samples = grid_calibration_samples

        self.save_hyperparameters()  # log hyperparameters

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def __step(self, batch: dict) -> Tuple:
        """
        Operates on a single batch of data.

        :param batch: The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
        :return: calculated loss, given values and predicted outputs
        """
        person_idx = batch['person_idx'].long()
        left_eye_image = batch['left_eye_image'].float()
        right_eye_image = batch['right_eye_image'].float()
        full_face_image = batch['full_face_image'].float()

        gaze_pitch = batch['gaze_pitch'].float()
        gaze_yaw = batch['gaze_yaw'].float()
        labels = torch.stack([gaze_pitch, gaze_yaw]).T

        outputs = self(person_idx, full_face_image, right_eye_image, left_eye_image)  # prediction on the base model
        loss = F.mse_loss(outputs, labels)

        return loss, labels, outputs

    def training_step(self, train_batch: dict, batch_idx: int) -> STEP_OUTPUT:
        loss, labels, outputs = self.__step(train_batch)

        self.log('train/loss', loss)
        self.log('train/angular_error', calc_angle_error(labels, outputs))

        return loss

    def validation_step(self, valid_batch: dict, batch_idx: int) -> STEP_OUTPUT:
        loss, labels, outputs = self.__step(valid_batch)

        self.log('valid/offset(k=0)/loss', loss)
        self.log('valid/offset(k=0)/angular_error', calc_angle_error(labels, outputs))

        return {'loss': loss, 'labels': labels, 'outputs': outputs, 'gaze_locations': valid_batch['gaze_location'], 'screen_sizes': valid_batch['screen_size']}

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.__log_and_plot_details(outputs, 'valid')

    def test_step(self, test_batch: dict, batch_idx: int) -> STEP_OUTPUT:
        loss, labels, outputs = self.__step(test_batch)

        self.log('test/offset(k=0)/loss', loss)
        self.log('test/offset(k=0)/angular_error', calc_angle_error(labels, outputs))

        return {'loss': loss, 'labels': labels, 'outputs': outputs, 'gaze_locations': test_batch['gaze_location'], 'screen_sizes': test_batch['screen_size']}

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.__log_and_plot_details(outputs, 'test')

    def __log_and_plot_details(self, outputs, tag: str):
        test_labels = torch.cat([output['labels'] for output in outputs])
        test_outputs = torch.cat([output['outputs'] for output in outputs])
        test_gaze_locations = torch.cat([output['gaze_locations'] for output in outputs])
        test_screen_sizes = torch.cat([output['screen_sizes'] for output in outputs])

        figure = plot_prediction_vs_ground_truth(test_labels, test_outputs, PitchYaw.PITCH)
        log_figure(self.logger, f'{tag}/offset(k=0)/pitch', figure, self.global_step)

        figure = plot_prediction_vs_ground_truth(test_labels, test_outputs, PitchYaw.YAW)
        log_figure(self.logger, f'{tag}/offset(k=0)/yaw', figure, self.global_step)

        # find calibration params
        last_x = 500
        calibration_train = test_outputs[:-last_x].cpu().detach().numpy()
        calibration_test = test_outputs[-last_x:].cpu().detach().numpy()

        calibration_train_labels = test_labels[:-last_x].cpu().detach().numpy()
        calibration_test_labels = test_labels[-last_x:].cpu().detach().numpy()

        gaze_locations_train = test_gaze_locations[:-last_x].cpu().detach().numpy()
        screen_sizes_train = test_screen_sizes[:-last_x].cpu().detach().numpy()

        if len(calibration_train) > 0:
            for k in self.k:
                if k <= 0:
                    continue
                calibrated_solutions = []

                num_calibration_runs = 500 if self.grid_calibration_samples else 10_000  # original results are both evaluated with 10,000 runs
                for calibration_run_idx in range(num_calibration_runs):  # get_each_of_one_grid_idx is slower than get_random_idx
                    np.random.seed(42 + calibration_run_idx)
                    calibration_sample_idxs = get_each_of_one_grid_idx(k, gaze_locations_train, screen_sizes_train) if self.grid_calibration_samples else get_random_idx(k, len(calibration_train))
                    calibration_points_x = np.asarray([calibration_train[idx] for idx in calibration_sample_idxs])
                    calibration_points_y = np.asarray([calibration_train_labels[idx] for idx in calibration_sample_idxs])

                    if self.adjust_slope:
                        m, b = np.polyfit(calibration_points_y[:, :1].reshape(-1), calibration_points_x[:, :1].reshape(-1), deg=1)
                        pitch_fixed = (calibration_test[:, :1] - b) * (1 / m)
                        m, b = np.polyfit(calibration_points_y[:, 1:].reshape(-1), calibration_points_x[:, 1:].reshape(-1), deg=1)
                        yaw_fixed = (calibration_test[:, 1:] - b) * (1 / m)
                    else:
                        mean_diff_pitch = (calibration_points_y[:, :1] - calibration_points_x[:, :1]).mean()  # mean offset
                        pitch_fixed = calibration_test[:, :1] + mean_diff_pitch
                        mean_diff_yaw = (calibration_points_y[:, 1:] - calibration_points_x[:, 1:]).mean()  # mean offset
                        yaw_fixed = calibration_test[:, 1:] + mean_diff_yaw

                    pitch_fixed, yaw_fixed = torch.Tensor(pitch_fixed), torch.Tensor(yaw_fixed)
                    outputs_fixed = torch.stack([pitch_fixed, yaw_fixed], dim=1).squeeze(-1)
                    calibrated_solutions.append(calc_angle_error(torch.Tensor(calibration_test_labels), outputs_fixed).item())

                self.log(f'{tag}/offset(k={k})/mean_angular_error', np.asarray(calibrated_solutions).mean())
                self.log(f'{tag}/offset(k={k})/std_angular_error', np.asarray(calibrated_solutions).std())

        # best case, with all calibration samples, all values except the last `last_x` values
        if self.adjust_slope:
            m, b = np.polyfit(calibration_train_labels[:, :1].reshape(-1), calibration_train[:, :1].reshape(-1), deg=1)
            pitch_fixed = torch.Tensor((calibration_test[:, :1] - b) * (1 / m))
            m, b = np.polyfit(calibration_train_labels[:, 1:].reshape(-1), calibration_train[:, 1:].reshape(-1), deg=1)
            yaw_fixed = torch.Tensor((calibration_test[:, 1:] - b) * (1 / m))
        else:
            mean_diff_pitch = (calibration_train_labels[:, :1] - calibration_train[:, :1]).mean()  # mean offset
            pitch_fixed = calibration_test[:, :1] + mean_diff_pitch
            mean_diff_yaw = (calibration_train_labels[:, 1:] - calibration_train[:, 1:]).mean()  # mean offset
            yaw_fixed = calibration_test[:, 1:] + mean_diff_yaw

        pitch_fixed, yaw_fixed = torch.Tensor(pitch_fixed), torch.Tensor(yaw_fixed)
        outputs_fixed = torch.stack([pitch_fixed, yaw_fixed], dim=1).squeeze(-1)
        calibration_test_labels = torch.Tensor(calibration_test_labels)
        self.log(f'{tag}/offset(k=all)/angular_error', calc_angle_error(calibration_test_labels, outputs_fixed))

        figure = plot_prediction_vs_ground_truth(calibration_test_labels, outputs_fixed, PitchYaw.PITCH)
        log_figure(self.logger, f'{tag}/offset(k=all)/pitch', figure, self.global_step)

        figure = plot_prediction_vs_ground_truth(calibration_test_labels, outputs_fixed, PitchYaw.YAW)
        log_figure(self.logger, f'{tag}/offset(k=all)/yaw', figure, self.global_step)


def main(path_to_data: str, validate_on_person: int, test_on_person: int, learning_rate: float, weight_decay: float, batch_size: int, k: int, adjust_slope: bool, grid_calibration_samples: bool):
    seed_everything(42)

    model = Model(learning_rate, weight_decay, k, adjust_slope, grid_calibration_samples)

    trainer = Trainer(
        gpus=1,
        max_epochs=50,
        default_root_dir='./saved_models/',
        logger=[
            TensorBoardLogger(save_dir="tb_logs"),
        ],
        benchmark=True,
    )

    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(path_to_data, validate_on_person, test_on_person, batch_size)
    trainer.fit(model, train_dataloader, valid_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path_to_data", type=str, default='./data')
    parser.add_argument("--validate_on_person", type=int, default=1)
    parser.add_argument("--test_on_person", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--k", type=int, default=[9, 128], nargs='+')
    parser.add_argument("--adjust_slope", type=bool, default=False)
    parser.add_argument("--grid_calibration_samples", type=bool, default=False)
    args = parser.parse_args()

    main(args.path_to_data, args.validate_on_person, args.test_on_person, args.learning_rate, args.weight_decay, args.batch_size, args.k, args.adjust_slope, args.grid_calibration_samples)
