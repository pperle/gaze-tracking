import glob
from argparse import ArgumentParser

import pandas as pd
import scipy.io


def check_mpii_gaze_not_on_screen(input_path: str, output_path: str) -> None:
    """
    Create CSV file with the filename of the images where the gaze is not on the screen.

    :param input_path: path to the original MPIIGaze dataset
    :param output_path: output dataset
    :return:
    """

    data = {'file_name': [], 'on_screen_gaze_position': [], 'monitor_pixels': []}

    for person_file_path in sorted(glob.glob(f'{input_path}/Data/Original/p*'), reverse=True):
        person = person_file_path.split('/')[-1]

        screen_size = scipy.io.loadmat(f'{input_path}/Data/Original/{person}/Calibration/screenSize.mat')
        screen_width_pixel = screen_size["width_pixel"].item()
        screen_height_pixel = screen_size["height_pixel"].item()

        for day_file_path in sorted(glob.glob(f'{person_file_path}/d*')):
            day = day_file_path.split('/')[-1]

            df = pd.read_csv(f'{day_file_path}/annotation.txt', sep=' ', header=None)
            for row_idx in range(len(df)):
                row = df.iloc[row_idx]
                on_screen_gaze_target = row[24:26].to_numpy().reshape(-1).astype(int)

                if not (0 <= on_screen_gaze_target[0] <= screen_width_pixel and 0 <= on_screen_gaze_target[1] <= screen_height_pixel):
                    file_name = f'{person}/{day}/{row_idx + 1:04d}.jpg'

                    data['file_name'].append(file_name)
                    data['on_screen_gaze_position'].append(list(on_screen_gaze_target))
                    data['monitor_pixels'].append([screen_width_pixel, screen_height_pixel])

    pd.DataFrame(data).to_csv(f'{output_path}/not_on_screen.csv', index=False)


def check_mpii_face_gaze_not_on_screen(input_path: str, output_path: str) -> None:
    """
    Create CSV file with the filename of the images where the gaze is not on the screen.

    :param input_path: path to the original MPIIFaceGaze dataset
    :param output_path: output dataset
    :return:
    """

    data = {'file_name': [], 'on_screen_gaze_position': [], 'monitor_pixels': []}

    for person_file_path in sorted(glob.glob(f'{input_path}/p*')):
        person = person_file_path.split('/')[-1]

        screen_size = scipy.io.loadmat(f'{input_path}/{person}/Calibration/screenSize.mat')
        screen_width_pixel = screen_size["width_pixel"].item()
        screen_height_pixel = screen_size["height_pixel"].item()

        df = pd.read_csv(f'{person_file_path}/{person}.txt', sep=' ', header=None)
        df_idx = 0

        for day_file_path in sorted(glob.glob(f'{person_file_path}/d*')):
            day = day_file_path.split('/')[-1]

            for image_file_path in sorted(glob.glob(f'{day_file_path}/*.jpg')):
                row = df.iloc[df_idx]
                on_screen_gaze_target = row[1:3].to_numpy().reshape(-1).astype(int)

                if not (0 <= on_screen_gaze_target[0] <= screen_width_pixel and 0 <= on_screen_gaze_target[1] <= screen_height_pixel):
                    file_name = f'{person}/{day}/{image_file_path.split("/")[-1]}'

                    data['file_name'].append(file_name)
                    data['on_screen_gaze_position'].append(list(on_screen_gaze_target))
                    data['monitor_pixels'].append([screen_width_pixel, screen_height_pixel])

                df_idx += 1

    pd.DataFrame(data).to_csv(f'{output_path}/not_on_screen.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, default='./MPIIFaceGaze')
    parser.add_argument("--output_path", type=str, default='./data')
    args = parser.parse_args()

    # check_mpiigaze('args.input_path, args.output_path)
    check_mpii_face_gaze_not_on_screen(args.input_path, args.output_path)
