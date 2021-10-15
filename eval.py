from argparse import ArgumentParser

from pytorch_lightning import seed_everything, Trainer

from dataset.mpii_face_gaze_dataset import get_dataloaders
from train import Model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path_to_checkpoints", type=str, default='./pretrained_models')
    parser.add_argument("--path_to_data", type=str, default='./data')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--k", type=int, default=[9, 128], nargs='+')
    parser.add_argument("--adjust_slope", type=bool, default=False)
    parser.add_argument("--grid_calibration_samples", type=bool, default=False)
    args = parser.parse_args()

    for person_idx in range(15):
        person = f'p{person_idx:02d}'

        seed_everything(42)
        print('grid_calibration_samples', args.grid_calibration_samples)
        model = Model.load_from_checkpoint(f'{args.path_to_checkpoints}/{person}.ckpt', k=args.k, adjust_slope=args.adjust_slope, grid_calibration_samples=args.grid_calibration_samples)

        trainer = Trainer(
            gpus=1,
            benchmark=True,
        )

        _, _, test_dataloader = get_dataloaders(args.path_to_data, 0, person_idx, args.batch_size)
        trainer.test(model, test_dataloader)
