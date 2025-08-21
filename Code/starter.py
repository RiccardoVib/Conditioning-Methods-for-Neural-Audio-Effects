# Copyright (C) 2024 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.
#
# If you use this code or any part of it in any program or publication, please acknowledge
# its authors by adding a reference to this publication:
#
# R. Simionato, 2024, "Conditioning Methods for Neural Audio Effects" in proceedings of Sound and Music Computing, Porto, Portugal.

from Training import train
import argparse


"""
main script

"""
def parse_args():
    parser = argparse.ArgumentParser(description='Trains an State-based network. Can also be used to run pure inference.')

    parser.add_argument('--model_save_dir', default='./models', type=str, nargs='?', help='Folder directory in which to store the trained models.')

    parser.add_argument('--data_dir', default='./datasets', type=str, nargs='?', help='Folder directory in which the datasets are stored.')

    parser.add_argument('--datasets', default=" ", type=str, nargs='+', help='The names of the datasets to use.')

    parser.add_argument('--epochs', default=60, type=int, nargs='?', help='Number of training epochs.')

    parser.add_argument('--batch_size', default=1, type=int, nargs='?', help='Batch size.')

    parser.add_argument('--mini_batch_size', default=2400, type=int, nargs='?', help='Mini batch size.')

    parser.add_argument('--units', default=8, nargs='+', type=int, help='Hidden layer sizes (amount of units) of the network.')

    parser.add_argument('--learning_rate', default=3e-4, type=float, nargs='?', help='Initial learning rate.')

    parser.add_argument('--order', default=1, type=int, nargs='?', help='Order of transformation (valid only if FiLM).')

    parser.add_argument('--technique', default='ExtraInp', type=str, nargs='?', help='Conditioning technique [ExtraInp, GAF, FILM-GLU, FILM-GCU].')

    parser.add_argument('--only_inference', default=False, type=bool, nargs='?', help='When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model.')

    return parser.parse_args()


def start_train(args):

    print("######### Preparing for training/inference #########")
    print("\n")
    train(data_dir=args.data_dir,
          model_save_dir=args.model_save_dir,
          save_folder=f'{args.model}_{args.dataset}_{args.technique}_{args.order}',
          dataset=args.datasets,
          epochs=args.epochs,
          b_size=args.batch_size,
          mini_batch_size=args.mini_batch_size,
          units=args.units,
          learning_rate=args.learning_rate,
          order=args.order,
          technique=args.technique,
          inference=args.only_inference)


def main():
    args = parse_args()
    start_train(args)

if __name__ == '__main__':
    main()