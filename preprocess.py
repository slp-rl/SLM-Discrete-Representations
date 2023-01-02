"""
Run this Script to pre-process the data for the app.
Example:
python preprocess.py --dense_models hubert --dense_models cpc --dense_models logmel --vocab 50 --vocab 100 --vocab 200

The example will pre-preprocess the data for HuBERT, CPC, and logmel using a vocabulary size of 50,100, and 200.

The Script will generate the following files:
 - assets/2d_area/{dense_models}_{vocab}.csv
 - assets/audio/{LIBRISPEECH|LJSPEECH}_{vocab}_{dense_modelsl}/{n}.wav
 - assets/dimension_reduction/{dense_models}_clusters.csv
 - assets/dimension_reduction/{dense_models}_{T-SNE|PCA}_{2|3}D.csv
 - assets/units_phonemes/{dense_models}_{vocab}._{counts|stats}.csv
"""

import argparse

from preprocessing.dimension_reduction import config_dimension_reduction
from preprocessing.encoders import ENCODERS
from preprocessing.units_to_audio import save_units_audio
from preprocessing.units_to_phonemes import save_units_phonemes
from preprocessing.voronoi import save_2d_area


def main():
    """
    Executes the main program logic.
    Parses command line arguments, configures the dimension reduction, generates and saves 2D areas, generates and saves
    units and phonemes data, and generates and saves audio files for units.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dense_models",
        type=str,
        required=True,
        help="Pre-trained dense models to use.",
        action="append",
    )
    parser.add_argument(
        "--vocab", type=int, required=True, help="Vocabulary size to use.", action="append"
    )
    parser.add_argument(
        "--skip_dimension_reduction", type=bool, required=False, help="Skip Dimension Reduction")

    parser.add_argument(
        "--only_dimension_reduction", type=bool, required=False, help="Only Dimension Reduction")
    parser.add_argument(
        "--dataset", type=str, default="BOTH", help="Dateset to use", choices=["BOTH", "LJSPEECH", "LIBRISPEECH"])
    args = parser.parse_args()

    dense_models = args.dense_models
    vocab_sizes = args.vocab
    if args.dataset == "BOTH":
        datasets = ["LJSPEECH", "LIBRISPEECH"]
    else:
        datasets = [args.dataset]
    for dense_model in dense_models:
        print(f"Starting {dense_model}")
        if not args.skip_dimension_reduction:
            config_dimension_reduction(dense_model, vocab_sizes)
        if args.only_dimension_reduction:
            continue
        assert dense_model in ENCODERS, "Dense model encoder is missing."
        for vocab_size in vocab_sizes:
            print(f"Starting {vocab_size}")
            save_2d_area(dense_model, vocab_size)
            save_units_phonemes(dense_model, vocab_size)
            for dataset in datasets:
                print(f"Starting {dataset}")
                save_units_audio(dense_model, vocab_size, dataset)


if __name__ == "__main__":
    main()

