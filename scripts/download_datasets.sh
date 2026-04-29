#!/bin/bash

set -e

source "$(dirname "$(readlink -f "$0")")/utils.sh"

DOWNLOAD_DIR="$SCRATCH/data"

DATASETS=(
    "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"
    "http://ann-benchmarks.com/gist-960-euclidean.hdf5"
    "http://ann-benchmarks.com/mnist-784-euclidean.hdf5"
    "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
)

for dataset in "${DATASETS[@]}"; do
    color_echo "yellow" "Downloading $dataset"
    filename=$(basename "$dataset")
    wget --continue --output-document "$DOWNLOAD_DIR/$filename" "$dataset"
    color_echo "green" "Downloaded $filename to $DOWNLOAD_DIR/$filename"
done

color_echo "green" "Downloaded ${#DATASETS[@]} datasets!"
