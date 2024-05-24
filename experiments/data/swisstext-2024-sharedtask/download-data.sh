#!/bin/sh

set -e

DIR=$(dirname $0)
DOWNLOAD_FILE_TRAIN="task1-train.jsonl"
DOWNLOAD_FILE_TEST="task1-test.jsonl"

main() {
    echo "Downloading SwissText 2024 Shared Task data..."
    # Download the data for the SwissText 2024 Shared Task
    # from https://github.com/ZurichNLP/sdg_swisstext_2024_sharedtask/tree/main
    wget -q --show-progress https://raw.githubusercontent.com/ZurichNLP/sdg_swisstext_2024_sharedtask/main/data/task1_train.jsonl -O "$DIR/$DOWNLOAD_FILE_TRAIN"
    wget -q --show-progress https://raw.githubusercontent.com/ZurichNLP/sdg_swisstext_2024_sharedtask/main/data/task1_test-covered.jsonl -O "$DIR/$DOWNLOAD_FILE_TEST"
}

# Run the main function
main