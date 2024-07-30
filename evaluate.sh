#!/bin/bash

# Default values
MODEL_FAMILY=""
MODEL_PATH=""
METHOD=""
OUTPUT_DIR="experiments/08-finalized-xai"
CUDA_VISIBLE_DEVICES=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model-family)
        MODEL_FAMILY="$2"
        shift # past argument
        shift # past value
        ;;
        --model-path)
        MODEL_PATH="$2"
        shift # past argument
        shift # past value
        ;;
        --method)
        METHOD="$2"
        shift # past argument
        shift # past value
        ;;
        --cuda-visible-devices)
        CUDA_VISIBLE_DEVICES="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Check if required arguments are provided
if [ -z "$MODEL_FAMILY" ] || [ -z "$MODEL_PATH" ] || [ -z "$METHOD" ]; then
    echo "Usage: $0 --model-family <family> --model-path <path> --method <method> [--cuda-visible-devices <devices>]"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set the output log file name
OUTPUT_LOG="${OUTPUT_DIR}/${MODEL_FAMILY}_${METHOD}.log"

# Print CUDA_VISIBLE_DEVICES to the log
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" > "$OUTPUT_LOG"

# Run the Python command with nohup and add a comment for identification
nohup env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES LOG_LEVEL=INFO HF_HOME=/srv/scratch2/dbielik/.cache/huggingface \
python -W ignore -m app.main \
  --model-family="$MODEL_FAMILY" \
  --model-path="$MODEL_PATH" \
  --data-source-target=local \
  --input-path=./experiments/data/zo_up_test.csv \
  --output-path=./heatmap_cli.pdf \
  --method="$METHOD" \
  --tfidf-corpus-path=experiments/data/osdg.csv \
  -o "$OUTPUT_DIR/evals" \
  evaluate \
  >> "$OUTPUT_LOG" 2>&1 & # Add a comment for process identification
echo $! # eval_${MODEL_FAMILY}_${METHOD}

# Get the process ID
PROC_ID=$!

echo "Evaluation started with process ID: $PROC_ID"
echo "Process comment: eval_${MODEL_FAMILY}_${METHOD}"
echo "Output is being written to $OUTPUT_LOG"
echo "Waiting for log file to be created..."

# Wait for the log file to be created (timeout after 5 seconds)
timeout=5
while [ ! -f "$OUTPUT_LOG" ] && [ $timeout -gt 0 ]; do
    sleep 1
    ((timeout--))
done

if [ -f "$OUTPUT_LOG" ]; then
    echo "Log file created. Streaming logs (press Ctrl+C to stop watching, the process will continue running):"
    echo "--------------------"
    # Stream the log file in real-time
    tail -f "$OUTPUT_LOG"
else
    echo "Timeout: Log file was not created within 5 seconds."
    echo "The process is still running with PID $PROC_ID"
    echo "You can check the log file later using: tail -f $OUTPUT_LOG"
fi