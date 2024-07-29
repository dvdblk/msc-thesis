import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import argparse
import os


def main(input_file, output_dir):
    # Load the dataset
    dataset = load_dataset("csv", data_files=input_file)

    # Convert to pandas DataFrame
    df = dataset["train"].to_pandas()

    # Convert SDG to 0-indexed int for stratification
    df["SDG_0indexed"] = df["sdg"].astype(int) - 1

    # Split the dataset
    train_data, test_data = train_test_split(
        df, test_size=0.3, stratify=df["SDG_0indexed"], random_state=1337
    )

    # Remove the temporary column used for stratification
    train_data = train_data.drop("SDG_0indexed", axis=1)
    test_data = test_data.drop("SDG_0indexed", axis=1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the base name of the input file without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Create output file names
    train_output = os.path.join(output_dir, f"{base_name}_train.csv")
    test_output = os.path.join(output_dir, f"{base_name}_test.csv")

    # Save to CSV
    train_data.to_csv(train_output, index=False)
    test_data.to_csv(test_output, index=False)

    print(f"Dataset splits have been saved to {train_output} and {test_output}")
    print(f"Columns in the saved files: {', '.join(train_data.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into train and test sets"
    )
    parser.add_argument(
        "--input-file", required=True, help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save the output CSV files"
    )

    args = parser.parse_args()

    main(args.input_file, args.output_dir)
