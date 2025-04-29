"""
This script converts a JSONL file to a CSV file with specific formatting.

The script reads a JSONL file containing records and converts it into a CSV file
with the following columns: sdg, abstract, id, sdg_desc_short, and sdg_desc_long.

Each record in the JSONL file is expected to have the following fields:
- SDG: The SDG number (integer).
- TITLE: The title of the record (string).
- ABSTRACT: The abstract of the record (string).
- ID: The unique identifier of the record (string).

The TITLE is prepended to the ABSTRACT in the "abstract" column of the CSV.

Example usage:
    python3 json2csv_zo_up.py input.jsonl output.csv

Replace `input.jsonl` with the path to your JSONL file and `output.csv` with the desired path for the output CSV file.

Docs: 
python3 json2csv_zo_up.py zo_up_sdg0.jsonl zo_up_sdg0.csv

python3 json2csv_zo_up.py zo_up_sdg17.jsonl zo_up_sdg17.csv
"""

import json
import csv
import argparse

# Define the CSV headers
csv_headers = ['sdg', 'abstract', 'id', 'sdg_desc_short', 'sdg_desc_long']

# SDG descriptions mapping
sdg_descriptions = {
    0: ("No SDG", "Dummy class for no SDG"),
    1: ("No Poverty", "Aims to end poverty in all its forms everywhere."),
    2: ("Zero Hunger", "Aims to end hunger, achieve food security and improved nutrition, and promote sustainable agriculture."),
    3: ("Good Health and Well-being", "Aims to ensure healthy lives and promote well-being for all at all ages."),
    4: ("Quality Education", "Aims to ensure inclusive and equitable quality education and promote lifelong learning opportunities for all."),
    5: ("Gender Equality", "Aims to achieve gender equality and empower all women and girls."),
    6: ("Clean Water and Sanitation", "Aims to ensure availability and sustainable management of water and sanitation for all."),
    7: ("Affordable and Clean Energy", "Aims to ensure access to affordable, reliable, sustainable, and clean energy for all."),
    8: ("Decent Work and Economic Growth", "Aims to promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all."),
    9: ("Industry, Innovation and Infrastructure", "Aims to build resilient infrastructure, promote inclusive and sustainable industrialization, and foster innovation."),
    10: ("Reduced Inequalities", "Aims to reduce inequality within and among countries."),
    11: ("Sustainable Cities and Communities", "Aims to make cities and human settlements inclusive, safe, resilient, and sustainable."),
    12: ("Responsible Consumption and Production", "Aims to ensure sustainable consumption and production patterns."),
    13: ("Climate Action", "Aims to take urgent action to combat climate change and its impacts."),
    14: ("Life Below Water", "Aims to conserve and sustainably use the oceans, seas, and marine resources for sustainable development."),
    15: ("Life on Land", "Aims to protect, restore and promote sustainable use of terrestrial ecosystems, sustainably manage forests, combat desertification, and halt and reverse land degradation and halt biodiversity loss."),
    16: ("Peace and Justice Strong Institutions", "Aims to promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable and inclusive institutions at all levels."),
    17: ("Partnerships for the Goals", "Aims to strengthen the means of implementation and revitalize the global partnership for sustainable development.")
}


def convert_jsonl_to_csv(input_file, output_file):
    """Convert a JSONL file to a CSV file with the specified format."""
    converted_count = 0  # Initialize a counter for converted records

    with open(input_file, 'r') as jsonl_file, open(output_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        #csv_writer.writerow(csv_headers)  # Write the header row

        for line in jsonl_file:
            record = json.loads(line.strip())
            ID = record.get('ID', '')
            if not ID:
                print("No ID found in the record. Skipping...")
                continue
            else:
                print(f"Processing record with ID: {ID}")
            converted_count += 1  # Increment the counter for each valid record
            sdg = record.get('SDG', 0)
            sdg_desc_short, sdg_desc_long = sdg_descriptions.get(sdg, ("", ""))
            title = record.get('TITLE', '').replace('\n', ' ').replace('\r', ' ')
            abstract = record.get('ABSTRACT', '').replace('\n', ' ').replace('\r', ' ')
            combined_abstract = f"{title}: {abstract}" if title and abstract else abstract
            csv_writer.writerow([
                sdg,
                combined_abstract,
                record.get('ID', ''),
                sdg_desc_short,
                sdg_desc_long
            ])

    print(f"Conversion completed. CSV file saved to {output_file}")
    print(f"Total records converted: {converted_count}")  # Report the total count

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert a JSONL file to a CSV file.")
    parser.add_argument("input_file", help="Path to the input JSONL file.")
    parser.add_argument("output_file", help="Path to the output CSV file.")

    # Parse arguments
    args = parser.parse_args()

    # Run the conversion
    convert_jsonl_to_csv(args.input_file, args.output_file)
