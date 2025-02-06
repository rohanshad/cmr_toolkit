'''
Generate sha256 checksums for pre-processed hdf5 files 
'''

import os
import hashlib
import pandas as pd
import argparse as ap

def compute_checksum(file_path):
    '''
    Compute the SHA256 checksum of a file.
    '''
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def generate_checksums(input_directory, output_csv):
    '''
    Generate checksums for all HDF5 files in a directory and save to a CSV.
    '''
    checksums = []

    print('------------------------------------')
    print(f'Generating checksums...')
    print('------------------------------------')

    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                checksum = compute_checksum(file_path)
                checksums.append({'file': file, 'checksum': checksum})

    df = pd.DataFrame(checksums)
    df.to_csv(output_csv, index=False)

    print(df)
    print('------------------------------------')
    print(f'Checksums saved to {output_csv}')
    print('------------------------------------')


def compare_checksums(new_csv, comparison_checksum_file):
    '''
    Compares checksums against a known "good" ground truth checksum list
    '''
    print('------------------------------------')
    print(f'Comparing checksums...')
    print('------------------------------------')

    df1 = pd.read_csv(new_csv)
    df2 = pd.read_csv(comparison_checksum_file)

    merged_df = df1.merge(df2, on="file", suffixes=('_file1', '_file2'))
    mismatches = merged_df[merged_df["checksum_file1"] != merged_df["checksum_file2"]]

    # Print mismatches
    if not mismatches.empty:
        print("Mismatched files found:")
        print('------------------------------------')
        print(mismatches)
        print('------------------------------------')
    else:
        print("All files have matching checksums.")
        return(True)



if __name__ == "__main__":

    parser = ap.ArgumentParser(description="Generate checksums for HDF5 files in a directory.")
    parser.add_argument('-f', '--comparison_checksum_file', required=False, help='csv file that has comparison checksums')
    parser.add_argument('-i', '--input_directory', required=True, help='Directory containing HDF5 files')
    parser.add_argument('-o', '--output_csv', required=True, help='Output CSV file to save checksums')
    args = parser.parse_args()

    generate_checksums(args.input_directory, args.output_csv)
    if args.comparison_checksum_file is not None:
        compare_checksums(args.output_csv, args.comparison_checksum_file)



