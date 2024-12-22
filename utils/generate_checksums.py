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


if __name__ == "__main__":

    parser = ap.ArgumentParser(description="Generate checksums for HDF5 files in a directory.")
    parser.add_argument('-i', '--input_directory', required=True, help='Directory containing HDF5 files')
    parser.add_argument('-o', '--output_csv', required=True, help='Output CSV file to save checksums')
    args = parser.parse_args()

    generate_checksums(args.input_directory, args.output_csv)
