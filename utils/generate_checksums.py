import os
import hashlib
import pandas as pd
import argparse

def compute_checksum(file_path):
    """Compute the SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def generate_checksums(directory, output_csv):
    """Generate checksums for all HDF5 files in a directory and save to a CSV."""
    checksums = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                checksum = compute_checksum(file_path)
                checksums.append({'file': file, 'checksum': checksum})

    df = pd.DataFrame(checksums)
    df.to_csv(output_csv, index=False)
    print(f"Checksums saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate checksums for HDF5 files in a directory.")
    parser.add_argument('-d', '--directory', required=True, help='Directory containing HDF5 files')
    parser.add_argument('-o', '--output_csv', required=True, help='Output CSV file to save checksums')
    args = parser.parse_args()

    generate_checksums(args.directory, args.output_csv)
