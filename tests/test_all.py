import unittest
import os
import numpy as np
import pandas as pd
from utils.tar_compressor import simple_tarcompress, dcm_tarcompress, csv_tarcompress
from utils.preprocess_mri import CMRI_PreProcessor
from utils.build_dataset import extract_series

class TestTarCompressor(unittest.TestCase):

    def setUp(self):
        self.root_dir = 'test_root'
        self.output_dir = 'test_output'
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        for directory in [self.root_dir, self.output_dir]:
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(directory)

    def test_simple_tarcompress(self):
        filename = 'test_folder'
        os.makedirs(os.path.join(self.root_dir, filename), exist_ok=True)
        simple_tarcompress(self.root_dir, filename, self.output_dir)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, filename + '.tgz')))

    def test_dcm_tarcompress(self):
        filename = 'test_dcm_folder'
        os.makedirs(os.path.join(self.root_dir, filename), exist_ok=True)
        dcm_tarcompress(self.root_dir, filename, self.output_dir)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, filename + '.tgz')))

    def test_csv_tarcompress(self):
        filename = 'test_csv_folder'
        os.makedirs(os.path.join(self.root_dir, filename), exist_ok=True)
        csv_reference = 'test_csv.csv'
        with open(csv_reference, 'w') as f:
            f.write('accession,anon_mrn,anon_accession\n')
        csv_tarcompress(self.root_dir, filename, self.output_dir, csv_reference)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, filename + '.tgz')))
        os.remove(csv_reference)

class TestCMRI_PreProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = CMRI_PreProcessor(root_dir='test_root', output_dir='test_output', framesize=256, institution_prefix='test', channels=3, compression='gzip')

    def test_dcm_to_array(self):
        input_file = 'test.dcm'
        # Create a mock DICOM file or use a library to generate one
        array, series, frame_loc, accession, mrn, unique_frame_index = self.processor.dcm_to_array(input_file)
        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.shape[0], self.processor.channels)

    def test_collate_arrays(self):
        dcm_subfolder = 'test_dcm_subfolder'
        # Create mock DICOM files in the subfolder
        collated_array, series, slice_frames, total_images, mrn, accession = self.processor.collate_arrays(dcm_subfolder)
        self.assertIsInstance(collated_array, np.ndarray)

    def test_array_to_h5(self):
        collated_array = np.random.rand(3, 10, 256, 256)
        series = 'test_series'
        slice_indices = [0, 5]
        total_images = 10
        mrn = 'test_mrn'
        accession = 'test_accession'
        self.processor.array_to_h5(collated_array, series, slice_indices, total_images, mrn, accession)
        self.assertTrue(os.path.exists(f'test_output/test_{mrn}/{accession}.h5'))

class TestBuildDataset(unittest.TestCase):

    def setUp(self):
        self.root_dir = 'test_root'
        self.output_dir = 'test_output'
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        for directory in [self.root_dir, self.output_dir]:
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(directory)

    def test_extract_series(self):
        folder_name = 'test_folder'
        os.makedirs(os.path.join(self.root_dir, folder_name), exist_ok=True)
        series = 'test_series'
        series_map = pd.DataFrame({'series_description': ['desc1'], 'cleaned_series_names': ['test_series']})
        extract_series(self.root_dir, folder_name, self.output_dir, series, series_map, 'gzip')
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, folder_name)))

if __name__ == '__main__':
    unittest.main()
