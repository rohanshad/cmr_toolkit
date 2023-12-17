# Cardiac MRI Toolkit (alpha)

[![arXiv](https://img.shields.io/badge/arXiv-2312.00357-b31b1b.svg)](https://arxiv.org/abs/2312.00357)

A set of preprocessing utilities for cardiac MRI studies to make deep learning projects less painful. Supplementary repository for work performed in the paper: [A Generalizable Deep Learning System for Cardiac MRI](https://arxiv.org/abs/2312.00357)


![summary_usage](https://github.com/rohanshad/cmr_toolkit/blob/5b6055dc059aeccb50bd78d106be4b88eccabe31/media/summary_usage.png)

1. Scripts to convert cardiac-mri dicom files to hdf5 filestores
2. Methods to standardize metadata (view, frames, slice frame index)
3. Utilties to plot / tar compress dicoms and extract metadata to double check if everything works

### Basic Workflow

```preprocess_mri.py``` reads in a folder full of dicom studies and converts them to hdf5. Data from each patient (MRN) is stored at a top level folder. Each unique scan (accession #) gets its own .hdf5 file. For each series / view from a 'SeriesDescription' we store the cine sequence as a 4d array ```[c, f, h, w]``` as a hdf5 dataset. To each hdf5 dataset we append attributes (total # of frames and if relevant, the slice index - 'ie. location of the short axis slice')

Scripts are all built to scale almost linearly to about 64 CPU cores for speed. Can process up to 100k MRI scans in less than 3 hours on 64 CPU cores.

```
upenn_Zx3da3244
├── Zx3da3581.h5
├── Gf3lv2173.h5
	├── 4CH_FIESTA_BH 		{data: 4d array} {attr: fps, total images}
	├── SAX_FIESTA_BH_1		{data: 4d array} {attr: fps, total images, slice frame index}
	├── SAX_FIESTA_BH_2		{data: 4d array} {attr: fps, total images, slice frame index}
	├── STACK_LV3CH_FIESTA_BH	{data: 4d array} {attr: fps, total images, slice frame index}
	
```

```build_dataset.py``` builds a copy of the hdf5 filestore above with standardized names for different views. The names are pulled from a separate csv file ```series_descriptions_master.csv```. This contains standard names for a large list of ```SeriesDescription``` tags seen in US based DICOM studies from Phillips, GE, and Siemens scanners. The filestores have the following structure thereafter, where the name of the datasets within the hdf5 files are now standardized series names (4CH, 2CH, 3CH, SAX). 

```
upenn_Zx3da3244
├── Zx3da3581.h5
├── Gf3lv2173.h5
		├── 4CH 	{data: 4d array} {attr: fps, total images}
		├── SAX		{data: 4d array} {attr: fps, total images, slice frame index}
		├── 3CH		{data: 4d array} {attr: fps, total images}
```

Chances are SeriesDescription tags from your dataset are not available in the ```series_descriptions_master.csv``` file I have provided. To build your own one specific to your institution, run ```dicom_metadata.py``` to process all your dicoms. You can then add the 'cleaned_series_name' manually thereafter. The 'counts' column is a good way of verifying you're capturing the most commonly occuring studies in your dataset. 

### Notes

There is a local file called ```local_config.yaml``` that I reference in some of the scripts. This is nothing but a simple text file that contains some hardcoded variables such as temporary folders, google / aws bucket names etc. This is just a unique configuration file for each laptop / server as different environments will have their own filesystems. Example below:

```
tmp_dir: 'tmp/dir/on_some_server'
bucket_name: unique_googlecloud_bucket
some_other_variable: 'local/folder/tree'
```

### Future Directions

Replace the mastersheet based approach with deep learning view recognition combined with simple heuristics for accurate view, sequence, and phase detection. Initial experiments show promising results. Codebase will be extended to incorporate T1 / T2 / LGE scans in the future. Will also incorporate CI / CD pipelines to keep new commits from breaking reproducibility in pre-processing existing datasets.
