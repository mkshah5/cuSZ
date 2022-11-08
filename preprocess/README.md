# Using Threshold and Grouping with CUDA
An implementation of threshold and grouping code for QTensor data. Note that cuSZ and cuSZx are not included, this code is only pre-/post-processing steps.

### Dependencies
- NVCOMP v2.2
- CUDA 11.0+

### Tested GPUs
- Pascal Titan Xp
- A100

### Compiling the Code
Use the following commmands (assuming NVCOMP added as library):
- NVCOMP-based: `nvcc threshold_nvcomp.cu -o threshold_nv -lnvcomp`
- Two-level bitmap compression: `nvcc threshold_twolevelbitmap.cu -o threshold_2lb`

### Running the code
Flags:
- `-z`: Do compression
- `-d <compressed file path>`: Do decompression
- `-T <absolute threshold>`: Required for any thresholding
- `-i <original file path>`: Designate original file (for decompression, specifies output file name)
- `-g <number of significant values>`: Do grouping, number of significant values required for decompression, otherwise use "0"
- `-L <data length>`: Specify number of data points
- `-F`: cast the output to single precision float (reads double precision input)
- `-N`: Run lossless compression on bitmap for either version
- `-R <r2r threshold>`: Specify relative-to-value-range threshold


### Examples
- Run LZ4-based compression, threshold, grouping pre-processing. Use R2R threshold of 0.005: `./threshold_nv -z -N -T 0.01 -R 0.005 -i real_tensor_d26.bin -L 67108864 -F -g 0`
- Run LZ4-based decompression post-processing with grouped values: `./threshold_nv -d real_tensor_d26.bin.threshold -i real_tensor_d26.bin -L 67108864 -g <number of significant values>`