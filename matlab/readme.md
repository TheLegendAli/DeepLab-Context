### MATLAB scripts

The MATLAB scripts are mainly used to evaluate the segmentation results, and used to read/write data in the stage of post-processing by DenseCRF.

## my_script folder

The folder my_script stores the MATLAB scripts used in the experiments.

Some useful scripts:

1. SetupEnv.m
    * Set up the environmental variables.  
    * You can also find the values of DenseCRF parameters found by our cross-validation in this script.
2. EvalSegResults.m
    * Evaluate the segmentation results.  
3. GetDenseCRFResult.m (saved under the folder densecrf/my_script)
    * Transform the CRF computed results from the format of __bin__ to __png__ format.
4. DownSampleFeature.m (saved under the folder densecrf/my_script)
    * Downsample the DCNN computed features for cross-validation.
