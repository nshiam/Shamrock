# Shamrock

## Description

Shamrock is a Photometric Redshift Estimator for Type Ia Supernova Light-Curves designed for the Dark Energy Survey and later for the Rubin Legacy Survey of Space and Time. 


## Installation
Shamrock can be installed by following the next steps. 
### 1. Clone the repository

Open the terminal at the directory that you want it installed and run:

```bash
git clone https://github.com/nshiam/Shamrock.git
```

Navigate into the project directory:

```bash
cd Shamrock
```

### 2. Set up the environment

Create the conda environment using the provided environment file.
(Note: Shamrock is still in developer mode, so this step might take a bit longer)


```bash
conda env create -f shamrock_environment.yml
```

This command will make an environment called shamrock. Activate the environment:

```bash
conda activate shamrock
```


### 3. Verify installation & Example Usage

To verify the installation and that everything is working as expected, I would suggest to test the code on some example light-curves.
The directory DES_EXAMPLE_LC contains some HEAD and PHOT files from DES, and I will show how to run it on these DES light curves. 
In the terminal:

```bash
# Ensure the Conda environment is activated
conda activate shamrock

# Check all the available inputs
python main.py -h

# Run the script on the DES_EXAMPLE_LC for the first 20 SNIDs, using 2 cores.
python main.py --FILE_PATH ./DES_EXAMPLE_LC/ --ZERO_POINT 27.5 --NCORES 2 --START 0 --END 20

```
Once this is done, go to ./DES_EXAMPLE_LC/ where the csv file with the photo-z estimates is located.


## Contact
Shamrock is still under development. For questions or further information, please contact:
N.Shiamtanis | N.Shiamtanis@soton.ac.uk
