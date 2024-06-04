# SFT_preprocessing

# Installation

**1. Clone the SFT_preprocessing repository.**

```
git clone https://github.com/chaneyn/SFT_preprocessing.git
cd SFT_preprocessing
```

**2. Create a conda environment named SFT from the spec-file. Note that the only current spec-file in the repository is for a linux64 machine.** 

```
conda create --name SFT_preprocessing --file spec-file.txt
source activate SFT_preprocessing
pip install git+https://github.com/chaneyn/geospatialtools@dev_nate
pip install mpi4py
```

# How to run

**0. Create JSON file**

The JSON file defines the domain, the location of the datasets, and the binning used in creating the SFTs. A unique JSON metadata file should be created for each SFT database. 

**1. Define domain**

The lat/lon boundaries and resolution defined in the json file are used to create a shapefile define the macroscale polygons (i.e., regular grid cells) over the selected domain. The land cover data is then used to determine grid cells that are only ocean; the final result is a corrected shapefile with only grid cells that have land.

```
mpirun -n 4 python driver.py -f json/test_region.json -s define_domain
```

**2. Preprocess datasets**

Each grid cell is processed and the corresponding data (e.g., elevation) are extracted at the preprocessing resolution (e.g., 3 arcsec). A database is made for each macroscale polygon that contains land over the domain

```
mpirun -n 4 python driver.py -f json/test_region.json -s preprocess_datasets
```

**3. Compute SFTs**

For each grid cell, the preprocessed data is read in and binned using a n-dimensional histogram. The SFTs are then defined from the histogram. Note that the defined binning of the histogram is the same for all grid cells ensuring the SFT definition is the same for all grid cells. The SFT fractions are calculated from the histogram and each fine-scale pixel (i.e., preprocessing resolution) is asigned a SFT and saved within the given macroscale polygons directory as a geotiff titled `sfts.tif`

```
mpirun -n 4 python driver.py -f json/test_region.json -s compute_sfts
```

**3. Finalize NetCDF**

The SFT databases calculated previously per grid cell are read in and used to assemble a domain NetCDF file that contains the fractional coverage of each grid cell which can then be readily Orchidee. 

```
mpirun -n 4 python driver.py -f json/test_region.json -s finalize_netcdf
```

# JSON file parameters

* **output_dir** - Directory where all the data will be processed and saved.
* **domain** - Structure that defines the bounding box (bbox) and macroscale spatial resolution (res; arcdegrees).
* **preprocessing_resolution** - Fine-scale spatial resolution in arcdegrees at which the preprocessing will be performed per macroscale grid cell. 
* **preprocessing_year** - Year for which preprocessing will be run. This will be used to choose the ESA CCI dataset year as well as the LUH2 data. 
* **input_files** - Location of the input_files used in the preprocessing. 
* **histogram_info** - Binning information used to create the histogram and thus the SFTs. 

# Histogram info

The variables and binning information used for each variable can be changed within the JSON file. The only constraint is that pft needs to be one of the variables. 

* Whether the variable is continuous (e.g., elevation) or class (e.g., pft) needs to be defined.
* For continuous data there must be at least 2 bin edges defined per variable; for class data there must be at least 1 bin value defined. 
* The binning can be linear or non-linear; however, the minimum and maximum values that can be found need to be bound by the defined bins if not fine-scale pixels can be excluded. 
* The current following variables can be used for binning: 
 * Elevation (elev; meters)
 * PFTs (pft; 1-15 are the ORCHIDEE PFTs, 16 is permanent snow/ice, and 17 is water)
 * Slope (slope; m/m)
 * Aspect S-N (aspect_sn)
 * Aspect W-E (aspect_ew)
 * Clay (clay; %)
 * Sand (sand; %)
 * Silt (silt; %)
 * South-North (sn)
 * West-East (ew)


# Output NetCDF file

* The `sfcfct` array defines the fractional coverage of each SFT per grid cell. The sum of sfcfct per grid cell should be 1. 
* The bin edges of the SFTs for each chosen continuous variable in the JSON file are included in the NetCDF file (min and max).
* The class value of the SFTs for each chosen class variable in the JSON file are included in the NetCDF file. 
