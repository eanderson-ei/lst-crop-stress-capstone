[![DOI](https://zenodo.org/badge/627146632.svg)](https://zenodo.org/badge/latestdoi/627146632)
# Remotely Sensed Crop Stress Early Indicator

Can satellite remote sensing detect early stage crop stress at high resolution across whole-farm scale?

Hydrosat produces a proprietary "fused" thermal infrared land surface temperature (LST) imagery product using sharpening and interpolation algorithms to produce near daily sub-30 m resolution imagery from a combination of MODIS, Sentinel and Landsat imaging platforms. This project aims to explore the input data and intermediate outputs of the algorithms used to produce the LST product in order to detect errors potentially introduced by data quality issues or the application of the algorithms while also exploring applications within agricultural contexts.

See the blog post [here](https://eriktuck.github.io/lst-crop-stress-capstone/).

# Collaborators and Acknowledgements

- [Erik Anderson](https://github.com/eriktuck)
- [Tyler Cruickshank](https://github.com/tcruicks)
- [Joe McGlinchy](https://github.com/joemcglinchy)
We thank Joe McGlinchy of Hydrosat for providing project guidance and data access.

# Environment
To reproduce this workflow, use the provided `environment.yml` file to create a `conda` environment (you should already have Python and `conda` installed on your system).

After cloning this directory to your system, use the following `bash` commands to create and activate the environment:

```bash
conda env create -f environment.yml
conda activate ea-lst
```

You must have Jupyter Notebook installed on your system to run this analysis. For installation instructions, consult the documentation [here](https://jupyter.org/).

Once Jupyter Notebook is installed, install the iPython kernel in the environment. With the environment activated, in bash use the command:

```bash
ipython kernel install --name ea-lst --user
```

The `--name` flag specifies the name the kernel will appear under when selecting it to use with Jupyter Notebooks. It's best to name it the same as the environment name.

# Data

For access to Hydrosat's proprietary products, you must receive credentials using the instructions described at the [Hydrosat Fusion Hub](https://hydrosat.github.io/fusion-hub-docs/intro.html). Add your credentials to the `secrets/` folder in a file called `creds.json` with the format:

```json
{
    "username":"",
    "password":""
}
```
See the [Hydrosat Fusion Hub Documentation](https://hydrosat.github.io/fusion-hub-docs/intro.html) for additional guidance.

Meteorological data are provided by [Ameriflux](https://ameriflux.lbl.gov/) and included in the project's `data/Ameriflux` directory.

# Workflows
Follow the instructions below to reproduce the workflows in the notebook `blog.ipynb` and convert the Jupyter Notebook to an HTML report.

## Run the analysis
To reproduce the analysis in `blog.ipynb`, first follow the instructions to set up the environment and access data, described above. 

Open the file `blog.ipynb` in a Jupyter Notebook and run all cells to repeat the analysis. To change the crop locations for the analysis of NDVI (`f1`) and the analysis of CATD (`f2`), change the `f1_met_tower` and `f2_met_tower` variables (see Configuration below to add your own locations). For information on how to launch a Jupyter Notebook, consult the documentation [here](https://jupyter.org/).

## Configuration

This project uses a configuration file to facilitate repeat workflows on new Ameriflux meteorological towers. To repeat the analysis on a new met tower, add the necessary information to the file `config.yml` by copy/pasting an existing configuration section and updating with the necessary parameters. You must include 
  - the longitude, latitude for the meteorological tower center point, 
  - the the longitude, latitude for the ag field center point, 
  - either a bounding box (list of four coordinates) or area of interest (list of four longitude, latitude pairs), and 
  - the parameters for reading the Ameriflux data. 
  
Note that Hydrosat does not currently include coverage for the entire CONUS area and so confirm the coverage of data for any new meteorological tower location.

## Convert to HTML report
To convert the file `blog.ipynb` to an HTML report, you must have the library `nbconvert` installed (note that this library is not included in this project's environment; read the [documentation](https://nbconvert.readthedocs.io/en/latest/) for installation instructions). 

Use the following bash command to convert the notebook and post to gh-pages:

```bash
jupyter nbconvert --to html_embed --no-input blog.ipynb
mv blog.html index.html
git commit -am "update blog"
git pull
git push origin main
git checkout gh-pages
git checkout main index.html
git commit -am "updating gh-pages"
git push origin gh-pages
git checkout main
```
