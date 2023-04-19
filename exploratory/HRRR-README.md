Downloading data from the HRRR.

We use the Herbie library to access HRRR data.
https://herbie.readthedocs.io/en/latest/index.html

First install Herbie using conda:
>>conda install -c conda-forge herbie-data

Check the settings especially for file download location:

Some default settings are set in the config.toml file. This file is automatically created the first time you import a Herbie module and is located at
    ${HOME}/.config/herbie/config.toml
You'll see a prompt to edit this file when you first import herbia via
>> from herbie import Herbie

The main thing you'll want to edit in this file is the directory in which downloaded files will be stored.
