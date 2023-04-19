SynopticPy is a python library that provides direct access to Synoptic met
database.

To use SynopticPy to access the Synoptic Data database you need to create
a free account on Synoptic and generate a token.  Then, the token can be used
to configure the SynopticPy library.

Configure SynopticPy with your token

SynopticPy needs to know your token.
The first time you import synoptic.services it will help you setup
your token in its config file.

Open python in a terminal and type the following:

>>import synoptic.services
You will be prompted with instructions for acquiring an API token,
and then it will ask you to input your token. Remember to enter
you API token and not your API key.

What is your Synoptic API token? >>>
The script updates a config file located at ~/.config/SynopticPy/config.toml.
