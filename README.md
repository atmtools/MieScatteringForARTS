# MieScatteringForARTS

Python scripts and modules for calculating the full spectrum scattering properties
for liquid and ice particles using the Mie theory.
This package was used to create the full spectrum scattering properties within
arts-xml-data package version >=2.6.7 [arts-xml-data](https://arts.mi.uni-hamburg.de/svn/rt/arts-xml-data/branches/arts-xml-data-2.6/).

It uses the miepython package to calculate the Mie scattering properties.
miepython:
    [miepython GitHub](https://github.com/scottprahl/miepython/)

## Requirements

- pyarts >=2.6.6
- miepython >=2.5.4

## Usage

Use the **generate_miescattering_ice.py** and **generate_miescattering_water.py** scripts to generate the full spectrum scattering properties for ice and liquid particles.
