# SICE_classes

# how to label:

features = ["dry_snow", "melted_snow", "flooded_snow","red_snow","bright_ice", "dark_ice"]

## dark_ice
use band 02, may be stretch range in QGIS, avoid bright patches, see examples of ok classes in [2019 example](https://github.com/jasonebox/SICE_classes/blob/main/Figs/2019-08-02_classes_SVM5bands_02_04_06_08_21.png)

## flooded_snow
use band 08 to band 02 ratio output by ./src/from 

## dry_snow
use band 21, straight forward to see brighest snow covered areas at the uppermost ice sheet elevations

## melted_snow
use band 21, straight forward to see darker snow covered areas as compared to the uppermost ice sheet elevations where the band 21 reflectance is high
