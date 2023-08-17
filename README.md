# SICE_classes

# how to label:

features = ["dry_snow", "melted_snow", "flooded_snow","red_snow","bright_ice", "dark_ice"]

## dark_ice
use band 02, mid way up western ablation area, see [Shimada et al 2016](https://www.frontiersin.org/articles/10.3389/feart.2016.00043/full)

try to include dark ice from far northwest

may be stretch range in QGIS, avoid bright patches, see examples of ok classes in [2019 example](https://github.com/jasonebox/SICE_classes/blob/main/Figs/2019-08-02_classes_SVM5bands_02_04_06_08_21.png)

## bright_ice
use band 02, may be stretch range in QGIS, is usually lower in elevation than the dark ice area and can be above the dark ice area

see examples of ok classes in [2019 example](https://github.com/jasonebox/SICE_classes/blob/main/Figs/2019-08-02_classes_SVM5bands_02_04_06_08_21.png)

## flooded_snow
use band 08 to band 02 ratio output by ./src/from/SICE_classes_SVC_v2.py

load the tif in QGIS, adjust contrast to from 1 to 1.6, then see the bright areas.

The profile below is NE ice sheet, across a blue area easily visible in the 'true color' RGB image also below
Areas above ~1.2 in the ratio are to be classified.
<img width="400" alt="image" src="https://github.com/jasonebox/SICE_classes/assets/32133350/05e4cb10-ea01-4d77-bf12-7c55c0b3104e">
<img width="200" alt="image" src="https://github.com/jasonebox/SICE_classes/assets/32133350/a349e61c-7f7d-4d06-ac5e-59e39d6e6f67">


Be more sure by checking using the RGB image that ./src/SICE_classes_SVC_v2.py outputs to the plot window. Note there is an issue to develop to make the RGB image output to a geotiff!

## dry_snow
use band 21, straight forward to see brighest snow covered areas at the uppermost ice sheet elevations

## melted_snow
use band 21, straight forward to see darker snow covered areas as compared to the uppermost ice sheet elevations where the band 21 reflectance is high

## red_snow

use band 08 to band 06 normalised difference ratio output by ./src/SICE_classes_SVC_v2.py outputs when user chooses do_generate_rasters=1

red snow areas are where the NDIX is above ~0.0. The magnitude is small, so it may be questionable how accurate this approach is. Let's check out the results.

In this example, there is a small red? area on the Sukkertoppen ice cap SW Greenland. Notice in the RGB image, also provided, the ice cap is not blue and there is blue flooded snow to the east.

<img width="500" alt="image" src="https://github.com/jasonebox/SICE_classes/assets/32133350/4e65808c-13ca-44c0-966b-8b6df7c8bc37">
<img width="500" alt="image" src="https://github.com/jasonebox/SICE_classes/assets/32133350/a9281fab-5d88-4544-959c-3454ba781d41">



