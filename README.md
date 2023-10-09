# SICE_classes

# how to label:

features = ["dry_snow", "melted_snow", "flooded_snow", "red_snow", "bright_ice", "dark_ice"]

The labling work is by generating shapefiles using QGIS named as above and stored in ./ROIs/Greenland in the appropriate dates. It is necessary to run ./src/SICE_classes_SVC_v2.py to generate the red to blue ratio image for flooded_snow and the NDIX image for red to green for red_snow. When running that top cell, select the correct date and see the output appear in ./SICE_rasters/ ... Open all rasters in QGIS to visualise and label as in instructions below. Just ask questions if any doubt and don't worry a lot about drawing the label polygons perfectly, more important is getting relatively large samples, i.e. >100 pixels... 1000 pixels should be enough but may be hard to find for some dates for the flooded snow and red_snow classes in particular. Talk with Rasmus about what to do when there are NO valid pixels for a given date.

30 Sep, 2023 screen records of Jason labeling images, editing some of what Jasper had done, expanding labeled areas

[video 1 classify red snow](https://youtu.be/YPBh6Y9hZAk)

[video 2 classify flooded and red snow](https://youtu.be/kDgHtQccaUo)

[video 3 classify all](https://youtu.be/_Z-4M6EW0dA)


## dark_ice
use band 02, mid way up western ablation area, see [Shimada et al 2016](https://www.frontiersin.org/articles/10.3389/feart.2016.00043/full)

try to include far northwest

may be stretch range in QGIS, avoid bright patches, see examples of ok classes in [2019 example](https://github.com/jasonebox/SICE_classes/blob/main/Figs/2019-08-02_classes_SVM5bands_02_04_06_08_21.png)

## bright_ice
use band 02, may be stretch range in QGIS, is usually lower in elevation than the dark ice area ~~and can be above the dark ice area~~

see examples of ok classes in [2019 example](https://github.com/jasonebox/SICE_classes/blob/main/Figs/2019-08-02_classes_SVM5bands_02_04_06_08_21.png)

## flooded_snow
later using RGB image instead of (also useful) band 08 to band 02 ratio output by ./src/from/SICE_classes_SVC_v2.py

When using band ratio, load the tif in QGIS, adjust contrast to from 1 to 1.6 to see the bright areas.

The profile below is NE ice sheet, across a blue area easily visible in the 'true color' RGB image also below
Areas above ~1.2 in the ratio are to be classified.
<img width="400" alt="image" src="https://github.com/jasonebox/SICE_classes/assets/32133350/05e4cb10-ea01-4d77-bf12-7c55c0b3104e">
<img width="200" alt="image" src="https://github.com/jasonebox/SICE_classes/assets/32133350/a349e61c-7f7d-4d06-ac5e-59e39d6e6f67">


### examples

2021-07-30 SSW Greenland ![image](https://github.com/jasonebox/SICE_classes/assets/32133350/c6307a7a-67a7-41dc-aa4a-4b6595d5679d)

2021-07-30 NNW Greenland ![image](https://github.com/jasonebox/SICE_classes/assets/32133350/3e039e5e-0ce6-4d6d-ab44-17ca28fa25db)

2022-07-31 NNW Greenland Sentinel 2 ![image](https://github.com/jasonebox/SICE_classes/assets/32133350/001e93ec-2802-4aa9-ba52-29ef9ec207ef)

2022-07-31 NNW Greenland Sentinel 3 ![image](https://github.com/jasonebox/SICE_classes/assets/32133350/cdc2c6f0-a95b-4645-b2ae-81974733e0ac)

## dry_snow
use band 21, straight forward to see brighest snow covered areas at the uppermost ice sheet elevations

## melted_snow
use band 21, straight forward to see darker snow covered areas as compared to the uppermost ice sheet elevations where the band 21 reflectance is high

## red_snow

the RGB image "area on the Sukkertoppen ice cap SW Greenland. Notice in the RGB image, also provided, the ice cap is not blue and there is blue flooded snow to the east."

![image](https://github.com/jasonebox/SICE_classes/assets/32133350/02cbef70-5f98-47f4-af6e-051fab2bab1f)

![image](https://github.com/jasonebox/SICE_classes/assets/32133350/aa091771-0a24-4553-bdb6-2c55888a7ec1)

revision 27/9. Either use where band 08 to band 06 normalised difference ratio is in the range 0.02 to 0.04, or as I (JEB) did using RGB image, peripheral areas are 'golden' and constrast strongly with 'blue' areas to the east... could the golden be either a real algae signal or something about light conditions in relatively small glaciated areas.

### August
use band 08 to band 06 normalised difference ratio output by ./src/SICE_classes_SVC_v2.py outputs when user chooses do_generate_rasters=1

red snow areas are where the NDIX is ~~above ~0.0~~ negative. The magnitude is small, ~-0.05 is a strong signal. Negative makes sense because the idea is that the reflectance of red snow would be higher and the NDXI is (green minus red) / (green + red)

In this example, there is a small red? area on the Sukkertoppen ice cap SW Greenland. Notice in the RGB image, also provided, the ice cap is not blue and there is blue flooded snow to the east.

<img width="500" alt="image" src="https://github.com/jasonebox/SICE_classes/assets/32133350/4e65808c-13ca-44c0-966b-8b6df7c8bc37">
<img width="500" alt="image" src="https://github.com/jasonebox/SICE_classes/assets/32133350/a9281fab-5d88-4544-959c-3454ba781d41">



