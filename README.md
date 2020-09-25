# Cloud and Cloud Shadow Mask v2 (Sentinel-2 MSI)

## Background

Sentinel 2 imagery provides spatially detailed (10 metre), frequently acquired (5 day revisit)   Remote sensing images play more and more important roles in a lot of applications. Such applications and other remote sensing activities depend on noise free remote sensing data. Beside sensor abnormality, clouds and clouds shadow pose significant challenges to obtaining desired results. Detecting cloud and cloud shadow pixels and removing them from inputs for remote sensing data is essential for all remote sensing data modelling algorithms.

## What this product offers

This product is a time series cloud and cloud shadow detection algorithm for Sentinel-2 surface reflectance data.It models time series of surface reflectance derived indices and calculates time series abnormality coefficients for pixels in the time series. It does not rely on predefined training data to generate complex models with many rule sets, which often work well for data similar to the training data while returning poor results for data contrasting to the training data. Instead, it identifies cloud and cloud shadows by detecting local abnormalities in temporal and spatial contexts from abnormality coefficients.

## Output format
 
This product uses time series analysis to identify clouds and cloud shadows so that they can be excluded from automated analysis.  The time series analysis algorithm classifies a Sentinel-2 pixel into one of four distinctive categories:
No observation ---> 0
Clear ---> 1
Cloud ---> 2
Cloud shadow ---> 3
 

## Applications

The cloud and cloud shadow mask for Sentinel-2 data can be applied to filtered out noisy data in any application using Sentinel-2 data as inputs. A remote sensing application loads Sentinel-2 data in conjunction with the Sentinel-2 cloud and cloud masks. Then any data which is not in the clear category can be identified and excluded from the inputs for the application. Some applications are:

Urban extent classification and change detection
Sentinel-2 geomedian data 



