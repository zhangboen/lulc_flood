# module load UDUNITS; module load GDAL/3.7.1-foss-2023a-spatialite; module load R/4.3.2-gfbf-2023a
# Rscript --no-restore --no-save calc_GRIT_catch_ave_MSWEP_1.r prName, shpName
library(conflicted)
library(sf)
library(terra)
library(exactextractr)
library(tidyverse)
library(tidyterra)
library(arrow)

func <- function(prName, shpName, method, outName) {
    if(file.exists(outName)) return(NA)

    data <- terra::rast(prName)

    poly0 <- st_read(shpName)
    poly0 <- st_transform(poly0, crs = 4326)

    ## Extract data
    if (method != 'ratio') {
        res <- exact_extract(data, poly0, fun = method, default_value = 0)
    } else {
        res1 <- exact_extract(data, poly0, fun = 'sum', default_value = 0)
        res2 <- exact_extract(data, poly0, fun = 'count')
        res <- res1 / res2
    }

    df <- data.frame(t(res))
    colnames(df) <- poly0$ohdb_id

    # save to parquet
    write.csv(df, outName)
}

method <- 'mean'
for (domain in c('NA','EU','SA','AS','AF','SP')) {
    for (year in 1992:2019) {
        prName <- paste0('../../data/GDP_1992_2019/', year, 'GDP.tif')
        shpName <- paste0('../basin_boundary/GRIT_full_catchment_', domain, '_EPSG8857_simplify_final_125km2_subset.gpkg')
        outName <- paste0('../data_gdp/', year, 'GDP_catch_ave_domain_new_', domain, '.csv')
        if (!file.exists(outName)) {
            func(prName, shpName, method, outName)
        }
    }
}


