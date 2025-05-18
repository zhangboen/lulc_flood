# module load UDUNITS; module load GDAL; module load R/4.3.2-gfbf-2023a
# Rscript --no-restore --no-save calc_GRIT_catch_ave_MSWEP_1.r prName, shpName
library(conflicted)
library(sf)
library(terra)
library(exactextractr)
library(tidyverse)
library(tidyterra)
library(arrow)

args <- commandArgs(trailingOnly = TRUE)
domain <- args[1]

func <- function(prName, shpName, method, outName) {
    if(file.exists(outName)) return(NA)

    data <- terra::rast(prName)

    poly0 <- st_read(shpName)
    poly0 <- st_transform(poly0, crs = 4326)

    ## Extract data
    if (method != 'ratio') {
        res <- exact_extract(data, poly0, fun = method)
    } else {
        res1 <- exact_extract(data, poly0, fun = 'sum')
        res2 <- exact_extract(data, poly0, fun = 'count')
        res <- res1 / res2
    }

    df <- data.frame(t(res))
    colnames(df) <- poly0$ohdb_id

    # save to parquet
    write.csv(df, outName)
}

method <- 'mean'
fnames <- list.files(path = '../../data/GHS_population', pattern = '\\.tif$', full.name = T)
for (prName in fnames) {
    prName0 <- basename(prName)
    shpName <- paste0('../basin_boundary/GRIT_full_catchment_', domain, '_EPSG8857_simplify_final_125km2_subset.gpkg')
    outName <- paste0('../data/GHS_population_catch_ave_domain_', domain, '_', substr(prName0, 1, nchar(prName0)-4), '.csv')
    if (!file.exists(outName)) {
        func(prName, shpName, method, outName)
    }
}


