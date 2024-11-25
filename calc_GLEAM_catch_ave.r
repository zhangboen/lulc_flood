# module load UDUNITS; module load GDAL; module load R/4.3.2-gfbf-2023a
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
fnames <- list.files(path = 'gleam_data', pattern = '.*?nc', full.name = T)
for (prName in fnames) {
    prName0 <- basename(prName)
    for (domain in c('AF','AS','SA','SP','NA','EU')) {
        shpName <- paste0('basin_boundary_new/GRIT_full_catchment_domain_', domain, '_EPSG4326_500km2.shp')
        outName <- paste0('gleam_data/GRIT_catch_ave_domain_', domain, '_', substr(prName0, 1, nchar(prName0)-3), '.csv')
        func(prName, shpName, method, outName)
    }
}


