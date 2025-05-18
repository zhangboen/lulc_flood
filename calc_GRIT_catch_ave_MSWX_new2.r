# module load UDUNITS; module load GDAL/3.7.1-foss-2023a-spatialite; module load R/4.3.2-gfbf-2023a
# Rscript --no-restore --no-save calc_GRIT_catch_ave_MSWEP_1.r prName, shpName

# This script is calculate antecedent conditions (ave and std) but only for those grid cells with rainfall greater than 0.1 mm

library(conflicted)
library(sf)
library(terra)
library(exactextractr)
library(tidyverse)
library(tidyterra)
library(arrow)
library(readr)
library(foreach)
library(doParallel)

args <- commandArgs(trailingOnly = TRUE)
varName <- args[1]
shpName <- args[2]
year <- as.integer(args[3])
outName <- args[4]
print(varName)
print(year)
print(shpName)

if(file.exists(outName)&(file.info(outName)$size>0)) quit(save="no")

poly0 <- st_read(shpName)

## Extract data
fnames = list.files(path = paste('../../data/MSWX/',varName,'Daily',sep = '/'), 
                    pattern = paste0(year,'[0-9][0-9][0-9].nc'),
                    full.name = T)
func <- function(fname) {
    cat(paste0("Processed ", fname, "\n"))
    data <- terra::rast(fname)
    # if (varName == 'P') {
    #     data[ data < 0.1 ] <- NA
    # }
    data <- project(data, 'EPSG:8857')

    res <- exact_extract(data, poly0, fun = c('mean','stdev'))

    df <- res %>%
    set_names(names(data))
    df <- as.data.frame(df)
    if ('ohdb_id' %in% colnames(poly0)) {
    rownames(df) <- poly0$ohdb_id
    } else {
    rownames(df) <- poly0$global_id
    }
    rowNames0 <- rownames(df)
    colNames0 <- c('ave','std')
    df <- data.table::transpose(df)
    colnames(df) <- rowNames0
    df['stat'] <- colNames0

    ## Add time by joining against variable name
    df['time'] <- terra::time(data)

    ## Remove index from variable name
    df[is.na(df)] <- 0
    return (df)
}

cl <- makeCluster(12)
registerDoParallel(cl)
df <- foreach(i = fnames, .combine = 'rbind', .packages = c('exactextractr','terra','tidyverse')) %dopar% {func(i)}
print(dim(df))
stopCluster(cl)

# save to parquet
write_csv(df, outName)
