# module load UDUNITS; module load GDAL/3.7.1-foss-2023a-spatialite; module load R/4.3.2-gfbf-2023a
# Rscript --no-restore --no-save calc_GRIT_catch_ave_MSWEP_1.r prName, shpName

# This script is calculate the annual statistics (not antecedent conditions) for each grid cell and then do the catchment-average

library(conflicted)
library(sf)
library(terra)
library(exactextractr)
library(tidyverse)
library(tidyterra)
library(arrow)
library(readr)
library(lubridate)
library(reshape2)
library(dplyr)

# define input arguments
args <- commandArgs(trailingOnly = TRUE)
varName <- args[1]
shpName <- args[2]
year0 <- as.integer(args[3])
outName <- paste0('../data_mswx/GRIT_catch_ave_temporalAveThenGridAve_', varName, '_', year0, '.csv')
print(varName)
print(shpName)
print(year0)

# if output exists, do not process the script
if(file.exists(outName)&(file.info(outName)$size>0)) quit(save="no")

# read all rasters during YEAR
if (varName %in% c('snowfall','snowmelt')) {
    rasters <- terra::rast(paste0('../../data/MSWX/Snow/', varName, '_', year0, '.nc'))
} else {
    fnames <- list.files(path = paste('../../data/MSWX/', varName, 'Daily/', sep = '/'), pattern = paste0(year0, '[0-9][0-9][0-9].nc'), full.name = T)
    dates <- sapply(fnames, function(x) strsplit(basename(x),'.',fixed=T)[[1]][1], USE.NAMES = F)
    rasters <- terra::rast(fnames)
    names(rasters) <- dates
}
rasters <- terra::crop(rasters, ext(-155, 179, -51, 73))
rasters <- terra::project(rasters, 'EPSG:8857', threads = TRUE)

# calculate pixel-based statistics
ras_sum <- terra::app(rasters, fun = sum, cores = 12)
ras_max <- terra::app(rasters, fun = max, cores = 12)
ras_mean <- terra::app(rasters, fun = mean, cores = 12)
ras_min <- terra::app(rasters, fun = min, cores = 12)

# read basin boundary
poly0 <- st_read(shpName)

# calculate catchment averages
res_sum <- exact_extract(ras_sum, poly0, 'mean')
res_max <- exact_extract(ras_max, poly0, 'mean')
res_min <- exact_extract(ras_min, poly0, 'mean')
res_mean <- exact_extract(ras_mean, poly0, 'mean')

res <- rbind(res_sum, res_max, res_min, res_mean)
res <- t(res)
res <- as.data.frame(res)
colnames(res) <- c('annSum','annMax','annMin','annAve')
res['ohdb_id'] <- poly0$ohdb_id

write.csv(res, outName, row.names = F)
