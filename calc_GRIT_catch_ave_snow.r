# module load UDUNITS; module load GDAL/3.7.1-foss-2023a-spatialite; module load R/4.3.2-gfbf-2023a
# Rscript --no-restore --no-save calc_GRIT_catch_ave_MSWEP_1.r prName, shpName
library(conflicted)
library(sf)
library(terra)
library(exactextractr)
library(tidyverse)
library(tidyterra)
library(arrow)
library(readr)

args <- commandArgs(trailingOnly = TRUE)
ncName <- args[1]
method <- 'mean'
shpName <- args[2]
outName <- args[3] 

if(file.exists(outName)&(file.info(outName)$size>0)) quit(save="no")

poly0 <- st_read(shpName)

## Extract data
data <- terra::rast(ncName)
data <- project(data, 'EPSG:8857')

# reproject
data[data == 0] <- NA

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
colNames0 <- colnames(df)
df <- data.table::transpose(df)
colnames(df) <- rowNames0
rownames(df) <- colNames0
df['stat'] <- c('mean', 'std')

## Add time by joining against variable name
df['time'] <- terra::time(data)

## Remove index from variable name
df[is.na(df)] <- 0

# save to csv
write_csv(df, outName)
