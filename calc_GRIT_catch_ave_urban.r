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
ncName <- args[1]
shpName <- args[2]
outName <- args[3]
print(ncName)
print(shpName)

if(file.exists(outName)&(file.info(outName)$size>0)) quit(save="no")

poly0 <- st_read(shpName)

## Extract data
data <- terra::rast(ncName)

# data <- ifel(data == 190, 1, 0)

data <- project(data, 'EPSG:8857')

if (st_crs(poly0)$epsg != 8857) {
  poly0 <- st_transform(poly0, 8857)
}

res <- exact_extract(data, poly0, fun = c('sum','count','mean'))

df <- res %>%
set_names(names(data))
df <- as.data.frame(df)
if ('ohdb_id' %in% colnames(poly0)) {
    rownames(df) <- poly0$ohdb_id
} else {
    rownames(df) <- poly0$global_id
}
rowNames0 <- rownames(df)
colNames0 <- c('sum','count','mean')
df <- data.table::transpose(df)
colnames(df) <- rowNames0
df['stat'] <- colNames0

# save to parquet
write_csv(df, outName)
