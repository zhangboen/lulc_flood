# module load UDUNITS; module load GDAL; module load R/4.3.2-gfbf-2023a
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
prName <- args[1]
shpName <- args[2]
method <- args[3]
outName <- args[4]
print(prName)
print(shpName)

if(file.exists(outName)&(file.info(outName)$size>0)) quit(save="no")

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
df['variable'] <- rownames(df)

## Add time by joining against variable name
times <- tibble(variable = names(data), time = time(data))
df <- df %>%
  left_join(times, by = "variable")

## Remove index from variable name
df <- subset(df, select = -c(variable))
df[is.na(df)] <- 0

# save to parquet
write_csv(df, outName)
