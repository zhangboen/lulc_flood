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
library(foreach)
library(doParallel)

args <- commandArgs(trailingOnly = TRUE)
varName <- args[1]
year <- as.integer(args[2])
shpName <- args[3]
method <- args[4]
outName <- args[5]
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
    data <- terra::rast(fname)
    data <- project(data, 'EPSG:8857')
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
    df['time'] <- terra::time(data)

    ## Remove index from variable name
    df <- subset(df, select = -c(variable))
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
