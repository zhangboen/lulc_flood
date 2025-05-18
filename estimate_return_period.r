# module load UDUNITS; module load GDAL/3.7.1-foss-2023a-spatialite; module load R/4.3.2-gfbf-2023a
library(extRemes)
library(dplyr)
library(foreach)
library(doParallel)

args <- commandArgs(trailingOnly = TRUE)
fname <- args[1]

df <- read.csv(fname)

funcHighFlow <- function(df0) {
    df1 <- df0 %>% 
        group_by(year) %>%
        summarize(Q = max(Q))
    x <- df1$Q
    fit <- fevd(x)
    prob_exceedance <- 1 - pextRemes(fit, q=df0$Q)
    # Calculate the return period
    rp <- 0.25 / prob_exceedance
    colName <- colnames(df0)[grep('date',colnames(df0))]
    out <- data.frame(rp=rp, ohdb_id=df0$ohdb_id)
    out[colName] <- df0[colName]
    return (out)
}

funcLowFlow <- function(df0) {
    df1 <- df0 %>% 
        group_by(year) %>%
        summarize(Q = min(Q))
    x <- df1$Q * -1
    fit <- fevd(x)
    prob_exceedance <- 1 - pextRemes(fit, q=df0$Q*-1)
    # Calculate the return period
    rp <- 0.25 / prob_exceedance
    colName <- colnames(df0)[grep('date',colnames(df0))]
    out <- data.frame(rp=rp, ohdb_id=df0$ohdb_id)
    out[colName] <- df0[colName]
    return (out)
}

cl <- makeCluster(12)
registerDoParallel(cl)
if (grepl('Qmax7', fname)) {
    df1 <- foreach(i = unique(df$ohdb_id), .combine = 'rbind', .packages = c('extRemes','dplyr')) %dopar% {funcHighFlow(df[df$ohdb_id==i,])}
} else {
    df1 <- foreach(i = unique(df$ohdb_id), .combine = 'rbind', .packages = c('extRemes','dplyr')) %dopar% {funcLowFlow(df[df$ohdb_id==i,])}
}
stopCluster(cl)

colName <- colnames(df)[grep('date',colnames(df))]
df <- merge(df, df1, by = c('ohdb_id',colName))

# save to parquet
write.csv(df, paste0(substr(fname, 1, nchar(fname)-4), '_rp.csv'), row.names = F)