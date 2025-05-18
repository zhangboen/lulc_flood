# module load UDUNITS; module load GDAL/3.7.1-foss-2023a-spatialite; module load R/4.3.2-gfbf-2023a
library(grf)
library(dplyr)
library(reshape2)

args <- commandArgs(trailingOnly = TRUE)
fname <- args[1]

df <- read.csv(fname)
print(length(unique(df$ohdb_id)))

if (!'HYBAS_ID'%in%colnames(df)) {
    s = read.csv('../data/basin_attributes_new.csv')
    df = merge(df, s[c('ohdb_id','HYBAS_ID')], by = 'ohdb_id')
}

# create label-encoding variable for basin id
x = data.frame(HYBAS_ID = unique(df$HYBAS_ID), basin_id=1:length(unique(df$HYBAS_ID)))
df = merge(df, x, by = 'HYBAS_ID')

# create label-encoding variable for dam purpose
x = data.frame(Main_Purpose=unique(df$Main_Purpose), Main_Purpose_id=1:length(unique(df$Main_Purpose)))
df = merge(df, x, by = 'Main_Purpose')

# create label-encoding variable for season 
x = data.frame(season=unique(df$season),season_id=1:length(unique(df$season)))
df = merge(df, x, by = 'season')
df['year'] = as.numeric(df$year)

# limit gauges to those with minimal influences of dams:
    # 2. percentage of reservoir area to catchment area less than 10
connect = read.csv('../data/basin_reservoir_darea_ratio.csv')
connect = connect[(connect$ratio>=10),]
df = df[!df$ohdb_id%in%(connect$ohdb_id),]
print(length(unique(df$ohdb_id)))

# limit gauges to those with no more than one year of no urban area
df1 <- df %>%
  group_by(ohdb_id,year) %>%
  summarize(
    ImperviousSurface = ImperviousSurface[1]
  ) %>%
  group_by(ohdb_id) %>%
  summarize(
    a = sum(ImperviousSurface==0)
  )
df1 = as.vector(df1[df1$a<=1,'ohdb_id'])$ohdb_id
df = df[df$ohdb_id%in%(df1),]
print(length(unique(df$ohdb_id)))

# add population
df_pop = read.csv('../data_population/GHS_population_catch_ave_1982-2023_cubic_interp.csv')
df_pop = melt(df_pop, id.vars = 'ohdb_id', variable.name = 'year', value.name = 'population')
df_pop[df_pop$population<1e-3,'population'] = 0
df_pop['year'] = as.numeric(substring(df_pop$year, 2, 5))
df = merge(df, df_pop, by = c('ohdb_id','year'))

predictors <- c('BDTICM', 'elevation', 'slope', 
                'aridity', 
                'sedimentary', 'plutonic', 'volcanic', 'metamorphic',
                'clay', 'sand', 'silt',
                'Porosity_x', 'logK_Ice_x',
                # 'ohdb_latitude', 'ohdb_longitude', 
                # 'year', 
                # 'climate', 
                # 'season_id',
                'basin_id', 
                # 'gauge_id',
                'p_3', 'tmax_3', 'tmin_3', 'swd_3', 'relhum_3', 'wind_3',
                'p_7', 'tmax_7', 'tmin_7', 'swd_7', 'relhum_7', 'wind_7',
                'p_15', 'tmax_15', 'tmin_15', 'swd_15', 'relhum_15', 'wind_15',
                'p_30', 'tmax_30', 'tmin_30', 'swd_30', 'relhum_30', 'wind_30',
                'p_365',
                # 'GDP', 
                'population',
                'runoff_ratio', 
                'slope_fdc', 
                'Q10_50', 
                'high_q_freq', 'low_q_freq', 'zero_q_freq', 'high_q_dur', 'low_q_dur', 
                'cv', 'BFI', 
                'noResRatio', 
                'FI', 'lagT', 'stream_elas', 'hfd_mean',
                'p_mean', 'tmax_ave', 'tmax_std',
                'forest', 
                # 'crop', 
                # 'grass', 'water', 'wetland',
                'res_darea_normalize', 'Year_ave', 'Main_Purpose_id'
)

X <- df[predictors]
W <- df$ImperviousSurface
Y <- df$Q / df$gritDarea
c.forest <- causal_forest(X, Y, W)

# Predict using the forest.
c.pred <- predict(c.forest,  estimate.variance = T)

df <- cbind(df, c.pred)

# estimate 95% confidence interval
tau_hat <- c.pred$predictions  # Estimated treatment effects
variance_hat <- c.pred$variance.estimates  # Variance of the estimates
# Standard error
se <- sqrt(variance_hat)
# 95% confidence interval
alpha <- 0.05
z_score <- qnorm(1 - alpha / 2)
ci_lower <- tau_hat - z_score * se
ci_upper <- tau_hat + z_score * se
df['te_lower'] <- ci_lower
df['te_upper'] <- ci_upper

outName = sub('final_dataset_', '', strsplit(basename(fname), '.', fixed = T)[1])
write.csv(df, paste0('../results/causal_'))