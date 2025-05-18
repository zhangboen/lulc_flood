module load UDUNITS; module load GDAL/3.7.1-foss-2023a-spatialite; module load R/4.3.2-gfbf-2023a
name=$1
Rscript -e "install.packages("${name}", lib="~/local/rlibs")"
