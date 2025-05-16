# This scripts calls all scripts to create tables for comparing forecasts of the abm and the baseline models (AR, ARX, VAR, VARX)
import BeforeIT as Bit
using LaTeXStrings, CSV, HDF5, MAT

# Clear old files
#foreach(rm, filter(endswith(".h5"), readdir("./analysis/tabs/",join=true)))
#foreach(rm, filter(endswith(".tex"), readdir("./analysis/tabs/",join=true)))

country = "netherlands"

include("./analysis_utils.jl")
include("./error_table_ar.jl")
include("./error_table_abm.jl")
include("./error_table_validation_var.jl")
include("./error_table_validation_abm.jl")


error_table_ar(country)
error_table_validation_var(country)

#error_table_abm(country)
error_table_abm(country, abmx = false, unconditional_forecasts = true)
error_table_abm(country, abmx = true, unconditional_forecasts = true)
error_table_validation_abm(country, abmx = false, unconditional_forecasts = true)
error_table_validation_abm(country, abmx = true, unconditional_forecasts = true)

