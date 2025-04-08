using MAT, Dates

dir = @__DIR__

nation = "netherlands"

remove_outliers = false

# Load calibration data (with figaro input-output tables)
calibration_data = matread(joinpath(dir, "calibration_data/" * nation * "/calibration/2010Q1.mat"))["calibration_data"]

# Load time series data
data = matread(joinpath(dir, "calibration_data/" * nation * "/data/1996.mat"))["data"]
ea = matread(joinpath(dir, "calibration_data/" * nation * "/ea/1996.mat"))["ea"]


if remove_outliers #TODO
    outlier_index = argmax(data["real_imports_quarterly"])[1] 
    data["real_imports_quarterly"][outlier_index] = (data["real_imports_quarterly"][outlier_index+1] + data["real_imports_quarterly"][outlier_index-1])/2
    outlier_index = argmax(data["real_exports_quarterly"][1:80])
    data["real_exports_quarterly"][outlier_index] = (data["real_exports_quarterly"][outlier_index+1] + data["real_exports_quarterly"][outlier_index-1])/2

    matwrite(joinpath(dir, "calibration_data/" * nation * "/data/1996.mat"), Dict("data" => data))
    matwrite(joinpath(dir, "calibration_data/" * nation * "/ea/1996.mat"), Dict("ea" => ea))
end


# add calibration times to the data
max_calibration_date = DateTime(2016, 12, 31)
estimation_date = DateTime(1996, 12, 31)


struct CalibrationDataNetherlands
    calibration::Dict{String, Any}
    data::Dict{String, Any}
    ea::Dict{String, Any}
    max_calibration_date::DateTime
    estimation_date::DateTime
end

const NETHERLANDS_CALIBRATION = CalibrationDataNetherlands(calibration_data, data, ea, max_calibration_date, estimation_date)
