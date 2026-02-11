
function error_table_abm(country::String, ea, data, quarters, horizons;
                         model_variant::String="base",
                         prediction_folder::String="abm_predictions")

    quarters_num = Bit.date2num.(quarters)
    number_quarters = length(quarters)
    max_year = year(quarters[end])

    number_horizons = length(horizons)
    number_variables = 5

    forecast = fill(NaN, number_quarters, number_horizons, number_variables)
    actual = fill(NaN, number_quarters, number_horizons, number_variables)

    # Build predictions path
    predictions_dir = "./data/$(country)/$(prediction_folder)"

    # Find number_of_seeds from the first available prediction file
    number_of_seeds = nothing
    for q in quarters_num
        fname = "$(predictions_dir)/$(year(Bit.num2date(q)))Q$(quarterofyear(Bit.num2date(q))).jld2"
        if isfile(fname)
            model = load(fname, "predictions_dict")
            number_of_seeds = size(model["real_gdp_quarterly"], 2)
            break
        end
    end
    if number_of_seeds === nothing
        @warn "No prediction files found for $country in $predictions_dir, skipping"
        return
    end

    for i in 1:number_quarters

        q = quarters_num[i]
        fname = "$(predictions_dir)/$(year(Bit.num2date(q)))Q$(quarterofyear(Bit.num2date(q))).jld2"

        # Skip quarters without prediction files (leave NaN)
        isfile(fname) || continue

        model = load(fname, "predictions_dict")

        for j in 1:number_horizons
            horizon = horizons[j]

            forecast_quarter_num = Bit.date2num(lastdayofmonth(Bit.num2date(q) + Month(3 * horizon)))
            Bit.num2date(forecast_quarter_num) > Date(max_year, 12, 31) && break

            # Skip if actual data doesn't cover this forecast quarter
            any(data["quarters_num"] .== forecast_quarter_num) || continue

            actual[i, j, :] = hcat(
                log.(data["real_gdp_quarterly"][data["quarters_num"] .== forecast_quarter_num]),
                log.(1 .+ data["gdp_deflator_growth_quarterly"][data["quarters_num"] .== forecast_quarter_num]),
                log.(data["real_household_consumption_quarterly"][data["quarters_num"] .== forecast_quarter_num]),
                log.(data["real_fixed_capitalformation_quarterly"][data["quarters_num"] .== forecast_quarter_num]),
                (1 .+ data["euribor"][data["quarters_num"] .== forecast_quarter_num]).^(1/4)
            )

            forecast[i, j, :] = hcat(
                log.(mean(model["real_gdp_quarterly"][repeat(model["quarters_num"] .== forecast_quarter_num,1,number_of_seeds)])),
                log.(1 .+ mean(model["gdp_deflator_growth_quarterly"][repeat(model["quarters_num"] .== forecast_quarter_num,1,number_of_seeds)])),
                log.(mean(model["real_household_consumption_quarterly"][repeat(model["quarters_num"] .== forecast_quarter_num,1,number_of_seeds)])),
                log.(mean(model["real_fixed_capitalformation_quarterly"][repeat(model["quarters_num"] .== forecast_quarter_num,1,number_of_seeds)])),
                (1 .+ mean(model["euribor"][repeat(model["quarters_num"] .== forecast_quarter_num,1,number_of_seeds)])).^(1/4)
            )
        end
    end

    # Save forecast to variant subfolder
    analysis_dir = "data/$(country)/analysis/$(model_variant)"
    mkpath(analysis_dir)
    save("$(analysis_dir)/forecast_abm.jld2", "forecast", forecast)
    create_bias_rmse_tables_abm(forecast, actual, horizons, "training", number_variables, country; model_variant=model_variant)
end

