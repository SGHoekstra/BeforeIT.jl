
function latexTableContent(input_data::Matrix{String}, tableRowLabels::Vector{String}, 
        dataFormat::String, tableColumnAlignment, tableBorders::Bool, booktabs::Bool, 
        makeCompleteLatexDocument::Bool)
    nrows, ncols = size(input_data)
    latex = []

    if makeCompleteLatexDocument
        push!(latex, "\\documentclass{article}")
        push!(latex, "\\begin{document}")
    end

    booktabs && push!(latex, "\\toprule")

    for row in 1:nrows
        row_content = [tableRowLabels[row]]
        for col in 1:ncols
            push!(row_content, input_data[row, col])
        end
        if row < nrows
            push!(latex, join(row_content, " & "), " \\\\ ")
        else
            push!(latex, join(row_content, " & "))
        end
    end

    booktabs && push!(latex, "\\bottomrule")
    makeCompleteLatexDocument && push!(latex, "\\end{document}")

    return latex
end

# Helper functions for LaTeX table creation and stars notation
function stars(p_value)
    if p_value < 0.01
        return "***"
    elseif p_value < 0.05
        return "**"
    elseif p_value < 0.1
        return "*"
    else
        return ""
    end
end

nanmean(x) = mean(filter(!isnan,x))
nanmean(x,y) = mapslices(nanmean,x; dims = y)

function calculate_forecast_errors(forecast, actual)
    error = forecast - actual
    rmse = dropdims(100 * sqrt.(nanmean(error.^2, 1)), dims=1)
    bias = dropdims(nanmean(error, 1), dims=1)
    return rmse, bias, error
end

function write_latex_table(filename, country, input_data_S, horizons)
    tableRowLabels = ["$(i)q" for i in horizons]
    dataFormat, tableColumnAlignment = "%.2f", "r"
    tableBorders, booktabs, makeCompleteLatexDocument = false, false, false

    latex = latexTableContent(input_data_S, tableRowLabels, dataFormat, tableColumnAlignment, tableBorders, booktabs, makeCompleteLatexDocument)

    open("data/$(country)/analysis/$(filename)", "w") do fid
        for line in latex
            write(fid, line * "\n")
        end
    end
end

function write_csv_table(filename, country, input_data, horizons, number_variables)
    """
    Write table data to CSV format for easy programmatic access.

    # Arguments
    - `filename`: CSV filename (e.g., "rmse_var.csv")
    - `country`: Country code
    - `input_data`: Matrix of numerical values (horizons × variables)
    - `horizons`: Vector of forecast horizons
    - `number_variables`: Number of variables (5 for base, 8 for validation)
    """

    # Variable names based on number of variables
    if number_variables == 5
        variable_names = ["Real GDP", "GDP Deflator Growth", "Real Consumption", "Real Investment", "Euribor"]
    elseif number_variables == 8
        variable_names = ["Real GDP", "GDP Deflator Growth", "Real Gov Consumption", "Real Exports", "Real Imports", "Real GDP (EA)", "GDP Deflator Growth (EA)", "Euribor"]
    else
        # Fallback for other cases
        variable_names = ["$(model_type)$i" for i in 1:number_variables]
    end

    # Create DataFrame with proper structure
    df = DataFrame()

    # Add horizon column
    df.Horizon = ["$(h)q" for h in horizons]

    # Add variable columns
    for (j, var_name) in enumerate(variable_names)
        if j <= size(input_data, 2)
            df[!, var_name] = input_data[:, j]
        else
            df[!, var_name] = fill(NaN, length(horizons))
        end
    end

    # Write CSV file
    csv_path = "data/$(country)/analysis/$(filename)"
    CSV.write(csv_path, df)

    @info "Saved CSV table: $csv_path"
end

function generate_dm_test_comparison(error1, error2, rmse1, rmse2, horizons, number_variables)
    input_data = -round.(100 * (rmse1 .- rmse2) ./ rmse2, digits=1)
    input_data_S = fill("", size(input_data))

    for j in 1:length(horizons)
        h = horizons[j]
        for l in 1:number_variables
            dm_error1 = view(error1, :, j, l)[map(!, isnan.(view(error1, :, j, l)))]
            dm_error2 = view(error2, :, j, l)[map(!, isnan.(view(error2, :, j, l)))]
            _, p_value = Bit.dmtest_modified(dm_error2, dm_error1, h)
            input_data_S[j, l] = string(input_data[j, l]) * "(" * string(round(p_value, digits=2)) * ", " * string(stars(p_value)) * ")"
        end
    end
    return input_data_S
end

function generate_mz_test_bias(error, actual, bias, horizons, number_variables)
    input_data = round.(bias, digits=4)
    input_data_S = fill("", size(input_data))

    for j in 1:length(horizons)
        h = horizons[j]
        for l in 1:number_variables
            mz_forecast = (view(error, :, j, l) + view(actual, :, j, l))[map(!, isnan.(view(error, :, j, l) + view(actual, :, j, l)))]
            mz_actual = view(actual, :, j, l)[map(!, isnan.(view(actual, :, j, l)))]
            _, _, p_value = Bit.mztest(mz_actual, mz_forecast)
            input_data_S[j, l] = string(input_data[j, l]) * " (" * string(round(p_value, digits=3)) * ", " * stars(p_value) * ")"
        end
    end
    return input_data_S
end

function create_bias_rmse_tables_abm(forecast, actual, horizons, type, number_variables, country)
    type_prefix = type == "validation" ? "validation_" : ""

    if type == "validation"
        comparison_model = "var"
    else
        comparison_model = "ar"
    end
    
    rmse_abm, bias_abm, error_abm = calculate_forecast_errors(forecast, actual)

    forecast_var = load("data/$(country)/analysis/forecast_$(type_prefix)$(comparison_model).jld2")["forecast"]
    rmse_var, _, error_var = calculate_forecast_errors(forecast_var, actual)

    # Save RMSE comparison data (ABM vs $(model_type))
    rmse_comparison_data_latex = generate_dm_test_comparison(error_abm, error_var, rmse_abm, rmse_var, horizons, number_variables)
    write_latex_table("rmse_$(type_prefix)abm.tex", country, rmse_comparison_data_latex, horizons)

    # For CSV, save just the numerical comparison values (percentage improvement)
    rmse_comparison_data_numeric = -round.(100 * (rmse_abm .- rmse_var) ./ rmse_var, digits=1)
    write_csv_table("rmse_$(type_prefix)abm.csv", country, rmse_comparison_data_numeric, horizons, number_variables)

    # Save bias data
    bias_data_latex = generate_mz_test_bias(error_abm, actual, bias_abm, horizons, number_variables)
    write_latex_table("bias_$(type_prefix)abm.tex", country, bias_data_latex, horizons)

    # For CSV, save just the numerical bias values (without p-values and stars)
    bias_data_numeric = round.(bias_abm, digits=4)
    write_csv_table("bias_$(type_prefix)abm.csv", country, bias_data_numeric, horizons, number_variables)

    return nothing
end

function create_bias_rmse_tables_var(forecast, actual, horizons, forecast_type, model_type, number_variables, k, country)
    type_prefix = forecast_type == "validation" ? "validation_" : ""

    if k == 1
        save("data/$(country)/analysis/forecast_$(type_prefix)$(model_type).jld2", "forecast", forecast)
        rmse_var, bias_var, error_var = calculate_forecast_errors(forecast, actual)

        # Save RMSE data
        input_data_rmse = round.(rmse_var, digits=2)
        write_latex_table("rmse_$(type_prefix)$(model_type).tex", country, string.(input_data_rmse), horizons)
        write_csv_table("rmse_$(type_prefix)$(model_type).csv", country, input_data_rmse, horizons, number_variables)

        # Save bias data
        bias_data_latex = generate_mz_test_bias(error_var, actual, bias_var, horizons, number_variables)
        write_latex_table("bias_$(type_prefix)$(model_type).tex", country, bias_data_latex, horizons)

        # For CSV, save just the numerical bias values (without p-values and stars)
        bias_data_numeric = round.(bias_var, digits=4)
        write_csv_table("bias_$(type_prefix)$(model_type).csv", country, bias_data_numeric, horizons, number_variables)

    else
        save("data/$(country)/analysis/forecast_$(type_prefix)$(model_type)_$(k).jld2", "forecast", forecast)
        rmse_var_k, _, error_var_k = calculate_forecast_errors(forecast, actual)

        forecast_base_var = load("data/$(country)/analysis/forecast_$(type_prefix)$(model_type).jld2")["forecast"]
        rmse_base_var, _, error_base_var = calculate_forecast_errors(forecast_base_var, actual)

        # Generate comparison data for LaTeX (with p-values and stars)
        rmse_comparison_data_latex = generate_dm_test_comparison(error_var_k, error_base_var, rmse_var_k, rmse_base_var, horizons, number_variables)
        write_latex_table("rmse_$(type_prefix)$(model_type)_$(k).tex", country, rmse_comparison_data_latex, horizons)

        # For CSV, save just the numerical comparison values (percentage improvement)
        rmse_comparison_data_numeric = -round.(100 * (rmse_var_k .- rmse_base_var) ./ rmse_base_var, digits=1)
        write_csv_table("rmse_$(type_prefix)$(model_type)_$(k).csv", country, rmse_comparison_data_numeric, horizons, number_variables)
    end
end