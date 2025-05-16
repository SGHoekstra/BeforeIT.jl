import BeforeIT as Bit
using FileIO, Plots, Statistics


function generate_inflation_decomposition(
    country::String = "netherlands";
    empirical_distribution::Bool = false,
    abmx::Bool = false,
    unconditional_forecasts::Bool = false,
    year_i::Int = 2019,
    quarter::Int = 4,
)
    """
    Generate visualizations decomposing inflation into its components across different sectors.

    Parameters:
    - country: The country to analyze
    - empirical_distribution: Whether to use empirical distribution
    - abmx: Whether to use abmx model (vs abm)
    - unconditional_forecasts: Whether to use unconditional forecasts
    - year_i: Year for analysis
    - quarter: Quarter for analysis
    """

    # Define the output structure based on parameters (without long_run)
    output_folder =
        "./analysis/figs/" *
        country *
        (empirical_distribution ? "/empirical" : "/calibrated") *
        (abmx ? (unconditional_forecasts ? "/abmx_uf" : "/abmx") : "/abm")

    # Create output directory
    mkpath(output_folder)

    # Load parameters and initial conditions
    parameter_path = "data/$(country)/parameters/" * string(year_i) * "Q" * string(quarter) * ".jld2"
    initial_conditions_path = "data/$(country)/initial_conditions/" * string(year_i) * "Q" * string(quarter) * ".jld2"

    # Check if files exist
    if !isfile(parameter_path)
        @warn "Parameters file not found: $parameter_path, exiting"
        return nothing
    end

    if !isfile(initial_conditions_path)
        @warn "Initial conditions file not found: $initial_conditions_path, exiting"
        return nothing
    end

    parameters = load(parameter_path)
    initial_conditions = load(initial_conditions_path)

    T = 16
    n_runs = 32

    model = Bit.init_model(parameters, initial_conditions, T)
    data_vector = Bit.ensemblerun(model, n_runs; multi_threading = true, abmx = abmx, conditional_forecast = !unconditional_forecasts)

    cost_push_inflation = reshape(data_vector.cost_push_inflation, T + 1, model.prop.I, n_runs)
    demand_pull_inflation = reshape(data_vector.demand_pull_inflation, T + 1, model.prop.I, n_runs)
    cost_push_inflation_labour = reshape(data_vector.cost_push_inflation_labour, T + 1, model.prop.I, n_runs)
    cost_push_inflation_capital = reshape(data_vector.cost_push_inflation_capital, T + 1, model.prop.I, n_runs)
    cost_push_inflation_material = reshape(data_vector.cost_push_inflation_material, T + 1, model.prop.I, n_runs)

    # Create the first decomposition plot
    inflation_decomposition = plot(layout = (3, 3), size = (900, 800))

    masks = [
        model.firms.G_i .∈ [1:3],
        model.firms.G_i .∈ [4:26],
        model.firms.G_i .== 27,
        model.firms.G_i .∈ [28:36],
        model.firms.G_i .∈ [37:40],
        model.firms.G_i .∈ [41:43],
        model.firms.G_i .== 44,
        model.firms.G_i .∈ [45:53],
        model.firms.G_i .∈ [58:62],
    ]

    plot_titles = ["A", "B,C,D,and E", "F", "G,F,I, and J", "K", "L", "M and N", "O,P and Q", "R and S"]
    variable_names = ["Cost-Push", "Demand-Pull", "Expectations"]
    colors = [:darkblue, :mediumseagreen, :darkorange]

    for (i, mask) in enumerate(masks)
        # Extract inflation components
        cost_push = vec(mean(mean(cost_push_inflation[:, mask, :], dims = 2), dims = 3))
        demand_pull = vec(mean(mean(demand_pull_inflation[:, mask, :], dims = 2), dims = 3))
        expectations = vec(mean(data_vector.aggregate_inflation_expectations, dims = 2))

        # Stack the data
        stacked_data = hcat(cost_push, demand_pull, expectations)

        # Calculate the total (sum of all components)
        total_inflation = sum(stacked_data, dims = 2)

        # Create bar plot
        groupedbar!(
            inflation_decomposition,
            stacked_data,
            bar_position = :stack,
            subplot = i,
            title = plot_titles[i],
            label = permutedims(variable_names),
            color = permutedims(colors),
            linewidth = 0,
        )

        # Add dashed line for total inflation
        plot!(
            inflation_decomposition,
            1:length(total_inflation),
            total_inflation,
            subplot = i,
            linewidth = 2,
            linestyle = :dash,
            color = :black,
            label = (i == 1 ? "Total Inflation" : nothing),
        )  # Only add label on first subplot
    end

    # Enhance the appearance
    plot!(
        inflation_decomposition,
        legend = true,
        legendfontsize = 8,
        background_color = :white,
        foreground_color = :grey,
        framestyle = :box,
        grid = false,
        tickfontsize = 8,
        titlefontsize = 10,
    )

    # Save the first decomposition plot
    savefig(inflation_decomposition, output_folder * "/inflation_decomposition.png")

    # Create the detailed decomposition plot
    inflation_decomposition_detailed = plot(layout = (3, 3), size = (900, 800))

    variable_names = ["Labour cost", "Capital cost", "Material cost", "Demand Pull", "Expectations"]
    colors = [:cornflowerblue, :indianred, :goldenrod, :mediumseagreen, :mediumpurple]

    for (i, mask) in enumerate(masks)
        # Extract inflation components
        cost_push_labour = vec(mean(mean(cost_push_inflation_labour[:, mask, :], dims = 2), dims = 3))
        cost_push_capital = vec(mean(mean(cost_push_inflation_capital[:, mask, :], dims = 2), dims = 3))
        cost_push_material = vec(mean(mean(cost_push_inflation_material[:, mask, :], dims = 2), dims = 3))
        demand_pull = vec(mean(mean(demand_pull_inflation[:, mask, :], dims = 2), dims = 3))
        expectations = vec(mean(data_vector.aggregate_inflation_expectations, dims = 2))

        # Stack the data
        stacked_data = hcat(cost_push_labour, cost_push_capital, cost_push_material, demand_pull, expectations)

        # Calculate the total (sum of all components)
        total_inflation = sum(stacked_data, dims = 2)

        # Create bar plot
        groupedbar!(
            inflation_decomposition_detailed,
            stacked_data,
            bar_position = :stack,
            subplot = i,
            title = plot_titles[i],
            label = permutedims(variable_names),
            color = permutedims(colors),
            linewidth = 0,
        )

        # Add dashed line for total inflation
        plot!(
            inflation_decomposition_detailed,
            1:length(total_inflation),
            total_inflation,
            subplot = i,
            linewidth = 2,
            linestyle = :dash,
            color = :black,
            label = (i == 1 ? "Total Inflation" : nothing),
        )  # Only add label on first subplot
    end

    # Enhance the appearance
    plot!(
        inflation_decomposition_detailed,
        legend = true,
        legendfontsize = 8,
        background_color = :white,
        foreground_color = :grey,
        framestyle = :box,
        grid = false,
        tickfontsize = 8,
        titlefontsize = 10,
    )

    # Save the detailed decomposition plot
    savefig(inflation_decomposition_detailed, output_folder * "/inflation_decomposition_detailed.png")

    # Return both plots for displaying in notebooks if needed
    return (inflation_decomposition, inflation_decomposition_detailed)
end

generate_inflation_decomposition("netherlands"; empirical_distribution = false, abmx = false, unconditional_forecasts = true, year_i = 2019, quarter = 4)
generate_inflation_decomposition("netherlands"; empirical_distribution = false, abmx = true, unconditional_forecasts = true, year_i = 2019, quarter = 4)