import BeforeIT as Bit
using FileIO, Plots, Statistics

function generate_inflation_histograms(
    country::String = "netherlands";
    empirical_distribution::Bool = false,
    abmx::Bool = false,
    unconditional_forecasts::Bool = false,
    year_i::Int = 2019,
    quarter::Int = 4,
)
    """
    Generate histograms of inflation across different sectors.

    Parameters:
    - country: The country to analyze
    - empirical_distribution: Whether to use empirical distribution
    - abmx: Whether to use abmx model (vs abm)
    - unconditional_forecasts: Whether to use unconditional forecasts
    - year_i: Year for analysis
    - quarter: Quarter for analysis
    """


    # Build folder path based on parameters
    model_folder =
        (empirical_distribution ? "/empirical" : "/calibrated") *
        (abmx ? (unconditional_forecasts ? "/abmx_uf" : "/abmx") : "/abm")

    # Define the output structure based on parameters (without long_run)
    output_folder =
        "./analysis/figs/" *
        country *
        (empirical_distribution ? "/empirical" : "/calibrated") *
        (abmx ? (unconditional_forecasts ? "/abmx_uf" : "/abmx") : "/abm")

    # Create output directory
    mkpath(output_folder)

    # Load parameters and initial conditions
    parameter_path =
        pwd() *
        "/src/utils/parameters_initial_conditions_data/$(country)/parameters/" *
        string(year_i) *
        "Q" *
        string(quarter) *
        ".jld2"
    initial_conditions_path =
        pwd() *
        "/src/utils/parameters_initial_conditions_data/$(country)/initial_conditions/" *
        string(year_i) *
        "Q" *
        string(quarter) *
        ".jld2"

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
    model = Bit.init_model(parameters, initial_conditions, T)

    if abmx
        model.prop.theta_UNION = 0.31
        model.prop.phi_DP = 0.83
        model.prop.phi_F_Q = 0.15
    else
        model.prop.theta_UNION = 0.25
        model.prop.phi_DP = 0.08
        model.prop.phi_F_Q = 0.01
    end

    data = Bit.ensemblerun(model, 1, abmx = abmx, conditional_forecast = !unconditional_forecasts)

    # Create a 3×3 grid of subplots
    inflation_histograms = plot(layout = (3, 3), size = (900, 800))

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

    colors =
        [:dodgerblue, :crimson, :forestgreen, :darkorchid, :darkorange, :turquoise, :hotpink, :sienna, :darkslategray]

    for (i, mask) in enumerate(masks)
        # Get data for this group
        group_data =
            (1 .+ data.demand_pull_inflation[T + 1, mask]) .* (1 .+ data.cost_push_inflation[T + 1, mask]) .*
            (1 .+ data.aggregate_inflation_expectations[T + 1]) .- 1

        # Remove outliers based on standard deviation
        group_data = group_data[abs.(group_data .- mean(group_data)) .< 1 * std(group_data)]

        # Ensure consistent binning and bin width across all plots
        histogram!(
            inflation_histograms,
            group_data,
            bins = 100,
            title = plot_titles[i],
            color = colors[i],
            fillalpha = 0.8,
            linecolor = :white,
            linewidth = 0.5,
            subplot = i,
            #xlims=(-0.1, 0.1),
            grid = true,
            framestyle = :box,
        )
    end

    # Improve overall appearance
    plot!(
        inflation_histograms,
        background_color = :white,
        foreground_color = :grey,
        guidefont = font(10),
        tickfont = font(8),
        legend = false,
    )

    # Save the plot
    savefig(inflation_histograms, output_folder * "/inflation_histograms.png")

    return inflation_histograms
end

generate_inflation_histograms("netherlands"; empirical_distribution = false, abmx = false, unconditional_forecasts = true, year_i = 2019, quarter = 4);
generate_inflation_histograms("netherlands"; empirical_distribution = false, abmx = true, unconditional_forecasts = true, year_i = 2019, quarter = 4);
