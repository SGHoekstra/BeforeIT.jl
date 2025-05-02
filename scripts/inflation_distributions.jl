import BeforeIT as Bit

using FileIO, Plots, Statistics

year_i = 2019
quarter = 4

country = "netherlands"
#country = "italy"

parameters = load(pwd() * "/src/utils/parameters_initial_conditions_data/$(country)/parameters/"* string(year_i) *"Q"* string(quarter) *".jld2");
initial_conditions = load(pwd() * "/src/utils/parameters_initial_conditions_data/$(country)/initial_conditions/"* string(year_i) *"Q"* string(quarter) *".jld2");

T = 16
model = Bit.init_model(parameters, initial_conditions, T)
data = Bit.ensemblerun(model, 1)


# Create a 3×3 grid of subplots
inflation_histograms = plot(layout=(3,3), size=(900, 800))

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

plot_titles = [
    "A",
    "B,C,D,and E",
    "F",
    "G,F,I, and J",
    "K",
    "L",
    "M and N",
    "O,P and Q",
    "R and S"
]

colors = [:dodgerblue, :crimson, :forestgreen, :darkorchid, :darkorange, 
          :turquoise, :hotpink, :sienna, :darkslategray]

for (i, mask) in enumerate(masks)
    # Get data for this group
    group_data = (1 .+ data.demand_pull_inflation[T+1,mask]) .*
                 (1 .+ data.cost_push_inflation[T+1,mask]) .*
                 (1 .+ data.aggregate_inflation_expectations[T+1]) .- 1
    
    # Remove outliers based on standard deviation
    group_data = group_data[abs.(group_data .- mean(group_data)) .< 1 * std(group_data)]

    # Ensure consistent binning and bin width across all plots
    histogram!(inflation_histograms, group_data, 
              bins=100,
              title=plot_titles[i], 
              color=colors[i],
              fillalpha=0.8,
              linecolor=:white,
              linewidth=0.5,
              subplot=i,
              #xlims=(-0.1, 0.1),
              grid=true,
              framestyle=:box)
end


# Improve overall appearance
plot!(inflation_histograms, 
      background_color=:white,
      foreground_color=:grey,
      guidefont=font(10),
      tickfont=font(8),
      legend=false)

inflation_histograms


