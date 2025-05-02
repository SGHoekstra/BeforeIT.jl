import BeforeIT as Bit

using FileIO, Plots, Statistics

year_i = 2019
quarter = 4

country = "netherlands"
#country = "italy"

parameters = load(pwd() * "/src/utils/parameters_initial_conditions_data/$(country)/parameters/"* string(year_i) *"Q"* string(quarter) *".jld2");
initial_conditions = load(pwd() * "/src/utils/parameters_initial_conditions_data/$(country)/initial_conditions/"* string(year_i) *"Q"* string(quarter) *".jld2");

T = 16
n_runs = 32

model = Bit.init_model(parameters, initial_conditions, T)
data_vector = Bit.ensemblerun(model, n_runs; multi_threading = true)

cost_push_inflation = reshape(data_vector.cost_push_inflation,T+1,model.prop.I,n_runs)
demand_pull_inflation = reshape(data_vector.demand_pull_inflation,T+1,model.prop.I,n_runs)
cost_push_inflation_labour = reshape(data_vector.cost_push_inflation_labour,T+1,model.prop.I,n_runs)
cost_push_inflation_capital = reshape(data_vector.cost_push_inflation_capital,T+1,model.prop.I,n_runs)
cost_push_inflation_material = reshape(data_vector.cost_push_inflation_material,T+1,model.prop.I,n_runs)


inflation_decomposition = plot(layout=(3,3), size=(900, 800))

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
variable_names = ["Cost-Push", "Demand-Pull", "Expectations"]
colors = [:darkblue, :mediumseagreen, :darkorange]


for (i, mask) in enumerate(masks)
    # Extract inflation components
    cost_push = vec(mean(mean(cost_push_inflation[:,mask,:],dims=2),dims=3))
    demand_pull = vec(mean(mean(demand_pull_inflation[:,mask,:],dims=2),dims=3))
    expectations = vec(mean(data_vector.aggregate_inflation_expectations,dims=2))
    
    # Stack the data
    stacked_data = hcat(cost_push, demand_pull, expectations)
    
    # Calculate the total (sum of all components)
    total_inflation = sum(stacked_data, dims=2)
    
    # Create bar plot
    groupedbar!(inflation_decomposition,
               stacked_data,
               bar_position = :stack,
               subplot=i,
               title=plot_titles[i],
               label=permutedims(variable_names),
               color=permutedims(colors),
               linewidth=0)
    
    # Add dashed line for total inflation
    plot!(inflation_decomposition,
          1:length(total_inflation), 
          total_inflation, 
          subplot=i,
          linewidth=2, 
          linestyle=:dash, 
          color=:black, 
          label=(i==1 ? "Total Inflation" : nothing))  # Only add label on first subplot
end

# Enhance the appearance
plot!(inflation_decomposition, 
      legend=true,
      legendfontsize=8,
      background_color=:white,
      foreground_color=:grey,
      framestyle=:box,
      grid=false,
      tickfontsize=8,
      titlefontsize=10)

inflation_decomposition

inflation_decomposition_detailed = plot(layout=(3,3), size=(900, 800))


variable_names = ["Labour cost", "Capital cost", "Material cost", "Demand Pull", "Expectations"]
colors = [:cornflowerblue, :indianred, :goldenrod, :mediumseagreen, :mediumpurple]

for (i, mask) in enumerate(masks)
    # Extract inflation components
    cost_push_labour = vec(mean(mean(cost_push_inflation_labour[:,mask,:],dims=2),dims=3))
    cost_push_capital = vec(mean(mean(cost_push_inflation_capital[:,mask,:],dims=2),dims=3))
    cost_push_material = vec(mean(mean(cost_push_inflation_material[:,mask,:],dims=2),dims=3))
    demand_pull = vec(mean(mean(demand_pull_inflation[:,mask,:],dims=2),dims=3))
    expectations = vec(mean(data_vector.aggregate_inflation_expectations,dims=2))
    
    # Stack the data
    stacked_data = hcat(cost_push_labour, cost_push_capital, cost_push_material, demand_pull, expectations)
    
    # Calculate the total (sum of all components)
    total_inflation = sum(stacked_data, dims=2)
    
    # Create bar plot
    groupedbar!(inflation_decomposition_detailed,
               stacked_data,
               bar_position = :stack,
               subplot=i,
               title=plot_titles[i],
               label=permutedims(variable_names),
               color=permutedims(colors),
               linewidth=0)
    
    # Add dashed line for total inflation
    plot!(inflation_decomposition_detailed,
          1:length(total_inflation), 
          total_inflation, 
          subplot=i,
          linewidth=2, 
          linestyle=:dash, 
          color=:black, 
          label=(i==1 ? "Total Inflation" : nothing))  # Only add label on first subplot
end

# Enhance the appearance
plot!(inflation_decomposition_detailed, 
      legend=true,
      legendfontsize=8,
      background_color=:white,
      foreground_color=:grey,
      framestyle=:box,
      grid=false,
      tickfontsize=8,
      titlefontsize=10)

      inflation_decomposition_detailed


