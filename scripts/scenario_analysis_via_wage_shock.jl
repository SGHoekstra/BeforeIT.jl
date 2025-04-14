# # Scenario analysis via wage shock

import BeforeIT as Bit
import StatsBase: mean, std
using Plots, FileIO, StatsPlots

year_i = 2010
quarter = 4

parameters = load(pwd() * "/src/utils/parameters_initial_conditions_data/netherlands/parameters/"* string(year_i) *"Q"* string(quarter) *".jld2");
initial_conditions = load(pwd() * "/src/utils/parameters_initial_conditions_data/netherlands/initial_conditions/"* string(year_i) *"Q"* string(quarter) *".jld2");

# Initialise the model and the data collector

T = 16
model = Bit.init_model(parameters, initial_conditions, T);

# Simulate the baseline model for T quarters, N_reps times, and collect the data

N_reps = 64
data_vec_baseline = Bit.ensemblerun(model, N_reps)

# Now, apply a shock to the model and simulate it again.
# A shock is simply a function that takes the model and changes some of
# its parameters for a specific time period.
# We do this by first defining a "struct" with useful attributes.
# For example, we can define an productivity and a consumption shock with the following structs


struct WageShock
    wage_multiplier::Float64    # wage multiplier
    shock_time::Int
end

# and then by making the structs callable functions that change the parameters of the model,
# this is done in Julia using the syntax below

# A permanent change in the wage by the factor s.wage_multiplier

function (s::WageShock)(model::Bit.Model)
   if model.agg.t == s.shock_time
        model.firms.w_bar_i = model.firms.w_bar_i .* s.wage_multiplier
   end
end


wage_shock = WageShock(1.04,4)


# Simulate the model with the shock

data_vec_shocked = Bit.ensemblerun(model, N_reps; shock = wage_shock)

# Compute mean and standard error of GDP for the baseline and shocked simulations

mean_gdp_baseline = mean(data_vec_baseline.real_gdp, dims = 2)
mean_gdp_shocked = mean(data_vec_shocked.real_gdp, dims = 2)
sem_gdp_baseline = std(data_vec_baseline.real_gdp, dims = 2) / sqrt(N_reps)
sem_gdp_shocked = std(data_vec_shocked.real_gdp, dims = 2) / sqrt(N_reps)

mean_inflation_baseline = mean(data_vec_baseline.nominal_gdp./data_vec_baseline.real_gdp, dims = 2)
mean_inflation_shocked = mean(data_vec_shocked.nominal_gdp./data_vec_shocked.real_gdp, dims = 2)
sem_inflation_baseline = std(data_vec_baseline.nominal_gdp./data_vec_baseline.real_gdp, dims = 2) / sqrt(N_reps)
sem_inflation_shocked = std(data_vec_shocked.nominal_gdp./data_vec_shocked.real_gdp, dims = 2) / sqrt(N_reps)

mean_wages_baseline = mean(data_vec_baseline.wages, dims = 2)
mean_wages_shocked = mean(data_vec_shocked.wages, dims = 2)
sem_wages_baseline = std(data_vec_baseline.wages, dims = 2) / sqrt(N_reps)
sem_wages_shocked = std(data_vec_shocked.wages, dims = 2) / sqrt(N_reps)

# Compute the ratio of shocked to baseline GDP

gdp_ratio = mean_gdp_shocked ./ mean_gdp_baseline

inflation_ratio = mean_inflation_shocked ./ mean_inflation_baseline

wages_ratio = mean_wages_shocked ./ mean_wages_baseline


# the standard error on a ratio of two variables is computed with the error propagation formula

sem_gdp_ratio = gdp_ratio .* ((sem_gdp_baseline ./ mean_gdp_baseline).^2 .+ (sem_gdp_shocked ./ mean_gdp_shocked).^2).^0.5

sem_inflation_ratio = inflation_ratio .* ((sem_inflation_baseline ./ mean_inflation_baseline).^2 .+ (sem_inflation_shocked ./ mean_inflation_shocked).^2).^0.5

sem_wages_ratio = wages_ratio .* ((sem_wages_baseline ./ mean_wages_baseline).^2 .+ (sem_wages_shocked ./ mean_wages_shocked).^2).^0.5

# Finally, we can plot the impulse response curve

plot(
    1:T+1,
    gdp_ratio,
    ribbon = sem_gdp_ratio,
    fillalpha = 0.2,
    label = "",
    xlabel = "quarters",
    ylabel = "GDP change",
)

plot(
    1:T+1,
    inflation_ratio,
    ribbon = sem_inflation_ratio,
    fillalpha = 0.2,
    label = "",
    xlabel = "quarters",
    ylabel = "inflation change",
)

plot(
    1:T+1,
    wages_ratio,
    ribbon = sem_wages_ratio,
    fillalpha = 0.2,
    label = "",
    xlabel = "quarters",
    ylabel = "wage change",
)

# We can save the figure using: savefig("gdp_shock.png")

cost_push_inflation_b = reshape(data_vec_baseline.cost_push_inflation,T+1,model.prop.I,N_reps)
demand_pull_inflation_b = reshape(data_vec_baseline.demand_pull_inflation,T+1,model.prop.I,N_reps)
cost_push_inflation_labour_b = reshape(data_vec_baseline.cost_push_inflation_labour,T+1,model.prop.I,N_reps)
cost_push_inflation_capital_b = reshape(data_vec_baseline.cost_push_inflation_capital,T+1,model.prop.I,N_reps)
cost_push_inflation_material_b = reshape(data_vec_baseline.cost_push_inflation_material,T+1,model.prop.I,N_reps)


cost_push_inflation_s = reshape(data_vec_shocked.cost_push_inflation,T+1,model.prop.I,N_reps)
demand_pull_inflation_s = reshape(data_vec_shocked.demand_pull_inflation,T+1,model.prop.I,N_reps)
cost_push_inflation_labour_s = reshape(data_vec_shocked.cost_push_inflation_labour,T+1,model.prop.I,N_reps)
cost_push_inflation_capital_s = reshape(data_vec_shocked.cost_push_inflation_capital,T+1,model.prop.I,N_reps)
cost_push_inflation_material_s = reshape(data_vec_shocked.cost_push_inflation_material,T+1,model.prop.I,N_reps)


inflation_decomposition_detailed = plot(layout=(3,3), size=(900, 800))


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

variable_names = ["Labour cost", "Capital cost", "Material cost", "Demand Pull", "Expectations"]
colors = [:cornflowerblue, :indianred, :goldenrod, :mediumseagreen, :mediumpurple]

for (i, mask) in enumerate(masks)
    # Extract inflation components
    diff_cost_push_labour = vec(mean(mean(cost_push_inflation_labour_s[:,mask,:],dims=2),dims=3)) - vec(mean(mean(cost_push_inflation_labour_b[:,mask,:],dims=2),dims=3))
    diff_cost_push_capital = vec(mean(mean(cost_push_inflation_capital_s[:,mask,:],dims=2),dims=3)) - vec(mean(mean(cost_push_inflation_capital_b[:,mask,:],dims=2),dims=3))
    diff_cost_push_material = vec(mean(mean(cost_push_inflation_material_s[:,mask,:],dims=2),dims=3)) - vec(mean(mean(cost_push_inflation_material_b[:,mask,:],dims=2),dims=3))
    diff_cost_push = vec(mean(mean(cost_push_inflation_s[:,mask,:],dims=2),dims=3)) - vec(mean(mean(cost_push_inflation_b[:,mask,:],dims=2),dims=3))
    diff_demand_pull = vec(mean(mean(demand_pull_inflation_s[:,mask,:],dims=2),dims=3)) - vec(mean(mean(demand_pull_inflation_b[:,mask,:],dims=2),dims=3))
    diff_expectations = vec(mean(data_vec_shocked.aggregate_inflation_expectations,dims=2)) - vec(mean(data_vec_baseline.aggregate_inflation_expectations,dims=2))
    



    # Stack the data
    stacked_data = hcat(diff_cost_push_labour, diff_cost_push_capital, diff_cost_push_material, diff_demand_pull, diff_expectations)
    
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


# Extract inflation components
diff_cost_push_labour = vec(mean(mean(cost_push_inflation_labour_s[:,:,:],dims=2),dims=3)) - vec(mean(mean(cost_push_inflation_labour_b[:,:,:],dims=2),dims=3))
diff_cost_push_capital = vec(mean(mean(cost_push_inflation_capital_s[:,:,:],dims=2),dims=3)) - vec(mean(mean(cost_push_inflation_capital_b[:,:,:],dims=2),dims=3))
diff_cost_push_material = vec(mean(mean(cost_push_inflation_material_s[:,:,:],dims=2),dims=3)) - vec(mean(mean(cost_push_inflation_material_b[:,:,:],dims=2),dims=3))
diff_demand_pull = vec(mean(mean(demand_pull_inflation_s[:,:,:],dims=2),dims=3)) - vec(mean(mean(demand_pull_inflation_b[:,:,:],dims=2),dims=3))
diff_expectations = vec(mean(data_vec_shocked.aggregate_inflation_expectations,dims=2)) - vec(mean(data_vec_baseline.aggregate_inflation_expectations,dims=2))

# Stack the data
stacked_data = hcat(diff_cost_push_labour, diff_cost_push_capital, diff_cost_push_material, diff_demand_pull, diff_expectations)

# Calculate the total (sum of all components)
total_inflation = sum(stacked_data, dims=2)

# Create bar plot
inflation_decomposition_detailed_all = groupedbar(
            stacked_data,
            bar_position = :stack,
            label=permutedims(variable_names),
            color=permutedims(colors),
            linewidth=0)

# Add dashed line for total inflation
plot!(inflation_decomposition_detailed_all,
        1:length(total_inflation), 
        total_inflation, 
        linewidth=2, 
        linestyle=:dash, 
        color=:black)  # Only add label on first subplot
