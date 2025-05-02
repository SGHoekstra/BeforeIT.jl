using Graphs, SimpleWeightedGraphs, GraphPlot, Colors, Random, Compose

# For reproducibility
Random.seed!(123)

# Create a weighted matrix (using smaller size for visualization)
n = 100  # Full size as in original code
w = rand(n, n)

# Create a weighted directed graph
g = SimpleWeightedDiGraph(w)

println("Graph info:")
println("Vertices: ", nv(g))
println("Edges: ", ne(g))

# Create a smaller subgraph for better visualization
n_small = 10  # We'll visualize a 10×10 subgraph
w_small = w[1:n_small, 1:n_small]
g_small = SimpleWeightedDiGraph(w_small)

# Collect all edges and weights manually
edge_weights = []
for i in 1:n_small
    for j in 1:n_small
        if w_small[i, j] > 0
            push!(edge_weights, w_small[i, j])
        end
    end
end

# Normalize the weights
if length(edge_weights) > 0
    max_weight = maximum(edge_weights)
    min_weight = minimum(edge_weights)
    normalized_weights = (edge_weights .- min_weight) ./ (max_weight - min_weight)
    
    # Generate colors based on weights (blue to red gradient)
    edge_colors = [weighted_color_mean(w, colorant"blue", colorant"red") for w in normalized_weights]
    
    # Edge widths based on weights (1.0 to 5.0)
    edge_widths = 1.0 .+ 4.0 .* normalized_weights
else
    edge_colors = ["black"]
    edge_widths = [1.0]
end

# Plot the small graph with weighted edges
plt = gplot(g, 
    nodefillc="lightgreen",
    nodestrokec="black",
    nodesize=0.1)

# Display the plot
plt