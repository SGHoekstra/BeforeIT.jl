using GraphCombinations, GraphMakie, CairoMakie
import GraphMakie.NetworkLayout as NL
import GraphCombinations as GC

# Function for plot settings (from the example)
function pltkwargs(g)
    (;
        layout=NL.Align(NL.Spring()),
        curve_distance=GC.gen_distances(g),
        curve_distance_usage=true,
    )
end

# Test with multiple self-loops on the same vertex
function test_multiple_selfloops()
    # Define edges with multiple self-loops on vertex 3
    edges = [
        1 => 3,    # Edge from vertex 1 to 3
        2 => 3,    # Edge from vertex 2 to 3
        3 => 3,    # Self-loop on vertex 3
        3 => 3,    # Self-loop on vertex 3 (Does not)
        3 => 4,    # Edge from vertex 3 to 4
        3 => 4,    # Edge from vertex 3 to 4
        4 => 3,    # Edge from vertex 4 to 3
        4 => 3,    # Edge from vertex 4 to 3
        4 => 4     # Self-loop on vertex 4
    ]
    
    # Build the graph
    g = GC.build_graph(edges)
    
    # Create a figure
    f = Figure(size=(800, 600))
    ax = Axis(f[1, 1], title="Testing Multiple Self-Loops")
    
    # Create the graph plot
    graphplot!(ax, g; pltkwargs(g)...)
    
    # Add labels for clarity
    ax.title = "GraphCombinations Multiple Self-Loops Test"
    
    hidedecorations!(ax)
    hidespines!(ax)
    ax.aspect = DataAspect()
    
    # Save and return the figure
    save("graphcombinations_multiple_selfloops.png", f)
    return f
end

# Run the test
test_multiple_selfloops()