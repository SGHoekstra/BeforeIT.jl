# # Essential use of BeforeIT

# We start by importing the BeforeIT library and other useful libraries.
import BeforeIT as Bit

using FileIO, Plots, Graphs, GraphMakie, CairoMakie, NetworkLayout, GeometryBasics
using DataStructures, LinearAlgebra

# Additional libraries for multigraph support
using Multigraphs
# Import SimpleEdge explicitly
using Graphs: SimpleEdge

# We then initialise the model loading some precomputed set of parameters and by specifying a number of epochs.
# In another tutorial we will illustrate how to compute parameters and initial conditions.

year_i = 2019
quarter = 4

country = "netherlands"
#country = "italy"

parameters = load(pwd() * "/data/$(country)/parameters/"* string(year_i) *"Q"* string(quarter) *".jld2");
initial_conditions = load(pwd() * "/data/$(country)/initial_conditions/"* string(year_i) *"Q"* string(quarter) *".jld2");

# To run mu
# We can now initialise the model, by specifying in advance the maximum number of epochs.

T = 12
model = Bit.init_model(parameters, initial_conditions, T; conditional_forecast = false)

# Simulate for a few steps if needed
for t_step in 1:1
    Bit.step!(model)
    # Bit.update_data!(data_results, model) # model.agg.t is now advanced
end

# --- Calculate flows at the current time step 't' ---
# model.agg.t now holds the current simulation time
current_flows = Bit.calculate_flows_at_t(model) # Assuming you have this function from previous step
Bit.check_stock_flow_consistency(model, current_flows)

# Define MultigraphWrap type for visualization that handles multiple edges between the same nodes
struct MultigraphWrap{T} <: AbstractGraph{T}
    g::DiMultigraph{T}                   # Underlying DiMultigraph, mainly for basic graph properties (nv, vertices)
    edge_magnitudes::Dict{Int, Float64}  # Map flow_idx to magnitude
    edge_labels::Dict{Int, String}       # Map flow_idx to label
    edge_sources::Dict{Int, Int}         # Map flow_idx to source node index
    edge_targets::Dict{Int, Int}         # Map flow_idx to target node index
    
    # For gen_distances: maps (src,dst) pairs to their list of expanded_edge_indices
    edge_indices::Dict{Tuple{Int,Int}, Vector{Int}}  
    
    # Tracking the expanded edges (SimpleEdge) returned by the edges() function
    expanded_edges::Vector{SimpleEdge{T}}
    
    # New map: expanded_edge_idx -> original_flow_idx
    map_expanded_idx_to_flow_idx::Dict{Int, Int} 
end

# Initialize with defaults
function MultigraphWrap(g::DiMultigraph{T}) where T
    edge_magnitudes = Dict{Int, Float64}()
    edge_labels = Dict{Int, String}()
    edge_sources = Dict{Int, Int}()
    edge_targets = Dict{Int, Int}()
    edge_indices = Dict{Tuple{Int,Int}, Vector{Int}}()
    expanded_edges = SimpleEdge{T}[]
    map_expanded_idx_to_flow_idx = Dict{Int, Int}() # Initialize new map
    
    return MultigraphWrap(g, edge_magnitudes, edge_labels, edge_sources, edge_targets, 
                         edge_indices, expanded_edges, map_expanded_idx_to_flow_idx)
end

# Implementation required methods for AbstractGraph interface
Base.eltype(g::MultigraphWrap) = eltype(g.g)
Graphs.edgetype(g::MultigraphWrap) = edgetype(g.g) # SimpleEdge{T} after expansion
Graphs.has_edge(g::MultigraphWrap, s, d) = has_edge(g.g, s, d) # delegates to underlying graph, indicates if *any* flow exists
Graphs.has_vertex(g::MultigraphWrap, i) = has_vertex(g.g, i)
Graphs.inneighbors(g::MultigraphWrap{T}, i) where T = inneighbors(g.g, i) # delegates, shows structural neighbors
Graphs.outneighbors(g::MultigraphWrap{T}, i) where T = outneighbors(g.g, i) # delegates, shows structural neighbors
Graphs.ne(g::MultigraphWrap) = length(g.expanded_edges)  # Use expanded edge count
Graphs.nv(g::MultigraphWrap) = nv(g.g)
Graphs.vertices(g::MultigraphWrap) = vertices(g.g)
Graphs.is_directed(g::MultigraphWrap) = is_directed(g.g)
Graphs.is_directed(::Type{<:MultigraphWrap}) = true

# Custom edges function that creates unique SimpleEdges for all flows
function Graphs.edges(g::MultigraphWrap)
    if isempty(g.expanded_edges)
        # Clear possibly stale data if re-generating
        empty!(g.expanded_edges)
        empty!(g.edge_indices)
        empty!(g.map_expanded_idx_to_flow_idx)

        # Iterate through original flow indices (assuming they are keys in edge_magnitudes)
        # Sort to ensure consistent order if that matters, though map iteration order is not guaranteed
        # Using actual flow indices stored is more robust
        
        # Determine the set of all flow indices from the stored dictionaries
        # This assumes that if a flow_idx exists in one dict, it exists in others relevant for this.
        all_flow_indices = sort(collect(keys(g.edge_magnitudes))) # or edge_labels, etc.

        for flow_idx in all_flow_indices
            src = g.edge_sources[flow_idx]
            dst = g.edge_targets[flow_idx]
            
            # Create a new SimpleEdge for this flow
            push!(g.expanded_edges, SimpleEdge(src, dst))
            current_expanded_edge_idx = length(g.expanded_edges)
            
            # Store mapping from this expanded_edge_idx to its original_flow_idx
            g.map_expanded_idx_to_flow_idx[current_expanded_edge_idx] = flow_idx
            
            # Populate g.edge_indices for gen_distances: (src,dst) -> list of expanded_edge_indices
            pair = (src, dst)
            if !haskey(g.edge_indices, pair)
                g.edge_indices[pair] = Int[]
            end
            push!(g.edge_indices[pair], current_expanded_edge_idx)
        end
        
        println("Created $(length(g.expanded_edges)) expanded edges from $(length(all_flow_indices)) original flows.")
        println("map_expanded_idx_to_flow_idx populated with $(length(g.map_expanded_idx_to_flow_idx)) entries.")
        println("edge_indices (for gen_distances): $(g.edge_indices)")
    end
    
    return g.expanded_edges
end


# Function to generate curve distances for edges, similar to GraphCombinations.gen_distances
function gen_distances(g::MultigraphWrap, node_positions)
    # ensure expanded_edges and edge_indices are populated
    # collect(Graphs.edges(g)) # this is usually called before gen_distances

    # edgearray = g.expanded_edges # Use already populated expanded_edges
    # distances = zeros(length(edgearray))
    distances = zeros(ne(g)) # ne(g) is length(g.expanded_edges)
    
    # Track which edges connect the same pair of nodes using the edge_indices map
    # g.edge_indices maps (src,dst) node index pair to a list of EXPANDED edge indices
    for (pair, indices_for_pair) in g.edge_indices # indices_for_pair are expanded_edge_indices
        src_node_idx, dst_node_idx = pair
        num_edges_for_pair = length(indices_for_pair)
        
        # Skip single edges unless they're self-loops
        if num_edges_for_pair <= 1 && src_node_idx != dst_node_idx
            continue
        end
        
        # For self-loops or multiple edges between the same nodes
        for (i, expanded_idx) in enumerate(indices_for_pair) # i is 1-based count within this parallel bundle
            if src_node_idx == dst_node_idx
                # Self-loops: use large positive/negative values alternating
                if mod(i, 2) == 0
                    distances[expanded_idx] = -0.75 - (div(i,2)) * 0.5 
                else
                    distances[expanded_idx] = 0.75 + (div(i-1,2)) * 0.5
                end
                # println("  Self-loop $(src_node_idx)->$(dst_node_idx) #$i (expanded edge $expanded_idx): curve = $(distances[expanded_idx])")
            else
                # Regular multiple edges
                if mod(i, 2) == 0
                    distances[expanded_idx] = -0.5 - (div(i,2)) * 0.25
                else
                    distances[expanded_idx] = 0.5 + (div(i-1,2)) * 0.25
                end
                # println("  Edge $(src_node_idx)->$(dst_node_idx) #$i (expanded edge $expanded_idx): curve = $(distances[expanded_idx])")
            end
        end
    end
    
    return distances
end

# --- PREPARE DATA FOR DIRECTED MULTIGRAPH ---
node_names = ["Firms", "Households", "Bank", "Government", "Central Bank", "RoW"]
num_nodes = length(node_names)

# Mapping between abbreviated names and full names
abbrev_map = Dict(
    "F" => "Firms",
    "H" => "Households",
    "B" => "Bank",
    "G" => "Government",
    "CB" => "Central Bank",
    "R" => "RoW",
    "RoW" => "RoW"
)

# Better flow abbreviation dictionary (previous version)
flow_abbrev = Dict(
    "Wages" => "Wages", # GrossWages
    "Dividends" => "Div",
    "Consumption" => "Cons", # ConsumptionGoods
    "HouseholdInvestment" => "H.Inv", # Tax_on_HouseholdInvestment
    "CorporateTax" => "Corp.Tax",
    "ProductTax" => "Prod.Tax",
    "ProductionTax" => "Prod.Tax", # Note: same as ProductTax in abbrev
    "EmployerSocialContributions" => "Emp.SocCont",
    "GovernmentConsumption" => "Gov.Cons", # Not in sample output, usually G_to_F_ConsumptionGoods
    "SubsidiesOnProducts" => "Subs.Prod",
    "SubsidiesOnProduction" => "Subs.Prod", # Note: same as SubsidiesOnProducts
    "IncomeTaxLabor" => "Inc.Tax",
    "IncomeTaxDividendsFirms" => "Inc.Tax.DivF",
    "IncomeTaxDividendsBank" => "Inc.Tax.DivB",
    "EmployeeSocialContributions" => "Emp.SocCont", # Note: same as EmployerSocialContributions
    "VAT" => "VAT", # VAT_on_Consumption
    "InvestmentTax" => "Inv.Tax", # Tax_on_HouseholdInvestment, F_to_G_InvestmentTax (if exists)
    "SocialBenefits" => "Soc.Ben",
    "NewLoans" => "New.Loans", # NewLoansGranted
    "LoanPrincipalRepayment" => "Loan.Repay",
    "LoanInterest" => "Loan.Int",
    "InterestOnFirmDeposits" => "Int.DepF",
    "InterestOnHouseholdDeposits" => "Int.DepH",
    "InterestOnHouseholdOverdrafts" => "Int.Ovrd",
    "Exports" => "Exports",
    "Imports" => "Imports",
    "InterestOnGovDebt" => "Int.Gov.Debt",
    "InterestOnReserves" => "Int.Res", # InterestOnBankReserves
    "InterestOnAdvances" => "Int.Adv", # InterestOnBankAdvances
    "NetGovBorrowing" => "Gov.Borrow",
    "FirmInvestmentGoods" => "Firm.Inv", # InvestmentGoodPurchases (F->F)
    "IntermediateGoods" => "Interm.Goods", # IntermediateGoodPurchases (F->F)
    "BankDividends" => "Bank Div",
    "BankCorporateTax" => "Bank Corp.Tax",
    "ExportTax" => "Export Tax",
    # Adding specific keys from output if missing
    "ConsumptionGoods" => "Cons",
    "InvestmentGoodPurchases" => "F.Inv", # Firms self-loop
    "IntermediateGoodPurchases" => "Interm.Goods", # Firms self-loop
    "GrossWages" => "Wages",
    "Tax_on_HouseholdInvestment" => "H.Inv.Tax",
    "VAT_on_Consumption" => "VAT",
    "InterestOnBankReserves" => "Int.Res",
    "InterestOnBankAdvances" => "Int.Adv",
    "NewLoansGranted" => "New Loans",
    "ProfitTransfer" => "Profit CB->G" # CB_to_G_ProfitTransfer
)


# Process flows and get labels
println("Processing flows for visualization (t=$(model.agg.t)):")

# Create a DiMultigraph (from Multigraphs.jl) for basic structure (nodes)
# This mg will also have edges added to it to reflect structural connections for has_edge etc.
mg_underlying = DiMultigraph(num_nodes) 
mg_wrap = MultigraphWrap(mg_underlying) # Pass it to the wrapper

# Initialize the flow index counter (unique for each flow)
flow_idx_counter = 0 # Renamed from original_edge_idx for clarity

# Initialize the flow descriptions aggregation dictionary
flow_descriptions_agg = Dict{Tuple{String, String}, Vector{Tuple{String, Float64}}}()

# Debug node name resolution
println("\nNode name mappings:")
for (abbrev, full_name) in abbrev_map
    println("  $abbrev => $full_name")
end

println("\nNode indices in visualization:")
for (i, name) in enumerate(node_names)
    println("  $i => $name")
end

# Track firms self-loops for special handling (using their flow_idx)
firms_idx = findfirst(==("Firms"), node_names)
intermediate_goods_flow_idx = 0 # Will store the flow_idx for intermediate goods
investment_goods_flow_idx = 0   # Will store the flow_idx for firm investment goods

# Process all flows from the dictionary
for (flow_name_sym, magnitude) in pairs(current_flows)
    flow_name_str = String(flow_name_sym)

    if abs(magnitude) < 1e-3 # Skip zero/tiny flows
        # println("  SKIPPING: $flow_name_str (magnitude: $magnitude)")
        continue
    end

    parts = split(flow_name_str, "_to_")
    if length(parts) != 2
        println("  SKIPPING: $flow_name_str (invalid format)")
        continue
    end

    source_abbrev = String(parts[1])
    target_parts = split(parts[2], "_", limit=2)

    if length(target_parts) != 2
        println("  SKIPPING: $flow_name_str (invalid target format)")
        continue
    end

    target_abbrev = String(target_parts[1])
    flow_description_key = target_parts[2] 

    source_name = get(abbrev_map, source_abbrev, "Unknown")
    target_name = get(abbrev_map, target_abbrev, "Unknown")

    if source_name == "Unknown" || target_name == "Unknown"
        println("  SKIPPING: $flow_name_str (unknown source/target: $source_abbrev/$target_abbrev)")
        continue
    end

    src_idx_num = findfirst(==(source_name), node_names)
    dst_idx_num = findfirst(==(target_name), node_names)

    # println("  Processing: $flow_name_str => $source_name ($src_idx_num) -> $target_name ($dst_idx_num)")

    if isnothing(src_idx_num) || isnothing(dst_idx_num)
        println("  SKIPPING: $flow_name_str (invalid node indices)")
        continue
    end
    
    if src_idx_num == dst_idx_num && source_name != "Firms" # Skip non-Firms self-loops
        # println("  SKIPPING: $flow_name_str (non-Firms self-loop)")
        continue
    end

    flow_idx_counter += 1 # This is the unique ID for this flow

    # Add edge to the underlying DiMultigraph (mg_underlying)
    # This updates multiplicities in mg_underlying, useful for has_edge, in/outneighbors if delegated.
    add_edge!(mg_wrap.g, src_idx_num, dst_idx_num, 1)

    # Store flow properties in MultigraphWrap, keyed by flow_idx_counter
    mg_wrap.edge_magnitudes[flow_idx_counter] = abs(magnitude)
    mg_wrap.edge_sources[flow_idx_counter] = src_idx_num
    mg_wrap.edge_targets[flow_idx_counter] = dst_idx_num

    flow_description_full = replace(flow_description_key, "_" => " ")
    short_desc = get(flow_abbrev, flow_description_key, "")
    if short_desc == "" # Fallback if key not in flow_abbrev
        # Try matching parts of the key
        for (k_fb, v_fb) in flow_abbrev
            if occursin(k_fb, flow_description_key) || occursin(k_fb, flow_description_full)
                short_desc = v_fb
                break
            end
        end
        if short_desc == "" # Ultimate fallback
            short_desc = length(flow_description_full) > 12 ? flow_description_full[1:10] * "..." : flow_description_full
        end
    end

    mag_value = round(Int, abs(magnitude))
    mag_str = if mag_value >= 1_000_000 string(round(mag_value / 1_000_000, digits=1)) * "M"
              elseif mag_value >= 1000 string(round(Int, mag_value / 1000)) * "K"
              else string(mag_value) end
    
    mg_wrap.edge_labels[flow_idx_counter] = "$short_desc: $mag_str"
    # println("  ADDED FLOW $(flow_idx_counter): $source_name -> $target_name ($short_desc: $mag_str) [From: $flow_name_str]")

    if src_idx_num == firms_idx && dst_idx_num == firms_idx
        if occursin("IntermediateGood", flow_description_key) || occursin("Interm.Goods", short_desc)
            intermediate_goods_flow_idx = flow_idx_counter
            # println("    Identified as intermediate goods self-loop (flow #$flow_idx_counter)")
        elseif occursin("InvestmentGood", flow_description_key) || occursin("Firm.Inv", short_desc) || occursin("F.Inv", short_desc)
            investment_goods_flow_idx = flow_idx_counter
            # println("    Identified as firm investment goods self-loop (flow #$flow_idx_counter)")
        end
    end

    pair_agg = (source_name, target_name)
    if !haskey(flow_descriptions_agg, pair_agg)
        flow_descriptions_agg[pair_agg] = []
    end
    push!(flow_descriptions_agg[pair_agg], (flow_description_full, abs(magnitude)))
end

println("\nMultigraph construction complete:")
println("  Nodes: $(nv(mg_wrap))")
# Number of structural edges in the underlying DiMultigraph (mg_wrap.g)
println("  Structural Edges in DiMultigraph: $(ne(mg_wrap.g))") 
println("  Total individual flows processed: $(flow_idx_counter)")


# --- CUSTOM LAYOUT ---
custom_positions = Dict(
    "Firms" => [5.0, 0.0], "Households" => [0.0, -5.0], "Bank" => [0.0, 5.0],        
    "Government" => [-5.0, 0.0], "Central Bank" => [-5.0, 5.0], "RoW" => [5.0, -5.0]           
)
fixed_layout = [Point2f(custom_positions[name]) for name in node_names] # Point2f for GraphMakie

# --- IMPROVED VISUALIZATION ---
fig = Figure(size = (1800, 1400))
ax = Axis(fig[1,1], title = "Multidigraph Flow Visualization (t=$(model.agg.t))")

# Force Graphs.edges(mg_wrap) to be called to initialize expanded_edges and mappings
collected_expanded_edges = collect(Graphs.edges(mg_wrap)) # This populates internal structures in mg_wrap
println("  Total Expanded Edges for plotting: $(ne(mg_wrap))")

curve_distances = gen_distances(mg_wrap, fixed_layout)

# println("\nFinal expanded edges in graph (showing first few):")
# for (i, e) in enumerate(collected_expanded_edges[1:min(5, end)])
#     src_name = node_names[e.src]
#     dst_name = node_names[e.dst]
#     flow_id = mg_wrap.map_expanded_idx_to_flow_idx[i]
#     println("  Exp. Edge $i: $src_name -> $dst_name (from flow #$flow_id)")
# end

direct_labels = String[]
direct_widths = Float64[]
direct_colors = []

# Iterate through expanded_edges (indices 1 to N_expanded)
for expanded_idx in 1:ne(mg_wrap) 
    original_flow_idx = mg_wrap.map_expanded_idx_to_flow_idx[expanded_idx] # Get original flow ID
    
    label = get(mg_wrap.edge_labels, original_flow_idx, "")
    magnitude = get(mg_wrap.edge_magnitudes, original_flow_idx, 0.0)
    
    width = 1.0
    min_width, max_width = 1.0, 12.0
    all_mags_values = collect(values(mg_wrap.edge_magnitudes))
    if !isempty(all_mags_values)
        pos_mags = filter(x -> x > 1e-9, all_mags_values)
        min_mag_val, max_mag_val = if !isempty(pos_mags): (minimum(pos_mags), maximum(pos_mags)) else (1.0,1.0) end
        
        if magnitude > 1e-9 && max_mag_val > min_mag_val
            width = min_width + (max_width - min_width) * (log1p(magnitude) - log1p(min_mag_val)) / (log1p(max_mag_val) - log1p(min_mag_val))
            width = clamp(width, min_width, max_width) # Ensure width is within bounds
        elseif magnitude > 1e-9 # Single positive magnitude or all same
             width = (min_width + max_width) / 2 
        end # else width remains 1.0 for zero/negative magnitudes (though we use abs(magnitude))
    end

    color = :slategray # Default color
    # Expanded edge's src/dst nodes
    # current_expanded_edge_simple = mg_wrap.expanded_edges[expanded_idx]
    # src_node_for_color = current_expanded_edge_simple.src
    # dst_node_for_color = current_expanded_edge_simple.dst
    src_node_for_color = mg_wrap.edge_sources[original_flow_idx]
    dst_node_for_color = mg_wrap.edge_targets[original_flow_idx]


    if src_node_for_color == firms_idx && dst_node_for_color == firms_idx # Firms self-loop
        if original_flow_idx == intermediate_goods_flow_idx
            color = :darkblue  # Intermediate goods
        elseif original_flow_idx == investment_goods_flow_idx
            color = :darkred   # Investment goods
        end
    elseif src_node_for_color == firms_idx && dst_node_for_color == findfirst(==("RoW"), node_names)
        color = :purple  # Exports
    elseif src_node_for_color == findfirst(==("RoW"), node_names) && dst_node_for_color == firms_idx
        color = :darkgreen  # Imports
    end
    
    push!(direct_labels, label)
    push!(direct_widths, max(1.0, width)) # Ensure minimum width
    push!(direct_colors, color)
    
    # println("  Expanded Edge $expanded_idx (Flow #$original_flow_idx) mapping: label=\"$label\", width=$(round(width, digits=2)), color=$color")
end

p = graphplot(mg_wrap, # Pass the MultigraphWrap instance
    layout = _ -> fixed_layout,
    nlabels = node_names,
    nlabels_align = [(:center, :center) for _ in 1:num_nodes], # Ensure this is a vector of tuples
    nlabels_distance = 70,
    nlabels_textsize = 36,
    node_size = [200 for _ in 1:num_nodes], # Ensure this is a vector
    node_color = [:lightskyblue, :lightgreen, :pink, :khaki, :lightcyan, :lavender],
    edge_width = direct_widths,
    edge_color = direct_colors,
    arrow_size = [25 for _ in 1:ne(mg_wrap)], # Ensure this is a vector if not scalar
    curve_distance = curve_distances,
    curve_distance_usage = true, # Important for making curve_distance work
    elabels = direct_labels,
    elabels_textsize = 32,
    elabels_distance = 15.0
)

# Configure self-loop attributes using expanded edge indices
# Find expanded indices for Firms self-loops using the map
intermediate_expanded_idx = 0
investment_expanded_idx = 0

if intermediate_goods_flow_idx > 0
    for (exp_idx, flow_idx) in mg_wrap.map_expanded_idx_to_flow_idx
        if flow_idx == intermediate_goods_flow_idx
            intermediate_expanded_idx = exp_idx
            break
        end
    end
end
if investment_goods_flow_idx > 0
     for (exp_idx, flow_idx) in mg_wrap.map_expanded_idx_to_flow_idx
        if flow_idx == investment_goods_flow_idx
            investment_expanded_idx = exp_idx
            break
        end
    end
end

# Apply self-loop customizations if found
if intermediate_expanded_idx > 0 || investment_expanded_idx > 0
    # Initialize DefaultDicts for self-loop properties
    # Note: GraphMakie might expect values for *all* edges if these are set globally,
    # or it might correctly use them only for actual self-loops.
    # For safety, let's create specific dictionaries for GraphMakie.
    # selfedge_size_dict = Dict{Int, Any}()
    # selfedge_direction_dict = Dict{Int, Any}()
    # selfedge_width_dict = Dict{Int, Any}()

    # if intermediate_expanded_idx > 0
    #     selfedge_size_dict[intermediate_expanded_idx] = 3.0
    #     selfedge_direction_dict[intermediate_expanded_idx] = Point2f(0.0, 1.0) # Up
    #     selfedge_width_dict[intermediate_expanded_idx] = 0.8*π
    #     println("  Configuring Intermediate Goods self-loop (expanded index $intermediate_expanded_idx)")
    # end
    # if investment_expanded_idx > 0
    #     selfedge_size_dict[investment_expanded_idx] = 1.5
    #     selfedge_direction_dict[investment_expanded_idx] = Point2f(0.0, -1.0) # Down
    #     selfedge_width_dict[investment_expanded_idx] = 0.5*π
    #     println("  Configuring Investment Goods self-loop (expanded index $investment_expanded_idx)")
    # end
    
    # p.selfedge_size = DefaultDict(Makie.automatic, selfedge_size_dict)
    # p.selfedge_direction = DefaultDict(Makie.automatic, selfedge_direction_dict)
    # p.selfedge_width = DefaultDict(Makie.automatic, selfedge_width_dict)
    # The selfedge_x attributes are often applied per-plot object, check GraphMakie docs for dynamic updates
    # For initial plot, pass them as keyword arguments if supported, or modify plot object `p`
    # This part seems to be tricky with GraphMakie if not setting for all self-loops or providing a vector.
    # The default automatic curving for self-loops via curve_distance might be sufficient.
    # If specific styling for these two self-loops is needed beyond color/width already handled,
    # this section might need more specific GraphMakie API usage.
    # For now, color and width are handled by direct_colors/direct_widths.
    # Curve distance already handles making them distinct.
    println("\nSelf-loops for Firms (if any) will be styled by color, width, and curve_distance.")
    if intermediate_expanded_idx > 0
         println("  Intermediate goods (expanded index $intermediate_expanded_idx) identified.")
    end
    if investment_expanded_idx > 0
        println("  Investment goods (expanded index $investment_expanded_idx) identified.")
    end

else
    println("\nNo specific Firms self-loops (Intermediate/Investment Goods) found or configured for special styling.")
end


# graphplot!(ax, p.plot, layout = _ -> fixed_layout) # p is already the plot object from graphplot
# This might not be needed if p = graphplot(...) already adds to current axis, or if p is the axis content.
# The return of graphplot(mg_wrap, ...) is usually the plot object itself.
# So, ax should contain p.

hidedecorations!(ax)
hidespines!(ax)

save("multiedge_dimultigraph_t$(model.agg.t).png", fig)
println("\nImproved multigraph visualization created: multiedge_dimultigraph_t$(model.agg.t).png")

# --- AGGREGATED VIEW WITH CORRECT DICTIONARY ---
fig2 = Figure(size = (1200, 1000))
ax2 = Axis(fig2[1,1], title = "Aggregated Flow Visualization (t=$(model.agg.t))")

agg_g = DiGraph(num_nodes) # SimpleDiGraph for aggregation
edge_weights_agg = Float64[]
edge_labels_agg = String[]

agg_flows = Dict{Tuple{String, String}, Float64}()
for (pair, flows_list) in flow_descriptions_agg
    agg_flows[pair] = sum(f[2] for f in flows_list)
end

# Sort for consistent edge ordering if SimpleGraph reorders
sorted_agg_flows = sort(collect(agg_flows), by=x->x[2], rev=true)

for (pair, total_mag) in sorted_agg_flows
    source_name, target_name = pair
    src_idx_agg = findfirst(==(source_name), node_names)
    dst_idx_agg = findfirst(==(target_name), node_names)
    
    if !isnothing(src_idx_agg) && !isnothing(dst_idx_agg) && src_idx_agg != dst_idx_agg # No self-loops in agg view
        add_edge!(agg_g, src_idx_agg, dst_idx_agg) # Add to SimpleDiGraph
        push!(edge_weights_agg, total_mag)
        
        total_val_int = round(Int, total_mag)
        total_str_agg = if total_val_int >= 1000 string(round(Int, total_val_int / 1000)) * "K"
                        else string(total_val_int) end
        push!(edge_labels_agg, total_str_agg)
    end
end

scaled_edge_widths_agg = if !isempty(edge_weights_agg)
    min_w_agg, max_w_agg = minimum(edge_weights_agg), maximum(edge_weights_agg)
    if min_w_agg == max_w_agg # all same weight
        [5.0 for _ in edge_weights_agg] # mid-range const width
    else
        2.0 .+ 10.0 .* (edge_weights_agg .- min_w_agg) ./ (max_w_agg - min_w_agg)
    end
else
    Float64[]
end

graphplot!(ax2, agg_g, # Plotting the SimpleDiGraph
    layout = _ -> fixed_layout,
    nlabels = node_names,
    nlabels_align = [(:center, :center) for _ in 1:num_nodes],
    nlabels_distance = 35,
    nlabels_textsize = 18,
    node_size = [60 for _ in 1:num_nodes],
    node_color = [:lightskyblue, :lightgreen, :pink, :khaki, :lightcyan, :lavender],
    edge_width = scaled_edge_widths_agg,
    edge_color = :slategray,
    arrow_size = [20 for _ in 1:Graphs.ne(agg_g)], # Match number of edges in agg_g
    elabels = edge_labels_agg,
    elabels_textsize = 18
)

hidedecorations!(ax2)
hidespines!(ax2)

save("multiedge_aggregated_t$(model.agg.t).png", fig2)
println("Aggregated view created: multiedge_aggregated_t$(model.agg.t).png")

println("\nBoth visualizations completed successfully.")