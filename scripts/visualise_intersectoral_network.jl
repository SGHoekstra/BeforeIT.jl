# # Essential use of BeforeIT

# We start by importing the BeforeIT library and other useful libraries.
import BeforeIT as Bit

using FileIO, Graphs, GraphMakie, CairoMakie, NetworkLayout, GeometryBasics # Plots is not strictly needed if only using Makie
using LinearAlgebra # DataStructures not strictly needed for this version

# Additional libraries for multigraph support
using Multigraphs
# Import SimpleEdge explicitly
using Graphs: SimpleEdge
using Makie: RGBA # For color type

# We then initialise the model loading some precomputed set of parameters and by specifying a number of epochs.
year_i = 2019
quarter = 4
country = "netherlands"

parameters = load(pwd() * "/data/$(country)/parameters/"* string(year_i) *"Q"* string(quarter) *".jld2");
initial_conditions = load(pwd() * "/data/$(country)/initial_conditions/"* string(year_i) *"Q"* string(quarter) *".jld2");

T = 12 # Max number of epochs
model = Bit.init_model(parameters, initial_conditions, T; conditional_forecast = false)

# Simulate for a few steps
for t_step in 1:1 # Simulating for 1 step, model.agg.t will be 2 (if starts at 1 and increments)
    Bit.step!(model)
end

current_flows = Bit.calculate_flows_at_t(model)
Bit.check_stock_flow_consistency(model, current_flows)

# Define MultigraphWrap type
struct MultigraphWrap{T} <: AbstractGraph{T}
    g::DiMultigraph{T}
    edge_magnitudes::Dict{Int, Float64}
    edge_labels::Dict{Int, String}
    edge_sources::Dict{Int, Int}
    edge_targets::Dict{Int, Int}
    edge_indices::Dict{Tuple{Int,Int}, Vector{Int}}
    expanded_edges::Vector{SimpleEdge{T}}
    map_expanded_idx_to_flow_idx::Dict{Int, Int}
end

function MultigraphWrap(g::DiMultigraph{T}) where T
    MultigraphWrap(g, Dict{Int, Float64}(), Dict{Int, String}(), Dict{Int, Int}(), Dict{Int, Int}(),
                   Dict{Tuple{Int,Int}, Vector{Int}}(), SimpleEdge{T}[], Dict{Int, Int}())
end

Base.eltype(g::MultigraphWrap) = eltype(g.g)
Graphs.edgetype(g::MultigraphWrap) = edgetype(g.g)
Graphs.has_edge(g::MultigraphWrap, s, d) = has_edge(g.g, s, d)
Graphs.has_vertex(g::MultigraphWrap, i) = has_vertex(g.g, i)
Graphs.inneighbors(g::MultigraphWrap{T}, i) where T = inneighbors(g.g, i)
Graphs.outneighbors(g::MultigraphWrap{T}, i) where T = outneighbors(g.g, i)
Graphs.ne(g::MultigraphWrap) = length(g.expanded_edges)
Graphs.nv(g::MultigraphWrap) = nv(g.g)
Graphs.vertices(g::MultigraphWrap) = vertices(g.g)
Graphs.is_directed(g::MultigraphWrap) = is_directed(g.g)
Graphs.is_directed(::Type{<:MultigraphWrap}) = true

function Graphs.edges(g::MultigraphWrap)
    # Populate only if empty or if the number of stored flows has changed
    if isempty(g.expanded_edges) || length(g.expanded_edges) != length(g.edge_magnitudes)
        empty!(g.expanded_edges)
        empty!(g.edge_indices)
        empty!(g.map_expanded_idx_to_flow_idx)

        all_flow_indices = sort(collect(keys(g.edge_magnitudes)))

        for flow_idx in all_flow_indices
            if !haskey(g.edge_sources, flow_idx) || !haskey(g.edge_targets, flow_idx)
                continue
            end
            src = g.edge_sources[flow_idx]
            dst = g.edge_targets[flow_idx]

            push!(g.expanded_edges, SimpleEdge(src, dst))
            current_expanded_edge_idx = length(g.expanded_edges)
            g.map_expanded_idx_to_flow_idx[current_expanded_edge_idx] = flow_idx

            pair = (src, dst)
            if !haskey(g.edge_indices, pair)
                g.edge_indices[pair] = Int[]
            end
            push!(g.edge_indices[pair], current_expanded_edge_idx)
        end
    end
    return g.expanded_edges
end

function gen_distances(g::MultigraphWrap, node_positions) # node_positions not used here
    if isempty(g.expanded_edges)
        collect(Graphs.edges(g)) # Ensure population
    end
    num_exp_edges = ne(g)
    distances = zeros(num_exp_edges)
    if num_exp_edges == 0 return distances end

    for (pair, indices_for_pair) in g.edge_indices
        src_node_idx, dst_node_idx = pair
        num_edges_for_pair = length(indices_for_pair)
        if num_edges_for_pair == 0 continue end

        # Find node_names for better logic, assuming node_names is globally available or passed
        # For simplicity, using indices directly for Firms (1) and RoW (6)
        firms_node_idx = 1 # Assuming Firms is node 1
        row_node_idx = 6   # Assuming RoW is node 6

        if num_edges_for_pair <= 1 && src_node_idx != dst_node_idx
            if src_node_idx == row_node_idx && dst_node_idx == firms_node_idx && !isempty(indices_for_pair)
                 distances[indices_for_pair[1]] = 1.2 # Special curve for RoW -> Firms
            end
            continue
        end

        for (i, expanded_idx) in enumerate(indices_for_pair)
            if !(1 <= expanded_idx <= num_exp_edges) continue end
            if src_node_idx == dst_node_idx # Self-loops
                if i == 1 distances[expanded_idx] = 2.8  # Larger for first self-loop
                elseif i == 2 distances[expanded_idx] = -2.0 # Distinct for second
                else distances[expanded_idx] = (iseven(i) ? -1 : 1) * (2.0 + (div(i-1,2)) * 0.6) end
            else # Regular multiple edges
                curve_val_base = 0.6; curve_increment = 0.3
                curve_val = curve_val_base + (div(i-1,2)) * curve_increment
                distances[expanded_idx] = (iseven(i) ? -1 : 1) * curve_val
            end
        end
    end
    return distances
end

# --- PREPARE DATA ---
node_names = ["Firms", "Households", "Bank", "Government", "Central Bank", "RoW"]
num_nodes = length(node_names)
abbrev_map = Dict("F"=>"Firms", "H"=>"Households", "B"=>"Bank", "G"=>"Government", "CB"=>"Central Bank", "R"=>"RoW", "RoW"=>"RoW")
flow_abbrev = Dict(
    "Wages"=>"Wages", "GrossWages"=>"Wages", "Dividends"=>"Div", "Consumption"=>"Cons", "ConsumptionGoods"=>"Cons",
    "HouseholdInvestment"=>"H.Inv", "Tax_on_HouseholdInvestment"=>"H.Inv.Tax", "CorporateTax"=>"Corp.Tax",
    "ProductTax"=>"Prod.Tax", "ProductionTax"=>"Prod.Tax", "EmployerSocialContributions"=>"Emp.SocCont",
    "EmployeeSocialContributions"=>"Emp.SocCont", "GovernmentConsumption"=>"Gov.Cons", "SubsidiesOnProducts"=>"Subs.Prod",
    "SubsidiesOnProduction"=>"Subs.Prod", "IncomeTaxLabor"=>"Inc.Tax", "IncomeTaxDividendsFirms"=>"Inc.Tax.DivF",
    "IncomeTaxDividendsBank"=>"Inc.Tax.DivB", "VAT"=>"VAT", "VAT_on_Consumption"=>"VAT", "InvestmentTax"=>"Inv.Tax",
    "SocialBenefits"=>"Soc.Ben", "NewLoans"=>"New.Loans", "NewLoansGranted"=>"New Loans", "LoanPrincipalRepayment"=>"Loan.Repay",
    "LoanInterest"=>"Loan.Int", "InterestOnFirmDeposits"=>"Int.DepF", "InterestOnHouseholdDeposits"=>"Int.DepH",
    "InterestOnHouseholdOverdrafts"=>"Int.Ovrd", "Exports"=>"Exports", "Imports"=>"Imports", "InterestOnGovDebt"=>"Int.Gov.Debt",
    "InterestOnReserves"=>"Int.Res", "InterestOnBankReserves"=>"Int.Res", "InterestOnAdvances"=>"Int.Adv",
    "InterestOnBankAdvances"=>"Int.Adv", "NetGovBorrowing"=>"Gov.Borrow", "FirmInvestmentGoods"=>"F.Inv",
    "InvestmentGoodPurchases"=>"F.Inv", "IntermediateGoods"=>"Interm.Goods", "IntermediateGoodPurchases"=>"Interm.Goods",
    "BankDividends"=>"Bank Div", "BankCorporateTax"=>"Bank Corp.Tax", "ExportTax"=>"Export Tax", "ProfitTransfer"=>"Profit CB->G"
)

println("Processing flows for visualization (t=$(model.agg.t)):")
mg_underlying = DiMultigraph(num_nodes)
mg_wrap = MultigraphWrap(mg_underlying)
flow_idx_counter = 0
flow_descriptions_agg = Dict{Tuple{String, String}, Vector{Tuple{String, Float64}}}()
firms_idx = findfirst(==("Firms"), node_names) # Should be 1
intermediate_goods_flow_idx = 0
investment_goods_flow_idx = 0

for (flow_name_sym, magnitude) in pairs(current_flows)
    flow_name_str = String(flow_name_sym)
    if abs(magnitude) < 1e-3 continue end
    parts = split(flow_name_str, "_to_"); if length(parts) != 2 continue end
    source_abbrev = String(parts[1])
    target_parts = split(parts[2], "_", limit=2); if length(target_parts) != 2 continue end
    target_abbrev = String(target_parts[1]); flow_description_key = target_parts[2]
    source_name = get(abbrev_map, source_abbrev, "Unknown"); target_name = get(abbrev_map, target_abbrev, "Unknown")
    if source_name == "Unknown" || target_name == "Unknown" continue end
    src_idx_num = findfirst(==(source_name), node_names); dst_idx_num = findfirst(==(target_name), node_names)
    if isnothing(src_idx_num) || isnothing(dst_idx_num) continue end
    if src_idx_num == dst_idx_num && src_idx_num != firms_idx continue end # Allow only Firms self-loops

    flow_idx_counter += 1
    add_edge!(mg_wrap.g, src_idx_num, dst_idx_num, 1)
    mg_wrap.edge_magnitudes[flow_idx_counter] = abs(magnitude)
    mg_wrap.edge_sources[flow_idx_counter] = src_idx_num
    mg_wrap.edge_targets[flow_idx_counter] = dst_idx_num

    flow_desc_full = replace(flow_description_key, "_" => " ")
    short_desc = get(flow_abbrev, flow_description_key, "")
    if isempty(short_desc)
        for (k,v) in flow_abbrev if occursin(k, flow_description_key) || occursin(k, flow_desc_full) short_desc=v; break end end
        if isempty(short_desc) short_desc = length(flow_desc_full) > 10 ? flow_desc_full[1:9]*"…" : flow_desc_full end
    end
    mag_val = round(Int, abs(magnitude))
    mag_str = mag_val >= 1_000_000 ? string(round(mag_val/1e6,digits=1))*"M" : (mag_val >= 1000 ? string(round(Int,mag_val/1e3))*"K" : string(mag_val))
    mg_wrap.edge_labels[flow_idx_counter] = "$short_desc: $mag_str"

    if !isnothing(firms_idx) && src_idx_num == firms_idx && dst_idx_num == firms_idx
        if occursin("IntermediateGood", flow_description_key) || occursin("Interm.Goods", short_desc)
            intermediate_goods_flow_idx = flow_idx_counter
        elseif occursin("InvestmentGood", flow_description_key) || occursin("F.Inv", short_desc)
            investment_goods_flow_idx = flow_idx_counter
        end
    end
    pair_agg = (source_name, target_name)
    if !haskey(flow_descriptions_agg, pair_agg) flow_descriptions_agg[pair_agg] = [] end
    push!(flow_descriptions_agg[pair_agg], (flow_desc_full, abs(magnitude)))
end
println("\nMultigraph construction complete. Nodes: $(nv(mg_wrap)), Flows processed: $(flow_idx_counter)")

# --- CUSTOM LAYOUT ---
custom_positions = Dict("Firms"=>[5.,0.], "Households"=>[0.,-5.], "Bank"=>[0.,5.], "Government"=>[-5.,0.], "Central Bank"=>[-5.,5.], "RoW"=>[5.,-5.])
fixed_layout = [Point2f(custom_positions[name]) for name in node_names]

# --- DEBUGGING & PREPARATION FOR DISAGGREGATED GRAPH ---
println("\n--- Debugging Disaggregated Graph Data ---")
# println("Node Names: ", node_names)
# println("Number of Nodes: ", num_nodes)

println("Calling Graphs.edges(mg_wrap) to populate internal structures...")
collected_expanded_edges = collect(Graphs.edges(mg_wrap)) # CRITICAL: This populates mg_wrap.expanded_edges etc.
println("  Number of expanded edges (ne(mg_wrap)): ", ne(mg_wrap))
if ne(mg_wrap) == 0 println("  WARNING: No expanded edges found. Graph will be blank.") end
# println("  map_expanded_idx_to_flow_idx (size): ", length(mg_wrap.map_expanded_idx_to_flow_idx))

println("\nGenerating curve_distances...")
curve_distances = gen_distances(mg_wrap, fixed_layout)
println("  Length of curve_distances: ", length(curve_distances))
if length(curve_distances) != ne(mg_wrap) println("  WARNING: Mismatch: curve_distances vs ne(mg_wrap)!") end
if !isempty(curve_distances) && any(isnan, curve_distances) println("  WARNING: NaN in curve_distances.") end

println("\nPreparing direct_labels, widths, colors...")
direct_labels = String[]; direct_widths = Float64[]; direct_colors = Union{Symbol, RGBA{Float32}}[]
intermediate_expanded_idx = 0; investment_expanded_idx = 0
if intermediate_goods_flow_idx > 0
    for (e_idx, f_idx) in mg_wrap.map_expanded_idx_to_flow_idx if f_idx == intermediate_goods_flow_idx intermediate_expanded_idx=e_idx; break end end
end
if investment_goods_flow_idx > 0
    for (e_idx, f_idx) in mg_wrap.map_expanded_idx_to_flow_idx if f_idx == investment_goods_flow_idx investment_expanded_idx=e_idx; break end end
end

for expanded_idx in 1:ne(mg_wrap)
    if !haskey(mg_wrap.map_expanded_idx_to_flow_idx, expanded_idx)
        push!(direct_labels,"Err"); push!(direct_widths,1.0); push!(direct_colors,:red); continue
    end
    original_flow_idx = mg_wrap.map_expanded_idx_to_flow_idx[expanded_idx]
    label = get(mg_wrap.edge_labels, original_flow_idx, "NoLabel")
    magnitude = get(mg_wrap.edge_magnitudes, original_flow_idx, 0.0)

    width = 1.0; min_w, max_w = 1.0, 12.0
    all_mags = collect(values(mg_wrap.edge_magnitudes))
    if !isempty(all_mags)
        pos_mags = filter(x->x > 1e-9, all_mags) # Semicolon here is fine, but removed for consistency
        # CORRECTED LINE FOR PARSEERROR
        min_mag, max_mag = if !isempty(pos_mags)
                               (minimum(pos_mags), maximum(pos_mags))
                           else
                               (1.0, 1.0)
                           end
        
        if magnitude > 1e-9 && max_mag > min_mag
            l_min=log1p(min_mag); l_max=log1p(max_mag); l_mag=log1p(magnitude)
            width = l_max > l_min ? min_w + (max_w-min_w)*(l_mag-l_min)/(l_max-l_min) : (min_w+max_w)/2
            width = clamp(width, min_w, max_w)
        elseif magnitude > 1e-9 # If all positive magnitudes are the same, or only one type of positive magnitude
            width = (min_w+max_w)/2 
        end
    end
    push!(direct_widths, max(1.0, width))
    push!(direct_labels, label)

    color_val::Union{Symbol, RGBA{Float32}} = :slategray
    src_node = mg_wrap.edge_sources[original_flow_idx]; dst_node = mg_wrap.edge_targets[original_flow_idx]
    row_idx = findfirst(==("RoW"), node_names) # Should be 6

    if !isnothing(firms_idx) && src_node == firms_idx && dst_node == firms_idx
        if expanded_idx == intermediate_expanded_idx && intermediate_expanded_idx != 0 color_val = :mediumblue
        elseif expanded_idx == investment_expanded_idx && investment_expanded_idx != 0 color_val = :firebrick
        end
    elseif !isnothing(firms_idx) && !isnothing(row_idx) && src_node == firms_idx && dst_node == row_idx color_val = :darkorchid
    elseif !isnothing(firms_idx) && !isnothing(row_idx) && src_node == row_idx && dst_node == firms_idx color_val = :forestgreen
    end
    push!(direct_colors, color_val)
end
println("  Lengths: Labels=$(length(direct_labels)), Widths=$(length(direct_widths)), Colors=$(length(direct_colors))")
if !(length(direct_labels)==ne(mg_wrap) && length(direct_widths)==ne(mg_wrap) && length(direct_colors)==ne(mg_wrap))
    println("  WARNING: Mismatch in edge property array lengths!")
end
# println("--- End Debugging ---")

# --- DISAGGREGATED VISUALIZATION ---
fig_disagg = Figure(size = (1900, 1500)) # Slightly larger
ax_disagg = Axis(fig_disagg[1,1], title = "Multidigraph Flow Visualization (t=$(model.agg.t))")

if ne(mg_wrap) > 0 && length(curve_distances) == ne(mg_wrap) && length(direct_labels) == ne(mg_wrap) &&
   length(direct_widths) == ne(mg_wrap) && length(direct_colors) == ne(mg_wrap)
    println("\nAttempting to plot disaggregated graph...")
    try
        graphplot!(ax_disagg, mg_wrap,
            layout = _ -> fixed_layout,
            nlabels = node_names,
            nlabels_align = [(:center, :center) for _ in 1:num_nodes],
            nlabels_distance = 75, nlabels_textsize = 38,
            node_size = [220 for _ in 1:num_nodes],
            node_color = [:lightskyblue, :lightgreen, :pink, :khaki, :lightcyan, :lavender],
            edge_width = direct_widths, edge_color = direct_colors,
            arrow_size = [28 for _ in 1:ne(mg_wrap)], # Increased arrow size
            curve_distance = curve_distances, curve_distance_usage = true,
            elabels = direct_labels, elabels_textsize = 30, # Slightly smaller labels
            elabels_distance = 18.0, elabels_rotation=false # Adjust rotation if needed
        )
        println("Disaggregated graphplot! command executed.")
    catch e
        println("ERROR during disaggregated graphplot!: ", e)
        showerror(stdout, e, catch_backtrace())
        println()
    end
    hidedecorations!(ax_disagg); hidespines!(ax_disagg)
    save("multiedge_dimultigraph_t$(model.agg.t).png", fig_disagg)
    println("\nDisaggregated multigraph visualization saved.")
else
    println("\nSKIPPING disaggregated graph plot due to no edges or mismatched properties.")
end

# --- AGGREGATED VISUALIZATION ---
fig_agg = Figure(size = (1200, 1000))
ax_agg = Axis(fig_agg[1,1], title = "Aggregated Flow Visualization (t=$(model.agg.t))")
agg_g = DiGraph(num_nodes); edge_weights_agg = Float64[]; edge_labels_agg = String[]
agg_flows = Dict{Tuple{String,String}, Float64}()
for (p,fs) in flow_descriptions_agg agg_flows[p] = sum(f[2] for f in fs) end
sorted_agg_flows = sort(collect(agg_flows), by=x->x[2], rev=true)

for (pair, total_mag) in sorted_agg_flows
    src_n, tgt_n = pair
    s_idx = findfirst(==(src_n),node_names); t_idx = findfirst(==(tgt_n),node_names)
    if !isnothing(s_idx) && !isnothing(t_idx) && s_idx != t_idx
        add_edge!(agg_g, s_idx, t_idx)
        push!(edge_weights_agg, total_mag)
        val_int = round(Int,total_mag)
        str_val = val_int >=1000 ? string(round(Int,val_int/1000))*"K" : string(val_int)
        push!(edge_labels_agg, str_val)
    end
end
scaled_widths_agg = if !isempty(edge_weights_agg)
    min_w_agg,max_w_agg = minimum(edge_weights_agg),maximum(edge_weights_agg) # Renamed to avoid conflict
    min_w_agg==max_w_agg ? [5.0 for _ in edge_weights_agg] : 2.0.+10.0.*(edge_weights_agg.-min_w_agg)./(max_w_agg-min_w_agg)
else Float64[] end

if Graphs.ne(agg_g) > 0 && length(scaled_widths_agg) == Graphs.ne(agg_g) && length(edge_labels_agg) == Graphs.ne(agg_g)
    graphplot!(ax_agg, agg_g, layout = _->fixed_layout,
        nlabels=node_names, nlabels_align=[(:center,:center) for _ in 1:num_nodes],
        nlabels_distance=35, nlabels_textsize=18, node_size=[60 for _ in 1:num_nodes],
        node_color=[:lightskyblue,:lightgreen,:pink,:khaki,:lightcyan,:lavender],
        edge_width=scaled_widths_agg, edge_color=:slategray,
        arrow_size=[20 for _ in 1:Graphs.ne(agg_g)],
        elabels=edge_labels_agg, elabels_textsize=18)
    hidedecorations!(ax_agg); hidespines!(ax_agg)
    save("multiedge_aggregated_t$(model.agg.t).png", fig_agg)
    println("Aggregated view created and saved.")
else
    println("Skipping aggregated graph plot due to no edges or mismatched properties.")
end

println("\nBoth visualizations attempted.")