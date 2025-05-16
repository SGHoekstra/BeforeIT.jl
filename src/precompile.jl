
using PrecompileTools

@setup_workload let
    country = "netherlands"
    parameters = load(pwd() * "/data/$(country)/parameters/2010Q1.jld2");
    initial_conditions = load(pwd() * "/data/$(country)/initial_conditions/2010Q1.jld2");
    
    T = 1
    @compile_workload let
        model = Bit.init_model(parameters, initial_conditions, T)
	data = Bit.init_data(model);
	Bit.step!(model)
	Bit.update_data!(data, model)
    end
end
