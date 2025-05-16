# In this tutorial we illustrate how to get predictions from the model for
# a number of quarters starting from previous simulations.

import BeforeIT as Bit

using FileIO, Dates
function get_predictions(;
    abmx = true,
    longrun = true,
    empirical_distribution = false,
    unconditional_forecasts = true, number_seeds = 10)

    year_ = 2010
    number_years = 10
    number_quarters = 4 * number_years

    if longrun
        horizon = 30
    else
        horizon = 12
    end

    # Load the real time series
    data = Bit.NETHERLANDS_CALIBRATION.data

    quarters_num = []
    year_m = year_
    for month in 4:3:((number_years + 1) * 12 + 1)
        year_m = year_ + (month ÷ 12)
        mont_m = month % 12
        date = DateTime(year_m, mont_m, 1) - Day(1)
        push!(quarters_num, Bit.date2num(date))
    end

    for i in 1:number_quarters
        quarter_num = quarters_num[i]
        Bit.get_predictions_from_sims(data, quarter_num, horizon, number_seeds;
        country = "netherlands", abmx = abmx, longrun = longrun, empirical_distribution = empirical_distribution, unconditional_forecasts = unconditional_forecasts)
    end
end

get_predictions(abmx = false, longrun = false, empirical_distribution = false, unconditional_forecasts = true)
get_predictions(abmx = true, longrun = false, empirical_distribution = false, unconditional_forecasts = true)

