# src/utils/get_intersectoral_flows.jl
import BeforeIT as Bit

export calculate_flows_at_t, check_stock_flow_consistency

"""
    calculate_flows_at_t(m::Bit.Model)

Calculates the aggregate nominal flows between major economic sectors
at the current time step `t` of the model `m`, strictly following the
SOURCE_to_TARGET_Description naming convention.

It assumes `m` reflects the state *after* all updates for period `t` are done,
and `_lagged` fields correctly hold `t-1` values (i.e., values at the start of period t).
"""
function calculate_flows_at_t(m::Bit.Model)
    # Unpack for convenience
    firms = m.firms
    w_act = m.w_act
    w_inact = m.w_inact
    bank = m.bank
    gov = m.gov
    cb = m.cb
    rotw = m.rotw
    agg = m.agg
    p = m.prop

    flows = Dict{Symbol, Float64}()
    P_bar_HH_eff = (1 + p.theta_UNION * max(agg.P_bar_HH - 1,0)) # Effective CPI for wages

    # --- Flows Originating from Firms (F) ---
    flows[:F_to_H_GrossWages] = sum(firms.w_i .* firms.N_i .* P_bar_HH_eff)
    flows[:F_to_H_Dividends] = sum(p.theta_DIV .* (1 .- p.tau_FIRM) .* Bit.pos.(firms.Pi_i))
    flows[:F_to_G_CorporateTax] = sum(p.tau_FIRM .* Bit.pos.(firms.Pi_i))
    flows[:F_to_G_ProductTax] = sum(firms.tau_Y_i .* firms.P_i .* firms.Y_i)
    flows[:F_to_G_ProductionTax] = sum(firms.tau_K_i .* firms.P_i .* firms.Y_i)
    flows[:F_to_G_EmployerSocialContributions] = sum(p.tau_SIF .* firms.w_i .* firms.N_i .* P_bar_HH_eff)
    flows[:F_to_B_LoanPrincipalRepayment] = sum(p.theta .* firms.L_i_lagged)
    flows[:F_to_B_LoanInterest] = sum(bank.r .* (firms.L_i_lagged .+ Bit.pos.(-firms.D_i_lagged)))
    # Value of exports sold BY Firms TO RoW
    flows[:F_to_R_Exports] = rotw.C_l
    # Value of intermediate goods purchased BY Firms. Assume primarily FROM Firms (simplification).
    flows[:F_to_F_IntermediateGoodPurchases] = sum(firms.DM_i .* firms.P_bar_i) # Total cost paid TO suppliers (F or R)
    # Value of investment goods purchased BY Firms. Assume primarily FROM Firms (simplification).
    flows[:F_to_F_InvestmentGoodPurchases] = sum(firms.P_CF_i .* firms.I_i) # Total cost paid TO suppliers (F)

    # --- Flows Originating from Households (H) ---
    # Value of consumption goods purchased BY Households FROM Firms
    total_C_h_nominal_goods_value = sum(w_act.C_h) + sum(w_inact.C_h) + sum(firms.C_h) + bank.C_h
    flows[:H_to_F_ConsumptionGoods] = total_C_h_nominal_goods_value
    # Value of investment goods purchased BY Households FROM Firms
    total_I_h_nominal_goods_value = sum(w_act.I_h) + sum(w_inact.I_h) + sum(firms.I_h) + bank.I_h
    flows[:H_to_F_InvestmentGoods] = total_I_h_nominal_goods_value
    # Taxes & Contributions paid BY Households TO Government
    taxable_wage_income_act_empl = sum(w_act.w_h[w_act.O_h .!= 0] .* (1 .- p.tau_SIW) .* P_bar_HH_eff)
    flows[:H_to_G_IncomeTaxLabor] = p.tau_INC * taxable_wage_income_act_empl
    taxable_dividend_income_firms = sum(p.theta_DIV .* (1 .- p.tau_FIRM) .* Bit.pos.(firms.Pi_i))
    flows[:H_to_G_IncomeTaxDividendsFirms] = p.tau_INC * taxable_dividend_income_firms
    taxable_dividend_income_bank = p.theta_DIV * (1 - p.tau_FIRM) * Bit.pos(bank.Pi_k)
    flows[:H_to_G_IncomeTaxDividendsBank] = p.tau_INC * taxable_dividend_income_bank
    flows[:H_to_G_EmployeeSocialContributions] = sum(w_act.w_h[w_act.O_h .!= 0] .* p.tau_SIW .* P_bar_HH_eff)
    flows[:H_to_G_VAT_on_Consumption] = p.tau_VAT * total_C_h_nominal_goods_value
    flows[:H_to_G_Tax_on_HouseholdInvestment] = p.tau_CF * total_I_h_nominal_goods_value
    # Interest paid BY Households TO Bank on overdrafts
    all_D_h_lagged_negative_sum = sum(Bit.pos.(-w_act.D_h_lagged)) + sum(Bit.pos.(-w_inact.D_h_lagged)) + sum(Bit.pos.(-firms.D_h_lagged)) + Bit.pos(-bank.D_h_lagged)
    flows[:H_to_B_InterestOnHouseholdOverdrafts] = bank.r * all_D_h_lagged_negative_sum

    # --- Flows Originating from Government (G) ---
    unemp_wages_for_UB_ref = sum(w_act.w_h[w_act.O_h .== 0]) # Reference wage sum for UB calculation
    flows[:G_to_H_SocialBenefits] = (p.H_inact * gov.sb_inact * agg.P_bar_HH) +
                                 (p.theta_UB * unemp_wages_for_UB_ref * agg.P_bar_HH) +
                                 (p.H * gov.sb_other * agg.P_bar_HH)
    # Value of goods/services purchased BY Government FROM Firms
    flows[:G_to_F_ConsumptionGoods] = gov.C_j
    # Interest paid BY Government TO Central Bank on debt
    flows[:G_to_CB_InterestOnGovDebt] = cb.r_G * gov.L_G_lagged

    # --- Flows Originating from Bank (B) ---
    flows[:B_to_F_NewLoansGranted] = sum(firms.DL_i)
    flows[:B_to_H_BankDividends] = p.theta_DIV * (1 - p.tau_FIRM) * Bit.pos(bank.Pi_k)
    flows[:B_to_G_BankCorporateTax] = p.tau_FIRM * Bit.pos(bank.Pi_k)
    # Interest paid BY Bank TO Firms on deposits
    flows[:B_to_F_InterestOnFirmDeposits] = sum(cb.r_bar .* Bit.pos.(firms.D_i_lagged))
    # Interest paid BY Bank TO Households on deposits
    all_D_h_lagged_positive_sum = sum(Bit.pos.(w_act.D_h_lagged)) + sum(Bit.pos.(w_inact.D_h_lagged)) + sum(Bit.pos.(firms.D_h_lagged)) + Bit.pos(bank.D_h_lagged)
    flows[:B_to_H_InterestOnHouseholdDeposits] = cb.r_bar * all_D_h_lagged_positive_sum
    # Interest paid BY Bank TO CentralBank on advances (negative reserves)
    flows[:B_to_CB_InterestOnBankAdvances] = cb.r_bar * Bit.pos(-bank.D_k_lagged)

    # --- Flows Originating from Central Bank (CB) ---
    # Interest paid BY CentralBank TO Bank on reserves (positive reserves)
    flows[:CB_to_B_InterestOnBankReserves] = cb.r_bar * Bit.pos(bank.D_k_lagged)
    # Profit transferred BY CentralBank TO Government (set to 0 based on current E_CB logic)
    flows[:CB_to_G_ProfitTransfer] = 0.0 # Assuming profits are retained in E_CB as per current model.

    # --- Flows Originating from Rest of the World (R) ---
    # Export tax paid BY RoW TO Government
    flows[:R_to_G_ExportTax] = p.tau_EXPORT * rotw.C_l
    # Value of imports sold BY RoW TO Firms (Assume Firms are main importers)
    flows[:R_to_F_Imports] = sum(rotw.P_m .* rotw.Q_m) # Total value of imports purchased

    return flows
end


"""
    check_stock_flow_consistency(m::Bit.Model, flows::Dict{Symbol, Float64}; tol=1e-6)

Checks stock-flow consistency for key financial stocks of each sector by comparing
the change in stock (Stock(t) - Stock(t-1)) with the net relevant flows during period t.

`m` is the model state *after* updates for period `t`.
`flows` is the dictionary of nominal flows for period `t` (strictly using X_to_Y convention).
`_lagged` fields in `m` must represent stocks at `t-1` (i.e., values at the start of period t).
"""
function check_stock_flow_consistency(m::Bit.Model, flows::Dict{Symbol, Float64}; tol=1e-6)
    discrepancies = Dict{Symbol, Float64}()
    p = m.prop

    # Helper function to safely get flow value (returns 0.0 if key missing)
    get_flow = (key) -> get(flows, key, 0.0)

    # --- Calculate Actual Stock Changes ---
    ΔD_h_actual = (sum(m.w_act.D_h) + sum(m.w_inact.D_h) + sum(m.firms.D_h) + m.bank.D_h) -
                  (sum(m.w_act.D_h_lagged) + sum(m.w_inact.D_h_lagged) + sum(m.firms.D_h_lagged) + m.bank.D_h_lagged)
    ΔD_i_actual = sum(m.firms.D_i .- m.firms.D_i_lagged)
    ΔL_i_actual = sum(m.firms.L_i .- m.firms.L_i_lagged)
    ΔE_k_actual = m.bank.E_k - m.bank.E_k_lagged
    ΔL_G_actual = m.gov.L_G - m.gov.L_G_lagged
    ΔD_RoW_actual = m.rotw.D_RoW - m.rotw.D_RoW_lagged
    ΔD_k_actual = m.bank.D_k - m.bank.D_k_lagged
    ΔE_CB_actual = m.cb.E_CB - m.cb.E_CB_lagged

    # --- Calculate Net Flows from Dictionary ---

    # 1. Household Total Deposits (Asset for H, Liability for B)
    hh_inflows = get_flow(:F_to_H_GrossWages) + get_flow(:F_to_H_Dividends) + get_flow(:B_to_H_BankDividends) +
                 get_flow(:G_to_H_SocialBenefits) + get_flow(:B_to_H_InterestOnHouseholdDeposits)
    hh_outflows = get_flow(:H_to_F_ConsumptionGoods) + get_flow(:H_to_F_InvestmentGoods) +
                  get_flow(:H_to_G_IncomeTaxLabor) + get_flow(:H_to_G_IncomeTaxDividendsFirms) + get_flow(:H_to_G_IncomeTaxDividendsBank) +
                  get_flow(:H_to_G_EmployeeSocialContributions) + get_flow(:H_to_G_VAT_on_Consumption) + get_flow(:H_to_G_Tax_on_HouseholdInvestment) +
                  get_flow(:H_to_B_InterestOnHouseholdOverdrafts)
    net_flow_H_deposits_calc = hh_inflows - hh_outflows
    discrepancies[:Household_TotalDeposits] = ΔD_h_actual - net_flow_H_deposits_calc

    # 2. Firm Total Deposits (Asset for F, Liability for B)
    # Inflows: Actual Sales Revenue, Interest Received, New Loans
    # Outflows: Labor Costs, Material Costs (total), Investment Costs (total), Taxes, Dividends, Interest Paid, Loan Repayment

    # Actual total sales revenue for domestic firms
    firm_actual_sales_revenue = sum(m.firms.P_i .* m.firms.Q_i)

    firm_inflows = firm_actual_sales_revenue + 
                   get_flow(:B_to_F_InterestOnFirmDeposits) + 
                   get_flow(:B_to_F_NewLoansGranted)
    
    # Outflows are payments by firms.
    # F_to_F_IntermediateGoodPurchases is total expenditure by firms on intermediates (domestic + imported).
    # F_to_F_InvestmentGoodPurchases is total expenditure by firms on investment goods (domestic + imported).
    firm_outflows = get_flow(:F_to_H_GrossWages) + 
                    get_flow(:F_to_G_EmployerSocialContributions) +
                    get_flow(:F_to_F_IntermediateGoodPurchases) + # This is sum(m.firms.DM_i .* m.firms.P_bar_i)
                    get_flow(:F_to_F_InvestmentGoodPurchases) +   # This is sum(m.firms.P_CF_i .* m.firms.I_i)
                    get_flow(:F_to_G_ProductTax) + 
                    get_flow(:F_to_G_ProductionTax) + 
                    get_flow(:F_to_G_CorporateTax) +
                    get_flow(:F_to_H_Dividends) +
                    get_flow(:F_to_B_LoanInterest) +
                    get_flow(:F_to_B_LoanPrincipalRepayment)
                    
    net_flow_F_deposits_calc = firm_inflows - firm_outflows
    discrepancies[:Firm_TotalDeposits] = ΔD_i_actual - net_flow_F_deposits_calc

    # 3. Firm Total Loans (Liability for F, Asset for B)
    net_flow_F_loans_calc = get_flow(:B_to_F_NewLoansGranted) - get_flow(:F_to_B_LoanPrincipalRepayment)
    discrepancies[:Firm_TotalLoans] = ΔL_i_actual - net_flow_F_loans_calc

    # 4. Bank Equity (Liability/Equity for B)
    bank_interest_income = get_flow(:F_to_B_LoanInterest) + get_flow(:H_to_B_InterestOnHouseholdOverdrafts) + get_flow(:CB_to_B_InterestOnBankReserves)
    bank_interest_expense = get_flow(:B_to_F_InterestOnFirmDeposits) + get_flow(:B_to_H_InterestOnHouseholdDeposits) + get_flow(:B_to_CB_InterestOnBankAdvances)
    bank_profit_calc = bank_interest_income - bank_interest_expense
    bank_retained_earnings_calc = bank_profit_calc - get_flow(:B_to_G_BankCorporateTax) - get_flow(:B_to_H_BankDividends)
    discrepancies[:Bank_Equity] = ΔE_k_actual - bank_retained_earnings_calc
    discrepancies[:Bank_Profit_Internal_vs_Flows] = m.bank.Pi_k - bank_profit_calc

    # 5. Government Debt (Liability for G, Asset for CB)
    gov_total_expenditure = get_flow(:G_to_F_ConsumptionGoods) + get_flow(:G_to_H_SocialBenefits) + get_flow(:G_to_CB_InterestOnGovDebt)
    gov_total_revenue = get_flow(:F_to_G_CorporateTax) + get_flow(:F_to_G_ProductTax) + get_flow(:F_to_G_ProductionTax) +
                        get_flow(:F_to_G_EmployerSocialContributions) +
                        get_flow(:H_to_G_IncomeTaxLabor) + get_flow(:H_to_G_IncomeTaxDividendsFirms) + get_flow(:H_to_G_IncomeTaxDividendsBank) +
                        get_flow(:H_to_G_EmployeeSocialContributions) + get_flow(:H_to_G_VAT_on_Consumption) +
                        get_flow(:H_to_G_Tax_on_HouseholdInvestment) + get_flow(:R_to_G_ExportTax) +
                        get_flow(:B_to_G_BankCorporateTax) + get_flow(:CB_to_G_ProfitTransfer)
    gov_deficit_calc = gov_total_expenditure - gov_total_revenue
    discrepancies[:Gov_Debt_L_G] = ΔL_G_actual - gov_deficit_calc
    discrepancies[:Gov_Revenue_Internal_vs_Flows] = m.gov.Y_G - gov_total_revenue

    # 6. RoW Net Financial Position (D_RoW with CB) (Asset for R, Liability for CB)
    net_flow_R_deposits_calc = get_flow(:R_to_F_Imports) + # Money IN for RoW from sales to F (and other domestic agents)
                               get_flow(:R_to_H_Imports_placeholder) + # Placeholder if HH/Gov buy imports directly from RoW (not via F)
                               get_flow(:R_to_G_Imports_placeholder) - # Placeholder
                               (get_flow(:F_to_R_Exports) + # Money OUT for RoW for purchases from F
                                get_flow(:H_to_R_Exports_placeholder) + # Placeholder
                                get_flow(:G_to_R_Exports_placeholder) + # Placeholder
                                get_flow(:R_to_G_ExportTax)) # Tax paid by RoW
    # Simplified if all trade is F-R or R-F:
    # R_to_F_Imports captures all goods sold by RoW to domestic market *if they become inputs/capital for Firms or sold via Firms*.
    # F_to_R_Exports captures all goods sold by Firms to RoW.
    # A more direct way: Value of (Imports by domestic economy) - Value of (Exports by domestic economy)
    # Domestic Imports = Money Flow from Domestic to RoW = (expenditure by F on imported intermediates/capital) + (expenditure by H on imported consumption/investment) + (expenditure by G on imported consumption)
    # Domestic Exports = Money Flow from RoW to Domestic = F_to_R_Exports (expenditure by RoW on domestic goods)
    # calculate_flows_at_t defines flows[:R_to_F_Imports] = sum(rotw.P_m .* rotw.Q_m). This is total import value sold by RoW.
    # calculate_flows_at_t defines flows[:F_to_R_Exports] = rotw.C_l. This is total export value purchased by RoW.
    net_trade_balance_for_RoW = get_flow(:R_to_F_Imports) - (get_flow(:F_to_R_Exports) + get_flow(:R_to_G_ExportTax))
    discrepancies[:RoW_NetPosition_D_RoW] = ΔD_RoW_actual - net_trade_balance_for_RoW


    # 7. Bank Net Position with CB (D_k) (Asset for B if >0, Liability if <0)
    net_flow_B_dk_calc_from_BS = ΔD_i_actual + ΔD_h_actual + ΔE_k_actual - ΔL_i_actual
    discrepancies[:Bank_CB_Position_Dk_from_BS] = ΔD_k_actual - net_flow_B_dk_calc_from_BS

    # 8. Central Bank Equity (E_CB) (Liability/Equity for CB)
    cb_interest_income = get_flow(:G_to_CB_InterestOnGovDebt) + get_flow(:B_to_CB_InterestOnBankAdvances)
    cb_interest_expense = get_flow(:CB_to_B_InterestOnBankReserves)
    cb_profit_calc = cb_interest_income - cb_interest_expense
    cb_retained_earnings_calc = cb_profit_calc - get_flow(:CB_to_G_ProfitTransfer)
    discrepancies[:CB_Equity] = ΔE_CB_actual - cb_retained_earnings_calc
    internal_Pi_CB = m.cb.r_G * m.gov.L_G_lagged - m.cb.r_bar * m.bank.D_k_lagged
    discrepancies[:CB_Profit_Internal_vs_Flows] = internal_Pi_CB - cb_profit_calc

    # Filter out discrepancies that are below tolerance
    keys_to_delete = Symbol[]
    for (key, value) in discrepancies
        if abs(value) < tol
            push!(keys_to_delete, key)
        end
    end
    for key in keys_to_delete
        delete!(discrepancies, key)
    end

    return discrepancies
end