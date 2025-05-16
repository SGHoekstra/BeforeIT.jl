import CommonSolve
using CommonSolve: step!
export step!

"""
    step!(model; multi_threading = false)

This function simulates a single epoch the economic model, updating various components of the model based
the interactions between different economic agents. It accepts a `model` object, which encapsulates the state for the
simulation, and an optional boolean parameter `multi_threading` to enable or disable multi-threading.

Key operations performed include:
- Financial adjustments for firms and banks, including insolvency checks and profit calculations.
- Economic expectations and adjustments, such as growth, inflation, and central bank rates.
- Labor and credit market operations, including wage updates and loan processing.
- Household economic activities, including consumption and investment budgeting.
- Government and international trade financial activities, including budgeting and trade balances.
- General market matching and accounting updates to reflect changes in economic indicators and positions.

The function updates the model in-place and does not return any value.
"""
function CommonSolve.step!(model::AbstractModel; multi_threading = false, shock = NoShock(), abmx = false, conditional_forecast = false)

    gov = model.gov # government
    cb = model.cb # central bank
    rotw = model.rotw # rest of the world
    firms = model.firms # firms
    bank = model.bank # bank
    w_act = model.w_act # active workers
    w_inact = model.w_inact # inactive workers
    agg = model.agg # aggregates
    prop = model.prop # model properties

    # return an error if t is greater than T
    if agg.t > prop.T + 1
        error("The model has already reached the final time step.")
    end

    # --- Start of Period t ---
    # FIRST: Update lagged variables to store the state from the end of t-1 (start of t)
    # These lagged values will be used in calculations during period t.
    # The main variables (e.g., w_act.D_h) still hold t-1 values at this point.

    # Update lagged household deposits
    w_act.D_h_lagged .= w_act.D_h
    w_inact.D_h_lagged .= w_inact.D_h
    firms.D_h_lagged .= firms.D_h
    bank.D_h_lagged = bank.D_h

    # Update lagged firm financial stocks
    firms.D_i_lagged .= firms.D_i
    firms.L_i_lagged .= firms.L_i
    firms.E_i_lagged .= firms.E_i # Assuming E_i_lagged exists

    # Update lagged bank financial stocks
    bank.D_k_lagged = bank.D_k
    bank.E_k_lagged = bank.E_k # Assuming E_k_lagged exists

    # Update lagged government debt
    gov.L_G_lagged = gov.L_G

    # Update lagged central bank equity
    cb.E_CB_lagged = cb.E_CB # Assuming E_CB_lagged exists

    # Update lagged RoW position
    rotw.D_RoW_lagged = rotw.D_RoW # Assuming D_RoW_lagged exists

    # --- NOW proceed with calculations for period t ---
    # Lagged variables now correctly hold start-of-period t values.

    Bit.finance_insolvent_firms!(firms, bank, model) # Uses lagged values if needed internally

    ####### GENERAL ESTIMATIONS #######

    # expectation on economic growth and inflation (often based on lagged aggregates)
    if abmx
        agg.Y_e, agg.gamma_e, agg.pi_e = Bit.growth_inflation_ARX(model; conditional_forecast = conditional_forecast)
    else
        agg.Y_e, agg.gamma_e, agg.pi_e = Bit.growth_inflation_expectations(model)
    end

    # update growth and inflation of economic area
    agg.epsilon_Y_EA, agg.epsilon_E, agg.epsilon_I = Bit.epsilon(prop.C)
    rotw.Y_EA, rotw.gamma_EA, rotw.pi_EA = Bit.growth_inflation_EA(rotw, model)

    # set central bank rate via the Taylor rule (uses lagged inflation/growth)
    cb.r_bar = Bit.central_bank_rate(cb, model)

    # apply an eventual shock to the model, the default does nothing
    shock(model)

    # update rate on loans and morgages (uses cb.r_bar)
    bank.r = Bit.bank_rate(bank, model)

    ####### FIRM EXPECTATIONS AND DECISIONS #######

    # compute firm quantity, price, investment etc. (based on expectations and lagged state)
    Q_s_i, I_d_i, DM_d_i, N_d_i, Pi_e_i, DL_d_i, K_e_i, L_e_i, P_i, pi_d_i, pi_c_i, pi_l_i, pi_k_i, pi_m_i =
        Bit.firms_expectations_and_decisions(firms, model) # Uses lagged stocks implicitly/explicitly

    firms.Q_s_i .= Q_s_i
    firms.I_d_i .= I_d_i
    firms.DM_d_i .= DM_d_i
    firms.N_d_i .= N_d_i
    firms.Pi_e_i .= Pi_e_i
    firms.P_i .= P_i
    firms.DL_d_i .= DL_d_i
    firms.K_e_i .= K_e_i
    firms.L_e_i .= L_e_i
    firms.pi_d_i .= pi_d_i
    firms.pi_c_i .= pi_c_i
    firms.pi_l_i .= pi_l_i
    firms.pi_k_i .= pi_k_i
    firms.pi_m_i .= pi_m_i

    ####### CREDIT MARKET, LABOUR MARKET AND PRODUCTION #######

    # firms acquire new loans (DL_i is flow for period t)
    firms.DL_i .= Bit.search_and_matching_credit(firms, model)

    # firms acquire labour
    N_i, Oh = Bit.search_and_matching_labour(firms, model)
    firms.N_i .= N_i
    w_act.O_h .= Oh

    # update wages and production (Y_i is output flow for period t)
    firms.w_i .= Bit.firms_wages(firms)
    firms.Y_i .= Bit.firms_production(firms)

    # update wages for workers
    Bit.update_workers_wages!(w_act, firms.w_i)


    ####### CONSUMPTION AND INVESTMENT BUDGET #######

    # update social benefits
    gov.sb_other, gov.sb_inact = Bit.gov_social_benefits(gov, model)

    # compute expected bank profits (uses lagged stocks for interest calcs)
    bank.Pi_e_k = Bit.bank_expected_profits(bank, model)

    # compute consumption and investment budget (uses lagged D_h for wealth effect etc.)
    C_d_h, I_d_h = Bit.households_budget_act(w_act, model)
    w_act.C_d_h .= C_d_h
    w_act.I_d_h .= I_d_h
    C_d_h, I_d_h = Bit.households_budget_inact(w_inact, model)
    w_inact.C_d_h .= C_d_h
    w_inact.I_d_h .= I_d_h
    C_d_h, I_d_h = Bit.households_budget(firms, model)
    firms.C_d_h .= C_d_h
    firms.I_d_h .= I_d_h
    bank.C_d_h, bank.I_d_h = Bit.households_budget(bank, model)


    ####### GOVERNMENT SPENDING BUDGET, IMPORT-EXPORT BUDGET #######

    # compute gov expenditure
    C_G, C_d_j = Bit.gov_expenditure(gov, model)
    gov.C_G = C_G
    gov.C_d_j .= C_d_j

    if !conditional_forecast
        agg.C_G[prop.T_prime + agg.t] = C_G
    end

    # compute demand for export and supply of imports
    C_E, Y_I, C_d_l, Y_m, P_m = Bit.rotw_import_export(rotw, model)
    rotw.C_E = C_E
    rotw.Y_I = Y_I
    rotw.C_d_l .= C_d_l
    rotw.Y_m .= Y_m
    rotw.P_m .= P_m

    if !conditional_forecast
        agg.C_E[prop.T_prime + agg.t] = C_E
        agg.Y_I[prop.T_prime + agg.t] = Y_I
    end

    ####### GENERAL SEARCH AND MATCHING FOR ALL GOODS #######

    Bit.search_and_matching!(model, multi_threading) # Determines actual flows (C_h, I_h, DM_i, I_i, C_j, C_l etc.)

    ####### FINAL GENERAL ACCOUNTING - Calculates flows and end-of-period t stocks #######

    # update inflation and price indices
    agg.pi_[prop.T_prime + agg.t], agg.P_bar = Bit.inflation_priceindex(firms.P_i, firms.Y_i, agg.P_bar)
    agg.P_bar_g .= Bit.sector_specific_priceindex(firms, rotw, prop.G)
    agg.P_bar_CF = sum(prop.products.b_CF_g .* agg.P_bar_g)
    agg.P_bar_HH = sum(prop.products.b_HH_g .* agg.P_bar_g)

    # update firms physical stocks (K(t), M(t), S(t))
    K_i, M_i, DS_i, S_i = Bit.firms_stocks(firms) # Uses I_i, DM_i flows from matching
    firms.K_i .= K_i # K(t)
    firms.M_i .= M_i # M(t)
    firms.DS_i .= DS_i
    firms.S_i .= S_i # S(t)

    # update firms profits (Pi_i(t), uses L_i_lagged, D_i_lagged implicitly/explicitly for interest etc.)
    firms.Pi_i .= Bit.firms_profits(firms, model) # Pi_i(t)

    # update bank profits (Pi_k(t), uses L_i_lagged, D_i_lagged, D_h_lagged, D_k_lagged for interest)
    bank.Pi_k = Bit.bank_profits(bank, model) # Pi_k(t)

    # update bank equity (E_k(t) = E_k_lagged + Retained_Pi_k(t))
    bank.E_k = Bit.bank_equity(bank, model) # E_k(t)

    # update households income (Y_h(t), includes interest on D_h_lagged)
    w_act.Y_h .= Bit.households_income_act(w_act, model)
    w_inact.Y_h .= Bit.households_income_inact(w_inact, model)
    firms.Y_h .= Bit.households_income(firms, model)
    bank.Y_h = Bit.households_income(bank, model)

    # update households deposits (D_h(t) = D_h_lagged + Y_h(t) - Expenditures(t))
    w_act.D_h .= Bit.households_deposits(w_act, model) # D_h(t)
    w_inact.D_h .= Bit.households_deposits(w_inact, model) # D_h(t)
    firms.D_h .= Bit.households_deposits(firms, model) # D_h(t)
    bank.D_h = Bit.households_deposits(bank, model) # D_h(t)

    # compute central bank equity (E_CB(t) = E_CB_lagged + Pi_CB(t), uses L_G_lagged, D_k_lagged for Pi_CB)
    cb.E_CB = Bit.central_bank_equity(cb, model) # E_CB(t)

    # compute government revenues (Y_G(t))
    gov.Y_G = Bit.gov_revenues(model) # Y_G(t)

    # compute government debt (L_G(t) = L_G_lagged + Deficit(t), uses L_G_lagged for interest)
    gov.L_G = Bit.gov_loans(gov, model) # L_G(t)

    # compute firms deposits (D_i(t) = D_i_lagged + DD_i(t), uses L_i_lagged, D_i_lagged for interest)
    firms.D_i .= Bit.firms_deposits(firms, model) # D_i(t)

    # compute firms loans (L_i(t) = L_i_lagged + DL_i(t) - Repayment(t), uses L_i_lagged for repayment)
    firms.L_i .= Bit.firms_loans(firms, model) # L_i(t)

    # compute firms equity (E_i(t) = E_i_lagged + Retained_Pi_i(t))
    firms.E_i .= Bit.firms_equity(firms, model) # E_i(t)

    # update RoW position (D_RoW(t) = D_RoW_lagged + NetFlow(t))
    rotw.D_RoW = Bit.rotw_deposits(rotw, model) # D_RoW(t)

    # update bank position with CB (D_k(t) calculated from BS identity using end-of-period stocks D_i(t), L_i(t), E_k(t), D_h(t))
    bank.D_k = Bit.bank_deposits(bank, model) # D_k(t)

    # update GDP
    agg.Y[prop.T_prime + agg.t] = sum(firms.Y_i) # GDP(t)

    # Increment time step for the next iteration
    agg.t += 1
end