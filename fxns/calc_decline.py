import streamlit as st, pandas as pd, numpy as np

@st.fragment()
def calculate_decline_rates(qi, qf, decline_type, b_factor=None, initial_decline=None, terminal_decline=None, max_months=600):
    """
    Calculate production rates using either exponential or hyperbolic decline.
    For hyperbolic decline, switches to exponential at terminal decline rate.
    Returns tuple of (rates array, decline_types array)
    """
    rates = []
    decline_types = []
    current_rate = qi
    # 189.5 nan EXP nan 0.32155217278441517 0.04
    if decline_type == 'EXP':  # Pure exponential decline
        monthly_decline = 1 - np.exp(-initial_decline/12)
        while current_rate > qf and len(rates) < max_months:
            rates.append(current_rate)
            current_rate *= (1 - monthly_decline)      
    else:  # Hyperbolic decline with terminal transition
        t = 0
        monthly_terminal = 1 - np.exp(-terminal_decline/12)
        while current_rate > qf and len(rates) < max_months:
            # Calculate current annual decline rate
            current_decline = initial_decline / (1 + b_factor * initial_decline * t/12)
            # Check for transition to terminal decline
            if current_decline <= terminal_decline:
                # Switch to exponential decline using terminal decline rate
                while current_rate > qf and len(rates) < max_months:
                    rates.append(current_rate)
                    current_rate *= (1 - monthly_terminal)
                break
            # Still in hyperbolic decline
            rates.append(current_rate)
            # Calculate next rate using hyperbolic formula
            current_rate = qi / np.power(1 + b_factor * initial_decline * (t + 1)/12, 1/b_factor)
            t += 1
    return np.array(rates)

@st.fragment()
##elii's decline calc
def calc_decline(prd, params):
    # Get the last production date
    last_prod_date = prd['PRODUCINGMONTH'].max()
    # Get forecast start dates from ECON_INPUT
    oil_fcst_start = params['FCST_START_OIL']
    gas_fcst_start = params['FCST_START_GAS']
    # Ensure fost_starecast start dates are on the first day of the month
    oil_fcst_start = oil_fcst_start.replace(day=1)
    gas_fcst_start = gas_fcst_start.replace(day=1)
    earliest_fcst_start = min(oil_fcst_start, gas_fcst_start)
    gas_qi = 0 if pd.isna(params.get('GAS_USER_QI')) else params.get('GAS_USER_QI')
    gas_qf = 0 if pd.isna(params.get('GAS_Q_MIN')) else params.get('GAS_Q_MIN')
    gas_decline_type = "EXP" if pd.isna(params.get('GAS_DECLINE_TYPE')) else params.get('GAS_DECLINE_TYPE')
    gas_decline = 0 if pd.isna(params.get('GAS_USER_DECLINE')) else params.get('GAS_USER_DECLINE')
    gas_b_factor = 0.01 if pd.isna(params.get('GAS_USER_B_FACTOR')) else params.get('GAS_USER_B_FACTOR')
    terminal_decline = 0 if pd.isna(params.get('GAS_D_MIN')) else params.get('GAS_D_MIN')

    gas_rates = calculate_decline_rates(gas_qi, 
                                        gas_qf, 
                                        gas_decline_type, 
                                        gas_b_factor, 
                                        gas_decline, 
                                        terminal_decline, 
                                        max_months=600
                                        )

    # Calculate the index offset for gas forecast start
    gas_offset = (gas_fcst_start - earliest_fcst_start).days // 30
    gas_offset = max(0, gas_offset)  # Ensure non-negative
    max_months = 600
    dates = pd.date_range(start=earliest_fcst_start, periods=max_months, freq='MS')
    well_fcst = pd.DataFrame({'PRODUCINGMONTH': dates})
    # Limit to available array length
    data_length = min(len(gas_rates), len(well_fcst) - gas_offset)
    # Insert gas forecast at the correct position
    well_fcst.loc[gas_offset:gas_offset+data_length-1, 'GasFcst_MCF'] = gas_rates[:data_length]
    # Add gas forecast start date for reference
    well_fcst['GasFcst_Start_Date'] = gas_fcst_start


    oil_qi = 0 if pd.isna(params.get('OIL_USER_QI')) else params.get('OIL_USER_QI')
    oil_qf = 0 if pd.isna(params.get('OIL_Q_MIN')) else params.get('OIL_Q_MIN')
    oil_decline_type = params.get('OIL_DECLINE_TYPE')
    oil_decline = 0 if pd.isna(params.get('OIL_USER_DECLINE')) else params.get('OIL_USER_DECLINE')
    oil_b_factor = 0.01 if pd.isna(params.get('OIL_USER_B_FACTOR')) else params.get('OIL_USER_B_FACTOR')
    oil_terminal_decline = 0 if pd.isna(params.get('OIL_D_MIN')) else params.get('OIL_D_MIN')

    oil_rates = calculate_decline_rates(
        qi=oil_qi,
        qf=oil_qf,
        decline_type=oil_decline_type,
        b_factor=oil_b_factor,
        initial_decline=oil_decline,
        terminal_decline=oil_terminal_decline, 
        max_months=600
    )

    # Calculate the index offset for oil forecast start
    oil_offset = (oil_fcst_start - earliest_fcst_start).days // 30
    oil_offset = max(0, oil_offset)  # Ensure non-negative
    data_length = min(len(oil_rates), len(well_fcst) - oil_offset)
    well_fcst.loc[oil_offset:oil_offset+data_length-1, 'OilFcst_BBL'] = oil_rates[:data_length]
    # Add oil forecast start date for reference
    well_fcst['OilFcst_Start_Date'] = oil_fcst_start

    # Initialize all possible forecast columns to ensure they exist
    forecast_cols = ['GasFcst_MCF', 'OilFcst_BBL']
    for col in forecast_cols:
        if col not in well_fcst.columns:
            well_fcst[col] = np.nan
            
    final_df = pd.merge(prd, well_fcst, on='PRODUCINGMONTH', how='outer')
    # Filter to end of 2050
    final_df = final_df[final_df['PRODUCINGMONTH'] <= '2050-12-31']
    final_df = final_df.sort_values('PRODUCINGMONTH')
    return final_df

# (2) Cache decline forecast calculation
@st.cache_data(ttl=300)
def cached_decline(prd, params):
    return calc_decline(prd, params)