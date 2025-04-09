##########################################
# SECTION 1: IMPORTS & SETUP
##########################################
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
import base64
import folium
from streamlit_folium import folium_static
import branca.colormap as cm
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
## delete below
import snowflake.connector
from snowflake.snowpark.session import Session

# Set Streamlit Page Configuration with wider layout
st.set_page_config(
    page_title="Well Data Viewer",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed for more space
)

# Make the app wider with custom CSS
st.markdown("""
<style>
    .block-container {
        max-width: 95% !important;
        padding-top: 2rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    .stButton>button {
        background-color: #0E3D59;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

# Import SingleWellForecast function
def SingleWellForecast(well_api, econ_input_df, raw_prod_data_df):
    """
    Generates a production forecast for a single well using either exponential or 
    hyperbolic decline based on decline type specified. Uses FCST_START_OIL and 
    FCST_START_GAS dates to determine when to start the respective forecasts.
    
    Parameters:
    -----------
    well_api : str
        The API/UWI identifier for the specific well to forecast
    econ_input_df : pandas.DataFrame
        DataFrame containing well parameters including decline curve inputs (ECON_INPUT)
    raw_prod_data_df : pandas.DataFrame
        DataFrame containing historical production data (RAW_PROD_DATA)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with historical and forecast data for the specified well
    """
    # Convert and clean DataFrames
    econ_input_df = econ_input_df.to_pandas() if not isinstance(econ_input_df, pd.DataFrame) else econ_input_df
    raw_prod_data_df = raw_prod_data_df.to_pandas() if not isinstance(raw_prod_data_df, pd.DataFrame) else raw_prod_data_df
    
    # Basic data preparation
    raw_prod_data_df.columns = raw_prod_data_df.columns.str.strip()
    econ_input_df.columns = econ_input_df.columns.str.strip()
    raw_prod_data_df['PRODUCINGMONTH'] = pd.to_datetime(raw_prod_data_df['PRODUCINGMONTH'])
    
    # Handle missing WATERPROD_BBL column
    if 'WATERPROD_BBL' not in raw_prod_data_df.columns:
        raw_prod_data_df['WATERPROD_BBL'] = 0
    
    # Filter to get only the selected well
    well_data = econ_input_df[econ_input_df['API_UWI'] == well_api]
    
    if well_data.empty:
        raise ValueError(f"Well API {well_api} not found in econ input")
    
    # Get the well's production history
    well_production = raw_prod_data_df[raw_prod_data_df['API_UWI'] == well_api]
    well_production = well_production[well_production['PRODUCINGMONTH'] >= '2018-01-01']
    
    if well_production.empty:
        raise ValueError(f"No production data found for well API {well_api}")
    
    # Aggregate production data by month (in case there are multiple entries)
    agg_columns = {
        'LIQUIDSPROD_BBL': 'sum',
        'GASPROD_MCF': 'sum'
    }
    
    # Add water only if it exists
    if 'WATERPROD_BBL' in well_production.columns:
        agg_columns['WATERPROD_BBL'] = 'sum'
    
    aggregated_data = well_production.groupby('PRODUCINGMONTH', as_index=False).agg(agg_columns)
    
    # Handle missing WATERPROD_BBL column in aggregated data
    if 'WATERPROD_BBL' not in aggregated_data.columns:
        aggregated_data['WATERPROD_BBL'] = 0
    
    # Calculate cumulative volumes
    aggregated_data = aggregated_data.sort_values('PRODUCINGMONTH')
    aggregated_data['CumLiquids_BBL'] = aggregated_data['LIQUIDSPROD_BBL'].cumsum()
    aggregated_data['CumGas_MCF'] = aggregated_data['GASPROD_MCF'].cumsum()
    aggregated_data['CumWater_BBL'] = aggregated_data['WATERPROD_BBL'].cumsum()
    
    def calculate_decline_rates(qi, qf, decline_type, b_factor=None, initial_decline=None, terminal_decline=None, max_months=600):
        """
        Calculate production rates using either exponential or hyperbolic decline.
        For hyperbolic decline, switches to exponential at terminal decline rate.
        Returns tuple of (rates array, decline_types array)
        """
        rates = []
        decline_types = []
        current_rate = qi
        
        if decline_type == 'E':  # Pure exponential decline
            monthly_decline = 1 - np.exp(-initial_decline/12)
            
            while current_rate > qf and len(rates) < max_months:
                rates.append(current_rate)
                decline_types.append('E')
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
                        decline_types.append('E')
                        current_rate *= (1 - monthly_terminal)
                    break
                
                # Still in hyperbolic decline
                rates.append(current_rate)
                decline_types.append('H')
                # Calculate next rate using hyperbolic formula
                current_rate = qi / np.power(1 + b_factor * initial_decline * (t + 1)/12, 1/b_factor)
                t += 1
        
        return np.array(rates), np.array(decline_types)
    
    # Generate forecasts
    well_row = well_data.iloc[0]
    
    try:
        # Get the last production date
        last_prod_date = well_production['PRODUCINGMONTH'].max()
        
        # Get forecast start dates from ECON_INPUT
        oil_fcst_start = pd.to_datetime(well_row['FCST_START_OIL']) if pd.notna(well_row['FCST_START_OIL']) else None
        gas_fcst_start = pd.to_datetime(well_row['FCST_START_GAS']) if pd.notna(well_row['FCST_START_GAS']) else None
        
        # If forecast start dates are not provided, use last production date + 1 month as default
        default_fcst_start = (pd.to_datetime(last_prod_date) + pd.DateOffset(months=1)).replace(day=1)
        
        # Use the specified dates or the default if not specified
        oil_fcst_start = oil_fcst_start if oil_fcst_start is not None else default_fcst_start
        gas_fcst_start = gas_fcst_start if gas_fcst_start is not None else default_fcst_start
        
        # Ensure forecast start dates are on the first day of the month
        oil_fcst_start = oil_fcst_start.replace(day=1)
        gas_fcst_start = gas_fcst_start.replace(day=1)
        
        # Find the earliest forecast start date to initialize the forecast period
        earliest_fcst_start = min(oil_fcst_start, gas_fcst_start)
        
        # Initialize forecast period - use longer period to ensure we capture all forecast data
        max_months = 600
        dates = pd.date_range(start=earliest_fcst_start, periods=max_months, freq='MS')
        
        # Initialize an empty DataFrame for this well's forecasts
        well_fcst = pd.DataFrame({'PRODUCINGMONTH': dates})
        has_forecast = False
        
        # Generate gas forecast
        try:
            # Determine gas initial rate (qi) - use user input if available, otherwise calculated
            gas_qi = float(well_row['GAS_USER_QI']) if pd.notna(well_row['GAS_USER_QI']) else float(well_row['GAS_CALC_QI'])
            gas_qf = float(well_row['GAS_Q_MIN'])
            
            if gas_qi > gas_qf:
                # # Use the gas forecast start date to determine the reference production data
                # last_gas_prod = well_production[well_production['PRODUCINGMONTH'] <= gas_fcst_start]['GASPROD_MCF'].tail(3)
                # if not last_gas_prod.empty:
                #     gas_qi = min(gas_qi, last_gas_prod.mean() * 1.1)
                
                gas_decline_type = well_row['GAS_DECLINE_TYPE']
                
                # Determine gas decline rate - use user input if available, otherwise empirical
                gas_decline = float(well_row['GAS_USER_DECLINE']) if pd.notna(well_row['GAS_USER_DECLINE']) else float(well_row['GAS_EMPIRICAL_DECLINE'])
                
                # Determine b-factor - use user input if available, otherwise calculated
                gas_b_factor = float(well_row['GAS_USER_B_FACTOR']) if pd.notna(well_row['GAS_USER_B_FACTOR']) else float(well_row['GAS_CALC_B_FACTOR'])
                
                gas_rates, gas_decline_types = calculate_decline_rates(
                    qi=gas_qi,
                    qf=gas_qf,
                    decline_type=gas_decline_type,
                    b_factor=gas_b_factor,
                    initial_decline=gas_decline,
                    terminal_decline=float(well_row['GAS_D_MIN'])
                )
                
                if len(gas_rates) > 0:
                    # Calculate the index offset for gas forecast start
                    gas_offset = (gas_fcst_start - earliest_fcst_start).days // 30
                    gas_offset = max(0, gas_offset)  # Ensure non-negative
                    
                    # Limit to available array length
                    data_length = min(len(gas_rates), len(well_fcst) - gas_offset)
                    
                    # Insert gas forecast at the correct position
                    well_fcst.loc[gas_offset:gas_offset+data_length-1, 'GasFcst_MCF'] = gas_rates[:data_length]
                    well_fcst.loc[gas_offset:gas_offset+data_length-1, 'Gas_Decline_Type'] = gas_decline_types[:data_length]
                    
                    # Add gas forecast start date for reference
                    well_fcst['GasFcst_Start_Date'] = gas_fcst_start
                    
                    has_forecast = True
                    
        except Exception as e:
            print(f"Error processing gas forecast for well {well_api}: {str(e)}")
        
        # Generate oil forecast
        try:
            # Determine oil initial rate (qi) - use user input if available, otherwise calculated
            oil_qi = float(well_row['OIL_USER_QI']) if pd.notna(well_row['OIL_USER_QI']) else float(well_row['OIL_CALC_QI'])
            oil_qf = float(well_row['OIL_Q_MIN'])
            
            if oil_qi > oil_qf:
                # Use the oil forecast start date to determine the reference production data
                last_oil_prod = well_production[well_production['PRODUCINGMONTH'] <= oil_fcst_start]['LIQUIDSPROD_BBL'].tail(3)
                if not last_oil_prod.empty:
                    oil_qi = min(oil_qi, last_oil_prod.mean() * 1.1)
                
                oil_decline_type = well_row['OIL_DECLINE_TYPE']
                
                # Determine oil decline rate - use user input if available, otherwise empirical
                oil_decline = float(well_row['OIL_USER_DECLINE']) if pd.notna(well_row['OIL_USER_DECLINE']) else float(well_row['OIL_EMPIRICAL_DECLINE'])
                
                # Determine b-factor - use user input if available, otherwise calculated
                oil_b_factor = float(well_row['OIL_USER_B_FACTOR']) if pd.notna(well_row['OIL_USER_B_FACTOR']) else float(well_row['OIL_CALC_B_FACTOR'])
                
                oil_rates, oil_decline_types = calculate_decline_rates(
                    qi=oil_qi,
                    qf=oil_qf,
                    decline_type=oil_decline_type,
                    b_factor=oil_b_factor,
                    initial_decline=oil_decline,
                    terminal_decline=float(well_row['OIL_D_MIN'])
                )
                
                if len(oil_rates) > 0:
                    # Calculate the index offset for oil forecast start
                    oil_offset = (oil_fcst_start - earliest_fcst_start).days // 30
                    oil_offset = max(0, oil_offset)  # Ensure non-negative
                    
                    # Limit to available array length
                    data_length = min(len(oil_rates), len(well_fcst) - oil_offset)
                    
                    # Insert oil forecast at the correct position
                    well_fcst.loc[oil_offset:oil_offset+data_length-1, 'OilFcst_BBL'] = oil_rates[:data_length]
                    well_fcst.loc[oil_offset:oil_offset+data_length-1, 'Oil_Decline_Type'] = oil_decline_types[:data_length]
                    
                    # Add oil forecast start date for reference
                    well_fcst['OilFcst_Start_Date'] = oil_fcst_start
                    
                    has_forecast = True
                    
        except Exception as e:
            print(f"Error processing oil forecast for well {well_api}: {str(e)}")
        
        # Only continue if we have either oil or gas forecast
        if has_forecast:
            # Initialize all possible forecast columns to ensure they exist
            forecast_cols = ['GasFcst_MCF', 'OilFcst_BBL']
            for col in forecast_cols:
                if col not in well_fcst.columns:
                    well_fcst[col] = np.nan
            
            # Remove rows where all forecast columns are NaN
            well_fcst = well_fcst.dropna(subset=forecast_cols, how='all')
            
            # Ensure decline type columns exist
            for col in ['Oil_Decline_Type', 'Gas_Decline_Type']:
                if col not in well_fcst.columns:
                    well_fcst[col] = np.nan
            
    except Exception as e:
        print(f"Error processing well {well_api}: {str(e)}")
        well_fcst = pd.DataFrame()
    
    # Combine forecasts with historical data
    if not well_fcst.empty:
        # Handle potential duplicate dates between historical and forecast data
        final_df = pd.merge(aggregated_data, well_fcst, on='PRODUCINGMONTH', how='outer')
    else:
        final_df = aggregated_data.copy()
        for col in ['OilFcst_BBL', 'GasFcst_MCF']:
            final_df[col] = np.nan
    
    # Sort by date for cumulative calculations
    final_df = final_df.sort_values('PRODUCINGMONTH')
    
    # Calculate forecast cumulatives correctly handling custom forecast start dates
    
    # For gas forecast cumulatives
    if 'GasFcst_MCF' in final_df.columns and final_df['GasFcst_MCF'].notna().any():
        # Find the gas forecast start date
        gas_fcst_start_date = final_df['GasFcst_Start_Date'].iloc[0] if 'GasFcst_Start_Date' in final_df.columns else None
        
        if gas_fcst_start_date is not None:
            # Get the last actual production before forecast start
            gas_hist_before_fcst = final_df[final_df['PRODUCINGMONTH'] < gas_fcst_start_date]
            last_cum_gas = 0 if gas_hist_before_fcst.empty else gas_hist_before_fcst['CumGas_MCF'].iloc[-1]
            
            # Calculate cumulative from forecast start date
            fcst_mask = final_df['PRODUCINGMONTH'] >= gas_fcst_start_date
            final_df.loc[fcst_mask, 'GasFcstCum_MCF'] = final_df.loc[fcst_mask, 'GasFcst_MCF'].fillna(0).cumsum() + last_cum_gas
        else:
            # Fallback to previous method if forecast start date isn't available
            gas_idx = final_df['CumGas_MCF'].last_valid_index()
            if gas_idx is not None:
                last_cum_gas = final_df.loc[gas_idx, 'CumGas_MCF']
                mask = final_df.index > gas_idx
                final_df.loc[mask, 'GasFcstCum_MCF'] = last_cum_gas + final_df.loc[mask, 'GasFcst_MCF'].fillna(0).cumsum()
    
    # For oil forecast cumulatives
    if 'OilFcst_BBL' in final_df.columns and final_df['OilFcst_BBL'].notna().any():
        # Find the oil forecast start date
        oil_fcst_start_date = final_df['OilFcst_Start_Date'].iloc[0] if 'OilFcst_Start_Date' in final_df.columns else None
        
        if oil_fcst_start_date is not None:
            # Get the last actual production before forecast start
            oil_hist_before_fcst = final_df[final_df['PRODUCINGMONTH'] < oil_fcst_start_date]
            last_cum_oil = 0 if oil_hist_before_fcst.empty else oil_hist_before_fcst['CumLiquids_BBL'].iloc[-1]
            
            # Calculate cumulative from forecast start date
            fcst_mask = final_df['PRODUCINGMONTH'] >= oil_fcst_start_date
            final_df.loc[fcst_mask, 'OilFcstCum_BBL'] = final_df.loc[fcst_mask, 'OilFcst_BBL'].fillna(0).cumsum() + last_cum_oil
        else:
            # Fallback to previous method if forecast start date isn't available
            oil_idx = final_df['CumLiquids_BBL'].last_valid_index()
            if oil_idx is not None:
                last_cum_oil = final_df.loc[oil_idx, 'CumLiquids_BBL']
                mask = final_df.index > oil_idx
                final_df.loc[mask, 'OilFcstCum_BBL'] = last_cum_oil + final_df.loc[mask, 'OilFcst_BBL'].fillna(0).cumsum()
    
    # Create blended columns - properly handling custom forecast start dates
    
    # For oil blend, use historical data before FCST_START_OIL and forecast data after
    oil_fcst_start_date = final_df['OilFcst_Start_Date'].iloc[0] if 'OilFcst_Start_Date' in final_df.columns else None
    if oil_fcst_start_date is not None:
        # Use historical production before forecast start, and forecast after
        final_df['Oil_Blend'] = np.nan
        before_fcst_mask = final_df['PRODUCINGMONTH'] < oil_fcst_start_date
        after_fcst_mask = final_df['PRODUCINGMONTH'] >= oil_fcst_start_date
        
        final_df.loc[before_fcst_mask, 'Oil_Blend'] = final_df.loc[before_fcst_mask, 'LIQUIDSPROD_BBL']
        final_df.loc[after_fcst_mask, 'Oil_Blend'] = final_df.loc[after_fcst_mask, 'OilFcst_BBL']
    else:
        # Fallback to the original blend approach
        final_df['Oil_Blend'] = final_df['LIQUIDSPROD_BBL'].fillna(final_df['OilFcst_BBL'])
    
    # For gas blend, use historical data before FCST_START_GAS and forecast data after
    gas_fcst_start_date = final_df['GasFcst_Start_Date'].iloc[0] if 'GasFcst_Start_Date' in final_df.columns else None
    if gas_fcst_start_date is not None:
        # Use historical production before forecast start, and forecast after
        final_df['Gas_Blend'] = np.nan
        before_fcst_mask = final_df['PRODUCINGMONTH'] < gas_fcst_start_date
        after_fcst_mask = final_df['PRODUCINGMONTH'] >= gas_fcst_start_date
        
        final_df.loc[before_fcst_mask, 'Gas_Blend'] = final_df.loc[before_fcst_mask, 'GASPROD_MCF']
        final_df.loc[after_fcst_mask, 'Gas_Blend'] = final_df.loc[after_fcst_mask, 'GasFcst_MCF']
    else:
        # Fallback to the original blend approach
        final_df['Gas_Blend'] = final_df['GASPROD_MCF'].fillna(final_df['GasFcst_MCF'])
    
    # Replace zeros with NaN for numeric columns
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
    final_df[numeric_cols] = final_df[numeric_cols].replace(0, np.nan)
    
    # Calculate blended cumulative columns for oil and gas
    if 'Oil_Blend' in final_df.columns:
        final_df['OilBlend_Cum_BBL'] = final_df['Oil_Blend'].fillna(0).cumsum()
    
    if 'Gas_Blend' in final_df.columns:
        final_df['GasBlend_Cum_MCF'] = final_df['Gas_Blend'].fillna(0).cumsum()
    
    # Filter to end of 2050
    final_df = final_df[final_df['PRODUCINGMONTH'] <= '2050-12-31']
    
    # Add End of Month date column
    final_df['EOM_Date'] = final_df['PRODUCINGMONTH'].dt.to_period('M').dt.to_timestamp('M')
    
    # Set column order
    col_order = ['PRODUCINGMONTH', 'EOM_Date',
                 'LIQUIDSPROD_BBL', 'GASPROD_MCF', 'WATERPROD_BBL',
                 'CumLiquids_BBL', 'CumGas_MCF', 'CumWater_BBL',
                 'OilFcst_BBL', 'GasFcst_MCF', 'OilFcstCum_BBL', 'GasFcstCum_MCF',
                 'Oil_Blend', 'Gas_Blend', 'OilBlend_Cum_BBL', 'GasBlend_Cum_MCF',
                 'Oil_Decline_Type', 'Gas_Decline_Type',
                 'OilFcst_Start_Date', 'GasFcst_Start_Date']
    
    # Ensure all columns exist and add any additional columns at the end
    existing_cols = [col for col in col_order if col in final_df.columns]
    additional_cols = [col for col in final_df.columns if col not in col_order]
    final_df = final_df[existing_cols + additional_cols]
    
    return final_df

##########################################
# SECTION 2: SIDEBAR & DATABASE CONNECTION
##########################################

##deleteit
@st.cache_resource
def get_snowflake_session():
    connection_parameters = {'user':"ELII",
        'password':"Elii123456789!",
        'account':"CMZNSCB-MU47932",
        'warehouse':"COMPUTE_WH",
        'database':"WELLS",
        'schema':"MINERALS"}
    return Session.builder.configs(connection_parameters).create()

session = get_snowflake_session()
##dontdeleteit

##########################################
# SECTION 3: UTILITY FUNCTIONS - DATABASE
##########################################

@st.cache_data(ttl=3600)  # Cache for an hour, adjust as needed
def load_owners():
    query = 'SELECT * FROM WELLS.MINERALS.DISTINCT_OWNERS'
    result = session.sql(query).to_pandas()
    return result
all_owners = load_owners()
owner_filter = st.text_input("Enter owner name (or part of it)", "XTO ENERGY INC \(MIN-WI\)")
if owner_filter:
    owners = all_owners[all_owners['Owner'].str.contains(owner_filter, case=False)]
    if owners.shape[0]<=100:
        owner = st.selectbox("Filter by Owner", options=owners)
        st.divider()
        @st.cache_data(ttl=600)
        def load_wells(owner):
            query = f"""
                SELECT DISTINCT "API_10_EXPLODED" 
                FROM WELLS.MINERALS.DI_TX_MINERAL_APPRAISALS_2023_EXPLODED
                WHERE "Owner" = '{owner}'
                ORDER BY "API_10_EXPLODED" """
            result = session.sql(query).to_pandas()
            return result
        well_ids = load_wells(owner)["API_10_EXPLODED"].tolist()

@st.cache_data(ttl=600)
def get_filtered_well_data(well_ids):
    """Query well input view based on list of API_10_EXPLODED (dashless IDs)"""
    try:
        # Make sure we have something to query
        if not well_ids or len(well_ids) == 0:
            return pd.DataFrame()

        # Convert well list to SQL-safe string of quoted IDs
        formatted_ids = ",".join(f"'{w}'" for w in well_ids)

        query = f"""
        SELECT API_UWI, WELLNAME, STATEPROVINCE, COUNTRY, COUNTY, FIRSTPRODDATE, LATITUDE, LONGITUDE,
               ENVOPERATOR, LEASE, ENVWELLSTATUS, ENVINTERVAL, TRAJECTORY, CUMGAS_MCF, CUMOIL_BBL, TOTALPRODUCINGMONTHS
        FROM wells.minerals.vw_well_input
        WHERE REPLACE(API_UWI, '-', '') IN ({formatted_ids})
        """

        result = session.sql(query).to_pandas()

        if result.empty:
            st.warning("No well data retrieved from database.")
            return None

        result["CUMOIL_BBL"] = result["CUMOIL_BBL"].fillna(0)
        result["CUMGAS_MCF"] = result["CUMGAS_MCF"].fillna(0)

        return result
    except Exception as e:
        st.error(f"Error fetching well data: {e}")
        return None


@st.cache_data(ttl=600)
def get_production_data(selected_wells):
    try:
        selected_wells_query_string = ", ".join(f"'{api}'" for api in selected_wells)
        query = f"""
            SELECT API_UWI, ProducingMonth, LIQUIDSPROD_BBL, GASPROD_MCF, WATERPROD_BBL
            FROM wells.minerals.raw_prod_data
            WHERE API_UWI IN ({selected_wells_query_string})
            ORDER BY API_UWI, ProducingMonth;
            """ 
        # Execute query and convert to pandas
        result = session.sql(query).to_pandas()
        # Debug information to understand what's being returned
        if len(result) == 0:
            print(f"No production data found")
        else:
            result['PRODUCINGMONTH'] = pd.to_datetime(result['PRODUCINGMONTH'])
            print(f"Found {len(result)} production records for selected wells")
        return result
    except Exception as e:
        print(f"Error fetching production data for selected wells: {str(e)}")
        return pd.DataFrame()
@st.cache_data(ttl=600)
def get_decline_parameters(selected_wells):
    selected_wells_query_string = ", ".join(f"'{api}'" for api in selected_wells)
    try:
        query = f"""
            SELECT *
            FROM wells.minerals.econ_input
            WHERE API_UWI IN ({selected_wells_query_string})
            ORDER BY API_UWI;
            """
        return session.sql(query).to_pandas()
    except Exception as e:
        st.error(f"Error fetching decline parameters: {e}")
        return pd.DataFrame()

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

# Fetch Well Data for Filtering
@st.cache_data(ttl=600)
def get_well_data():
    """Get well data using direct query execution"""
    try:
        # from snowflake.snowpark.context import get_active_session
        # session = get_active_session()
        
        query = """
        SELECT API_UWI, WELLNAME, STATEPROVINCE, COUNTRY, COUNTY, FIRSTPRODDATE, LATITUDE, LONGITUDE,
               ENVOPERATOR, LEASE, ENVWELLSTATUS, ENVINTERVAL, TRAJECTORY, CUMGAS_MCF, CUMOIL_BBL, TOTALPRODUCINGMONTHS
        FROM wells.minerals.vw_well_input
        """
        
        # Execute the query directly using the session
        result = session.sql(query).to_pandas()
        
        if result.empty:
            st.warning("No well data retrieved from database.")
            return None
            
        # Fill NaNs in the CUMOIL_BBL and CUMGAS_MCF columns with 0
        result["CUMOIL_BBL"] = result["CUMOIL_BBL"].fillna(0)
        result["CUMGAS_MCF"] = result["CUMGAS_MCF"].fillna(0)
        
        return result
    except Exception as e:
        st.error(f"Error fetching well data: {e}")
        return None

# Update database record function
def update_database_record(table_name, primary_key, key_value, update_values, table_columns):
    """Execute update SQL using Snowpark Session"""
    try:
        # Handle DATE fields (convert empty strings to NULL)
        update_values = {
            col: f"'{value}'" if value else "NULL"
            if table_columns.get(col, "") != "DATE" else "NULL" if value == "" else f"'{value}'"
            for col, value in update_values.items()
        }

        set_clause = ", ".join([f"{col} = {value}" for col, value in update_values.items()])
        sql = f"UPDATE {table_name} SET {set_clause} WHERE {primary_key} = '{key_value}'"

        # from snowflake.snowpark.context import get_active_session
        # session = get_active_session()
        session.sql(sql).collect()
        return True, "Record updated successfully!"
    except Exception as e:
        return False, f"Error updating record: {e}"

##########################################
# SECTION 4: UTILITY FUNCTIONS - PRODUCTION DATA
##########################################
# Fetch Production Data
# @st.cache_data(ttl=600)
# def get_production_data(api_uwi):
#     try:
#         # from snowflake.snowpark.context import get_active_session
#         # session = get_active_session()
        
#         query = f"""
#         SELECT API_UWI, ProducingMonth, LIQUIDSPROD_BBL, GASPROD_MCF
#         FROM wells.minerals.raw_prod_data
#         WHERE API_UWI = '{api_uwi}'
#         ORDER BY ProducingMonth;
#         """
        
#         # Execute query and convert to pandas
#         result = session.sql(query).to_pandas()
        
#         # Debug information to understand what's being returned
#         if len(result) == 0:
#             print(f"No production data found for API_UWI: {api_uwi}")
#         else:
#             print(f"Found {len(result)} production records for API_UWI: {api_uwi}")
        
#         return result
#     except Exception as e:
#         print(f"Error fetching production data for {api_uwi}: {str(e)}")
#         return pd.DataFrame()

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

# Get the last production date for a well, separately for oil and gas
def get_last_production_dates(prd):
    # Get production data and convert directly to pandas DataFrame
    raw_data = prd
    
    if raw_data.empty:
        print(f"No production data")
        return None, None
    
    try:
        # Print the first few rows to debug
        print(f"First few rows of production data:")
        print(raw_data.head())
        
        # Ensure LIQUIDSPROD_BBL and GASPROD_MCF are numeric
        raw_data['LIQUIDSPROD_BBL'] = pd.to_numeric(raw_data['LIQUIDSPROD_BBL'], errors='coerce').fillna(0)
        raw_data['GASPROD_MCF'] = pd.to_numeric(raw_data['GASPROD_MCF'], errors='coerce').fillna(0)
        
        # Find the last non-zero oil production
        non_zero_oil = raw_data[raw_data['LIQUIDSPROD_BBL'] > 0]
        last_oil_date = None
        if not non_zero_oil.empty:
            # Sort by date to get the most recent
            non_zero_oil = non_zero_oil.sort_values('PRODUCINGMONTH')
            last_oil_date = non_zero_oil['PRODUCINGMONTH'].iloc[-1]
            print(f"Found last oil date: {last_oil_date}")
        
        # Find the last non-zero gas production
        non_zero_gas = raw_data[raw_data['GASPROD_MCF'] > 0]
        last_gas_date = None
        if not non_zero_gas.empty:
            # Sort by date to get the most recent
            non_zero_gas = non_zero_gas.sort_values('PRODUCINGMONTH')
            last_gas_date = non_zero_gas['PRODUCINGMONTH'].iloc[-1]
            print(f"Found last gas date: {last_gas_date}")
        
        return last_oil_date, last_gas_date
    except Exception as e:
        print(f"Error in get_last_production_dates: {e}")
        # Print the column names for debugging
        print(f"Column names: {raw_data.columns.tolist()}")
        return None, None

##########################################
# SECTION 5: UTILITY FUNCTIONS - CALCULATIONS
##########################################
# Calculate Oil_Calc_Qi and Gas_Calc_Qi based on 3-month average around the last production date
def calculate_qi_values(production_data, last_oil_date, last_gas_date):
    raw_data = production_data
    if raw_data.empty:
        print(f"No production data when calculating Qi values")
        return 0, 0
    try:      
        # Ensure LIQUIDSPROD_BBL and GASPROD_MCF are numeric
        raw_data['LIQUIDSPROD_BBL'] = pd.to_numeric(raw_data['LIQUIDSPROD_BBL'], errors='coerce').fillna(0)
        raw_data['GASPROD_MCF'] = pd.to_numeric(raw_data['GASPROD_MCF'], errors='coerce').fillna(0)
        
        # Calculate Oil Qi
        oil_qi = 0
        if last_oil_date is not None:
            try:
                # Convert the date column to datetime if it isn't already
                if not pd.api.types.is_datetime64_any_dtype(raw_data['PRODUCINGMONTH']):
                    raw_data['PRODUCINGMONTH'] = pd.to_datetime(raw_data['PRODUCINGMONTH'], errors='coerce')
                
                # Ensure last_oil_date is datetime
                if not isinstance(last_oil_date, pd.Timestamp):
                    last_oil_date = pd.to_datetime(last_oil_date)
                
                # Calculate window - use 6 month window centered on last production date
                start_date = last_oil_date - pd.DateOffset(months=3)
                end_date = last_oil_date + pd.DateOffset(months=3)
                
                # Create window around last production date
                window_data = raw_data[
                    (raw_data['PRODUCINGMONTH'] >= start_date) & 
                    (raw_data['PRODUCINGMONTH'] <= end_date)
                ]
                
                # Calculate average of non-zero oil production in the window
                oil_window = window_data[window_data['LIQUIDSPROD_BBL'] > 0]
                if not oil_window.empty:
                    oil_qi = oil_window['LIQUIDSPROD_BBL'].mean()
                    print(f"Calculated oil Qi from {len(oil_window)} records: {oil_qi}")
            except Exception as e:
                print(f"Error calculating oil Qi: {e}")
        
        # Calculate Gas Qi
        gas_qi = 0
        if last_gas_date is not None:
            try:
                # Convert the date column to datetime if it isn't already
                if not pd.api.types.is_datetime64_any_dtype(raw_data['PRODUCINGMONTH']):
                    raw_data['PRODUCINGMONTH'] = pd.to_datetime(raw_data['PRODUCINGMONTH'], errors='coerce')
                
                # Ensure last_gas_date is datetime
                if not isinstance(last_gas_date, pd.Timestamp):
                    last_gas_date = pd.to_datetime(last_gas_date)
                
                # Calculate window - use 6 month window centered on last production date
                start_date = last_gas_date - pd.DateOffset(months=3)
                end_date = last_gas_date + pd.DateOffset(months=3)
                
                # Create window around last production date
                window_data = raw_data[
                    (raw_data['PRODUCINGMONTH'] >= start_date) & 
                    (raw_data['PRODUCINGMONTH'] <= end_date)
                ]
                
                # Calculate average of non-zero gas production in the window
                gas_window = window_data[window_data['GASPROD_MCF'] > 0]
                if not gas_window.empty:
                    gas_qi = gas_window['GASPROD_MCF'].mean()
                    print(f"Calculated gas Qi from {len(gas_window)} records: {gas_qi}")
            except Exception as e:
                print(f"Error calculating gas Qi: {e}")
        
        return oil_qi, gas_qi
    except Exception as e:
        print(f"Error in calculate_qi_values: {e}")
        return 0, 0

# Calculate Decline Fit with User-Defined Constants
def calculate_decline_fit(production_df, months=12, default_decline=0.06, min_decline=0.06, max_decline=0.98):
    if production_df.empty:
        return default_decline, default_decline
    
    # Sort by date if not already sorted
    if 'PRODUCINGMONTH' in production_df.columns:
        production_df = production_df.sort_values('PRODUCINGMONTH')
    
    # Limit to the most recent "months" number of records
    if len(production_df) > months:
        production_df = production_df.tail(months)
        
    # Create a copy for oil calculations
    oil_df = production_df.copy()
    oil_df['GASPROD_MCF'] = oil_df['LIQUIDSPROD_BBL']  # Use liquid production in place of gas for oil calculations

    def decline_rate(rates):
        # Need at least 6 points for calculation
        if len(rates) < 6:
            return default_decline
            
        # Create a smoothed version of the data
        smoothed = pd.Series(rates).rolling(6, center=True).mean().dropna()
        
        if len(smoothed) < 2:
            return default_decline
            
        # Calculate decline from first to last point in smoothed data
        # Avoid division by zero
        if smoothed.iloc[0] <= 0:
            return default_decline
            
        decline = (smoothed.iloc[0] - smoothed.iloc[-1]) / smoothed.iloc[0]
        
        # Convert to annual rate and ensure it's within bounds
        if decline < 0:  # Handle increasing production
            return min_decline
            
        # Calculate annualized decline rate
        annual_decline = 1 - (1 - decline) ** (12 / (len(smoothed) - 1))
        
        # Ensure result is within specified bounds
        return min(max(annual_decline, min_decline), max_decline)

    # Calculate decline rates
    oil_decline = decline_rate(oil_df['LIQUIDSPROD_BBL'].values)
    gas_decline = decline_rate(production_df['GASPROD_MCF'].values)

    return oil_decline, gas_decline

# Function to fix common forecast data issues
def fix_forecast_data_issues(well_row):
    """
    Fix common issues with forecast parameters that might prevent successful forecasting
    """
    fixed_row = well_row.copy()
    
    # Fix decline types (ensure they are E or H)
    if pd.isna(fixed_row['OIL_DECLINE_TYPE']) or fixed_row['OIL_DECLINE_TYPE'] not in ['E', 'H', 'EXP', 'HYP']:
        fixed_row['OIL_DECLINE_TYPE'] = 'E'  # Default to exponential
    elif fixed_row['OIL_DECLINE_TYPE'] == 'EXP':
        fixed_row['OIL_DECLINE_TYPE'] = 'E'
    elif fixed_row['OIL_DECLINE_TYPE'] == 'HYP':
        fixed_row['OIL_DECLINE_TYPE'] = 'H'
    
    if pd.isna(fixed_row['GAS_DECLINE_TYPE']) or fixed_row['GAS_DECLINE_TYPE'] not in ['E', 'H', 'EXP', 'HYP']:
        fixed_row['GAS_DECLINE_TYPE'] = 'E'  # Default to exponential
    elif fixed_row['GAS_DECLINE_TYPE'] == 'EXP':
        fixed_row['GAS_DECLINE_TYPE'] = 'E'
    elif fixed_row['GAS_DECLINE_TYPE'] == 'HYP':
        fixed_row['GAS_DECLINE_TYPE'] = 'H'
    
    # Ensure decline rates are valid
    if pd.isna(fixed_row['OIL_USER_DECLINE']) and pd.isna(fixed_row['OIL_EMPIRICAL_DECLINE']):
        fixed_row['OIL_EMPIRICAL_DECLINE'] = 0.06  # Default value
    
    if pd.isna(fixed_row['GAS_USER_DECLINE']) and pd.isna(fixed_row['GAS_EMPIRICAL_DECLINE']):
        fixed_row['GAS_EMPIRICAL_DECLINE'] = 0.06  # Default value
    
    # Ensure b-factors are valid for hyperbolic decline
    if fixed_row['OIL_DECLINE_TYPE'] == 'H':
        if pd.isna(fixed_row['OIL_USER_B_FACTOR']) and pd.isna(fixed_row['OIL_CALC_B_FACTOR']):
            fixed_row['OIL_CALC_B_FACTOR'] = 1.0  # Default value
    
    if fixed_row['GAS_DECLINE_TYPE'] == 'H':
        if pd.isna(fixed_row['GAS_USER_B_FACTOR']) and pd.isna(fixed_row['GAS_CALC_B_FACTOR']):
            fixed_row['GAS_CALC_B_FACTOR'] = 1.0  # Default value
    
    # Ensure minimum values are valid
    if pd.isna(fixed_row['OIL_Q_MIN']) or fixed_row['OIL_Q_MIN'] <= 0:
        fixed_row['OIL_Q_MIN'] = 1.0  # Default value
    
    if pd.isna(fixed_row['GAS_Q_MIN']) or fixed_row['GAS_Q_MIN'] <= 0:
        fixed_row['GAS_Q_MIN'] = 10.0  # Default value
    
    if pd.isna(fixed_row['OIL_D_MIN']) or fixed_row['OIL_D_MIN'] <= 0:
        fixed_row['OIL_D_MIN'] = 0.06  # Default value
    
    if pd.isna(fixed_row['GAS_D_MIN']) or fixed_row['GAS_D_MIN'] <= 0:
        fixed_row['GAS_D_MIN'] = 0.06  # Default value
    
    return fixed_row

# Function to validate forecast parameters
def validate_forecast_parameters(well_row):
    """
    Validates that all necessary parameters for forecasting are present and valid.
    Returns (is_valid, message)
    """
    try:
        # Check oil parameters
        oil_valid = True
        oil_messages = []
        
        # Check initial rate (qi)
        if pd.isna(well_row['OIL_USER_QI']) and pd.isna(well_row['OIL_CALC_QI']):
            oil_valid = False
            oil_messages.append("No oil initial rate (Qi) available")
        elif pd.notna(well_row['OIL_USER_QI']) and float(well_row['OIL_USER_QI']) <= 0:
            oil_valid = False
            oil_messages.append(f"Invalid oil initial rate: {well_row['OIL_USER_QI']}")
        elif pd.notna(well_row['OIL_CALC_QI']) and float(well_row['OIL_CALC_QI']) <= 0:
            oil_valid = False
            oil_messages.append(f"Invalid calculated oil initial rate: {well_row['OIL_CALC_QI']}")
            
        # Check decline rate
        if pd.isna(well_row['OIL_USER_DECLINE']) and pd.isna(well_row['OIL_EMPIRICAL_DECLINE']):
            oil_valid = False
            oil_messages.append("No oil decline rate available")
        
        # Check decline type
        if pd.isna(well_row['OIL_DECLINE_TYPE']) or well_row['OIL_DECLINE_TYPE'] not in ['E', 'H', 'EXP', 'HYP']:
            oil_valid = False
            oil_messages.append(f"Invalid oil decline type: {well_row['OIL_DECLINE_TYPE']}")
            
        # Check minimum rate
        if pd.isna(well_row['OIL_Q_MIN']) or float(well_row['OIL_Q_MIN']) <= 0:
            oil_valid = False
            oil_messages.append(f"Invalid oil minimum rate: {well_row['OIL_Q_MIN']}")
            
        # Check gas parameters similarly
        gas_valid = True
        gas_messages = []
        
        # Check initial rate (qi)
        if pd.isna(well_row['GAS_USER_QI']) and pd.isna(well_row['GAS_CALC_QI']):
            gas_valid = False
            gas_messages.append("No gas initial rate (Qi) available")
        elif pd.notna(well_row['GAS_USER_QI']) and float(well_row['GAS_USER_QI']) <= 0:
            gas_valid = False
            gas_messages.append(f"Invalid gas initial rate: {well_row['GAS_USER_QI']}")
        elif pd.notna(well_row['GAS_CALC_QI']) and float(well_row['GAS_CALC_QI']) <= 0:
            gas_valid = False
            gas_messages.append(f"Invalid calculated gas initial rate: {well_row['GAS_CALC_QI']}")
            
        # Check decline rate
        if pd.isna(well_row['GAS_USER_DECLINE']) and pd.isna(well_row['GAS_EMPIRICAL_DECLINE']):
            gas_valid = False
            gas_messages.append("No gas decline rate available")
        
        # Check decline type
        if pd.isna(well_row['GAS_DECLINE_TYPE']) or well_row['GAS_DECLINE_TYPE'] not in ['E', 'H', 'EXP', 'HYP']:
            gas_valid = False
            gas_messages.append(f"Invalid gas decline type: {well_row['GAS_DECLINE_TYPE']}")
            
        # Check minimum rate
        if pd.isna(well_row['GAS_Q_MIN']) or float(well_row['GAS_Q_MIN']) <= 0:
            gas_valid = False
            gas_messages.append(f"Invalid gas minimum rate: {well_row['GAS_Q_MIN']}")
            
        # At least one of oil or gas needs to be valid for a forecast
        if not oil_valid and not gas_valid:
            return False, "Neither oil nor gas parameters are valid for forecasting: " + \
                   "; ".join(oil_messages + gas_messages)
        
        return True, "Parameters valid for forecasting"
    
    except Exception as e:
        return False, f"Error validating forecast parameters: {str(e)}"

##########################################
# SECTION 6: SESSION STATE INITIALIZATION
##########################################
# Initialize variables for decline calculation constants for single well
if 'single_months_for_calc' not in st.session_state:
    st.session_state['single_months_for_calc'] = 24
if 'single_default_decline' not in st.session_state:
    st.session_state['single_default_decline'] = 0.06
if 'single_min_decline' not in st.session_state:
    st.session_state['single_min_decline'] = 0.06
if 'single_max_decline' not in st.session_state:
    st.session_state['single_max_decline'] = 0.98

# Initialize session state for calculated declines if it doesn't exist
if 'calculated_declines' not in st.session_state:
    st.session_state['calculated_declines'] = {}

# Initialize session state for selected wells
if 'selected_wells' not in st.session_state:
    st.session_state['selected_wells'] = []

# Initialize session state for filtered wells
if 'filtered_wells' not in st.session_state:
    st.session_state['filtered_wells'] = None

# Initialize session state for forecast settings
if 'forecast_enabled' not in st.session_state:
    st.session_state['forecast_enabled'] = True
if 'forecast_years' not in st.session_state:
    st.session_state['forecast_years'] = 25
if 'show_oil_forecast' not in st.session_state:
    st.session_state['show_oil_forecast'] = True
if 'show_gas_forecast' not in st.session_state:
    st.session_state['show_gas_forecast'] = True
if 'forecast_data' not in st.session_state:
    st.session_state['forecast_data'] = {}

# Initialize session state for pending form values
if 'pending_form_values' not in st.session_state:
    st.session_state['pending_form_values'] = {}

##########################################
# SECTION 7: FIELD DEFINITIONS
##########################################
# Define ordered oil and gas parameter fields
oil_fields = [
    "OIL_EMPIRICAL_DECLINE",
    "OIL_PROPHET_DECLINE",
    "OIL_DECLINE_TYPE",
    "OIL_USER_DECLINE",
    "OIL_CALC_QI",
    "OIL_USER_QI",
    "OIL_CALC_B_FACTOR",
    "OIL_USER_B_FACTOR",
    "OIL_D_MIN",
    "OIL_Q_MIN",
    "OIL_FCST_YRS"
]

gas_fields = [
    "GAS_EMPIRICAL_DECLINE",
    "GAS_PROPHET_DECLINE",
    "GAS_DECLINE_TYPE",
    "GAS_USER_DECLINE",
    "GAS_CALC_QI",
    "GAS_USER_QI",
    "GAS_CALC_B_FACTOR",
    "GAS_USER_B_FACTOR",
    "GAS_D_MIN",
    "GAS_Q_MIN",
    "GAS_FCST_YRS"
]

# Function to filter and order fields in table_columns
def get_ordered_fields(fields, all_columns):
    return [(field, all_columns.get(field)) for field in fields if field in all_columns]

##########################################
# SECTION 8: WELL FILTER AND MAP SECTION
##########################################
# Create a two-column layout with filters on left and map on right
filter_column, map_column = st.columns([1, 1])

# Put all filters in the left column
with filter_column:
    st.subheader("Filter Wells")
    
    # Load well data for filtering
    well_data = get_filtered_well_data(well_ids)
    print(f'rows in well_data:{well_data.shape[0]}')
    if well_data is not None:
        
        # Initialize filtered_wells to be the original well data
        filtered_wells = well_data.copy()
        
        ## set two columns for filters
        filter_column_col1, filter_column_col2 = st.columns([1, 1])
        with filter_column_col1:
            # Define filter columns in the specified order
            filters = [
                "ENVOPERATOR",
                "API_UWI",
                "ENVWELLSTATUS",
            ]
            # Apply filters - using full width since we're in a column already
            for filter in filters:
                if filter in well_data.columns:
                    # Get unique values for the current filter, based on data filtered so far
                    unique_values = sorted(filtered_wells[filter].dropna().unique().tolist())
                    # Create multiselect widget
                    selected_values = st.multiselect(
                        f"Select {filter}:",
                        options=unique_values,
                        default=[]
                    )
                    # Apply filter if values are selected
                    if selected_values:
                        filtered_wells = filtered_wells[filtered_wells[filter].isin(selected_values)]
                        st.session_state['filtered_wells'] = filtered_wells
                        st.session_state['selected_wells'] = filtered_wells["API_UWI"].tolist()

            # Add range sliders for production data
            # Oil production range slider
            if "CUMOIL_BBL" in well_data.columns:
                max_oil_value = int(well_data["CUMOIL_BBL"].max())
                oil_range = st.slider(
                    "Total Oil Production (BBL)", 
                    min_value=0, 
                    max_value=max_oil_value, 
                    value=(0, max_oil_value)
                )
                filtered_wells = filtered_wells[(filtered_wells["CUMOIL_BBL"] >= oil_range[0]) & (filtered_wells["CUMOIL_BBL"] <= oil_range[1])]
                st.session_state['filtered_wells'] = filtered_wells
                st.session_state['selected_wells'] = filtered_wells["API_UWI"].tolist()

        with filter_column_col2:
            # Define filter columns in the specified order
            filters = [
                "WELLNAME", 
                "TRAJECTORY", 
                "COUNTY", 
            ]
            # Apply filters - using full width since we're in a column already
            for filter in filters:
                if filter in well_data.columns:
                    # Get unique values for the current filter, based on data filtered so far
                    unique_values = sorted(filtered_wells[filter].dropna().unique().tolist())
                    
                    # Create multiselect widget
                    selected_values = st.multiselect(
                        f"Select {filter}:",
                        options=unique_values,
                        default=[]
                    )
                    
                    # Apply filter if values are selected
                    if selected_values:
                        filtered_wells = filtered_wells[filtered_wells[filter].isin(selected_values)]
                        st.session_state['filtered_wells'] = filtered_wells
                        st.session_state['selected_wells'] = filtered_wells["API_UWI"].tolist()
            # Gas production range slider
            if "CUMGAS_MCF" in well_data.columns:
                max_gas_value = int(well_data["CUMGAS_MCF"].max())
                gas_range = st.slider(
                    "Total Gas Production (MCF)", 
                    min_value=0, 
                    max_value=max_gas_value, 
                    value=(0, max_gas_value)
                )
                filtered_wells = filtered_wells[(filtered_wells["CUMGAS_MCF"] >= gas_range[0]) & (filtered_wells["CUMGAS_MCF"] <= gas_range[1])]
                st.session_state['filtered_wells'] = filtered_wells
                st.session_state['selected_wells'] = filtered_wells["API_UWI"].tolist()

        # Total producing months range slider
        if "TOTALPRODUCINGMONTHS" in well_data.columns:
            max_months = int(well_data["TOTALPRODUCINGMONTHS"].max())
            months_range = st.slider(
                "Total Producing Months", 
                min_value=0, 
                max_value=max_months, 
                value=(0, max_months)
            )
            filtered_wells = filtered_wells[(filtered_wells["TOTALPRODUCINGMONTHS"] >= months_range[0]) & (filtered_wells["TOTALPRODUCINGMONTHS"] <= months_range[1])]
            
        # Display filtered well count
        st.write(f"Filtered Wells: {len(filtered_wells)}")
        st.write('')
        st.write('')

        # Show a sample of filtered wells in a table below the map
        with st.expander("Filtered Wells Sample", expanded=False):
            display_cols = ["API_UWI", "WELLNAME", "ENVOPERATOR", "COUNTY", "CUMOIL_BBL", "CUMGAS_MCF"]
            display_cols = [col for col in display_cols if col in filtered_wells.columns]
            st.dataframe(filtered_wells[display_cols].head(10000), height = 250)
    else:
        st.warning("Could not load well data for filtering.")

# Put the map in the right column
with map_column:
    st.subheader("Well Map")
    
    # Only proceed if we have well data and filtered wells
    if 'filtered_wells' in locals() and not filtered_wells.empty:
        if 'LATITUDE' in filtered_wells.columns and 'LONGITUDE' in filtered_wells.columns:
            # Create a copy with just the needed columns
            map_data = filtered_wells[['LATITUDE', 'LONGITUDE']].copy()
            
            # Rename columns to what st.map expects
            map_data.columns = ['latitude', 'longitude']
            
            # Drop any rows with missing coordinates
            map_data = map_data.dropna()
            
            if not map_data.empty:
                # Simple fixed-size map - no color or size variations
                st.map(map_data)
                
                # Add information about the map
                st.info(f"Map shows {len(map_data)} filtered wells.")
                
            else:
                st.warning("Cannot display map: No valid coordinate data available")
        else:
            st.warning("Cannot display map: Latitude or longitude data is missing")
    else:
        st.info("No filtered wells to display. Use the filters on the left to select wells.")
st.divider()
# Add visualization of top operators and counties after the two-column layout
# This is outside the columns to use full width
if 'filtered_wells' in locals() and not filtered_wells.empty:
    st.subheader("Well Distribution")
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2 = st.tabs(["By Operator", "By County"])
    
    with viz_tab1:
        operator_counts = filtered_wells["ENVOPERATOR"].value_counts().reset_index()
        operator_counts.columns = ["Operator", "Count"]
        # Limit to top 10 operators
        if len(operator_counts) > 10:
            operator_counts = operator_counts.head(10)
            title_operator = "Top 10 Operators"
        else:
            title_operator = "Operators"

        chart_operator = alt.Chart(operator_counts).mark_bar().encode(
            x=alt.X("Count:Q", title="Count"),
            y=alt.Y("Operator:N", title="Operator", sort="-x"),
            tooltip=["Operator", "Count"]
        ).properties(title=title_operator, height=250)
        
        st.altair_chart(chart_operator, use_container_width=True)
    
    with viz_tab2:
        county_counts = filtered_wells["COUNTY"].value_counts().reset_index()
        county_counts.columns = ["County", "Count"]
        # Limit to top 10 counties
        if len(county_counts) > 10:
            county_counts = county_counts.head(10)
            title_county = "Top 10 Counties"
        else:
            title_county = "Counties"

        chart_county = alt.Chart(county_counts).mark_bar().encode(
            x=alt.X("Count:Q", title="Count"),
            y=alt.Y("County:N", title="County", sort="-x"),
            tooltip=["County", "Count"]
        ).properties(title=title_county, height=250)
        
        st.altair_chart(chart_county, use_container_width=True)
        
##########################################
# SECTION 10A: SINGLE WELL UPDATING - LEFT COLUMN
##########################################
st.divider()
st.title("Single Well Data Viewer & Editor")

if st.session_state['filtered_wells'] is not None:

    ##query for prd and decline parameters for filtered wells
    wells_list = st.session_state['filtered_wells']['API_UWI'].tolist()
    raw_prod_data_df = get_production_data(wells_list)
    decline_parameters = get_decline_parameters(wells_list)
    # Create a DataFrame with API and Well Name
    wells_data = []
    for well in wells_list:
        # Get the well name from the filtered wells DataFrame
        well_name = st.session_state['filtered_wells'].loc[
            st.session_state['filtered_wells']['API_UWI'] == well, 'WELLNAME'
        ].values[0] if 'WELLNAME' in st.session_state['filtered_wells'].columns else "N/A"
        
        wells_data.append({
            'API': well,
            'Well Name': well_name,
            # 'Current': well == selected_key
        })
    wells_df = pd.DataFrame(wells_data)

    ## limit to only rwos with production data
    wells_df = wells_df[wells_df['API'].isin(raw_prod_data_df['API_UWI'])]
    ##default select first well in list
    selected_key = wells_df.iloc[0]['API']
    record = decline_parameters[decline_parameters['API_UWI'] == selected_key].iloc[0]
    record_well_data = filtered_wells[filtered_wells['API_UWI'] == selected_key].iloc[0]
else:
    st.info("No wells filtered. Use the filter section to select wells.")
    selected_key = None

if not decline_parameters.empty and 'API_UWI' in decline_parameters.columns:
    # Create a three-column layout with API selection on the left
    left_column, middle_column, right_column = st.columns([0.75, 1, 2])
    with left_column:
        st.markdown("## Well Selection")
        # Display a list of wells below the dropdown
        if st.session_state['filtered_wells'] is not None:
            # Configure AG Grid
            gb = GridOptionsBuilder.from_dataframe(wells_df)
            # gb.configure_selection('single')  # 'multiple' for multiselect
            gb.configure_selection(selection_mode='single', use_checkbox=False)
            grid_options = gb.build()
            grid_options['preSelectedRows'] = [0] 
            grid_response = AgGrid(
                wells_df,
                gridOptions=grid_options,
                # update_mode=GridUpdateMode.SELECTION_CHANGED,
                # height=200,
                allow_unsafe_jscode=True,
                theme='streamlit'
            )
            try:
                selected_key = grid_response['selected_rows']['API'].values[0]
            except:
                selected_key = wells_df.iloc[0]['API']

    ## calc and collect forecast df
    prd = raw_prod_data_df[raw_prod_data_df['API_UWI'] == selected_key].copy()
    params = decline_parameters[decline_parameters['API_UWI'] == selected_key].copy().iloc[0].to_dict()
##########################################
# SECTION 10B: SINGLE WELL UPDATING - MIDDLE COLUMN
##########################################
    
    with middle_column:
        st.markdown("## Decline Parameters")
        with st.expander('Gas Decline Parameters', expanded=False):
            ## DECLINE TYPE SELECTOR
            gas_decl_type_options = ['EXP','HYP']
            default_index_gas_decl_typ = gas_decl_type_options.index(params['GAS_DECLINE_TYPE'])
            params['GAS_DECLINE_TYPE'] = st.radio("Choose one:", gas_decl_type_options, index = default_index_gas_decl_typ, horizontal = True)

            mid_left_col, mid_right_col = st.columns([1, 1])
            with mid_left_col:
                ## forecast start date month selector
                # Set the range of the slider
                start_date = prd['PRODUCINGMONTH'].min().date()#datetime(2015, 1, 1).date() 
                end_date = datetime.today().date()

                # Generate a list of dates in 1-month increments
                dates = []
                current = start_date
                while current <= end_date:
                    dates.append(current)
                    current += relativedelta(months=1)

                # Show the slider
                params['FCST_START_GAS'] = st.select_slider(
                    "Forecast Start Date",
                    options=dates,
                    value=params['FCST_START_GAS'],
                    format_func=lambda d: d.strftime("%b %Y"),
                    key = 'gas_start_date'
                )
                ## gas qi user input
                params['GAS_USER_QI'] = st.slider(
                    f"Qi",
                    min_value=10.0,
                    max_value=max(prd['GASPROD_MCF']),
                    value=params['GAS_USER_QI'],
                    step=1.0,
                    key = 'GAS_USER_QI',
                    # help="Select the initial production rate (Qi) for the forecast."
                )
                

            with mid_right_col:
                ## gas Di user input
                ## range conditional on Hyp or Exp
                if params['GAS_DECLINE_TYPE'] == 'HYP':
                    params['GAS_USER_DECLINE'] = st.slider(
                        f"Di",
                        min_value=0.005,
                        max_value=0.99,
                        value=0.4 if pd.isna(params['GAS_USER_DECLINE']) else params['GAS_USER_DECLINE'],
                        step=0.01,
                        key = 'GAS_USER_DECLINE_HYP',
                        # help="Select the initial decline rate (Di) for the forecast."
                    )
                    params['GAS_USER_B_FACTOR'] = st.slider(
                        f"b-factor",
                        min_value=0.01,
                        max_value=1.0,
                        value=0.4 if pd.isna(params['GAS_USER_B_FACTOR']) else params['GAS_USER_B_FACTOR'],
                        step=0.01,
                        key = 'GAS_USER_B_FACTOR_HYP',
                        # help="Select the initial decline rate (Di) for the forecast."
                    )
                else:
                    params['GAS_USER_DECLINE'] = st.slider(
                        f"Di",
                        min_value=0.005,
                        max_value=1.6,
                        value=0.4 if pd.isna(params['GAS_USER_DECLINE']) else params['GAS_USER_DECLINE'],
                        step=0.001,
                        key = 'GAS_USER_DECLINE_EXP',
                        # help="Select the initial decline rate (Di) for the forecast."
                    )
        with st.expander('Oil Decline Parameters', expanded=False):
            ## DECLINE TYPE SELECTOR
            gas_decl_type_options = ['EXP','HYP']
            default_index_gas_decl_typ = gas_decl_type_options.index(params['OIL_DECLINE_TYPE'])
            params['OIL_DECLINE_TYPE'] = st.radio("Choose one:", gas_decl_type_options, index = default_index_gas_decl_typ, key = 'OIL_DECLINE_TYPE', horizontal = True)

            mid_left_col, mid_right_col = st.columns([1, 1])
            with mid_left_col:
                ## forecast start date month selector
                # Set the range of the slider
                start_date = prd['PRODUCINGMONTH'].min().date()
                end_date = datetime.today().date()

                # Generate a list of dates in 1-month increments
                dates = []
                current = start_date
                while current <= end_date:
                    dates.append(current)
                    current += relativedelta(months=1)

                # Show the slider
                params['FCST_START_OIL'] = st.select_slider(
                    "Forecast Start Date",
                    options=dates,
                    value=params['FCST_START_OIL'],
                    format_func=lambda d: d.strftime("%b %Y"),
                    key = 'oil_start_date'
                )
                ## gas qi user input
                params['OIL_USER_QI'] = st.slider(
                    f"Qi",
                    min_value=1.0,
                    max_value=max(prd['LIQUIDSPROD_BBL']) if max(prd['LIQUIDSPROD_BBL'])>1 else 2.0,
                    value=params['OIL_USER_QI'],
                    step=1.0,
                    # help="Select the initial production rate (Qi) for the forecast."
                )
                

            with mid_right_col:
                ## gas Di user input
                ## range conditional on Hyp or Exp
                if params['OIL_DECLINE_TYPE'] == 'HYP':
                    params['OIL_USER_DECLINE'] = st.slider(
                        f"Di",
                        min_value=0.005,
                        max_value=0.99,
                        value=0.4 if pd.isna(params['OIL_USER_DECLINE']) else params['OIL_USER_DECLINE'],
                        step=0.01,
                        key = 'oil_decline_hyp'
                        # help="Select the initial decline rate (Di) for the forecast."
                    )
                else:
                    params['OIL_USER_DECLINE'] = st.slider(
                        f"Di",
                        min_value=0.005,
                        max_value=1.6,
                        value=0.4 if pd.isna(params['OIL_USER_DECLINE']) else params['OIL_USER_DECLINE'],
                        step=0.001,
                        key = 'oil_decline_exp'
                        # help="Select the initial decline rate (Di) for the forecast."
                    )
                ## gas b-factor user input
                ## range conditional on Hyp or Exp
                if params['OIL_DECLINE_TYPE'] == 'HYP':
                    params['OIL_USER_B_FACTOR'] = st.slider(
                        f"b-factor",
                        min_value=0.01,
                        max_value=1.0,
                        value=0.4 if pd.isna(params['OIL_USER_B_FACTOR']) else params['OIL_USER_B_FACTOR'],
                        step=0.01,
                        key = 'oil_b_factor_hyp'
                        # help="Select the initial decline rate (Di) for the forecast."
                    )
        with st.expander('Other Parameters', expanded=False):
            with st.form("other_params_form", clear_on_submit=False):
                # Show text input with default value
                new_scenario = st.text_input("ECON_SCENARIO", value=params['ECON_SCENARIO'], key='ECON_SCENARIO')

                submit_other = st.form_submit_button("Update Other Parameters", type="primary", use_container_width=True)

                if submit_other:
                    try:
                        # Build SQL statement
                        update_sql = f"""
                            UPDATE wells.minerals.econ_input
                            SET ECON_SCENARIO = '{new_scenario}'
                            WHERE API_UWI = '{params['API_UWI']}'
                        """
                        session.sql(update_sql).collect()
                        st.success("ECON_SCENARIO updated successfully.")
                        # Optionally update local value too
                        params['ECON_SCENARIO'] = new_scenario
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update ECON_SCENARIO: {e}")
               # Submit button
        if st.button("Update Database with Current Parameters", use_container_width=True):
            try:
                # Extract primary key
                primary_key = "API_UWI"
                pk_value = params[primary_key]

                # Remove primary key from update dict
                update_fields = {k: v for k, v in params.items() if k != primary_key}

                # Build SET clause
                set_clauses = []
                for col, val in update_fields.items():
                    if val is None or pd.isna(val):
                        set_clauses.append(f"{col} = NULL")
                    elif isinstance(val, str):
                        set_clauses.append(f"{col} = '{val}'")
                    elif isinstance(val, date):
                        set_clauses.append(f"{col} = '{val.isoformat()}'")
                    else:
                        set_clauses.append(f"{col} = {val}")

                set_clause = ",\n    ".join(set_clauses)

                # Construct final SQL
                update_sql = f"""
                UPDATE wells.minerals.econ_input
                SET
                    {set_clause}
                WHERE
                    {primary_key} = '{pk_value}'
                """

                # Execute it (assuming you have a working session)
                session.sql(update_sql).collect()

                st.success("Database updated successfully.")

            except Exception as e:
                st.error(f"Update failed: {e}")
        with st.expander('Fast Edit Status', expanded=False):
            # Add FAST_EDIT toggle buttons above the chart
            st.markdown("### Fast Edit Status")
            fast_edit_col1, fast_edit_col2 = st.columns(2)
            # Make a fresh query to the database to get the current value of FAST_EDIT
            try:
                # from snowflake.snowpark.context import get_active_session
                # session = get_active_session()
                
                # Query to get the latest FAST_EDIT status directly from database
                query = f"SELECT FAST_EDIT FROM 'ECON_INPUT' WHERE 'API_UWI' = '{selected_key}'"
                fast_edit_result = session.sql(query).collect()
                
                # Get current value (use 0 as default if not found or null)
                if fast_edit_result and len(fast_edit_result) > 0:
                    current_fast_edit = fast_edit_result[0]["FAST_EDIT"] if fast_edit_result[0]["FAST_EDIT"] is not None else 0
                else:
                    current_fast_edit = 0
            except Exception as e:
                # Fallback to value from record if query fails
                current_fast_edit = record.get('FAST_EDIT', 0) if 'FAST_EDIT' in record else 0
            # Convert to integer to ensure proper comparison
            try:
                current_fast_edit = int(current_fast_edit)
            except (ValueError, TypeError):
                current_fast_edit = 0
            fast_edit_status = "In Fast Edit List" if current_fast_edit == 1 else "Not In Fast Edit List"
            with fast_edit_col1:
                add_button = st.button("📋 Add to Fast Edit", use_container_width=True, 
                                help="Set FAST_EDIT to 1 for this well", 
                                type="primary" if current_fast_edit != 1 else "secondary",
                                disabled=current_fast_edit == 1)
            with fast_edit_col2:
                remove_button = st.button("🗑️ Remove from Fast Edit", use_container_width=True, 
                                help="Set FAST_EDIT to 0 for this well",
                                type="primary" if current_fast_edit == 1 else "secondary",
                                disabled=current_fast_edit != 1)
            # Display current status with color
            st.markdown(
                f"""
                <div style="
                    background-color: {'#D4EDDA' if current_fast_edit == 1 else '#F8D7DA'}; 
                    padding: 10px; 
                    border-radius: 5px; 
                    text-align: center;
                    margin-top: 4px;
                    ">
                    <span style="
                        color: {'#155724' if current_fast_edit == 1 else '#721C24'}; 
                        font-weight: bold;
                        ">
                        {fast_edit_status}
                    </span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            # Handle button actions after defining all the interface elements
            if add_button:
                # Update FAST_EDIT to 1
                success, message = update_database_record(
                    'ECON_INPUT',
                    'API_UWI',
                    selected_key,
                    {"FAST_EDIT": "1"},
                    params.keys()
                )
                if success:
                    st.success(f"Added {selected_key} to Fast Edit list")
                    # Clear all cached data to ensure fresh data on reload
                    st.cache_data.clear()
                    # Use JavaScript to reload the page completely
                    st.rerun()
                else:
                    st.error(message)       
            if remove_button:
                # Update FAST_EDIT to 0
                success, message = update_database_record(
                    'ECON_INPUT',
                    'API_UWI',
                    selected_key,
                    {"FAST_EDIT": "0"},
                    params.keys()
                )
                if success:
                    st.success(f"Removed {selected_key} from Fast Edit list")
                    # Clear all cached data to ensure fresh data on reload
                    st.cache_data.clear()
                    # Use JavaScript to reload the page completely
                    st.rerun()
                else:
                    st.error(message)


















    with right_column:
        st.markdown("## Forecast Chart")
        chart_data = calc_decline(prd, params)
        chart_data[['LIQUIDSPROD_BBL', 'GASPROD_MCF']] = chart_data[['LIQUIDSPROD_BBL', 'GASPROD_MCF']].replace(0, None)
        chart_data = chart_data.rename(columns={
            'LIQUIDSPROD_BBL': 'Oil (BBL)',
            'GASPROD_MCF': 'Gas (MCF)',
            'OilFcst_BBL': 'Oil Forecast (BBL)',
            'GasFcst_MCF': 'Gas Forecast (MCF)'
        })
        # Melt the DataFrame for Altair
        melted_data = chart_data.melt(
            id_vars=['PRODUCINGMONTH'], 
            value_vars=['Oil (BBL)', 'Gas (MCF)', 'Oil Forecast (BBL)', 'Gas Forecast (MCF)'],
            var_name='Production Type', 
            value_name='Volume'
        )
        melted_data['Production Month'] = melted_data['PRODUCINGMONTH']
        melted_data['legend_group'] = ['Oil (BBL)' if 'Oil' in x else 'Gas (MCF)' for x in melted_data['Production Type']]
        color_mapping = {
            'Oil (BBL)': '#1e8f4e',        # Green for oil
            'Gas (MCF)': '#d62728',         # Red for gas
            'Oil Forecast (BBL)': '#1e8f4e',  # Same green for oil forecast
            'Gas Forecast (MCF)': '#d62728'   # Same red for gas forecast
        }
        # Add selection based on legend
        # highlight = alt.selection_multi(name = 'prod_type', fields=['legend_group'], bind='legend')
        chart = alt.Chart(melted_data).encode(
            x=alt.X('Production Month:T', title='Production Month'),
            y=alt.Y('Volume:Q', scale= alt.Scale(type='log', domainMin=1), title='check'),
            color=alt.Color('legend_group:N', 
                            scale=alt.Scale(domain=list(color_mapping.keys()), 
                                            range=list(color_mapping.values())),
                            legend=alt.Legend(title='Production Type',
                                                orient='right',
                                                offset=-105,
                                                labelFontSize=11,
                                                values=['Oil (BBL)', 'Gas (MCF)']
                                                )),
            # opacity=alt.condition(highlight, alt.value(1), alt.value(0.1)),
            tooltip=['Production Month', 'Production Type', 'Volume']
        )#.add_selection(highlight)
        # Create separate line marks for historical data (solid) and forecast data (dashed)
        historical_chart = chart.transform_filter(
            alt.FieldOneOfPredicate(field='Production Type', oneOf=['Oil (BBL)', 'Gas (MCF)'])
        ).mark_line(point=True)

        forecast_chart = chart.transform_filter(
            alt.FieldOneOfPredicate(field='Production Type', oneOf=['Oil Forecast (BBL)', 'Gas Forecast (MCF)'])
        ).mark_line(point=True, strokeDash=[6, 2])  # Dashed line for forecast

        # Combine the charts
        final_chart = (historical_chart + forecast_chart).properties(
            # title=f"{wells_df[wells_df['API']==selected_key]['Well Name'].values[0]} (API: {selected_key})",
            height=400
        ).interactive()

        # Get forecast start dates
        oil_start_date = params['FCST_START_OIL']
        gas_start_date = params['FCST_START_GAS']

        oil_rule = alt.Chart(pd.DataFrame({'date': [oil_start_date]})).mark_rule(
            color='green',
            strokeDash=[6, 4],
            strokeWidth=1.5
        ).encode(
            x='date:T',
            tooltip=[alt.Tooltip('date:T', title='Oil Forecast Start')]
        )
        gas_rule = alt.Chart(pd.DataFrame({'date': [gas_start_date]})).mark_rule(
            color='red',
            strokeDash=[6, 4],
            strokeWidth=1.5
        ).encode(
            x='date:T',
            tooltip=[alt.Tooltip('date:T', title='Gas Forecast Start')]
        )

        final_chart = final_chart + oil_rule + gas_rule
        ## well info above chart
        record_well_data = filtered_wells[filtered_wells['API_UWI'] == selected_key].iloc[0]
        # st.markdown(f"{record_well_data['API_UWI']}")
        st.markdown(f"**Operator:** {record_well_data['ENVOPERATOR']} &nbsp;\
                     **Well Name:** {record_well_data['WELLNAME']}")
        st.markdown(f"**Status:** {record_well_data['ENVWELLSTATUS']} &nbsp;\
                    **First PRD:** {record_well_data['FIRSTPRODDATE']}")
    
        # Display the chart
        st.altair_chart(final_chart, use_container_width=True)

# st.write(params)
# st.write(prd)
# st.write(melted_data)