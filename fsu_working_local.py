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
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
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
                # Use the gas forecast start date to determine the reference production data
                last_gas_prod = well_production[well_production['PRODUCINGMONTH'] <= gas_fcst_start]['GASPROD_MCF'].tail(3)
                if not last_gas_prod.empty:
                    gas_qi = min(gas_qi, last_gas_prod.mean() * 1.1)
                
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
# Sidebar Inputs
st.sidebar.title("Well Data Viewer")
table_name = st.sidebar.text_input("Enter table name", "ECON_INPUT")
primary_key = st.sidebar.text_input("Primary Key Column", "API_UWI")

# Use the active session directly (Snowflake Native App approach)
# @st.cache_resource
# def get_session():
#     """Get the current Snowpark session"""
#     try:
#         from snowflake.snowpark.context import get_active_session
#         return get_active_session()
#     except Exception as e:
#         st.error(f"Error getting Snowflake session: {e}")
#         return None
## Get session
# session = get_session()

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
# Get Table Columns
@st.cache_data(ttl=600)
def get_table_columns(table):
    try:
        # from snowflake.snowpark.context import get_active_session
        # session = get_active_session()
        
        query = f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}' ORDER BY ORDINAL_POSITION"
        df = session.sql(query).to_pandas()
        if not df.empty:
            return df.set_index("COLUMN_NAME")["DATA_TYPE"].to_dict()
        return {}
    except Exception as e:
        st.error(f"Error getting table columns: {e}")
        return {}

# Fetch Table Data
@st.cache_data(ttl=60)
def get_table_data(table, where_clause=""):
    try:
        # from snowflake.snowpark.context import get_active_session
        # session = get_active_session()
        
        query = f"SELECT * FROM {table} {f'WHERE {where_clause}' if where_clause else ''} LIMIT 1000"
        return session.sql(query).to_pandas()
    except Exception as e:
        st.error(f"Error fetching table data: {e}")
        return pd.DataFrame()

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
@st.cache_data(ttl=600)
def get_production_data(api_uwi):
    try:
        # from snowflake.snowpark.context import get_active_session
        # session = get_active_session()
        
        query = f"""
        SELECT API_UWI, ProducingMonth, LIQUIDSPROD_BBL, GASPROD_MCF
        FROM wells.minerals.raw_prod_data
        WHERE API_UWI = '{api_uwi}'
        ORDER BY ProducingMonth;
        """
        
        # Execute query and convert to pandas
        result = session.sql(query).to_pandas()
        
        # Debug information to understand what's being returned
        if len(result) == 0:
            print(f"No production data found for API_UWI: {api_uwi}")
        else:
            print(f"Found {len(result)} production records for API_UWI: {api_uwi}")
        
        return result
    except Exception as e:
        print(f"Error fetching production data for {api_uwi}: {str(e)}")
        return pd.DataFrame()

# Get the last production date for a well, separately for oil and gas
def get_last_production_dates(api_uwi):
    # Get production data and convert directly to pandas DataFrame
    raw_data = get_production_data(api_uwi)
    
    if raw_data.empty:
        print(f"No production data for {api_uwi}")
        return None, None
    
    try:
        # Print the first few rows to debug
        print(f"First few rows of production data for {api_uwi}:")
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
    st.session_state['forecast_enabled'] = False
if 'forecast_years' not in st.session_state:
    st.session_state['forecast_years'] = 5
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
    well_data = get_well_data()
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
# SECTION 9: SIDEBAR FILTERS & DATA DISPLAY
##########################################
# Sidebar Filters for original functionality
if table_name:
    table_columns = get_table_columns(table_name)
else:
    table_columns = {}

if table_columns:
    st.sidebar.subheader("Table Filters")
    filter_values = {col: st.sidebar.text_input(f"Filter by {col}") for col in list(table_columns.keys())[:3]}
    where_clause = " AND ".join([f"{col} LIKE '%%{value}%%'" for col, value in filter_values.items() if value])
else:
    where_clause = ""

# Fetch & Display Data
if table_name:
    data = get_table_data(table_name, where_clause)
    
    # Apply additional well filtering if wells have been selected from the map
    if st.session_state['filtered_wells'] is not None and primary_key in data.columns:
        # Get the list of selected API_UWIs
        selected_api_uwis = st.session_state['selected_wells']
        
        # Filter the data to only include the selected wells
        if selected_api_uwis:
            data = data[data[primary_key].isin(selected_api_uwis)]
            table_title = f"{table_name} Data (Filtered to {len(data)} selected wells)"
        else:
            table_title = f"{table_name} Data ({len(data)} rows)"
    else:
        table_title = f"{table_name} Data ({len(data)} rows)"
    
    # Make the data table section minimizable with an expander
    with st.expander(table_title, expanded=False):
        # Create two columns - one for econ input table, one for production data
        econ_col, prod_col = st.columns([1, 1])
        
        with econ_col:
            st.subheader("Econ Input Data")
            st.dataframe(data)
        
        with prod_col:
            st.subheader("Production Data")
            
            # Add a well selector for the production data
            if primary_key in data.columns and not data.empty:
                prod_well = st.selectbox("Select Well for Production Data", 
                                        options=data[primary_key].tolist(),
                                        index=0,
                                        key="prod_data_well_selector")
                
                # Get production data for selected well
                prod_data = get_production_data(prod_well)
                
                if prod_data.empty:
                    st.warning(f"No production data available for {prod_well}")
                else:
                    # Calculate statistics
                    total_oil = prod_data['LIQUIDSPROD_BBL'].sum()
                    total_gas = prod_data['GASPROD_MCF'].sum()
                    
                    # Display statistics and data
                    st.markdown(f"**Records:** {len(prod_data)} | **Oil:** {total_oil:,.0f} BBL | **Gas:** {total_gas:,.0f} MCF")
                    
                    # Format date column for display
                    if 'PRODUCINGMONTH' in prod_data.columns:
                        prod_data['PRODUCINGMONTH'] = pd.to_datetime(prod_data['PRODUCINGMONTH']).dt.strftime('%Y-%m-%d')
                    
                    # Display sorted data
                    st.dataframe(prod_data.sort_values('PRODUCINGMONTH', ascending=False))
                    
                    # Add download button for CSV
                    csv = prod_data.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{prod_well}_production.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning(f"Primary key '{primary_key}' not found in the table or data is empty.")
else:
    data = pd.DataFrame()
    st.warning("Please enter a table name")

##########################################
# SECTION 10A: SINGLE WELL UPDATING - LEFT COLUMN
##########################################
st.divider()
st.title("Single Well Data Viewer & Editor")
# Display well information at the top
st.markdown("## Well Information")

if st.session_state['filtered_wells'] is not None:
    wells_list = st.session_state['filtered_wells'][primary_key].tolist()
    # Create a DataFrame with API and Well Name
    wells_data = []
    for well in wells_list:
        # Get the well name from the filtered wells DataFrame
        well_name = st.session_state['filtered_wells'].loc[
            st.session_state['filtered_wells'][primary_key] == well, 'WELLNAME'
        ].values[0] if 'WELLNAME' in st.session_state['filtered_wells'].columns else "N/A"
        
        wells_data.append({
            'API': well,
            'Well Name': well_name,
            # 'Current': well == selected_key
        })
    wells_df = pd.DataFrame(wells_data)
    ## limit to only rwos with production data
    wells_df = wells_df[wells_df['API'].isin(data['API_UWI'])]
    ##default select first well in list
    selected_key = wells_df.iloc[0]['API']
    record = data[data[primary_key] == selected_key].iloc[0]
    record_well_data = filtered_wells[filtered_wells[primary_key] == selected_key].iloc[0]
    well_info_col1, well_info_col2, well_info_col3, well_info_col4 = st.columns(4)
    with well_info_col1:
        st.markdown(f"**Operator:** {record_well_data['ENVOPERATOR']}")
    with well_info_col2:
        st.markdown(f"**Well Name:** {record_well_data['WELLNAME']}")
    with well_info_col3:
        st.markdown(f"**Status:** {record_well_data['ENVWELLSTATUS']}")
    with well_info_col4:
        st.markdown(f"**First Production:** {record_well_data['FIRSTPRODDATE']}")
else:
    st.info("No wells filtered. Use the filter section to select wells.")
    selected_key = None
st.divider()
if not data.empty and primary_key in data.columns:
    # Create a three-column layout with API selection on the left
    left_column, middle_column, right_column = st.columns([1, 1.5, 2])
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

##########################################
# SECTION 10B: SINGLE WELL UPDATING - MIDDLE COLUMN
##########################################

    # if selected_key:
    # Get production data and last production dates once for use throughout this section
    production_data = get_production_data(selected_key)
    last_oil_date, last_gas_date = get_last_production_dates(selected_key)
    # Calculate Qi values once
    oil_qi, gas_qi = calculate_qi_values(production_data, last_oil_date, last_gas_date)
    with middle_column: 
        st.markdown("## Edit Decline Parameters")            
        # Oil parameters form in expander
        with st.expander("Oil Parameters", expanded=False):
            # Get ordered fields that actually exist in the table
            existing_oil_fields = get_ordered_fields(oil_fields, table_columns)
            
            with st.form("oil_update_form", clear_on_submit=False):
                st.subheader("Oil Parameters")
                
                # Create a User Inputs section at the top
                st.markdown("### User Inputs")
                user_input_col1, user_input_col2 = st.columns(2)
                
                oil_values = {}
                
                # 1. Oil Forecast Start Date - placed at the top in the first column
                with user_input_col1:
                    # Get the current oil forecast start date from the record
                    current_oil_fcst_start = record.get('FCST_START_OIL', None)
                    if current_oil_fcst_start is not None:
                        try:
                            current_oil_fcst_start = pd.to_datetime(current_oil_fcst_start).strftime('%Y-%m-%d')
                        except:
                            current_oil_fcst_start = ""
                    else:
                        current_oil_fcst_start = ""
                    
                    # Create a date picker for oil forecast start date
                    oil_fcst_start = st.date_input(
                        "Forecast Start Date",
                        value=pd.to_datetime(current_oil_fcst_start) if current_oil_fcst_start else None,
                        help="Date to start oil forecast. If blank, will use last production date + 1 month."
                    )
                    
                    # Convert to string for database update
                    if oil_fcst_start is not None:
                        oil_values['FCST_START_OIL'] = oil_fcst_start.strftime('%Y-%m-%d')
                    else:
                        oil_values['FCST_START_OIL'] = ""
                
                    # # 2. User Qi
                    # current_user_qi = record.get('OIL_USER_QI', '')
                    # oil_values['OIL_USER_QI'] = st.text_input(
                    #     f"User Qi (Current: {current_user_qi})",
                    #     value=str(current_user_qi) if current_user_qi else ""
                    # )

                    # 2. User Qi
                    if not production_data.empty:
                        # Calculate the min and max production values from historical data
                        non_zero_production = production_data[production_data['LIQUIDSPROD_BBL'] > 0]
                        min_production = non_zero_production['LIQUIDSPROD_BBL'].min() if not non_zero_production.empty else 0
                        max_production = production_data['LIQUIDSPROD_BBL'].max()
                    else:
                        # Default values if no production data is available
                        min_production = 1
                        max_production = 1000

                    current_user_qi = record.get('OIL_USER_QI', min_production)
                    oil_values['OIL_USER_QI'] = st.slider(
                        f"User Qi (Current: {current_user_qi})",
                        min_value=float(min_production),
                        max_value=float(max_production),
                        value=float(current_user_qi) if current_user_qi else float(min_production),
                        step=1.0,
                        help="Select the initial production rate (Qi) for the forecast."
                    )
                
                with user_input_col2:
                    # 3. User Decline
                    current_user_decline = record.get('OIL_USER_DECLINE', '')
                    oil_values['OIL_USER_DECLINE'] = st.text_input(
                        f"User DI (Current: {current_user_decline})",
                        value=str(current_user_decline) if current_user_decline else ""
                    )
                    
                    # 4. User B Factor
                    current_user_b_factor = record.get('OIL_USER_B_FACTOR', '')
                    oil_values['OIL_USER_B_FACTOR'] = st.text_input(
                        f"User B Factor (Current: {current_user_b_factor})",
                        value=str(current_user_b_factor) if current_user_b_factor else ""
                    )
                
                # Automatically populate OIL_CALC_QI if available
                if oil_qi > 0:
                    st.info(f"Calculated Oil Initial Rate (Qi): {oil_qi:.2f}")
                    oil_values['OIL_CALC_QI'] = str(oil_qi)
                
                # Other oil parameters (not in the top 4)
                st.markdown("### Other Oil Parameters")
                other_oil_fields = [field for field, _ in existing_oil_fields 
                                    if field not in ['OIL_USER_QI', 'OIL_USER_DECLINE', 
                                                    'OIL_USER_B_FACTOR', 'FCST_START_OIL']]
                
                # Create 2 columns for the remaining parameters
                other_col1, other_col2 = st.columns(2)
                half_point = len(other_oil_fields) // 2
                
                # First half in first column
                with other_col1:
                    for field in other_oil_fields[:half_point]:
                        current_value = record[field] if field in record else ""
                        oil_values[field] = st.text_input(
                            f"{field} (Current: {current_value})", 
                            value=str(current_value) if current_value else ""
                        )
                
                # Second half in second column
                with other_col2:
                    for field in other_oil_fields[half_point:]:
                        current_value = record[field] if field in record else ""
                        oil_values[field] = st.text_input(
                            f"{field} (Current: {current_value})", 
                            value=str(current_value) if current_value else ""
                        )
                
                # Add form submission handler
                submit_oil = st.form_submit_button("Update Oil Parameters", type="primary", use_container_width=True)
                
                if submit_oil:
                    # Store oil values in session state for the current well
                    if selected_key not in st.session_state['pending_form_values']:
                        st.session_state['pending_form_values'][selected_key] = {}
                    
                    # Update the oil values in the pending form values
                    st.session_state['pending_form_values'][selected_key].update(oil_values)
                    
                    # Display success message
                    st.success("Oil parameters updated in memory. Click 'Update All' to save to database.")
                    
                    # Set a flag to trigger live forecast update if enabled
                    if st.session_state['forecast_enabled'] and st.session_state['forecast_controls']['live_forecast']:
                        st.session_state['refresh_live_forecast'] = True
                        st.rerun()
        
        # Gas parameters form in expander
        with st.expander("Gas Parameters", expanded=False):
            # Get ordered fields that actually exist in the table
            existing_gas_fields = get_ordered_fields(gas_fields, table_columns)
            
            with st.form("gas_update_form", clear_on_submit=False):
                st.subheader("Gas Parameters")
                
                # Create a User Inputs section at the top
                st.markdown("### User Inputs")
                user_input_col1, user_input_col2 = st.columns(2)
                
                gas_values = {}
                
                # 1. Gas Forecast Start Date - placed at the top in the first column
                with user_input_col1:
                    # Get the current gas forecast start date from the record
                    current_gas_fcst_start = record.get('FCST_START_GAS', None)
                    if current_gas_fcst_start is not None:
                        try:
                            current_gas_fcst_start = pd.to_datetime(current_gas_fcst_start).strftime('%Y-%m-%d')
                        except:
                            current_gas_fcst_start = ""
                    else:
                        current_gas_fcst_start = ""
                    
                    # Create a date picker for gas forecast start date
                    gas_fcst_start = st.date_input(
                        "Forecast Start Date",
                        value=pd.to_datetime(current_gas_fcst_start) if current_gas_fcst_start else None,
                        help="Date to start gas forecast. If blank, will use last production date + 1 month."
                    )
                    
                    # Convert to string for database update
                    if gas_fcst_start is not None:
                        gas_values['FCST_START_GAS'] = gas_fcst_start.strftime('%Y-%m-%d')
                    else:
                        gas_values['FCST_START_GAS'] = ""
                
                    # 2. User Qi
                    current_user_qi = record.get('GAS_USER_QI', '')
                    gas_values['GAS_USER_QI'] = st.text_input(
                        f"User Qi (Current: {current_user_qi})",
                        value=str(current_user_qi) if current_user_qi else ""
                    )
                
                with user_input_col2:
                    # 3. User Decline
                    current_user_decline = record.get('GAS_USER_DECLINE', '')
                    gas_values['GAS_USER_DECLINE'] = st.text_input(
                        f"User DI (Current: {current_user_decline})",
                        value=str(current_user_decline) if current_user_decline else ""
                    )
                    
                    # 4. User B Factor
                    current_user_b_factor = record.get('GAS_USER_B_FACTOR', '')
                    gas_values['GAS_USER_B_FACTOR'] = st.text_input(
                        f"User B Factor (Current: {current_user_b_factor})",
                        value=str(current_user_b_factor) if current_user_b_factor else ""
                    )
                
                # Automatically populate GAS_CALC_QI if available
                if gas_qi > 0:
                    st.info(f"Calculated Gas Initial Rate (Qi): {gas_qi:.2f}")
                    gas_values['GAS_CALC_QI'] = str(gas_qi)
                
                # Other gas parameters (not in the top 4)
                st.markdown("### Other Gas Parameters")
                other_gas_fields = [field for field, _ in existing_gas_fields 
                                    if field not in ['GAS_USER_QI', 'GAS_USER_DECLINE', 
                                                    'GAS_USER_B_FACTOR', 'FCST_START_GAS']]
                
                # Create 2 columns for the remaining parameters
                other_col1, other_col2 = st.columns(2)
                half_point = len(other_gas_fields) // 2
                
                # First half in first column
                with other_col1:
                    for field in other_gas_fields[:half_point]:
                        current_value = record[field] if field in record else ""
                        gas_values[field] = st.text_input(
                            f"{field} (Current: {current_value})", 
                            value=str(current_value) if current_value else ""
                        )
                
                # Second half in second column
                with other_col2:
                    for field in other_gas_fields[half_point:]:
                        current_value = record[field] if field in record else ""
                        gas_values[field] = st.text_input(
                            f"{field} (Current: {current_value})", 
                            value=str(current_value) if current_value else ""
                        )
                
                # Add form submission handler    
                submit_gas = st.form_submit_button("Update Gas Parameters", type="primary", use_container_width=True)
                
                if submit_gas:
                    # Store gas values in session state for the current well
                    if selected_key not in st.session_state['pending_form_values']:
                        st.session_state['pending_form_values'][selected_key] = {}
                    
                    # Update the gas values in the pending form values
                    st.session_state['pending_form_values'][selected_key].update(gas_values)
                    
                    # Display success message
                    st.success("Gas parameters updated in memory. Click 'Update All' to save to database.")
                    
                    # Set a flag to trigger live forecast update if enabled
                    if st.session_state['forecast_enabled'] and st.session_state['forecast_controls']['live_forecast']:
                        st.session_state['refresh_live_forecast'] = True
                        st.rerun()
        
        # Other fields expander
        with st.expander("Other Parameters", expanded=False):
            with st.form("other_update_form", clear_on_submit=False):
                st.subheader("Other Parameters")
                other_fields = {}
                for col in table_columns:
                    if (col != primary_key and 
                        col.lower() != 'lease' and
                        col not in oil_values and 
                        col not in gas_values):
                        other_fields[col] = st.text_input(
                            f"{col} (Current: {record[col]})", 
                            value=str(record[col]) if record[col] else ""
                        )
                
                submit_other = st.form_submit_button("Update Other Parameters", type="primary", use_container_width=True)
                
                if submit_other:
                    # Store other values in session state for the current well
                    if selected_key not in st.session_state['pending_form_values']:
                        st.session_state['pending_form_values'][selected_key] = {}
                    
                    # Update the other values in the pending form values
                    st.session_state['pending_form_values'][selected_key].update(other_fields)
                    
                    # Display success message
                    st.success("Other parameters updated in memory. Click 'Update All' to save to database.")
        
        # Add FAST_EDIT toggle buttons above the chart
        st.markdown("### Fast Edit Status")
        fast_edit_col1, fast_edit_col2 = st.columns(2)
        # Make a fresh query to the database to get the current value of FAST_EDIT
        try:
            # from snowflake.snowpark.context import get_active_session
            # session = get_active_session()
            
            # Query to get the latest FAST_EDIT status directly from database
            query = f"SELECT FAST_EDIT FROM {table_name} WHERE {primary_key} = '{selected_key}'"
            fast_edit_result = session.sql(query).collect()
            
            # Get current value (use 0 as default if not found or null)
            if fast_edit_result and len(fast_edit_result) > 0:
                current_fast_edit = fast_edit_result[0]["FAST_EDIT"] if fast_edit_result[0]["FAST_EDIT"] is not None else 0
            else:
                current_fast_edit = 0
        except Exception as e:
            # Fallback to value from record if query fails
            current_fast_edit = record.get('FAST_EDIT', 0) if 'FAST_EDIT' in record else 0
            st.warning(f"Using cached data for FAST_EDIT status. Latest status may not be reflected.")
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
                table_name,
                primary_key,
                selected_key,
                {"FAST_EDIT": "1"},
                table_columns
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
                table_name,
                primary_key,
                selected_key,
                {"FAST_EDIT": "0"},
                table_columns
            )
            if success:
                st.success(f"Removed {selected_key} from Fast Edit list")
                # Clear all cached data to ensure fresh data on reload
                st.cache_data.clear()
                # Use JavaScript to reload the page completely
                st.rerun()
            else:
                st.error(message)
                
        # Calculate Decline Rates section
        with st.expander("Calculate Decline Rates", expanded=False):
            st.subheader("Decline Calculation Parameters")
            # Add a small section for the decline constants
            decline_col1, decline_col2 = st.columns(2)
            with decline_col1:
                st.session_state['single_months_for_calc'] = st.slider(
                    "Months for Calculation", 
                    6, 60, st.session_state['single_months_for_calc'],
                    help="Number of most recent months to use for this well"
                )
                st.session_state['single_default_decline'] = st.number_input(
                    "Default Rate", 
                    value=st.session_state['single_default_decline'], 
                    format="%.6f",
                    help="Fallback rate when insufficient data"
                )
            with decline_col2:
                st.session_state['single_min_decline'] = st.number_input(
                    "Minimum Rate", 
                    value=st.session_state['single_min_decline'], 
                    format="%.6f",
                    help="Minimum allowed decline rate"
                )
                st.session_state['single_max_decline'] = st.number_input(
                    "Maximum Rate", 
                    value=st.session_state['single_max_decline'], 
                    format="%.6f",
                    help="Maximum allowed decline rate"
                )
            
            # Calculate button for this well in a prominent position
            if st.button("Calculate Decline Rates", key="single_calc", use_container_width=True):
                with st.spinner(f"Calculating decline rates for {selected_key}..."):
                    if production_data.empty:
                        st.warning(f"No production data found for {selected_key}")
                    else:
                        # Calculate decline rates using single well parameters
                        oil_decline, gas_decline = calculate_decline_fit(
                            production_data, st.session_state['single_months_for_calc'], 
                            st.session_state['single_default_decline'], 
                            st.session_state['single_min_decline'], 
                            st.session_state['single_max_decline']
                        )
                        
                        # Store in session state
                        st.session_state['calculated_declines'][selected_key] = (oil_decline, gas_decline)
                        
                        # Display results
                        st.success("Decline rates calculated")
            
            # Show calculated rates if available in a more prominent way
            if selected_key in st.session_state['calculated_declines']:
                oil_decline, gas_decline = st.session_state['calculated_declines'][selected_key]
                oil_col1, gas_col1 = st.columns(2)
                with oil_col1:
                    st.metric("Oil Decline Rate", f"{oil_decline:.6f}")
                with gas_col1:
                    st.metric("Gas Decline Rate", f"{gas_decline:.6f}")
       
        # Update and navigation buttons
        st.markdown("### Actions")        
        # with update_col1:
        update_button = st.button("Save Decline Parameters to Database", key="update_record_btn", use_container_width=True)
        update_next = None
        # with update_col2:
        #     update_next = st.button("Save & Next Well", key="save_next_btn", use_container_width=True)
            
        # Option to apply calculated decline rates
        apply_calc_decline = st.checkbox("Apply calculated decline rates (if available)")
        
        # Combine all update values for use with the buttons
        update_values = {**oil_values, **gas_values, **other_fields}
        
        # Add last production dates if those fields exist
        if ('LAST_OIL_DATE' in table_columns or 'LAST_GAS_DATE' in table_columns or 
            'LAST_PROD_DATE' in table_columns):
            
            # Update individual date fields
            if last_oil_date is not None and 'LAST_OIL_DATE' in table_columns:
                st.info(f"Last Oil Production Date: {last_oil_date}")
                update_values['LAST_OIL_DATE'] = str(last_oil_date)
            if last_gas_date is not None and 'LAST_GAS_DATE' in table_columns:
                st.info(f"Last Gas Production Date: {last_gas_date}")
                update_values['LAST_GAS_DATE'] = str(last_gas_date)
            
            # Determine and update the most recent production date
            if 'LAST_PROD_DATE' in table_columns:
                most_recent_date = None
                if last_oil_date is not None and last_gas_date is not None:
                    most_recent_date = max(last_oil_date, last_gas_date)
                elif last_oil_date is not None:
                    most_recent_date = last_oil_date
                elif last_gas_date is not None:
                    most_recent_date = last_gas_date
                
                if most_recent_date is not None:
                    st.info(f"Last Production Date (Most Recent): {most_recent_date}")
                    update_values['LAST_PROD_DATE'] = str(most_recent_date)
                    
        # Handle update buttons
        if update_button or update_next:
            # Get pending form values if they exist
            if selected_key in st.session_state['pending_form_values']:
                pending_values = st.session_state['pending_form_values'][selected_key]
                update_values.update(pending_values)
            
            # Apply decline rates if checkbox is selected and values exist in session state
            if apply_calc_decline and selected_key in st.session_state['calculated_declines']:
                oil_decline, gas_decline = st.session_state['calculated_declines'][selected_key]
                update_values["OIL_EMPIRICAL_DECLINE"] = str(oil_decline)
                update_values["GAS_EMPIRICAL_DECLINE"] = str(gas_decline)

            # Call the update function
            success, message = update_database_record(
                table_name, 
                primary_key, 
                selected_key, 
                update_values, 
                table_columns
            )
            
            if success:
                st.success(f"Record {primary_key} = {selected_key} {message}")
                # Clear pending form values for this well
                if selected_key in st.session_state['pending_form_values']:
                    del st.session_state['pending_form_values'][selected_key]
                
                # Clear cache data to ensure fresh information
                st.cache_data.clear()
                
                # If it was the "Save & Next" button, find and load the next well
                if update_next:
                    # Get a fresh copy of the data to avoid stale lists
                    fresh_data = get_table_data(table_name, where_clause)
                    
                    # Apply filtering if wells are filtered
                    if st.session_state['filtered_wells'] is not None and primary_key in fresh_data.columns:
                        selected_api_uwis = st.session_state['selected_wells']
                        if selected_api_uwis:
                            fresh_data = fresh_data[fresh_data[primary_key].isin(selected_api_uwis)]
                    
                    # Get wells into a clean, sorted list
                    wells = sorted(fresh_data[primary_key].unique().tolist())
                    
                    # Find the next well
                    if selected_key in wells:
                        current_idx = wells.index(selected_key)
                        next_idx = (current_idx + 1) % len(wells)
                        next_well = wells[next_idx]
                        
                        # Set flag for auto-generating forecast
                        if st.session_state['forecast_enabled']:
                            st.session_state['auto_generate_forecast'] = True
                            
                        # Debug information
                        st.session_state['next_well_to_load'] = next_well
                        st.session_state['debug_info'] = {
                            'current_well': selected_key,
                            'next_well': next_well,
                            'current_idx': current_idx,
                            'next_idx': next_idx,
                            'total_wells': len(wells)
                        }
                        
                        # Store everything needed for the next load
                        st.session_state['next_well_data'] = {
                            'well_to_load': next_well,
                            'auto_generate_forecast': st.session_state['forecast_enabled'],
                            'current_time': pd.Timestamp.now().isoformat()  # Add timestamp to force refresh
                        }
                        
                        # Use a safer rerun approach
                        st.rerun()
                    else:
                        st.warning(f"Current well {selected_key} not found in well list. Cannot navigate.")
            else:
                st.error(message)

##########################################
# SECTION 10C: SINGLE WELL UPDATING - RIGHT COLUMN
##########################################

    # Data visualization in right column
    with right_column:
        st.markdown("## Production Data Visualization")
        time_selection = "All Data"
        show_oil = True
        show_gas = True
        log_scale  = True
        filter_zeros = True
        # Add forecast controls
        with st.expander("Forecast Controls", expanded=False):
            forecast_col1, forecast_col2 = st.columns(2)

            # Store forecast control settings in session state to persist them
            if 'forecast_controls' not in st.session_state:
                st.session_state['forecast_controls'] = {
                    'enabled': st.session_state['forecast_enabled'],
                    'show_oil': st.session_state['show_oil_forecast'],
                    'show_gas': st.session_state['show_gas_forecast'],
                    'live_forecast': True,
                    'time_range': "All Data",
                    'log_scale': True,
                    'show_oil': True,
                    'show_gas': True,
                    'filter_zeros': True
                }
            st.session_state['forecast_controls']['enabled'] = True
            st.session_state['auto_generate_forecast'] = True
            with forecast_col1:

                st.session_state['forecast_controls']['enabled'] = st.checkbox(
                    "Enable Forecast", 
                    st.session_state['forecast_controls']['enabled']
                )
                st.session_state['forecast_enabled'] = st.session_state['forecast_controls']['enabled']
                
                if st.session_state['forecast_enabled']:
                    st.session_state['forecast_controls']['show_oil'] = st.checkbox(
                        "Show Oil Forecast", 
                        st.session_state['forecast_controls']['show_oil']
                    )
                    st.session_state['show_oil_forecast'] = st.session_state['forecast_controls']['show_oil']
                    
                    st.session_state['forecast_controls']['show_gas'] = st.checkbox(
                        "Show Gas Forecast", 
                        st.session_state['forecast_controls']['show_gas']
                    )
                    st.session_state['show_gas_forecast'] = st.session_state['forecast_controls']['show_gas']
                    
                    st.session_state['forecast_controls']['live_forecast'] = st.checkbox(
                        "Live Forecast", 
                        value=st.session_state['forecast_controls']['live_forecast'],
                        help="Automatically update forecast when parameters change"
                    )
                    live_forecast = st.session_state['forecast_controls']['live_forecast']

            with forecast_col2:
                if st.session_state['forecast_enabled']:
                    st.session_state['forecast_years'] = st.slider(
                        "Forecast Years", 
                        1, 20, 
                        st.session_state['forecast_years']
                    )
                    
                    if not live_forecast:
                        generate_forecast = st.button("Generate Forecast", key="gen_forecast_btn")
                        
                        if generate_forecast or 'auto_generate_forecast' in st.session_state:
                            # Clear the auto-generate flag if it was set
                            if 'auto_generate_forecast' in st.session_state:
                                del st.session_state['auto_generate_forecast']
                                
                            # Handle forecast generation logic
                            try:
                                # Get the well row data
                                well_row = data[data[primary_key] == selected_key].iloc[0]
                                
                                # Fix common data issues
                                fixed_well_row = fix_forecast_data_issues(well_row)
                                
                                # Create a temporary dataframe with the fixed row
                                temp_data = data.copy()
                                
                                try:
                                    # Find the index of the row we want to replace
                                    idx = temp_data[temp_data[primary_key] == selected_key].index
                                    
                                    if len(idx) == 0:
                                        st.error(f"No row found with {primary_key} = {selected_key}")
                                    elif len(idx) > 1:
                                        st.warning(f"Multiple rows found with {primary_key} = {selected_key}, updating first occurrence")
                                        idx = idx[0]  # Take just the first match
                                    else:
                                        idx = idx[0]  # Single match
                                        
                                    # Get the columns we want to update
                                    common_cols = list(set(temp_data.columns) & set(fixed_well_row.index))
                                    
                                    # Update each column individually
                                    for col in common_cols:
                                        temp_data.at[idx, col] = fixed_well_row[col]
                                except Exception as e:
                                    st.error(f"Error updating row: {str(e)}")
                                    import traceback
                                    st.code(traceback.format_exc())
                                
                                # Validate the parameters
                                valid_params, validation_message = validate_forecast_parameters(fixed_well_row)
                                
                                if not valid_params:
                                    st.error(f"Cannot generate forecast: {validation_message}")
                                    st.info("Please update the parameters to valid values and try again.")
                                else:
                                    with st.spinner("Generating forecast..."):
                                        try:
                                            # Call the SingleWellForecast function
                                            forecast_df = SingleWellForecast(
                                                selected_key,  # API_UWI of selected well
                                                temp_data,  # ECON_INPUT data with fixes
                                                production_data  # Production data for the well
                                            )
                                            
                                            # Check if forecast was actually generated
                                            if forecast_df is not None and not forecast_df.empty:
                                                forecast_cols = ['OilFcst_BBL', 'GasFcst_MCF']
                                                has_forecast_data = any(col in forecast_df.columns and forecast_df[col].notna().any() for col in forecast_cols)
                                                
                                                if has_forecast_data:
                                                    # Store in session state
                                                    st.session_state['forecast_data'][selected_key] = forecast_df
                                                    st.success("Forecast generated successfully!")
                                                else:
                                                    st.warning("Forecast generated but contains no forecast data. Check parameter values.")
                                            else:
                                                st.error("Failed to generate forecast. Check parameter values.")
                                        except Exception as e:
                                            st.error(f"Error generating forecast: {str(e)}")
                                            # Add more detailed error information
                                            import traceback
                                            st.code(traceback.format_exc())
                            except Exception as e:
                                st.error(f"Error in forecast preparation: {str(e)}")
        
        # Generate live forecast if enabled
        if st.session_state['forecast_enabled'] and 'live_forecast' in locals() and live_forecast:
            try:
                # Get the well row data
                well_row = data[data[primary_key] == selected_key].iloc[0]
                
                # Create a copy of the well row that we can modify
                modified_well_row = well_row.copy()
                
                # Apply any pending form values if they exist
                if selected_key in st.session_state['pending_form_values']:
                    pending_values = st.session_state['pending_form_values'][selected_key]
                    for key, value in pending_values.items():
                        if key in modified_well_row.index:
                            # Convert numeric values appropriately
                            try:
                                if key.endswith('_QI') or key.endswith('_DECLINE') or key.endswith('_B_FACTOR') or key.endswith('_D_MIN') or key.endswith('_Q_MIN'):
                                    modified_well_row[key] = float(value) if value else modified_well_row[key]
                                else:
                                    modified_well_row[key] = value
                            except ValueError:
                                # If conversion fails, use the original value
                                st.warning(f"Could not convert value '{value}' for field '{key}' to number. Using original value.")
                
                # Fix common data issues with the modified row
                fixed_well_row = fix_forecast_data_issues(modified_well_row)
                
                # Create a temporary dataframe with the fixed row
                temp_data = data.copy()
                
                try:
                    # Find the index of the row we want to replace
                    idx = temp_data[temp_data[primary_key] == selected_key].index
                    
                    if len(idx) == 0:
                        st.error(f"No row found with {primary_key} = {selected_key}")
                    elif len(idx) > 1:
                        st.warning(f"Multiple rows found with {primary_key} = {selected_key}, updating first occurrence")
                        idx = idx[0]  # Take just the first match
                    else:
                        idx = idx[0]  # Single match
                        
                    # Get the columns we want to update
                    common_cols = list(set(temp_data.columns) & set(fixed_well_row.index))
                    
                    # Update each column individually
                    for col in common_cols:
                        temp_data.at[idx, col] = fixed_well_row[col]
                except Exception as e:
                    st.error(f"Error updating row: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                
                # Generate live forecast
                with st.spinner("Updating live forecast..."):
                    # Validate parameters first
                    valid_params, validation_message = validate_forecast_parameters(fixed_well_row)
                    
                    if not valid_params:
                        st.warning(f"Cannot generate live forecast: {validation_message}")
                    else:
                        # Clear any refresh flag
                        if 'refresh_live_forecast' in st.session_state:
                            del st.session_state['refresh_live_forecast']
                        
                        live_forecast_df = SingleWellForecast(
                            selected_key,
                            temp_data,
                            production_data
                        )
                        # Store in session state under a different key
                        st.session_state['live_forecast_data'] = live_forecast_df
            except Exception as e:
                st.error(f"Error updating live forecast: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Production data visualization
        if production_data.empty:
            st.warning(f"No production data available for {selected_key}")
        else:
            # Ensure data types are correct and dates are in datetime format
            production_data['PRODUCINGMONTH'] = pd.to_datetime(production_data['PRODUCINGMONTH'])
            production_data['LIQUIDSPROD_BBL'] = pd.to_numeric(production_data['LIQUIDSPROD_BBL'], errors='coerce').fillna(0)
            production_data['GASPROD_MCF'] = pd.to_numeric(production_data['GASPROD_MCF'], errors='coerce').fillna(0)
            
            # Sort the data by date
            production_data = production_data.sort_values('PRODUCINGMONTH')
            
            # Filter data based on time selection
            if time_selection != "All Data":
                end_date = production_data['PRODUCINGMONTH'].max()
                if time_selection == "Last Year":
                    start_date = end_date - pd.DateOffset(years=1)
                elif time_selection == "Last 2 Years":
                    start_date = end_date - pd.DateOffset(years=2)
                elif time_selection == "Last 5 Years":
                    start_date = end_date - pd.DateOffset(years=5)
                
                filtered_data = production_data[production_data['PRODUCINGMONTH'] >= start_date]
                if not filtered_data.empty:
                    production_data = filtered_data
                else:
                    st.warning(f"No data available for the selected time range. Showing all data.")
            
            # Create a DataFrame for visualization with selected series
            chart_data = pd.DataFrame()
            chart_data['Production Month'] = production_data['PRODUCINGMONTH']
            
            if show_oil:
                if filter_zeros and log_scale:
                    # Filter out zero values for oil when using log scale
                    oil_data = production_data[production_data['LIQUIDSPROD_BBL'] > 0]
                    if not oil_data.empty:
                        chart_data['Oil (BBL)'] = pd.Series(dtype='float64')  # Initialize as empty series
                        # Map oil data to chart_data based on matching dates
                        for idx, row in oil_data.iterrows():
                            date_mask = chart_data['Production Month'] == row['PRODUCINGMONTH']
                            if date_mask.any():
                                chart_data.loc[date_mask, 'Oil (BBL)'] = row['LIQUIDSPROD_BBL']
                else:
                    chart_data['Oil (BBL)'] = production_data['LIQUIDSPROD_BBL']
            
            if show_gas:
                if filter_zeros and log_scale:
                    # Filter out zero values for gas when using log scale
                    gas_data = production_data[production_data['GASPROD_MCF'] > 0]
                    if not gas_data.empty:
                        chart_data['Gas (MCF)'] = pd.Series(dtype='float64')  # Initialize as empty series
                        # Map gas data to chart_data based on matching dates
                        for idx, row in gas_data.iterrows():
                            date_mask = chart_data['Production Month'] == row['PRODUCINGMONTH']
                            if date_mask.any():
                                chart_data.loc[date_mask, 'Gas (MCF)'] = row['GASPROD_MCF']
                else:
                    chart_data['Gas (MCF)'] = production_data['GASPROD_MCF']
            
            # Melt the dataframe to create a format suitable for Altair
            if ('Oil (BBL)' in chart_data.columns or 'Gas (MCF)' in chart_data.columns):
                id_vars = ['Production Month']
                value_vars = []
                if 'Oil (BBL)' in chart_data.columns:
                    value_vars.append('Oil (BBL)')
                if 'Gas (MCF)' in chart_data.columns:
                    value_vars.append('Gas (MCF)')
                
                melted_data = chart_data.melt(
                    id_vars=id_vars,
                    value_vars=value_vars,
                    var_name='Production Type',
                    value_name='Volume'
                )
                
                # Drop rows with NaN Volume (filtered zeros)
                melted_data = melted_data.dropna(subset=['Volume'])
                
                # Create scale based on log toggle
                if log_scale:
                    y_scale = alt.Scale(type='log', domainMin=1)
                    y_title = 'Production Volume (Log Scale)'
                else:
                    y_scale = alt.Scale(zero=True)
                    y_title = 'Production Volume'
                
                # Include forecast data if enabled and available
                forecast_df = None
                if st.session_state['forecast_enabled']:
                    # Decision tree for forecast data source:
                    # 1. Use live forecast if it's enabled
                    # 2. Otherwise use saved forecast if available
                    if 'live_forecast' in locals() and live_forecast and 'live_forecast_data' in st.session_state:
                        forecast_df = st.session_state['live_forecast_data']
                        forecast_source = "Live forecast"
                    elif selected_key in st.session_state['forecast_data']:
                        forecast_df = st.session_state['forecast_data'][selected_key]
                        forecast_source = "Saved forecast"
                    
                    if forecast_df is not None:
                        # Limit forecast to specified number of years
                        max_forecast_date = pd.to_datetime(melted_data['Production Month'].max()) + pd.DateOffset(years=st.session_state['forecast_years'])
                        forecast_df = forecast_df[forecast_df['PRODUCINGMONTH'] <= max_forecast_date]
                        
                        # Create forecast data for visualization
                        forecast_chart_data = pd.DataFrame()
                        forecast_chart_data['Production Month'] = forecast_df['PRODUCINGMONTH']
                        
                        if st.session_state['show_oil_forecast'] and 'OilFcst_BBL' in forecast_df.columns:
                            forecast_chart_data['Oil Forecast (BBL)'] = forecast_df['OilFcst_BBL']
                        
                        if st.session_state['show_gas_forecast'] and 'GasFcst_MCF' in forecast_df.columns:
                            forecast_chart_data['Gas Forecast (MCF)'] = forecast_df['GasFcst_MCF']
                        
                        # Melt the forecast dataframe
                        forecast_value_vars = []
                        if 'Oil Forecast (BBL)' in forecast_chart_data.columns:
                            forecast_value_vars.append('Oil Forecast (BBL)')
                        if 'Gas Forecast (MCF)' in forecast_chart_data.columns:
                            forecast_value_vars.append('Gas Forecast (MCF)')
                        
                        if forecast_value_vars:
                            melted_forecast = forecast_chart_data.melt(
                                id_vars=['Production Month'],
                                value_vars=forecast_value_vars,
                                var_name='Production Type',
                                value_name='Volume'
                            )
                            
                            # Drop any NaN values in the forecast data
                            melted_forecast = melted_forecast.dropna(subset=['Volume'])
                            
                            # Append to the historical data
                            melted_data = pd.concat([melted_data, melted_forecast], ignore_index=True)
                
                # Update the color mapping to use green for oil and red for gas
                color_mapping = {
                    'Oil (BBL)': '#1e8f4e',        # Green for oil
                    'Gas (MCF)': '#d62728',         # Red for gas
                    'Oil Forecast (BBL)': '#1e8f4e',  # Same green for oil forecast
                    'Gas Forecast (MCF)': '#d62728'   # Same red for gas forecast
                }
                
                chart = alt.Chart(melted_data).encode(
                    x=alt.X('Production Month:T', title='Production Month'),
                    y=alt.Y('Volume:Q', scale=y_scale, title=y_title),
                    color=alt.Color('Production Type:N', 
                                    scale=alt.Scale(domain=list(color_mapping.keys()), 
                                                    range=list(color_mapping.values())),
                                    legend=alt.Legend(title='Production Type',
                                                        orient='right',
                                                        offset=-105,
                                                        labelFontSize=11,
                                                        values=['Oil (BBL)', 'Gas (MCF)']
                                                        )),
                    tooltip=['Production Month', 'Production Type', 'Volume']
                )
                
                # Create separate line marks for historical data (solid) and forecast data (dashed)
                historical_chart = chart.transform_filter(
                    alt.FieldOneOfPredicate(field='Production Type', oneOf=['Oil (BBL)', 'Gas (MCF)'])
                ).mark_line(point=True)
                
                forecast_chart = chart.transform_filter(
                    alt.FieldOneOfPredicate(field='Production Type', oneOf=['Oil Forecast (BBL)', 'Gas Forecast (MCF)'])
                ).mark_line(point=True, strokeDash=[6, 2])  # Dashed line for forecast
                
                # Create chart title with forecast indicator if applicable
                chart_title = f"Production History for {selected_key}"
                if forecast_df is not None and 'forecast_source' in locals():
                    chart_title = f"Production History and Forecast for {selected_key} ({forecast_source})"
                
                # Combine the charts
                final_chart = (historical_chart + forecast_chart).properties(
                    title=chart_title,
                    height=400
                ).interactive()
                
                # Add forecast start date markers if applicable
                if st.session_state['forecast_enabled'] and forecast_df is not None:
                    # Get forecast start dates
                    oil_start_date = None
                    gas_start_date = None
                    
                    if 'OilFcst_Start_Date' in forecast_df.columns:
                        oil_start_dates = forecast_df['OilFcst_Start_Date'].dropna().unique()
                        if len(oil_start_dates) > 0:
                            oil_start_date = pd.to_datetime(oil_start_dates[0])
                    
                    if 'GasFcst_Start_Date' in forecast_df.columns:
                        gas_start_dates = forecast_df['GasFcst_Start_Date'].dropna().unique()
                        if len(gas_start_dates) > 0:
                            gas_start_date = pd.to_datetime(gas_start_dates[0])
                    
                    # Create vertical rules for forecast start dates if they exist
                    oil_rule = None
                    gas_rule = None
                    
                    # Create a rule for oil forecast start date
                    if oil_start_date is not None:
                        oil_rule = alt.Chart(pd.DataFrame({'date': [oil_start_date]})).mark_rule(
                            color='green',
                            strokeDash=[6, 4],
                            strokeWidth=1.5
                        ).encode(
                            x='date:T',
                            tooltip=[alt.Tooltip('date:T', title='Oil Forecast Start')]
                        )
                    
                    # Create a rule for gas forecast start date
                    if gas_start_date is not None:
                        gas_rule = alt.Chart(pd.DataFrame({'date': [gas_start_date]})).mark_rule(
                            color='red',
                            strokeDash=[6, 4],
                            strokeWidth=1.5
                        ).encode(
                            x='date:T',
                            tooltip=[alt.Tooltip('date:T', title='Gas Forecast Start')]
                        )
                    
                    # Add the rules to the chart if they exist
                    if oil_rule is not None and gas_rule is not None:
                        final_chart = final_chart + oil_rule + gas_rule
                    elif oil_rule is not None:
                        final_chart = final_chart + oil_rule
                    elif gas_rule is not None:
                        final_chart = final_chart + gas_rule
                    

                
                # Display the chart
                st.altair_chart(final_chart, use_container_width=True)
                # Add a textual note about forecast start dates
                if oil_start_date is not None or gas_start_date is not None:
                    # st.markdown("### Forecast Start Dates")
                    
                    fcst_text_col1, fcst_text_col2 = st.columns(2)
                    
                    with fcst_text_col1:
                        if oil_start_date is not None:
                            st.markdown(f"**Oil Forecast Starts:** {oil_start_date.strftime('%Y-%m-%d')}")
                        else:
                            st.markdown("**Oil Forecast:** Using default start date")
                    
                    with fcst_text_col2:
                        if gas_start_date is not None:
                            st.markdown(f"**Gas Forecast Starts:** {gas_start_date.strftime('%Y-%m-%d')}")
                        else:
                            st.markdown("**Gas Forecast:** Using default start date")                
                # Add a note about zero filtering if active
                if log_scale and filter_zeros:
                    st.caption("Note: Zero values are filtered out when using log scale to prevent display artifacts.")
            else:
                st.warning("Please select at least one data series to display (Oil or Gas)")
            # Visualization controls
            st.markdown("### Visualization Controls")
            control_col1, control_col2 = st.columns(2)
            
            with control_col1:
                log_scale = st.checkbox("Log Scale Y-Axis", True)  # Default to True
                # Add time range selector (last X months/years)
                time_options = ["All Data", "Last Year", "Last 2 Years", "Last 5 Years"]
                time_selection = st.selectbox("Time Range", time_options)            
            with control_col2:
            # with control_col3:
                # Add option to show/hide oil or gas
                show_oil = st.checkbox("Show Oil", True)
                show_gas = st.checkbox("Show Gas", True)
                # Add option to filter zeros when using log scale
                filter_zeros = st.checkbox("Filter Zero Values", True)
                # Display production statistics and dates in a well-organized format below the plot
            with st.expander("Production Statistics", expanded=False):
                # Create two columns for oil and gas statistics
                stat_col1, stat_col2 = st.columns(2)
                
                with stat_col1:
                    st.markdown("### Oil Statistics")
                    st.metric("Total Oil (BBL)", f"{production_data['LIQUIDSPROD_BBL'].sum():,.0f}")
                    st.metric("Avg Oil (BBL/month)", f"{production_data['LIQUIDSPROD_BBL'].mean():,.0f}")
                
                with stat_col2:
                    st.markdown("### Gas Statistics")
                    st.metric("Total Gas (MCF)", f"{production_data['GASPROD_MCF'].sum():,.0f}")
                    st.metric("Avg Gas (MCF/month)", f"{production_data['GASPROD_MCF'].mean():,.0f}")
                
                # Display all dates and rates in organized sections
                date_col1, date_col2 = st.columns(2)
                
                with date_col1:
                    st.markdown("### Production Dates")
                    if last_oil_date is not None:
                        st.markdown(f"**Last Oil Date:** {last_oil_date.strftime('%Y-%m-%d')}")
                    if last_gas_date is not None:
                        st.markdown(f"**Last Gas Date:** {last_gas_date.strftime('%Y-%m-%d')}")
                        
                    # Calculate overall last production date
                    most_recent_date = None
                    if last_oil_date is not None and last_gas_date is not None:
                        most_recent_date = max(last_oil_date, last_gas_date)
                    elif last_oil_date is not None:
                        most_recent_date = last_oil_date
                    elif last_gas_date is not None:
                        most_recent_date = last_gas_date
                        
                    if most_recent_date is not None:
                        st.markdown(f"**Last Production Date:** {most_recent_date.strftime('%Y-%m-%d')}")
                
                with date_col2:
                    # Initial Rates (Qi)
                    st.markdown("### Initial Rates (Qi)")
                    if oil_qi > 0:
                        st.markdown(f"**Oil Qi:** {oil_qi:.2f} BBL/month")
                    if gas_qi > 0:
                        st.markdown(f"**Gas Qi:** {gas_qi:.2f} MCF/month")
            
            # Add forecast statistics section
            if st.session_state['forecast_enabled']:
                # Determine which forecast to use for statistics
                forecast_df = None
                if 'live_forecast' in locals() and live_forecast and 'live_forecast_data' in st.session_state:
                    forecast_df = st.session_state['live_forecast_data']
                    forecast_label = "Live Forecast"
                elif selected_key in st.session_state['forecast_data']:
                    forecast_df = st.session_state['forecast_data'][selected_key]
                    forecast_label = "Saved Forecast"
                
                if forecast_df is not None:
                    with st.expander("Forecast Statistics", expanded=False):
                        st.markdown(f"### {forecast_label} Statistics")
                        
                        # Calculate forecast totals
                        oil_forecast_total = forecast_df['OilFcst_BBL'].sum() if 'OilFcst_BBL' in forecast_df.columns else 0
                        gas_forecast_total = forecast_df['GasFcst_MCF'].sum() if 'GasFcst_MCF' in forecast_df.columns else 0
                        
                        # Calculate forecast years based on monthly data
                        forecast_months = len(forecast_df)
                        forecast_years = forecast_months / 12
                        
                        # Create three columns for oil, gas, and general forecast stats
                        fcst_col1, fcst_col2, fcst_col3 = st.columns(3)
                        
                        with fcst_col1:
                            st.markdown("#### Oil Forecast")
                            st.metric("Total Forecast Oil (BBL)", f"{oil_forecast_total:,.0f}")
                            
                            # Calculate EUR (Estimated Ultimate Recovery) for oil
                            total_historical_oil = production_data['LIQUIDSPROD_BBL'].sum()
                            oil_eur = total_historical_oil + oil_forecast_total
                            st.metric("Oil EUR (BBL)", f"{oil_eur:,.0f}")
                        
                        with fcst_col2:
                            st.markdown("#### Gas Forecast")
                            st.metric("Total Forecast Gas (MCF)", f"{gas_forecast_total:,.0f}")
                            
                            # Calculate EUR for gas
                            total_historical_gas = production_data['GASPROD_MCF'].sum()
                            gas_eur = total_historical_gas + gas_forecast_total
                            st.metric("Gas EUR (MCF)", f"{gas_eur:,.0f}")
                        
                        with fcst_col3:
                            st.markdown("#### Forecast Timeline")
                            st.metric("Forecast Months", f"{forecast_months}")
                            st.metric("Forecast Years", f"{forecast_years:.1f}")
                            
                            # Get forecast period from the well parameters if available
                            forecast_period = None
                            if "OIL_FCST_YRS" in oil_values and oil_values["OIL_FCST_YRS"] not in ["", "nan", "None"]:
                                try:
                                    forecast_period = float(oil_values["OIL_FCST_YRS"])
                                except (ValueError, TypeError):
                                    pass
                            elif "GAS_FCST_YRS" in gas_values and gas_values["GAS_FCST_YRS"] not in ["", "nan", "None"]:
                                try:
                                    forecast_period = float(gas_values["GAS_FCST_YRS"])
                                except (ValueError, TypeError):
                                    pass
                            
                            if forecast_period:
                                st.metric("Target Forecast Period (Years)", f"{forecast_period}")
                        
                        # Show forecast parameters used
                        show_forecast_params = st.checkbox("Show Forecast Parameters", value=False)
                        if show_forecast_params:
                            st.markdown("### Forecast Parameters")
                            param_col1, param_col2 = st.columns(2)
                            
                            # For the Oil Parameters section:
                            with param_col1:
                                st.markdown("#### Oil Parameters")
                                
                                # Safely get the first forecast value if it exists
                                first_oil_fcst = None
                                if "OilFcst_BBL" in forecast_df.columns and not forecast_df["OilFcst_BBL"].empty:
                                    first_oil_fcst_idx = forecast_df["OilFcst_BBL"].first_valid_index()
                                    if first_oil_fcst_idx is not None:
                                        first_oil_fcst = forecast_df.loc[first_oil_fcst_idx, "OilFcst_BBL"]
                                
                                # Check for pending form values first, then use row values
                                if selected_key in st.session_state['pending_form_values']:
                                    pending = st.session_state['pending_form_values'][selected_key]
                                    
                                    decline_type = pending.get("OIL_DECLINE_TYPE", well_row.get("OIL_DECLINE_TYPE", "EXP"))
                                    decline_rate = pending.get("OIL_USER_DECLINE", pending.get("OIL_EMPIRICAL_DECLINE", 
                                                        well_row.get("OIL_USER_DECLINE", well_row.get("OIL_EMPIRICAL_DECLINE", "N/A"))))
                                    b_factor = pending.get("OIL_USER_B_FACTOR", pending.get("OIL_CALC_B_FACTOR", 
                                                well_row.get("OIL_USER_B_FACTOR", well_row.get("OIL_CALC_B_FACTOR", "N/A"))))
                                    terminal_decline = pending.get("OIL_D_MIN", well_row.get("OIL_D_MIN", "N/A"))
                                    min_rate = pending.get("OIL_Q_MIN", well_row.get("OIL_Q_MIN", "N/A"))
                                else:
                                    decline_type = well_row.get("OIL_DECLINE_TYPE", "EXP")
                                    decline_rate = well_row.get("OIL_USER_DECLINE", well_row.get("OIL_EMPIRICAL_DECLINE", "N/A"))
                                    b_factor = well_row.get("OIL_USER_B_FACTOR", well_row.get("OIL_CALC_B_FACTOR", "N/A"))
                                    terminal_decline = well_row.get("OIL_D_MIN", "N/A")
                                    min_rate = well_row.get("OIL_Q_MIN", "N/A")
                                
                                oil_params = {
                                    "Initial Rate (Qi)": f"{first_oil_fcst:.2f}" if first_oil_fcst is not None else "N/A",
                                    "Decline Type": decline_type,
                                    "Decline Rate": decline_rate,
                                    "B Factor": b_factor,
                                    "Terminal Decline": terminal_decline,
                                    "Minimum Rate": min_rate
                                }
                                
                                for param, value in oil_params.items():
                                    st.markdown(f"**{param}:** {value}")

                            # For the Gas Parameters section:
                            with param_col2:
                                st.markdown("#### Gas Parameters")
                                
                                # Safely get the first forecast value if it exists
                                first_gas_fcst = None
                                if "GasFcst_MCF" in forecast_df.columns and not forecast_df["GasFcst_MCF"].empty:
                                    first_gas_fcst_idx = forecast_df["GasFcst_MCF"].first_valid_index()
                                    if first_gas_fcst_idx is not None:
                                        first_gas_fcst = forecast_df.loc[first_gas_fcst_idx, "GasFcst_MCF"]
                                
                                # Check for pending form values first, then use row values
                                if selected_key in st.session_state['pending_form_values']:
                                    pending = st.session_state['pending_form_values'][selected_key]
                                    
                                    decline_type = pending.get("GAS_DECLINE_TYPE", well_row.get("GAS_DECLINE_TYPE", "EXP"))
                                    decline_rate = pending.get("GAS_USER_DECLINE", pending.get("GAS_EMPIRICAL_DECLINE", 
                                                        well_row.get("GAS_USER_DECLINE", well_row.get("GAS_EMPIRICAL_DECLINE", "N/A"))))
                                    b_factor = pending.get("GAS_USER_B_FACTOR", pending.get("GAS_CALC_B_FACTOR", 
                                                well_row.get("GAS_USER_B_FACTOR", well_row.get("GAS_CALC_B_FACTOR", "N/A"))))
                                    terminal_decline = pending.get("GAS_D_MIN", well_row.get("GAS_D_MIN", "N/A"))
                                    min_rate = pending.get("GAS_Q_MIN", well_row.get("GAS_Q_MIN", "N/A"))
                                else:
                                    decline_type = well_row.get("GAS_DECLINE_TYPE", "EXP")
                                    decline_rate = well_row.get("GAS_USER_DECLINE", well_row.get("GAS_EMPIRICAL_DECLINE", "N/A"))
                                    b_factor = well_row.get("GAS_USER_B_FACTOR", well_row.get("GAS_CALC_B_FACTOR", "N/A"))
                                    terminal_decline = well_row.get("GAS_D_MIN", "N/A")
                                    min_rate = well_row.get("GAS_Q_MIN", "N/A")
                                
                                gas_params = {
                                    "Initial Rate (Qi)": f"{first_gas_fcst:.2f}" if first_gas_fcst is not None else "N/A",
                                    "Decline Type": decline_type,
                                    "Decline Rate": decline_rate,
                                    "B Factor": b_factor,
                                    "Terminal Decline": terminal_decline,
                                    "Minimum Rate": min_rate
                                }
                                
                                for param, value in gas_params.items():
                                    st.markdown(f"**{param}:** {value}")
                        
                        # Add a download button for the forecast data
                        st.markdown("### Download Forecast")
                        csv = forecast_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="{selected_key}_forecast.csv">Download Forecast CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
        
else:
    st.warning("No data available or primary key not found in table")