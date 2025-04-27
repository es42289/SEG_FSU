##########################################
# SECTION 1: IMPORTS & SETUP
##########################################
import streamlit as st, pandas as pd, altair as alt
from datetime import datetime
from fxns.fxns import get_snowflake_session, load_owners, load_wells, get_filtered_well_data, get_production_data, get_decline_parameters, make_alias, update_database_record
from fxns.calc_decline import calculate_decline_rates, calc_decline, cached_decline
from fxns.sliders_dca import sliders_and_chart
from fxns.maps import map_wells

# Set Streamlit Page Configuration with wider layout
st.set_page_config(
    page_title="Single Well Decline Editor",
    layout="wide",
    page_icon="🛢️",
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

##########################################
# SECTION 6: SESSION STATE INITIALIZATION
##########################################
# Initialize variables for decline calculation constants for single well

if 'selected_wells' not in st.session_state:
    st.session_state['selected_wells'] = []

if 'filtered_wells' not in st.session_state:
    st.session_state['filtered_wells'] = None

if 'selected_well' not in st.session_state:
    st.session_state['selected_well'] = None

##########################################
# SECTION 8: WELL FILTER AND MAP SECTION
##########################################
try:
    st.logo('logo.png')
except:
    pass
## title
st.title("Single Well Decline Editor")

# Create a two-column layout with filters on left and map on right
filter_column, map_column = st.columns([1, 1])

# Put all filters in the left column
with filter_column:        
    st.subheader("Filter Wells")
    ## set two columns for filters
    filter_column_col1, filter_column_col2 = st.columns([1, 1])
    with filter_column_col1:
        ## load the distinct owners list for the selection drop down
        owners = load_owners()
        ## make labels for the owner selection dropdown
        owners["alias"] = owners.apply(lambda row: make_alias(row["Owner"], row["WELL_COUNT"]), axis=1)
        ## create alias to owner dictionary
        alias_to_owner = dict(zip(owners["alias"], owners["Owner"]))
        ## selection box for owner
        selected_alias = st.multiselect("Filter by Owner", options=list(alias_to_owner.keys()))
        ## convert the alias back to the owner name
        owners = [alias_to_owner[x] for x in selected_alias]
        ## collect the wells associated to the owner
        start = datetime.now()##delete
        well_ids = load_wells(owners)["API_10_EXPLODED"].tolist()    
        
        # Load owner well header data
        well_data = get_filtered_well_data(well_ids)
        # Check if well_data is not None and has data
        if well_data is not None:
            # Initialize filtered_wells to be the original well data
            filtered_wells = well_data.copy()
            # Define filter columns in the specified order
            filters = [
                "ENVOPERATOR",
                "API_UWI",
                "ENVWELLSTATUS",
                "WELLNAME"
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

        with filter_column_col2:
            # Define filter columns in the specified order
            filters = [
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
    # else:
    #     st.warning("Could not load well data for filtering.")
# Put the map in the right column
with map_column:
    st.subheader("Well Map")
    
    # Only proceed if we have well data and filtered wells
    if 'filtered_wells' in locals() and not filtered_wells.empty:
        if 'LATITUDE' in filtered_wells.columns and 'LONGITUDE' in filtered_wells.columns:
            # Create a copy with just the needed columns
            map_data = filtered_wells.copy()
            try:
                map_wells(map_data)
                # Add information about the map
                st.info(f"Map shows {len(map_data)} filtered wells sized by Cum. BOE.")
            except:
                st.warning("Cannot display map: No valid coordinate data available")
    else:
        st.info("No filtered wells to display. Use the filters on the left to select wells.")

st.divider()
# Add visualization of top operators and counties after the two-column layout
# This is outside the columns to use full width
if 'filtered_wells' in locals() and not filtered_wells.empty:
    st.subheader("Well Distribution")
    
    # Create tabs for different visualizations
    # viz_tab1, viz_tab2 = st.tabs(["By Operator", "By County"])
    viz_tab1, viz_tab2 = st.columns([1,1])
    
    with viz_tab1:
        operator_counts = filtered_wells["ENVOPERATOR"].value_counts().reset_index()
        operator_counts.columns = ["Operator", "Count"]
        # Limit to top 10 operators
        if len(operator_counts) > 10:
            operator_counts = operator_counts.head(10)
            title_operator = "Top 10 Operators"
        else:
            title_operator = "Operators"

        chart_operator = alt.Chart(operator_counts).mark_bar(color="rgb(91,111,149)").encode(
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

        chart_county = alt.Chart(county_counts).mark_bar(color = "rgb(140,195,199)").encode(
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
    # raw_prod_data_df = get_production_data(wells_list)
    # decline_parameters = get_decline_parameters(wells_list)
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

    # ## limit to only rwos with production data
    # wells_df = wells_df[wells_df['API'].isin(raw_prod_data_df['API_UWI'])]
    ##default select first well in list
    # selected_key = wells_df.iloc[0]['API']
    st.session_state['selected_well'] = wells_df.iloc[0]['API']
    # selected_key = st.session_state['selected_well']
    raw_prod_data_df = get_production_data([st.session_state['selected_well']])
    decline_parameters = get_decline_parameters([st.session_state['selected_well']])
    record = decline_parameters[decline_parameters['API_UWI'] == st.session_state['selected_well']].iloc[0]
    record_well_data = filtered_wells[filtered_wells['API_UWI'] == st.session_state['selected_well']].iloc[0]
else:
    st.info("FIlter wells further to narrow results.")
    st.session_state['selected_well'] = None

if not decline_parameters.empty and 'API_UWI' in decline_parameters.columns:
    chart_data, params, prd = sliders_and_chart(wells_df, record_well_data, raw_prod_data_df, decline_parameters)


# st.write(params)
# st.write(prd)
# # st.write(melted_data)
# st.write(chart_data)