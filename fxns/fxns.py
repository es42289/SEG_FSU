import streamlit as st, pandas as pd
from snowflake.snowpark import Session

def get_snowflake_session():
    connection_parameters = {'user':"ELII",
        'password':"Elii123456789!",
        'account':"CMZNSCB-MU47932",
        'warehouse':"COMPUTE_WH",
        'database':"WELLS",
        'schema':"MINERALS"}
    return Session.builder.configs(connection_parameters).create()

@st.cache_data(ttl=3600)  # Cache for an hour, adjust as needed
def load_owners():
    session = get_snowflake_session()
    query = 'SELECT * FROM WELLS.MINERALS.DISTINCT_OWNERS_WITH_DATA ORDER BY WELL_COUNT DESC;'
    result = session.sql(query).to_pandas()
    session.close()
    return result

@st.cache_data(ttl=600)
def load_wells(owners):
    session = get_snowflake_session()
    # If owners is None or empty, return all wells
    if not owners:
        query = """
            SELECT DISTINCT "API_10_EXPLODED"
            FROM WELLS.MINERALS.DI_TX_MINERAL_APPRAISALS_2023_EXPLODED
            ORDER BY "API_10_EXPLODED"
        """
    else:
        # Handle single string or list of owners
        if isinstance(owners, str):
            owners = [owners]
        # Escape quotes and format for SQL IN clause
        formatted_owners = ", ".join(f"'{owner}'" for owner in owners)
        query = f"""
            SELECT DISTINCT "API_10_EXPLODED" 
            FROM WELLS.MINERALS.DI_TX_MINERAL_APPRAISALS_2023_EXPLODED
            WHERE "Owner" IN ({formatted_owners})
            ORDER BY "API_10_EXPLODED" """
    result = session.sql(query).to_pandas()
    session.close()
    return result

@st.cache_data(ttl=600)
def get_filtered_well_data(well_ids):
    session = get_snowflake_session()
    try:
        # Make sure we have something to query
        if not well_ids or len(well_ids) == 0:
            return pd.DataFrame()
        if len(well_ids) < 200000:
            # Convert well list to SQL-safe string of quoted IDs
            formatted_ids = ",".join(f"'{w}'" for w in well_ids)
            query = f"""
            SELECT API_UWI, WELLNAME, STATEPROVINCE, COUNTRY, COUNTY, FIRSTPRODDATE, LATITUDE, LONGITUDE,
                ENVOPERATOR, LEASE, ENVWELLSTATUS, ENVINTERVAL, TRAJECTORY, CUMGAS_MCF, CUMOIL_BBL, TOTALPRODUCINGMONTHS
            FROM wells.minerals.vw_well_input
            WHERE REPLACE(API_UWI, '-', '') IN ({formatted_ids})
            """
            result = session.sql(query).to_pandas()
        else:
            query = f"""
            SELECT API_UWI, WELLNAME, STATEPROVINCE, COUNTRY, COUNTY, FIRSTPRODDATE, LATITUDE, LONGITUDE,
                ENVOPERATOR, LEASE, ENVWELLSTATUS, ENVINTERVAL, TRAJECTORY, CUMGAS_MCF, CUMOIL_BBL, TOTALPRODUCINGMONTHS
            FROM wells.minerals.vw_well_input
            """
            result = session.sql(query).to_pandas()
        if result.empty:
            st.warning("No well data retrieved from database.")
            return None
        result["CUMOIL_BBL"] = result["CUMOIL_BBL"].fillna(0)
        result["CUMGAS_MCF"] = result["CUMGAS_MCF"].fillna(0)
        session.close()
        return result
    except Exception as e:
        st.error(f"Error fetching well data: {e}")
        session.close()
        return pd.DataFrame()
    
@st.cache_data(ttl=600)
def get_production_data(selected_wells):
    session = get_snowflake_session()
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
            session.close()
        return result
    except Exception as e:
        print(f"Error fetching production data for selected wells: {str(e)}")
        session.close()
        return pd.DataFrame()
    
# @st.cache_data(ttl=600)
def get_decline_parameters(selected_wells):
    session = get_snowflake_session()
    selected_wells_query_string = ", ".join(f"'{api}'" for api in selected_wells)
    try:
        query = f"""
            SELECT *
            FROM wells.minerals.econ_input
            WHERE API_UWI IN ({selected_wells_query_string})
            ORDER BY API_UWI;
            """
        result = session.sql(query).to_pandas()
        session.close()
        return result
    except Exception as e:
        st.error(f"Error fetching decline parameters: {e}")
        session.close()
        return pd.DataFrame()
    
def make_alias(owner, count):
    well_word = "Well" if count == 1 else "Wells"
    return f"{owner} ({count} {well_word})"

# Update database record function
def update_database_record(params):
    session = get_snowflake_session()
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