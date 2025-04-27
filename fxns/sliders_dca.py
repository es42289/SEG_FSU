import streamlit as st
import pandas as pd
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from fxns.calc_decline import calc_decline
import base64

@st.fragment()
def sliders_and_chart(wells_df, record_well_data, raw_prod_data_df, decline_parameters):
    # Create a three-column layout with API selection on the left
    left_column, middle_column, right_column = st.columns([1, 1, 1])
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
                st.session_state['selected_well'] = grid_response['selected_rows']['API'].values[0]
                raw_prod_data_df = get_production_data([st.session_state['selected_well']])
                decline_parameters = get_decline_parameters([st.session_state['selected_well']])
            except:
                st.session_state['selected_well'] = wells_df.iloc[0]['API']

    ## calc and collect forecast df
    prd = raw_prod_data_df[raw_prod_data_df['API_UWI'] == st.session_state['selected_well']].copy()
    params = decline_parameters[decline_parameters['API_UWI'] == st.session_state['selected_well']].copy().iloc[0].to_dict()
    with middle_column:
        st.markdown("## Decline Parameters")
        with st.expander('Gas Decline Parameters', expanded=False):
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
                    min_value=max(1.0, 0.75*min(prd[prd['PRODUCINGMONTH']>datetime(params['FCST_START_GAS'].year, params['FCST_START_GAS'].month, params['FCST_START_GAS'].day)]['GASPROD_MCF'])),
                    max_value=max(2.0,1.25*max(prd[prd['PRODUCINGMONTH']>datetime(params['FCST_START_GAS'].year, params['FCST_START_GAS'].month, params['FCST_START_GAS'].day)]['GASPROD_MCF'])),
                    value=prd[prd['PRODUCINGMONTH']==datetime(params['FCST_START_GAS'].year, params['FCST_START_GAS'].month, params['FCST_START_GAS'].day)]['GASPROD_MCF'].values[0],
                    step=1.0,
                    key = 'GAS_USER_QI',
                    help="Select the initial production rate (Qi) for the forecast."
                )

                ## gas qmin user input
                params['GAS_Q_MIN'] = st.slider(
                    f"Qmin",
                    min_value = 10.0,
                    max_value = 1000.0,
                    value = 30.0,
                    step=10.0,
                    key = 'GAS_Q_MIN',
                    help="Select the minimum production rate for the well"
                )

            with mid_right_col:
                ## DECLINE TYPE SELECTOR
                gas_decl_type_options = ['EXP','HYP']
                default_index_gas_decl_typ = gas_decl_type_options.index(params['GAS_DECLINE_TYPE'])
                params['GAS_DECLINE_TYPE'] = st.radio("Choose one:", gas_decl_type_options, index = default_index_gas_decl_typ, horizontal = True)
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
                    min_value=max(1.0, 0.75*min(prd[prd['PRODUCINGMONTH']>datetime(params['FCST_START_OIL'].year, params['FCST_START_OIL'].month, params['FCST_START_OIL'].day)]['LIQUIDSPROD_BBL'])),
                    max_value=max(1.0,1.25*max(prd[prd['PRODUCINGMONTH']>datetime(params['FCST_START_OIL'].year, params['FCST_START_OIL'].month, params['FCST_START_OIL'].day)]['LIQUIDSPROD_BBL'])),
                    value=prd[prd['PRODUCINGMONTH']==datetime(params['FCST_START_OIL'].year, params['FCST_START_OIL'].month, params['FCST_START_OIL'].day)]['LIQUIDSPROD_BBL'].values[0],
                    step=1.0,
                    key = 'oil_user_qi',
                    # help="Select the initial production rate (Qi) for the forecast."
                )
                ## gas qmin user input
                params['OIL_Q_MIN'] = st.slider(
                    f"Qmin",
                    min_value = 0.0,
                    max_value = 100.0,
                    value = 5.0,
                    step=1.0,
                    key = 'OIL_Q_MIN',
                    help="Select the minimum production rate for the well"
                )
            with mid_right_col:
                ## DECLINE TYPE SELECTOR
                gas_decl_type_options = ['EXP','HYP']
                default_index_gas_decl_typ = gas_decl_type_options.index(params['OIL_DECLINE_TYPE'])
                params['OIL_DECLINE_TYPE'] = st.radio("Choose one:", gas_decl_type_options, index = default_index_gas_decl_typ, key = 'OIL_DECLINE_TYPE', horizontal = True)

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
        chart_data = calc_decline(prd, params)  
        chart_data[['LIQUIDSPROD_BBL', 'GASPROD_MCF']] = chart_data[['LIQUIDSPROD_BBL', 'GASPROD_MCF']].replace(0, None)
        chart_data = chart_data.rename(columns={
            'LIQUIDSPROD_BBL': 'Oil (BBL)',
            'GASPROD_MCF': 'Gas (MCF)',
            'OilFcst_BBL': 'Oil Forecast (BBL)',
            'GasFcst_MCF': 'Gas Forecast (MCF)'
        })
        with st.expander("Forecast Statistics", expanded=False):
            # Calculate forecast totals
            gas_forecast_total = chart_data['Gas Forecast (MCF)'].sum()
            
            # Calculate forecast years based on monthly data
            forecast_months = len(prd)
            forecast_years = forecast_months / 12
            
            # Create three columns for oil, gas, and general forecast stats
            fcst_col1, fcst_col2 = st.columns(2)
            
            with fcst_col1:
                st.markdown("#### Gas Statistics")
                gas_hist_total = chart_data['Gas (MCF)'].sum()
                gas_fcst_total = chart_data[chart_data['PRODUCINGMONTH']>datetime.today()]['Gas Forecast (MCF)'].sum()
                last_gas_date = chart_data[~chart_data['Gas (MCF)'].isna()]['PRODUCINGMONTH'].max()
                gas_eur = gas_hist_total + gas_fcst_total + chart_data[\
                    (chart_data['PRODUCINGMONTH']> datetime(last_gas_date.year, last_gas_date.month, last_gas_date.day))\
                    &\
                    (chart_data['PRODUCINGMONTH']< datetime.today())\
                        ]['Gas Forecast (MCF)'].sum()

                st.metric("Historic Gas Total (MCF)", f"{gas_hist_total:,.0f}")
                st.metric("Gas EUR (MCF)", f"{gas_eur:,.0f}")
                st.metric(f"Forecasted Gas Recovery (starting {date(date.today().year, date.today().month+1, 1)})", f"{gas_fcst_total:,.0f}")
                st.metric(f"Forecasted Qmin Date", str(chart_data[chart_data['Gas Forecast (MCF)']>0]['PRODUCINGMONTH'].max().date()))
                # st.markdown(f"<div style='font-size:24px;'>Gas is <b>{1-(gas_fcst_total/gas_eur):.0%}</b> Depleted</div>", unsafe_allow_html=True)

            with fcst_col2:
                st.markdown("#### Oil Statistics")
                oil_hist_total = chart_data['Oil (BBL)'].sum()
                oil_fcst_total = chart_data[chart_data['PRODUCINGMONTH']>datetime.today()]['Oil Forecast (BBL)'].sum()
                last_oil_date = chart_data[~chart_data['Oil (BBL)'].isna()]['PRODUCINGMONTH'].max()
                oil_eur = oil_hist_total + oil_fcst_total + chart_data[\
                    (chart_data['PRODUCINGMONTH']> datetime(last_gas_date.year, last_gas_date.month, last_gas_date.day))\
                    &\
                    (chart_data['PRODUCINGMONTH']< datetime.today())\
                        ]['Oil Forecast (BBL)'].sum()
                st.metric("Historic Oil Total (MCF)", f"{oil_hist_total:,.0f}")
                st.metric("Oil EUR (MCF)", f"{oil_eur:,.0f}")
                st.metric(f"Forecasted Oil Recovery (starting {date(date.today().year, date.today().month+1, 1)})", f"{oil_fcst_total:,.0f}")
                st.metric(f"Forecasted Qmin Date", str(chart_data[chart_data['Oil Forecast (BBL)']>0]['PRODUCINGMONTH'].max().date()))
                # st.markdown(f"<div style='font-size:24px;'>Oil is <b>{1-(oil_fcst_total/oil_eur):.0%}</b> Depleted</div>", unsafe_allow_html=True)
        
            # Add a download button for the forecast data
            st.markdown("### Download Forecast")
            csv = prd.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            download_well = st.session_state['selected_well']
            href = f'<a href="data:file/csv;base64,{b64}" download="{download_well}_forecast.csv">Download Forecast CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        st.divider()     
        if st.button("Save decline Parameters", key="save_decline", help="Save decline parameters to the ECON_INPUT table in Snowflake"):
            update_database_record(params)
            decline_parameters = get_decline_parameters([st.session_state['selected_well']])
    with right_column:
        st.markdown("## Forecast Chart")
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
        chart = alt.Chart(melted_data).encode(
            x=alt.X('Production Month:T', title='Production Month'),
            y=alt.Y('Volume:Q', scale= alt.Scale(type='log', domainMin=1), title='MCF or BBL per Month'),
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
        ).mark_line(point=False, strokeDash=[6, 3, 3, 2], strokeWidth = 2)  # Dashed line for forecast

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
            # strokeDash=[6, 4],
            strokeWidth=1.5,
            opacity = 0.2
        ).encode(
            x='date:T',
            tooltip=[alt.Tooltip('date:T', title='Oil Forecast Start')]
        )
        gas_rule = alt.Chart(pd.DataFrame({'date': [gas_start_date]})).mark_rule(
            color='red',
            # strokeDash=[6, 4],
            strokeWidth=1.5,
            opacity = 0.2
        ).encode(
            x='date:T',
            tooltip=[alt.Tooltip('date:T', title='Gas Forecast Start')]
        )

        final_chart = final_chart + oil_rule + gas_rule
        ## well info above chart
        # record_well_data = filtered_wells[filtered_wells['API_UWI'] == st.session_state['selected_well']].iloc[0]
        # st.markdown(f"{record_well_data['API_UWI']}")
        st.markdown(f"**Operator:** {record_well_data['ENVOPERATOR']} &nbsp;\
                    **Well Name:** {record_well_data['WELLNAME']}")
        st.markdown(f"**Status:** {record_well_data['ENVWELLSTATUS']} &nbsp;\
                    **First PRD:** {record_well_data['FIRSTPRODDATE']}")

        # Display the chart
        st.altair_chart(final_chart, use_container_width=True)
    return chart_data, params, prd