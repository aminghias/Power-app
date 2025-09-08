import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from power_optimizer_modified_n6a import PowerOptimizer, PowerSource, BatteryEnergyStorageSystem

# Page config
st.set_page_config(
    page_title="Power Optimization System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .source-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }
    .bess-card {
        border: 2px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8fff8;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'power_sources' not in st.session_state:
        st.session_state.power_sources = {}
    if 'bess_config' not in st.session_state:
        st.session_state.bess_config = {}
    if 'global_params' not in st.session_state:
        st.session_state.global_params = {
            'wind_speed': 10.0,
            'fuel_pressure': 3.0,
            'fuel_level': 50.0,
            'gas_pressure': 2.0,
            'ghi': 600.0
        }

def create_mock_optimizer():
    """Create a mock optimizer without database connection"""
    # Mock db config for standalone operation
    host='enerlytics.cm2egm0j3xhd.ap-south-1.rds.amazonaws.com'
    port=3306
    user='admin'
    password='zN5mDVC9yG6gj2XnG6NY'
    database='Enerlytics_DB'
    # host=
    # port=
    # user=
    # password=
    # database=
    db_config = {
        'host': host,
        'port': port,
        'user': user,
        'password': password,
        'database': database
    }
    return PowerOptimizer(db_config)

def save_power_source_data(source_name, active_power, reactive_power):
    """Save power source data to CSV file"""
    data = {
        'active_power': [active_power],
        'reactive_power': [reactive_power]
    }
    csv_file = f"{source_name}_data.csv"
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file

def create_temp_config(sources_config, bess_config):
    """Create site configuration JSON file"""
    config = {
        "grid_connected": True,
        "allow_grid_feed": True,
        "bess_systems": bess_config,
        "installed_sources": sources_config
    }
    
    with open("temp_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    return config

def main():
    st.markdown('<h1 class="main-header">⚡ APEX Adaptive Power Exchange - Power Optimization System</h1>', unsafe_allow_html=True)
    
    initialize_session_state()
    
    st.sidebar.header("🔧 Global Settings")
    
    st.sidebar.subheader("Carbon Pricing")
    carbon_cost_pkr_kg = st.sidebar.number_input("Standard Carbon Cost (PKR/kg CO2)", 
                                                  min_value=0.0, value=50.0, 
                                                  help="Standard carbon pricing for Pakistan")
    
    st.sidebar.subheader("Operational Parameters")
    wind_speed = st.sidebar.number_input("Wind Speed (m/s)", min_value=0.0, value=10.0, 
                                        help="Wind speed for wind turbine operation (3.5-25 m/s)")
    fuel_pressure = st.sidebar.number_input("Fuel Pressure (bar)", min_value=0.0, value=3.0, 
                                           help="Fuel pressure for diesel generators (>2 bar)")
    fuel_level = st.sidebar.number_input("Fuel Level (%)", min_value=0.0, max_value=100.0, value=50.0, 
                                        help="Fuel level for diesel generators (>10%)")
    gas_pressure = st.sidebar.number_input("Gas Pressure (bar)", min_value=0.0, value=2.0, 
                                          help="Gas pressure for gas generators (>1 bar)")
    ghi = st.sidebar.number_input("Global Horizontal Irradiance (W/m²)", min_value=0.0, value=600.0, 
                                  help="GHI for solar panels (>100 W/m²)")
    
    st.session_state.global_params.update({
        'wind_speed': wind_speed,
        'fuel_pressure': fuel_pressure,
        'fuel_level': fuel_level,
        'gas_pressure': gas_pressure,
        'ghi': ghi
    })
    
    st.sidebar.subheader("Reliability Parameters")
    max_peak_load = st.sidebar.number_input("Max Peak Running Load (kW)", min_value=0.0, value=1000.0)
    critical_load = st.sidebar.number_input("Total Critical Load (kW)", min_value=0.0, value=500.0)
    tripping_cost = st.sidebar.number_input("Tripping Cost (PKR)", min_value=0.0, value=100000.0)
    production_loss = st.sidebar.number_input("Production Loss Cost (PKR/hour)", min_value=0.0, value=50000.0)
    
    st.sidebar.subheader("Source Counts")
    num_solar = st.sidebar.number_input("Solar Systems", min_value=0, max_value=5, value=1)
    num_wind = st.sidebar.number_input("Wind Turbines", min_value=0, max_value=5, value=0)
    num_diesel = st.sidebar.number_input("Diesel Generators", min_value=0, max_value=10, value=1)
    num_gas = st.sidebar.number_input("Gas Generators", min_value=0, max_value=10, value=3)
    num_grid = st.sidebar.number_input("Grid Connections", min_value=0, max_value=5, value=1)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏭 Power Sources", "🔋 BESS Configuration", "⚙️ Optimization", "📊 Results & Analytics"])
    
    with tab1:
        st.header("Power Sources Configuration")
        
        source_types = {
            'Solar': {'default_prod_cost': 3.0, 'default_carbon': 0.05, 'default_min': 0.0, 'default_max': 500.0, 'default_active': 150.0, 'default_reactive': 0.0, 'default_rel': 9.0},
            'Wind': {'default_prod_cost': 4.0, 'default_carbon': 0.03, 'default_min': 0.0, 'default_max': 600.0, 'default_active': 200.0, 'default_reactive': 0.0, 'default_rel': 8.5},
            'Diesel': {'default_prod_cost': 20.0, 'default_carbon': 0.8, 'default_min': 100.0, 'default_max': 500.0, 'default_active': 100.0, 'default_reactive': 50.0, 'default_rel': 7.0},
            'Gas': {'default_prod_cost': 15.0, 'default_carbon': 0.5, 'default_min': 100.0, 'default_max': 500.0, 'default_active': 100.0, 'default_reactive': 50.0, 'default_rel': 8.0},
            'Grid': {'default_prod_cost': 12.0, 'default_carbon': 0.6, 'default_min': 0.0, 'default_max': 1500.0, 'default_active': 100.0, 'default_reactive': 50.0, 'default_rel': 5.0},
        }
        
        sources_data = []
        
        for type_name, num in [('Solar', num_solar), ('Wind', num_wind), ('Diesel', num_diesel), ('Gas', num_gas), ('Grid', num_grid)]:
            if num > 0:
                with st.expander(f"🔌 {type_name} Configuration ( {num} units )", expanded=True):
                    production_cost = st.number_input(f"{type_name} Production Cost (PKR/kWh)", min_value=0.0, value=source_types[type_name]['default_prod_cost'], key=f"prod_cost_{type_name}")
                    carbon_emission = st.number_input(f"{type_name} Carbon Emission (kg CO2/kWh)", min_value=0.0, value=source_types[type_name]['default_carbon'], key=f"carbon_{type_name}")
                    min_cap = st.number_input(f"{type_name} Min Capacity per Unit (kW)", min_value=0.0, value=source_types[type_name]['default_min'], key=f"min_{type_name}")
                    max_cap = st.number_input(f"{type_name} Max Capacity per Unit (kW)", min_value=0.0, value=source_types[type_name]['default_max'], key=f"max_{type_name}")
                    reliability = st.number_input(f"{type_name} Reliability Score (1-10)", min_value=1.0, max_value=10.0, value=source_types[type_name]['default_rel'], key=f"rel_{type_name}")
                    
                    for i in range(1, num + 1):
                        source_name = f"{type_name}_{i}" if num > 1 else type_name
                        
                        st.subheader(f"{source_name}")
                        
                        with st.container():
                            st.markdown('<div class="source-card">', unsafe_allow_html=True)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                active_power = st.number_input(
                                    "Active Power (kW)", 
                                    min_value=0.0, 
                                    value=source_types[type_name]['default_active'],
                                    key=f"active_{source_name}"
                                )
                            with col_b:
                                reactive_power = st.number_input(
                                    "Reactive Power (kVAR)", 
                                    min_value=0.0, 
                                    value=source_types[type_name]['default_reactive'],
                                    key=f"reactive_{source_name}"
                                )
                            
                            total_cost = production_cost + (carbon_emission * carbon_cost_pkr_kg)
                            st.info(f"Total Cost: {total_cost:.2f} PKR/kWh (Production: {production_cost} + Carbon: {carbon_emission * carbon_cost_pkr_kg:.2f})")
                            
                            # Display availability status based on global parameters
                            if type_name.lower() == 'wind':
                                available = 3.5 <= wind_speed <= 25
                                st.info(f"Wind Speed: {wind_speed} m/s - {'Available' if available else 'Unavailable'}")
                            elif type_name.lower() == 'diesel':
                                available = fuel_pressure > 2 and fuel_level > 10
                                st.info(f"Fuel Pressure: {fuel_pressure} bar, Fuel Level: {fuel_level}% - {'Available' if available else 'Unavailable'}")
                            elif type_name.lower() == 'gas':
                                available = gas_pressure > 1
                                st.info(f"Gas Pressure: {gas_pressure} bar - {'Available' if available else 'Unavailable'}")
                            elif type_name.lower() == 'solar':
                                available = ghi > 100
                                st.info(f"GHI: {ghi} W/m² - {'Available' if available else 'Unavailable'}")
                            elif type_name.lower() == 'grid':
                                st.info("Grid: Always Available")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.session_state.power_sources[source_name] = {
                                'active_power': active_power,
                                'reactive_power': reactive_power,
                                'production_cost': production_cost,
                                'carbon_emission': carbon_emission,
                                'total_cost': total_cost,
                                'min_capacity': min_cap,
                                'max_capacity': max_cap,
                                'reliability_score': reliability
                            }
                            
                            save_power_source_data(source_name, active_power, reactive_power)
                            
                            sources_data.append({
                                'name': source_name,
                                'production_cost': production_cost,
                                'carbon_emission': carbon_emission,
                                'min_capacity': min_cap,
                                'max_capacity': max_cap,
                                'reliability_score': reliability,
                                'device_id': 1 if type_name == 'Grid' else None
                            })
    
    with tab2:
        st.header("Battery Energy Storage System (BESS)")
        
        st.subheader("🔋 BESS Configuration")
        
        num_bess = st.number_input("Number of BESS Units", min_value=0, max_value=10, value=1)
        
        if num_bess > 0:
            st.subheader("💰 BESS Cost Configuration (Shared for all units)")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bess_production_cost = st.number_input("BESS Operation Cost (PKR/kWh)", 
                                                       min_value=0.0, value=2.0,
                                                       help="Cost for BESS operation including maintenance")
            with col2:
                bess_carbon_emission = st.number_input("BESS Carbon Emission (kg CO2/kWh)", 
                                                       min_value=0.0, value=0.1,
                                                       help="Carbon footprint for BESS operation")
            with col3:
                bess_reliability = st.number_input("BESS Reliability Score (1-10)", 
                                                   min_value=1.0, max_value=10.0, value=9.0)
            
            bess_total_cost = bess_production_cost + (bess_carbon_emission * carbon_cost_pkr_kg)
            st.info(f"BESS Total Cost: {bess_total_cost:.2f} PKR/kWh (Operation: {bess_production_cost} + Carbon: {bess_carbon_emission * carbon_cost_pkr_kg:.2f})")
            
            bess_config = []
            bess_status_data = []
            
            for i in range(1, num_bess + 1):
                bess_name = f"BESS_{i}" if num_bess > 1 else "BESS"
                
                with st.expander(f"🔋 {bess_name} Configuration", expanded=True):
                    st.markdown('<div class="bess-card">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        bess_capacity = st.number_input("Battery Capacity (kWh)", min_value=0.0, value=1000.0, key=f"cap_{bess_name}")
                        power_rating = st.number_input("Power Rating (kW)", min_value=0.0, value=500.0, key=f"rating_{bess_name}")
                        power_input = st.number_input("Current Power Input/Output (kW)", 
                                                    min_value=-power_rating, max_value=power_rating, 
                                                    value=0.0, 
                                                    help="Positive for charging, negative for discharging", key=f"input_{bess_name}")
                    
                    with col2:
                        current_soc = st.slider("Current State of Charge (%)", 0, 100, 65, key=f"soc_{bess_name}")
                        discharge_threshold = st.slider("Discharge Threshold (%)", 0, 100, 50, key=f"dis_{bess_name}")
                        charge_threshold = st.slider("Charge Threshold (%)", 0, 100, 85, key=f"charge_{bess_name}")
                    
                    available = 20 < current_soc < 95  # Assuming min_soc=20, max_soc=95
                    st.info(f"SOC: {current_soc}% - {'Available' if available else 'Unavailable'}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    bess_config.append({
                        "name": bess_name,
                        "capacity_kwh": bess_capacity,
                        "power_rating_kw": power_rating,
                        "current_soc": current_soc,
                        "discharge_threshold": discharge_threshold,
                        "charge_threshold": charge_threshold,
                        "power_input": power_input,
                        "production_cost": bess_production_cost,
                        "carbon_emission": bess_carbon_emission,
                        "reliability_score": bess_reliability
                    })
                    
                    available_energy = (current_soc / 100) * bess_capacity
                    if power_input > 0:
                        status = "🔋 Charging"
                    elif power_input < 0:
                        status = "⚡ Discharging"
                    else:
                        status = "⏸️ Standby"
                    bess_status_data.append({
                        "Unit": bess_name,
                        "Capacity": f"{bess_capacity} kWh",
                        "SOC": f"{current_soc}%",
                        "Available Energy": f"{available_energy:.1f} kWh",
                        "Status": status,
                        "Available": "Yes" if available else "No"
                    })
            
            st.session_state.bess_config = bess_config
            
            if bess_status_data:
                st.subheader("📊 BESS Status")
                st.dataframe(pd.DataFrame(bess_status_data), use_container_width=True)
    
    with tab3:
        st.header("Power Optimization Engine")
        
        if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
            with st.spinner("Running power optimization..."):
                try:
                    create_temp_config(sources_data, st.session_state.bess_config)
                    
                    optimizer = create_mock_optimizer()
                    optimizer.carbon_cost_pkr_kg = carbon_cost_pkr_kg
                    optimizer.set_global_params(
                        wind_speed=wind_speed,
                        fuel_pressure=fuel_pressure,
                        fuel_level=fuel_level,
                        gas_pressure=gas_pressure,
                        ghi=ghi
                    )
                    optimizer.max_peak_load = max_peak_load
                    optimizer.critical_load = critical_load
                    optimizer.tripping_cost = tripping_cost
                    optimizer.production_loss_hourly = production_loss
                    
                    optimizer.initialize_sources(site_id="dummy")
                    
                    # Run cost optimization
                    optimizer.optimize_power_allocation_cost()
                    results_df_cost, total_current_cost, total_savings_hr_cost = optimizer.generate_results()
                    
                    # Run reliability optimization
                    optimizer.optimize_power_allocation_reliability()
                    results_df_rel, _, total_savings_hr_rel = optimizer.generate_results()
                    
                    # Decide which to use
                    cost_opt = results_df_cost[results_df_cost['ENERGY SOURCE'] == 'TOTAL']['COST OPTIMIZED COST/HR'].values[0]
                    cost_rel = results_df_rel[results_df_rel['ENERGY SOURCE'] == 'TOTAL']['RELIABILITY OPTIMIZED COST/HR'].values[0]
                    cost_diff = cost_rel - cost_opt
                    total_loss = optimizer.tripping_cost + optimizer.production_loss_hourly
                    
                    if total_loss > 100*cost_diff:
                        results_df = results_df_rel
                        total_savings_hr = total_savings_hr_rel
                        choice = "reliability"
                        reason = "loss > extra cost"
                    else:
                        results_df = results_df_cost
                        total_savings_hr = total_savings_hr_cost
                        choice = "cost"
                        reason = "loss <= extra cost"
                    
                    recommendations = optimizer.generate_recommendations(results_df)
                    recommendations += f"\n\nChosen mode: {choice} because {reason}"
                    
                    st.session_state.optimizer = optimizer
                    st.session_state.results = results_df
                    st.session_state.recommendations = recommendations
                    
                    st.success("✅ Optimization completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Optimization failed: {str(e)}")
                    st.exception(e)
        
        if st.session_state.power_sources:
            st.subheader("📋 Current Configuration Summary")
            
            total_current_active = sum(source['active_power'] for source in st.session_state.power_sources.values())
            total_current_reactive = sum(source['reactive_power'] for source in st.session_state.power_sources.values())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Active Load", f"{total_current_active:.1f} kW")
            with col2:
                st.metric("Total Reactive Load", f"{total_current_reactive:.1f} kVAR")
            with col3:
                if st.session_state.bess_config:
                    total_bess_power = sum(b['power_input'] for b in st.session_state.bess_config)
                    bess_status = "Charging" if total_bess_power > 0 else "Discharging" if total_bess_power < 0 else "Standby"
                    st.metric("BESS Overall Status", bess_status)
    
 
    with tab4:
        st.header("Optimization Results & Analytics")

        if st.session_state.results is not None:
            st.subheader("📊 Optimization Results")
            
            display_df = st.session_state.results.copy()
            st.dataframe(display_df, use_container_width=True)
            
            st.subheader("🎯 Key Metrics")
            
            total_row = display_df[display_df['ENERGY SOURCE'] == 'TOTAL'].squeeze()

            # import streamlit as st

            def display_metrics(total_row):
                # Define color scheme
                primary_color = "#1f77b4"  # Blue for normal cost
                cost_opt_color = "#2ca02c"  # Green for cost-optimized
                rel_opt_color = "#ff7f0e"  # Orange for reliability-optimized

                # First row: Normal Cost
                st.markdown("### Current Cost")
                col1 = st.columns(1)[0]
                with col1:
                    st.metric(
                        label="Current Cost",
                        value=f"PKR {total_row['CURRENT COST/HR']:,.0f}",
                        delta=None,
                        label_visibility="visible"
                    )

                # Second row: Cost Optimized Metrics
                st.markdown("### Cost Optimized Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Cost Optimized Cost",
                        value=f"PKR {total_row['COST OPTIMIZED COST/HR']:,.0f}",
                        delta=None,
                        label_visibility="visible"
                    )
                
                with col2:
                    savings_c = total_row['CURRENT COST/HR'] - total_row['COST OPTIMIZED COST/HR']
                    st.metric(
                        label="Cost Optimized Savings",
                        value=f"PKR {savings_c:,.0f}",
                        delta=f"{savings_c:,.0f} PKR saved",
                        delta_color="normal",
                        label_visibility="visible"
                    )
                
                with col3:
                    savings_percent_c = (savings_c / total_row['CURRENT COST/HR']) * 100 if total_row['CURRENT COST/HR'] > 0 else 0
                    st.metric(
                        label="Cost Optimized Savings %",
                        value=f"{savings_percent_c:.1f}%",
                        delta=f"{savings_percent_c:.1f}% savings",
                        delta_color="normal",
                        label_visibility="visible"
                    )

                # Third row: Reliability Optimized Metrics
                st.markdown("### Reliability Optimized Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Reliability Optimized Cost",
                        value=f"PKR {total_row['RELIABILITY OPTIMIZED COST/HR']:,.0f}",
                        delta=None,
                        label_visibility="visible"
                    )
                
                with col2:
                    savings_r = total_row['CURRENT COST/HR'] - total_row['RELIABILITY OPTIMIZED COST/HR']
                    st.metric(
                        label="Reliability Optimized Savings",
                        value=f"PKR {savings_r:,.0f}",
                        delta=f"{savings_r:,.0f} PKR saved",
                        delta_color="normal",
                        label_visibility="visible"
                    )
                
                with col3:
                    savings_percent_r = (savings_r / total_row['CURRENT COST/HR']) * 100 if total_row['CURRENT COST/HR'] > 0 else 0
                    st.metric(
                        label="Reliability Optimized Savings %",
                        value=f"{savings_percent_r:.1f}%",
                        delta=f"{savings_percent_r:.1f}% savings",
                        delta_color="normal",
                        label_visibility="visible"
                    )

            display_metrics(total_row)


            st.subheader("📈 Power Allocation Charts")
            
            chart_data = display_df[display_df['ENERGY SOURCE'] != 'TOTAL'].copy()
            
            # Cost Optimized Charts
            col1c, col2c = st.columns(2)
            
            with col1c:
                fig_power_c = go.Figure(data=[
                    go.Bar(name='Current', x=chart_data['ENERGY SOURCE'], y=chart_data['CURRENT LOAD (kW)']),
                    go.Bar(name='Cost Optimized', x=chart_data['ENERGY SOURCE'], y=chart_data['COST OPTIMIZED LOAD (kW)'])
                ])
                fig_power_c.update_layout(title="Current vs Cost Optimized Active Power", 
                                        xaxis_title="Sources", yaxis_title="Power (kW)")
                st.plotly_chart(fig_power_c, use_container_width=True)
            
            with col2c:
                fig_cost_c = go.Figure(data=[
                    go.Bar(name='Current Cost', x=chart_data['ENERGY SOURCE'], y=chart_data['CURRENT COST/HR']),
                    go.Bar(name='Cost Optimized Cost', x=chart_data['ENERGY SOURCE'], y=chart_data['COST OPTIMIZED COST/HR'])
                ])
                fig_cost_c.update_layout(title="Current vs Cost Optimized Cost", 
                                    xaxis_title="Sources", yaxis_title="Cost (PKR/HR)")
                st.plotly_chart(fig_cost_c, use_container_width=True)

            # Reliability Optimized Charts
            col1r, col2r = st.columns(2)

            with col1r:
                fig_power_r = go.Figure(data=[
                    go.Bar(name='Current', x=chart_data['ENERGY SOURCE'], y=chart_data['CURRENT LOAD (kW)']),
                    go.Bar(name='Reliability Optimized', x=chart_data['ENERGY SOURCE'], y=chart_data['RELIABILITY OPTIMIZED LOAD (kW)'])
                ])
                fig_power_r.update_layout(title="Current vs Reliability Optimized Active Power", 
                                        xaxis_title="Sources", yaxis_title="Power (kW)")
                st.plotly_chart(fig_power_r, use_container_width=True)

            with col2r:
                fig_cost_r = go.Figure(data=[
                    go.Bar(name='Current Cost', x=chart_data['ENERGY SOURCE'], y=chart_data['CURRENT COST/HR']),
                    go.Bar(name='Reliability Optimized Cost', x=chart_data['ENERGY SOURCE'], y=chart_data['RELIABILITY OPTIMIZED COST/HR'])
                ])
                fig_cost_r.update_layout(title="Current vs Reliability Optimized Cost", 
                                    xaxis_title="Sources", yaxis_title="Cost (PKR/HR)")
                st.plotly_chart(fig_cost_r, use_container_width=True)

            # BESS Analysis
            bess_data = chart_data[chart_data['ENERGY SOURCE'].str.contains('BESS', na=False)]
            if not bess_data.empty:
                st.subheader("🔋 BESS Operation Analysis")
                
                fig_bess = go.Figure()
                
                for _, row in bess_data.iterrows():
                    # Check if columns exist before accessing
                    charge_power = row.get('OPTIMIZED CHARGE (kW)', 0) or row.get('COST OPTIMIZED CHARGE (kW)', 0)
                    discharge_power = row.get('OPTIMIZED DISCHARGE (kW)', 0) or row.get('COST OPTIMIZED DISCHARGE (kW)', 0)
                    
                    fig_bess.add_trace(go.Bar(
                        name=f"{row['ENERGY SOURCE']} Charge",
                        x=[row['ENERGY SOURCE']],
                        y=[charge_power],
                        marker_color='green'
                    ))
                    
                    fig_bess.add_trace(go.Bar(
                        name=f"{row['ENERGY SOURCE']} Discharge",
                        x=[row['ENERGY SOURCE']],
                        y=[-discharge_power],
                        marker_color='red'
                    ))
                
                fig_bess.update_layout(
                    title="BESS Charge/Discharge Profile",
                    xaxis_title="BESS Units",
                    yaxis_title="Power (kW)",
                    barmode='relative'
                )
                st.plotly_chart(fig_bess, use_container_width=True)
            
            st.subheader("💡 Optimization Recommendations")
            if hasattr(st.session_state, 'recommendations') and st.session_state.recommendations:
                st.markdown(st.session_state.recommendations)
            else:
                st.info("No specific recommendations available. The optimization results show the most efficient power allocation based on your configured parameters.")
            
            st.subheader("💾 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name=f"power_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_results = display_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=json_results,
                    file_name=f"power_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        else:
            st.info("👈 Configure your power sources and run optimization to see results")
            
            st.subheader("📋 Sample Expected Output")
            st.write("After running optimization, you'll see:")
            st.write("- Detailed power allocation table with priority and total score")
            st.write("- Cost savings analysis with unified cost calculation")
            st.write("- BESS charging/discharging status and recommendations")
            st.write("- Interactive charts and visualizations")
            st.write("- Downloadable results in CSV/JSON format")

   
    st.markdown("---")
    st.markdown("**Power Optimization System** - Optimizing energy efficiency and cost with unified carbon pricing")
if __name__ == "__main__":
    main()