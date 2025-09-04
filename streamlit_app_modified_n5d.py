import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from power_optimizer_modified_n5d import PowerOptimizer, PowerSource, BatteryEnergyStorageSystem

# Page config
st.set_page_config(
    page_title="Power Optimization System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database configuration
db_config = {
    'user': 'root',
    'password': 'password',
    'host': 'localhost',
    'database': 'power_optimization'
}

# Initialize PowerOptimizer
optimizer = PowerOptimizer(db_config)

# Sidebar
st.sidebar.title("Power Optimization System")
st.sidebar.markdown("Configure and run power optimization for your energy sources.")

# Site selection
site_id = st.sidebar.selectbox("Select Site ID", [1, 2, 3], index=0)

# Global parameters input
st.sidebar.subheader("Global Parameters")
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=10.0)
fuel_pressure = st.sidebar.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=10.0, value=3.0)
fuel_level = st.sidebar.number_input("Fuel Level (%)", min_value=0.0, max_value=100.0, value=50.0)
gas_pressure = st.sidebar.number_input("Gas Pressure (bar)", min_value=0.0, max_value=10.0, value=2.0)
ghi = st.sidebar.number_input("Global Horizontal Irradiance (W/m²)", min_value=0.0, max_value=1500.0, value=600.0)

# Set global parameters
optimizer.set_global_params(
    wind_speed=wind_speed,
    fuel_pressure=fuel_pressure,
    fuel_level=fuel_level,
    gas_pressure=gas_pressure,
    ghi=ghi
)

# Load configuration
optimizer.initialize_sources(site_id)

# Main content
st.title("Power Optimization Dashboard")
st.markdown(f"**Site ID: {site_id}** | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# System Status
st.header("System Status")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Active Load (kW)", f"{sum(s.current_active_load for s in optimizer.sources):.2f}")
with col2:
    st.metric("Total Reactive Load (kVAR)", f"{sum(s.current_reactive_load for s in optimizer.sources):.2f}")
with col3:
    st.metric("Grid Status", "Connected" if optimizer.grid_connected else "Disconnected")

# Source Status
st.subheader("Source Status")
source_data = []
for source in optimizer.sources:
    source_data.append({
        "Source": source.name,
        "Active Power (kW)": round(source.current_active_load, 2),
        "Reactive Power (kVAR)": round(source.current_reactive_load, 2),
        "Status": "Active" if source.available else "Inactive",
        "Reliability Score": round(source.reliability_score, 2)
    })
source_df = pd.DataFrame(source_data)
st.dataframe(source_df, use_container_width=True)

# BESS Status
st.subheader("BESS Status")
bess_data = []
for bess in optimizer.bess_systems:
    bess_data.append({
        "BESS": bess.name,
        "SOC (%)": round(bess.current_soc, 2),
        "Mode": bess.mode.capitalize(),
        "Power Input (kW)": round(bess.current_power_input, 2),
        "Status": "Active" if bess.available else "Inactive"
    })
bess_df = pd.DataFrame(bess_data)
st.dataframe(bess_df, use_container_width=True)

# Optimization Parameters
st.header("Optimization Parameters")
col1, col2 = st.columns(2)
with col1:
    optimizer.max_peak_load = st.number_input("Max Peak Load (kW)", min_value=0.0, value=1000.0)
    optimizer.critical_load = st.number_input("Critical Load (kW)", min_value=0.0, value=500.0)
with col2:
    optimizer.tripping_cost = st.number_input("Tripping Cost (PKR)", min_value=0.0, value=50000.0)
    optimizer.production_loss_hourly = st.number_input("Production Loss Hourly (PKR)", min_value=0.0, value=10000.0)

# Run Optimization
if st.button("Run Optimization"):
    with st.spinner("Optimizing power allocation..."):
        results_df, total_current_cost, total_savings_hr = optimizer.generate_results()
        
        # Display Optimization Results
        st.header("Optimization Results")
        st.markdown(f"**Chosen Optimization Mode: {optimizer.optimized_mode.capitalize()}**")
        
        # Format numeric columns
        numeric_columns = [
            'CURRENT LOAD (kW)', 'COST OPT LOAD (kW)', 'REL OPT LOAD (kW)',
            'CURRENT KVAR (kVAR)', 'COST OPT KVAR (kVAR)', 'REL OPT KVAR (kVAR)',
            'TOTAL COST (PKR/kWh)', 'PRODUCTION COST (PKR/kWh)', 'CARBON COST (PKR/kWh)',
            'CURRENT COST/HR', 'COST OPT COST/HR', 'REL OPT COST/HR',
            'COST OPT CHARGE (kW)', 'COST OPT DISCHARGE (kW)',
            'REL OPT CHARGE (kW)', 'REL OPT DISCHARGE (kW)',
            'COST GRID FEED (kW)', 'REL GRID FEED (kW)',
            'RELIABILITY SCORE', 'EFFICIENCY SCORE'
        ]
        for col in numeric_columns:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        
        st.dataframe(results_df, use_container_width=True)
        
        # Visualization
        st.subheader("Power Allocation Comparison")
        fig = go.Figure()
        
        # Current Load
        fig.add_trace(go.Bar(
            x=results_df['ENERGY SOURCE'],
            y=results_df['CURRENT LOAD (kW)'].astype(float),
            name='Current Load (kW)',
            marker_color='blue'
        ))
        
        # Cost-Optimized Load
        fig.add_trace(go.Bar(
            x=results_df['ENERGY SOURCE'],
            y=results_df['COST OPT LOAD (kW)'].astype(float),
            name='Cost-Optimized Load (kW)',
            marker_color='green'
        ))
        
        # Reliability-Optimized Load
        fig.add_trace(go.Bar(
            x=results_df['ENERGY SOURCE'],
            y=results_df['REL OPT LOAD (kW)'].astype(float),
            name='Reliability-Optimized Load (kW)',
            marker_color='orange'
        ))
        
        fig.update_layout(
            barmode='group',
            title="Active Power Allocation Comparison",
            xaxis_title="Energy Source",
            yaxis_title="Power (kW)",
            legend_title="Load Type"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Reactive Power Comparison
        st.subheader("Reactive Power Allocation Comparison")
        fig_kvar = go.Figure()
        
        fig_kvar.add_trace(go.Bar(
            x=results_df['ENERGY SOURCE'],
            y=results_df['CURRENT KVAR (kVAR)'].astype(float),
            name='Current KVAR',
            marker_color='blue'
        ))
        
        fig_kvar.add_trace(go.Bar(
            x=results_df['ENERGY SOURCE'],
            y=results_df['COST OPT KVAR (kVAR)'].astype(float),
            name='Cost-Optimized KVAR',
            marker_color='green'
        ))
        
        fig_kvar.add_trace(go.Bar(
            x=results_df['ENERGY SOURCE'],
            y=results_df['REL OPT KVAR (kVAR)'].astype(float),
            name='Reliability-Optimized KVAR',
            marker_color='orange'
        ))
        
        fig_kvar.update_layout(
            barmode='group',
            title="Reactive Power Allocation Comparison",
            xaxis_title="Energy Source",
            yaxis_title="Reactive Power (kVAR)",
            legend_title="KVAR Type"
        )
        st.plotly_chart(fig_kvar, use_container_width=True)
        
        # Cost Comparison
        st.subheader("Cost Comparison")
        fig_cost = go.Figure()
        
        fig_cost.add_trace(go.Bar(
            x=results_df['ENERGY SOURCE'],
            y=results_df['CURRENT COST/HR'].astype(float),
            name='Current Cost/Hr',
            marker_color='blue'
        ))
        
        fig_cost.add_trace(go.Bar(
            x=results_df['ENERGY SOURCE'],
            y=results_df['COST OPT COST/HR'].astype(float),
            name='Cost-Optimized Cost/Hr',
            marker_color='green'
        ))
        
        fig_cost.add_trace(go.Bar(
            x=results_df['ENERGY SOURCE'],
            y=results_df['REL OPT COST/HR'].astype(float),
            name='Reliability-Optimized Cost/Hr',
            marker_color='orange'
        ))
        
        fig_cost.update_layout(
            barmode='group',
            title="Hourly Cost Comparison",
            xaxis_title="Energy Source",
            yaxis_title="Cost (PKR/hr)",
            legend_title="Cost Type"
        )
        st.plotly_chart(fig_cost, use_container_width=True)
        
        # Recommendations
        st.subheader("Optimization Recommendations")
        recommendations = optimizer.generate_recommendations(results_df)
        st.markdown(recommendations)

# Configuration File Upload
st.sidebar.subheader("Upload Configuration")
uploaded_file = st.sidebar.file_uploader("Choose a JSON config file", type="json")
if uploaded_file is not None:
    config_data = json.load(uploaded_file)
    with open("temp_config.json", "w") as f:
        json.dump(config_data, f)
    st.sidebar.success("Configuration uploaded successfully!")