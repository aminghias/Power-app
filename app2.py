# app.py
import streamlit as st
import json
import pandas as pd
import os
import plotly.express as px
from power_optimizer_b import PowerOptimizer  # Import from the modular file

# Load default config from JSON
def load_default_config():
    config_file = "inputs_site_148.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return {
        "cost_weight": 0.5,
        "carbon_weight": 0.5,
        "grid_connected": True,
        "allow_grid_feed": True,
        "bess_systems": [],
        "installed_sources": []
    }

# Streamlit App
st.title("Power Optimization Dashboard")

st.markdown("""
This app optimizes power allocation based on your inputs. 
- Edit the weights, grid settings, BESS, and sources below.
- Provide current active/reactive power readings for each source.
- Click 'Run Optimization' to get results and dashboard.
""")

# Expandable section for inputs
with st.expander("Edit Configurations and Inputs", expanded=True):
    # Load defaults
    default_config = load_default_config()

    # Section 1: Weights and Grid Settings
    st.header("Weights and Grid Settings")
    cost_weight = st.number_input("Cost Weight", min_value=0.0, max_value=1.0, value=default_config.get("cost_weight", 0.5))
    carbon_weight = st.number_input("Carbon Weight", min_value=0.0, max_value=1.0, value=default_config.get("carbon_weight", 0.5))
    grid_connected = st.checkbox("Grid Connected", value=default_config.get("grid_connected", True))
    allow_grid_feed = st.checkbox("Allow Grid Feed", value=default_config.get("allow_grid_feed", True))

    # Section 2: BESS Systems
    st.header("BESS Systems")
    bess_systems = []
    num_bess = st.number_input("Number of BESS Systems", min_value=0, max_value=10, value=len(default_config.get("bess_systems", [])))
    for i in range(num_bess):
        st.subheader(f"BESS {i+1}")
        default_bess = default_config["bess_systems"][i] if i < len(default_config["bess_systems"]) else {}
        name = st.text_input(f"Name (BESS {i+1})", value=default_bess.get("name", f"BESS_{i+1}"))
        capacity_kwh = st.number_input(f"Capacity (kWh) (BESS {i+1})", value=default_bess.get("capacity_kwh", 1000.0))
        power_rating_kw = st.number_input(f"Power Rating (kW) (BESS {i+1})", value=default_bess.get("power_rating_kw", 500.0))
        current_soc = st.number_input(f"Current SOC (%) (BESS {i+1})", value=default_bess.get("current_soc", 65.0))
        discharge_threshold = st.number_input(f"Discharge Threshold (%) (BESS {i+1})", value=default_bess.get("discharge_threshold", 50.0))
        bess_systems.append({
            "name": name,
            "capacity_kwh": capacity_kwh,
            "power_rating_kw": power_rating_kw,
            "current_soc": current_soc,
            "discharge_threshold": discharge_threshold
        })

    # Section 3: Installed Sources
    st.header("Installed Sources")
    installed_sources = []
    num_sources = st.number_input("Number of Sources", min_value=0, max_value=20, value=len(default_config.get("installed_sources", [])))
    for i in range(num_sources):
        st.subheader(f"Source {i+1}")
        default_source = default_config["installed_sources"][i] if i < len(default_config["installed_sources"]) else {}
        name = st.text_input(f"Name (Source {i+1})", value=default_source.get("name", f"Source_{i+1}"))
        cost = st.number_input(f"Cost ($/kWh) (Source {i+1})", value=default_source.get("cost", 10.0))
        min_capacity = st.number_input(f"Min Capacity (kW) (Source {i+1})", value=default_source.get("min_capacity", 0.0))
        max_capacity = st.number_input(f"Max Capacity (kW) (Source {i+1})", value=default_source.get("max_capacity", 1000.0))
        carbon_footprint = st.number_input(f"Carbon Footprint (gm CO2/kWh) (Source {i+1})", value=default_source.get("carbon_footprint", 0.1))
        installed_sources.append({
            "name": name,
            "cost": cost,
            "min_capacity": min_capacity,
            "max_capacity": max_capacity,
            "carbon_footprint": carbon_footprint
        })

    # Section 4: Current Power Readings
    st.header("Current Power Readings")
    power_readings = {}
    for source in installed_sources:
        st.subheader(f"Readings for {source['name']}")
        active_power = st.number_input(f"Active Power (kW) for {source['name']}", value=0.0)
        reactive_power = st.number_input(f"Reactive Power (kVAR) for {source['name']}", value=0.0)
        power_readings[source['name']] = {"active_power": active_power, "reactive_power": reactive_power}

# Run Button
if st.button("Run Optimization"):
    # Save inputs to temp config
    config = {
        "cost_weight": cost_weight,
        "carbon_weight": carbon_weight,
        "grid_connected": grid_connected,
        "allow_grid_feed": allow_grid_feed,
        "bess_systems": bess_systems,
        "installed_sources": installed_sources
    }
    with open("temp_config.json", "w") as f:
        json.dump(config, f)

    # Simulate CSV creation
    for name, readings in power_readings.items():
        data = {"active_power": [readings["active_power"]], "reactive_power": [readings["reactive_power"]]}
        df = pd.DataFrame(data)
        df.to_csv(f"{name}_data.csv", index=False)

    # Initialize optimizer
    dummy_db_config = {}
    optimizer = PowerOptimizer(dummy_db_config)
    optimizer.cost_weight = cost_weight
    optimizer.carbon_weight = carbon_weight
    optimizer.grid_connected = grid_connected
    optimizer.allow_grid_feed = allow_grid_feed

    # Run
    optimizer.initialize_sources(148)
    optimizer.optimize_power_allocation()

    # Get results
    results_df, total_current_cost, total_savings_hr = optimizer.generate_results()
    recommendations = optimizer.generate_recommendations(results_df)

    # Clean up
    os.remove("temp_config.json")
    for name in power_readings:
        if os.path.exists(f"{name}_data.csv"):
            os.remove(f"{name}_data.csv")

    # Display Dashboard
    st.sidebar.title("Energy Sources")
    for source in optimizer.sources:
        st.sidebar.metric(label=source.name, value=f"{source.current_active_load:.0f} kW", delta=f"${source.cost:.2f}")

    tabs = st.tabs(["Current Load", "Optimized Load"])

    with tabs[0]:
        st.subheader("Current Load Distribution")
        col1, col2, col3 = st.columns(3)
        total_capacity = sum(s.current_active_load for s in optimizer.sources)
        col1.metric("Total Capacity", f"{total_capacity:.0f} kW")
        col2.metric("Hourly Cost", f"${total_current_cost:.0f}")
        col3.metric("Potential Savings", f"${total_savings_hr:.0f}/hour")

        col_pie, col_bar = st.columns(2)
        with col_pie:
            pie_data = pd.DataFrame({"Source": [s.name for s in optimizer.sources], "Load (kW)": [s.current_active_load for s in optimizer.sources]})
            fig_pie = px.pie(pie_data, values="Load (kW)", names="Source", title="Energy Distribution (kW)")
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_bar:
            bar_data = pd.DataFrame({"Source": [s.name for s in optimizer.sources], "Cost per kWh ($)": [s.cost for s in optimizer.sources]})
            fig_bar = px.bar(bar_data, x="Source", y="Cost per kWh ($)", title="Cost Analysis ($/kWh)")
            st.plotly_chart(fig_bar, use_container_width=True)

    with tabs[1]:
        st.subheader("Optimized Load Distribution")
        col1, col2, col3 = st.columns(3)
        total_opt_capacity = sum(s.optimized_active_load for s in optimizer.sources)
        opt_hourly_cost = total_current_cost - total_savings_hr
        col1.metric("Total Capacity", f"{total_opt_capacity:.0f} kW")
        col2.metric("Hourly Cost", f"${opt_hourly_cost:.0f}")
        col3.metric("Potential Savings", f"${total_savings_hr:.0f}/hour")

        col_pie, col_bar = st.columns(2)
        with col_pie:
            pie_data_opt = pd.DataFrame({"Source": [s.name for s in optimizer.sources], "Load (kW)": [s.optimized_active_load for s in optimizer.sources]})
            fig_pie_opt = px.pie(pie_data_opt, values="Load (kW)", names="Source", title="Energy Distribution (kW)")
            st.plotly_chart(fig_pie_opt, use_container_width=True)
        with col_bar:
            st.plotly_chart(fig_bar, use_container_width=True)

    st.header("Cost Management & Analysis")
    st.dataframe(results_df)

    col_table, col_summary = st.columns([3, 1])
    with col_summary:
        st.subheader("Savings Summary")
        st.metric("Hourly Savings", f"${total_savings_hr:.0f}")
        daily = total_savings_hr * 24
        st.metric("Daily Savings", f"${daily:.0f}")
        annual = daily * 365
        st.metric("Annual Savings", f"${annual:.0f}")
        reduction = (total_savings_hr / total_current_cost * 100) if total_current_cost > 0 else 0
        st.metric("Cost Reduction", f"{reduction:.1f}%")

    st.header("Recommendations")
    st.markdown(recommendations)

    st.subheader("Technology Stack")
    st.write("React Energy Dashboard Tech Stack")  # Placeholder for technology stack