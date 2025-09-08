import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

with tab4:
    st.header("Optimization Results & Analytics")

    if st.session_state.results is not None:
        st.subheader("📊 Optimization Results")
        
        display_df = st.session_state.results.copy()
        st.dataframe(display_df, use_container_width=True)
        
        st.subheader("🎯 Key Metrics")
        
        total_row = display_df[display_df['ENERGY SOURCE'] == 'TOTAL'].squeeze()
        
        # CSS for metric cards
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 15px;
        }
        .metric-card h3 {
            margin: 0;
            font-size: 1.2em;
            color: #333;
        }
        .metric-card h2 {
            margin: 10px 0 0;
            font-size: 1.5em;
            color: #007bff;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            
            # Current Cost
            st.markdown(f"""
            <div class="metric-card">
                <h3>Current Cost</h3>
                <h2>PKR {total_row['CURRENT COST/HR']:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Cost Optimized Cost
            st.markdown(f"""
            <div class="metric-card">
                <h3>Cost Optimized Cost</h3>
                <h2>PKR {total_row['COST OPTIMIZED COST/HR']:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Cost Optimized Cost Savings
            savings_c = total_row['CURRENT COST/HR'] - total_row['COST OPTIMIZED COST/HR']
            st.markdown(f"""
            <div class="metric-card">
                <h3>Cost Optimized Cost Savings</h3>
                <h2>PKR {savings_c:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Reliability Optimized Cost
            st.markdown(f"""
            <div class="metric-card">
                <h3>Reliability Optimized Cost</h3>
                <h2>PKR {total_row['RELIABILITY OPTIMIZED COST/HR']:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Reliability Optimized Cost Savings
            savings_r = total_row['CURRENT COST/HR'] - total_row['RELIABILITY OPTIMIZED COST/HR']
            st.markdown(f"""
            <div class="metric-card">
                <h3>Reliability Optimized Cost Savings</h3>
                <h2>PKR {savings_r:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Cost Optimized Savings %
            savings_percent_c = (savings_c / total_row['CURRENT COST/HR']) * 100 if total_row['CURRENT COST/HR'] > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>Cost Optimized Savings %</h3>
                <h2>{savings_percent_c:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Reliability Optimized Savings %
            savings_percent_r = (savings_r / total_row['CURRENT COST/HR']) * 100 if total_row['CURRENT COST/HR'] > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>Reliability Optimized Savings %</h3>
                <h2>{savings_percent_r:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

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