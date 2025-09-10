def generate_results(self):
    """Generate comprehensive results including all sources and BESS"""
    results = []
    total_current_load = 0
    total_cost_optimized_load = 0
    total_current_cost = 0
    total_cost_optimized_cost = 0
    total_current_kvar = 0
    total_cost_optimized_kvar = 0
    total_grid_feed = 0
    total_rel_optimized_kvar = 0
    total_rel_optimized_load = 0
    total_rel_optimized_cost = 0

    # Process all power sources
    for source in self.sources:
        current_load = source.current_active_load
        cost_optimized_load = source.optimized_cost_active_load
        rel_optimized_load = source.optimized_rel_active_load
        cost_per_kwh = source.total_cost

        current_kvar = source.current_reactive_load
        cost_optimized_kvar = getattr(source, 'optimized_cost_reactive_load', 0)
        rel_optimized_kvar = getattr(source, 'optimized_rel_reactive_load', 0)

        current_cost_hr = current_load * cost_per_kwh
        
        # Handle grid feed power correctly
        grid_feed_power = getattr(source, 'grid_feed_power', 0)
        if source.name.lower() == 'grid' and grid_feed_power > 0:
            # For grid source, when feeding to grid, we show net consumption
            cost_effective_optimized = cost_optimized_load - grid_feed_power
            rel_effective_optimized = rel_optimized_load - grid_feed_power
            # Cost calculation: if net is negative (feeding to grid), cost should be zero or negative
            cost_optimized_cost_hr = cost_effective_optimized * cost_per_kwh if cost_effective_optimized > 0 else 0
            rel_optimized_cost_hr = rel_effective_optimized * cost_per_kwh if rel_effective_optimized > 0 else 0
            total_grid_feed += grid_feed_power
        else:
            cost_effective_optimized = cost_optimized_load
            rel_effective_optimized = rel_optimized_load
            cost_optimized_cost_hr = cost_optimized_load * cost_per_kwh
            rel_optimized_cost_hr = rel_optimized_load * cost_per_kwh

        # Calculate efficiency score if not already calculated
        efficiency_score = getattr(source, 'efficiency_score', 0)
        if efficiency_score == 0:
            try:
                efficiency_score = self.calculate_efficiency_score(source)
            except:
                efficiency_score = 0

        row = {
            'ENERGY SOURCE': source.name,
            'CURRENT LOAD (kW)': round(current_load, 2),
            'COST OPTIMIZED LOAD (kW)': round(cost_effective_optimized, 2),
            'RELIABILITY OPTIMIZED LOAD (kW)': round(rel_effective_optimized, 2),
            'CURRENT KVAR (kVAR)': round(current_kvar, 2),
            'COST OPTIMIZED KVAR (kVAR)': round(cost_optimized_kvar, 2),
            'RELIABILITY OPTIMIZED KVAR (kVAR)': round(rel_optimized_kvar, 2),
            'TOTAL COST (PKR/kWh)': round(cost_per_kwh, 2),
            'PRODUCTION COST (PKR/kWh)': round(getattr(source, 'production_cost', 0), 2),
            'CARBON COST (PKR/kWh)': round(getattr(source, 'carbon_emission', 0) * getattr(self, 'carbon_cost_pkr_kg', 0), 2),
            'CURRENT COST/HR': round(current_cost_hr, 2),
            'COST OPTIMIZED COST/HR': round(cost_optimized_cost_hr, 2),
            'RELIABILITY OPTIMIZED COST/HR': round(rel_optimized_cost_hr, 2),
            'COST OPTIMIZED CHARGE (kW)': 0.0,
            'RELIABILITY OPTIMIZED CHARGE (kW)': 0.0,
            'COST OPTIMIZED DISCHARGE (kW)': 0.0,
            'RELIABILITY OPTIMIZED DISCHARGE (kW)': 0.0,
            'GRID FEED (kW)': round(grid_feed_power, 2),
            'RELIABILITY SCORE': round(getattr(source, 'reliability_score', 0), 2),
            'EFFICIENCY SCORE': round(efficiency_score, 2),
            'STATUS': 'Active' if getattr(source, 'available', True) else 'Inactive'
        }
        
        results.append(row)
        
        total_current_load += current_load
        total_cost_optimized_load += cost_effective_optimized
        total_rel_optimized_load += rel_effective_optimized
        total_current_cost += current_cost_hr
        total_cost_optimized_cost += cost_optimized_cost_hr
        total_rel_optimized_cost += rel_optimized_cost_hr
        total_current_kvar += current_kvar
        total_cost_optimized_kvar += cost_optimized_kvar
        total_rel_optimized_kvar += rel_optimized_kvar

    # Process all BESS systems
    for bess in self.bess_systems:
        current_bess_discharge = abs(getattr(bess, 'current_power_input', 0)) if getattr(bess, 'current_power_input', 0) < 0 else 0
        cost_optimized_discharge = getattr(bess, 'optimized_discharge_power', 0)
        rel_optimized_discharge = getattr(bess, 'optimized_discharge_power', 0)

        # BESS reactive power handling
        current_kvar = getattr(bess, 'current_reactive_load', 0)
        cost_optimized_kvar = getattr(bess, 'optimized_cost_reactive_load', 0)
        rel_optimized_kvar = getattr(bess, 'optimized_rel_reactive_load', cost_optimized_kvar)  # Use cost optimized if rel not available

        # BESS cost calculations
        try:
            current_cost_hr = bess.get_current_operating_cost()
        except (AttributeError, TypeError):
            current_cost_hr = current_bess_discharge * getattr(bess, 'total_cost', 0)
        
        try:
            cost_optimized_cost_hr = bess.get_optimized_operating_cost()
        except (AttributeError, TypeError):
            # Calculate based on discharge power and cost
            cost_optimized_cost_hr = cost_optimized_discharge * getattr(bess, 'total_cost', 0)
        
        try:
            rel_optimized_cost_hr = bess.get_optimized_operating_cost()
        except (AttributeError, TypeError):
            rel_optimized_cost_hr = rel_optimized_discharge * getattr(bess, 'total_cost', 0)
        
        # BESS status determination
        mode = getattr(bess, 'mode', 'standby')
        current_soc = getattr(bess, 'current_soc', 0)
        
        if mode == 'charging':
            status = f'Charging (SOC: {current_soc}%)'
        elif mode == 'discharging':
            status = f'Discharging (SOC: {current_soc}%)'
        else:
            status = f'Standby (SOC: {current_soc}%)'
        
        # Calculate BESS efficiency score
        try:
            bess_efficiency_score = getattr(bess, 'efficiency_score', 0)
            if bess_efficiency_score == 0:
                # For BESS, efficiency could be based on round-trip efficiency and cost
                bess_efficiency_score = getattr(bess, 'total_cost', 0) + (11 - getattr(bess, 'reliability_score', 5))
        except:
            bess_efficiency_score = 0
        
        row = {
            'ENERGY SOURCE': bess.name,
            'CURRENT LOAD (kW)': round(current_bess_discharge, 2),
            'COST OPTIMIZED LOAD (kW)': round(cost_optimized_discharge, 2),
            'RELIABILITY OPTIMIZED LOAD (kW)': round(rel_optimized_discharge, 2),
            'CURRENT KVAR (kVAR)': round(current_kvar, 2),
            'COST OPTIMIZED KVAR (kVAR)': round(cost_optimized_kvar, 2),
            'RELIABILITY OPTIMIZED KVAR (kVAR)': round(rel_optimized_kvar, 2),
            'TOTAL COST (PKR/kWh)': round(getattr(bess, 'total_cost', 0), 2),
            'PRODUCTION COST (PKR/kWh)': round(getattr(bess, 'production_cost', 0), 2),
            'CARBON COST (PKR/kWh)': round(getattr(bess, 'carbon_emission', 0) * getattr(self, 'carbon_cost_pkr_kg', 0), 2),
            'CURRENT COST/HR': round(current_cost_hr, 2),
            'COST OPTIMIZED COST/HR': round(cost_optimized_cost_hr, 2),
            'RELIABILITY OPTIMIZED COST/HR': round(rel_optimized_cost_hr, 2),
            'COST OPTIMIZED CHARGE (kW)': round(getattr(bess, 'optimized_charge_power', 0), 2),
            'RELIABILITY OPTIMIZED CHARGE (kW)': round(getattr(bess, 'optimized_charge_power', 0), 2),
            'COST OPTIMIZED DISCHARGE (kW)': round(cost_optimized_discharge, 2),
            'RELIABILITY OPTIMIZED DISCHARGE (kW)': round(rel_optimized_discharge, 2),
            'GRID FEED (kW)': 0.0,
            'RELIABILITY SCORE': round(getattr(bess, 'reliability_score', 0), 2),
            'EFFICIENCY SCORE': round(bess_efficiency_score, 2),
            'STATUS': status
        }
        
        results.append(row)
        
        # Add BESS discharge to totals (positive contribution to load)
        total_current_load += current_bess_discharge
        total_cost_optimized_load += cost_optimized_discharge
        total_rel_optimized_load += rel_optimized_discharge
        total_current_cost += current_cost_hr
        total_cost_optimized_cost += cost_optimized_cost_hr
        total_rel_optimized_cost += rel_optimized_cost_hr
        total_current_kvar += current_kvar
        total_cost_optimized_kvar += cost_optimized_kvar
        total_rel_optimized_kvar += rel_optimized_kvar

    # Calculate total savings
    total_cost_savings_hr = total_current_cost - total_cost_optimized_cost
    total_rel_savings_hr = total_current_cost - total_rel_optimized_cost

    # Add total row
    total_row = {
        'ENERGY SOURCE': 'TOTAL',
        'CURRENT LOAD (kW)': round(total_current_load, 2),
        'COST OPTIMIZED LOAD (kW)': round(total_cost_optimized_load, 2),
        'RELIABILITY OPTIMIZED LOAD (kW)': round(total_rel_optimized_load, 2),
        'CURRENT KVAR (kVAR)': round(total_current_kvar, 2),
        'COST OPTIMIZED KVAR (kVAR)': round(total_cost_optimized_kvar, 2),
        'RELIABILITY OPTIMIZED KVAR (kVAR)': round(total_rel_optimized_kvar, 2),
        'TOTAL COST (PKR/kWh)': '',
        'PRODUCTION COST (PKR/kWh)': '',
        'CARBON COST (PKR/kWh)': '',
        'CURRENT COST/HR': round(total_current_cost, 2),
        'COST OPTIMIZED COST/HR': round(total_cost_optimized_cost, 2),
        'RELIABILITY OPTIMIZED COST/HR': round(total_rel_optimized_cost, 2),
        'COST OPTIMIZED CHARGE (kW)': round(sum(getattr(b, 'optimized_charge_power', 0) for b in self.bess_systems), 2),
        'RELIABILITY OPTIMIZED CHARGE (kW)': round(sum(getattr(b, 'optimized_charge_power', 0) for b in self.bess_systems), 2),
        'COST OPTIMIZED DISCHARGE (kW)': round(sum(getattr(b, 'optimized_discharge_power', 0) for b in self.bess_systems), 2),
        'RELIABILITY OPTIMIZED DISCHARGE (kW)': round(sum(getattr(b, 'optimized_discharge_power', 0) for b in self.bess_systems), 2),
        'GRID FEED (kW)': round(total_grid_feed, 2),
        'RELIABILITY SCORE': '',
        'EFFICIENCY SCORE': '',
        'STATUS': f'Cost Savings: PKR {total_cost_savings_hr:.2f}/hr, Rel Savings: PKR {total_rel_savings_hr:.2f}/hr'
    }
    
    results.append(total_row)

    # Create DataFrame and add priority ranking
    df_r = pd.DataFrame(results)
    
    # Handle efficiency score ranking safely
    try:
        # Convert efficiency score to numeric, handling empty strings and non-numeric values
        df_r['EFFICIENCY SCORE'] = pd.to_numeric(df_r['EFFICIENCY SCORE'], errors='coerce')
        # Create priority ranking (lower efficiency score = higher priority)
        df_r['priority'] = df_r['EFFICIENCY SCORE'].rank(method='min', na_option='bottom').fillna(999).astype(int)
    except Exception as e:
        print(f"Warning: Could not calculate priority ranking: {e}")
        df_r['priority'] = 0

    return df_r, total_current_cost, total_cost_savings_hr

def generate_recommendations(self, results_df):
    """Generate optimization recommendations"""
    recommendations = []
    
    # Source-specific recommendations
    for _, row in results_df.iterrows():
        source = row['ENERGY SOURCE']
        if source == 'TOTAL':
            continue
        
        current_kw = row['CURRENT LOAD (kW)']
        cost_optimized_kw = row['COST OPTIMIZED LOAD (kW)']
        rel_optimized_kw = row['RELIABILITY OPTIMIZED LOAD (kW)']
        
        # Choose optimization strategy based on which has better results
        cost_savings = (row['CURRENT COST/HR'] - row['COST OPTIMIZED COST/HR']) if row['CURRENT COST/HR'] != '' else 0
        rel_savings = (row['CURRENT COST/HR'] - row['RELIABILITY OPTIMIZED COST/HR']) if row['CURRENT COST/HR'] != '' else 0
        
        # Use cost optimization by default, but switch to reliability if it's significantly better
        if rel_savings > cost_savings * 1.1:  # 10% threshold for switching
            optimized_kw = rel_optimized_kw
            optimization_type = "reliability-focused"
        else:
            optimized_kw = cost_optimized_kw
            optimization_type = "cost-focused"

        rec = f"**{source}** ({optimization_type}):\n"
        
        if 'BESS' in source.upper():
            charge_kw = row.get('COST OPTIMIZED CHARGE (kW)', 0)
            discharge_kw = row.get('COST OPTIMIZED DISCHARGE (kW)', 0)
            
            if charge_kw > 0:
                rec += f"• Charge at {charge_kw:.1f} kW to store energy for later use\n"
            elif discharge_kw > 0:
                rec += f"• Discharge at {discharge_kw:.1f} kW to provide {discharge_kw:.1f} kW to system load\n"
            else:
                rec += f"• Maintain standby mode - no charging or discharging needed\n"
            
            rec += f"• Current status: {row['STATUS']}\n"
            rec += f"• Reliability score: {row['RELIABILITY SCORE']}/10\n"
            
            # Add efficiency information
            if row['EFFICIENCY SCORE'] != '' and row['EFFICIENCY SCORE'] > 0:
                rec += f"• Efficiency ranking: #{row.get('priority', 'N/A')}\n"
                
        else:
            load_change = optimized_kw - current_kw
            if abs(load_change) < 0.1:  # Minimal change threshold
                rec += f"• Maintain current load of {current_kw:.1f} kW (optimal)\n"
            elif load_change > 0:
                rec += f"• Increase load from {current_kw:.1f} kW to {optimized_kw:.1f} kW (+{load_change:.1f} kW)\n"
            else:
                rec += f"• Reduce load from {current_kw:.1f} kW to {optimized_kw:.1f} kW ({load_change:.1f} kW)\n"
            
            rec += f"• Reliability score: {row['RELIABILITY SCORE']}/10\n"
            
            # Add cost information
            current_cost = row['CURRENT COST/HR']
            optimized_cost = row['COST OPTIMIZED COST/HR'] if optimization_type == "cost-focused" else row['RELIABILITY OPTIMIZED COST/HR']
            if current_cost != '' and optimized_cost != '':
                savings = current_cost - optimized_cost
                if abs(savings) >= 0.01:
                    rec += f"• Cost impact: PKR {savings:+.2f}/hr\n"
            
            # Add grid feed information
            grid_feed = row.get('GRID FEED (kW)', 0)
            if grid_feed > 0:
                rec += f"• Grid feed: {grid_feed:.1f} kW excess power fed back to grid\n"
            
            # Add efficiency ranking
            if row['EFFICIENCY SCORE'] != '' and row['EFFICIENCY SCORE'] > 0:
                rec += f"• Efficiency ranking: #{row.get('priority', 'N/A')} (lower is better)\n"
        
        recommendations.append(rec)
    
    # Overall system summary
    try:
        total_row = results_df[results_df['ENERGY SOURCE'] == 'TOTAL'].iloc[0]
        
        summary = f"""
**OPTIMIZATION SUMMARY**:
• Total current load: {total_row['CURRENT LOAD (kW)']} kW
• Cost optimized load: {total_row['COST OPTIMIZED LOAD (kW)']} kW
• Reliability optimized load: {total_row['RELIABILITY OPTIMIZED LOAD (kW)']} kW
• Current hourly cost: PKR {total_row['CURRENT COST/HR']:,.2f}
• Cost optimized hourly cost: PKR {total_row['COST OPTIMIZED COST/HR']:,.2f}
• Reliability optimized hourly cost: PKR {total_row['RELIABILITY OPTIMIZED COST/HR']:,.2f}
• Cost optimization savings: PKR {total_row['CURRENT COST/HR'] - total_row['COST OPTIMIZED COST/HR']:,.2f}/hr
• Reliability optimization savings: PKR {total_row['CURRENT COST/HR'] - total_row['RELIABILITY OPTIMIZED COST/HR']:,.2f}/hr
• Total BESS charge power: {total_row['COST OPTIMIZED CHARGE (kW)']} kW
• Total BESS discharge power: {total_row['COST OPTIMIZED DISCHARGE (kW)']} kW
• Grid feed: {total_row['GRID FEED (kW)']} kW
• System prioritizes balanced cost optimization with reliability considerations
        """
        
        recommendations.append(summary)
        
    except Exception as e:
        print(f"Warning: Could not generate summary: {e}")
        recommendations.append("**OPTIMIZATION SUMMARY**: Summary generation encountered an error")
    
    return '\n\n'.join(recommendations)