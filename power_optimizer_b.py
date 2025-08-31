import pandas as pd
import numpy as np
import json
import os

class BatteryEnergyStorageSystem:
    """Battery Energy Storage System class"""
    def __init__(self, name, capacity_kwh, power_rating_kw, charge_efficiency=0.95, 
                 discharge_efficiency=0.95, min_soc=20, max_soc=95, current_soc=50):
        self.name = name
        self.capacity_kwh = capacity_kwh
        self.power_rating_kw = power_rating_kw
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.current_soc = current_soc
        self.discharge_threshold = 50
        self.available = True
        
        self.mode = 'standby'
        self.grid_synced = True
        
        self.current_charge_power = 0
        self.current_discharge_power = 0
        
        self.optimized_charge_power = 0
        self.optimized_discharge_power = 0
        
        self.grid_feed_power = 0
        self.grid_import_power = 0
        
    def get_available_charge_capacity(self):
        available_soc = self.max_soc - self.current_soc
        return (available_soc / 100) * self.capacity_kwh
    
    def get_available_discharge_capacity(self):
        available_soc = self.current_soc - self.min_soc
        return (available_soc / 100) * self.capacity_kwh
    
    def can_charge(self, power_kw):
        if self.current_soc >= self.max_soc:
            return False
        max_charge_power = min(self.power_rating_kw, 
                              self.get_available_charge_capacity())
        return power_kw <= max_charge_power
    
    def can_discharge(self, power_kw):
        if self.current_soc <= self.min_soc:
            return False
        max_discharge_power = min(self.power_rating_kw, 
                                 self.get_available_discharge_capacity())
        return power_kw <= max_discharge_power
    
    def should_discharge(self):
        return self.current_soc >= self.discharge_threshold

class PowerSource:
    """Power source class with optimization capabilities"""
    def __init__(self, name, cost, carbon_footprint, min_capacity, max_capacity, 
                 power_reading=0, reactive_power_reading=0, source_type='conventional'):
        self.name = name
        self.cost = cost
        self.carbon_footprint = carbon_footprint
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.power_reading = power_reading
        self.reactive_power_reading = reactive_power_reading
        self.source_type = source_type
        self.available = True
        self.reliability_score = 10
        
        self.current_active_load = power_reading
        self.current_reactive_load = reactive_power_reading
        
        self.optimized_active_load = 0
        self.optimized_reactive_load = 0
        
        self.efficiency_score = 0

class PowerOptimizer:
    """Main power optimization system"""

    def __init__(self, db_config):
        self.db_config = db_config
        self.sources = []
        self.bess_systems = []
        self.cost_weight = 0.6
        self.carbon_weight = 0.2
        self.reliability_weight = 0.2
        
        self.grid_connected = True
        self.allow_grid_feed = True
        self.grid_feed_limit = 1000
        
        self.total_load_demand = 0
        self.total_reactive_demand = 0
        
    def load_sources_config(self, site_id):
        config_file = f"temp_config.json"
        if not os.path.exists(config_file):
            print(f"Configuration file {config_file} not found")
            return []
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            self.cost_weight = config.get('cost_weight', 0.6)
            self.carbon_weight = config.get('carbon_weight', 0.2)
            self.reliability_weight = config.get('reliability_weight', 0.2)
            
            return config.get('installed_sources', [])
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return []
        
    def load_bess_config(self, site_id):
        config_file = f"temp_config.json"
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config.get('bess_systems', [])
        except Exception as e:
            print(f"Error loading BESS configuration: {e}")
            return []

    def initialize_sources(self, site_id):
        sources_config = self.load_sources_config(site_id)
        if not sources_config:
            print("No sources configuration found")
            return
        
        self.sources = []
        
        for config in sources_config:
            source_type = 'conventional'
            if config['name'].lower() in ['solar', 'wind']:
                source_type = 'renewable'
            elif config['name'].lower() == 'grid':
                source_type = 'grid'
            elif 'bess' in config['name'].lower():
                source_type = 'storage'
                
            source = PowerSource(
                name=config['name'],
                cost=float(config['cost']),
                carbon_footprint=float(config['carbon_footprint']),
                min_capacity=float(config['min_capacity']),
                max_capacity=float(config['max_capacity']),
                source_type=source_type
            )
            
            csv_file = f"{source.name}_data.csv"
            if os.path.exists(csv_file):
                source_data = pd.read_csv(csv_file)
                source.power_reading = source_data['active_power'].iloc[0]
                source.reactive_power_reading = source_data['reactive_power'].iloc[0]
                source.current_active_load = source_data['active_power'].iloc[0]
                source.current_reactive_load = source_data['reactive_power'].iloc[0]
                
                print('-----------------------------------------')
                print(f"Source: {source.name}")
                print(f"Active Power: {source.power_reading} kW")
                print(f"Reactive Power: {source.reactive_power_reading} kVAR")
                print('-----------------------------------------')
            
            source.available = True
            self.sources.append(source)
        
        bess_config = self.load_bess_config(site_id)
        self.bess_systems = []
        
        for config in bess_config:
            bess = BatteryEnergyStorageSystem(
                name=config['name'],
                capacity_kwh=float(config['capacity_kwh']),
                power_rating_kw=float(config['power_rating_kw']),
                current_soc=float(config.get('current_soc', 50))
            )
            bess.discharge_threshold = float(config.get('discharge_threshold', 50))
            bess.grid_synced = self.grid_connected
            self.bess_systems.append(bess)
            
            print('-----------------------------------------')
            print(f"BESS: {bess.name}")
            print(f"Capacity: {bess.capacity_kwh} kWh")
            print(f"Power Rating: {bess.power_rating_kw} kW")
            print(f"Current SOC: {bess.current_soc}%")
            print('-----------------------------------------')

    def calculate_efficiency_score(self, source):
        return (self.cost_weight * source.cost) + (self.carbon_weight * source.carbon_footprint) + (self.reliability_weight / source.reliability_score)

    def optimize_power_allocation(self):
        available_sources = [s for s in self.sources if s.available]
        
        if not available_sources:
            print("No available power sources")
            return
        
        total_active_load = sum(s.current_active_load for s in available_sources)
        total_reactive_load = sum(s.current_reactive_load for s in available_sources)
        
        print(f"Total active load: {total_active_load:.2f} kW")
        print(f"Total reactive load: {total_reactive_load:.2f} kVAR")
        
        for source in available_sources:
            source.efficiency_score = self.calculate_efficiency_score(source)
        
        available_sources.sort(key=lambda x: x.efficiency_score)
        
        for source in available_sources:
            source.optimized_active_load = 0
            source.optimized_reactive_load = 0
        
        remaining_active_load = total_active_load
        self.allocate_active_power(available_sources, remaining_active_load)
        
        remaining_reactive_load = total_reactive_load
        self.allocate_reactive_power(available_sources, remaining_reactive_load)
        
        self.apply_load_sharing(available_sources)
        
        # Handle BESS optimization
        total_bess_contribution = 0
        for bess in self.bess_systems:
            source = next((s for s in self.sources if s.name == bess.name), None)
            if source:
                if bess.should_discharge():
                    discharge = min(bess.power_rating_kw, bess.get_available_discharge_capacity())
                    bess.optimized_discharge_power = discharge
                    bess.optimized_charge_power = 0
                    bess.mode = 'discharging'
                    source.optimized_active_load += discharge
                    total_bess_contribution += discharge
                else:
                    charge = min(bess.power_rating_kw, bess.get_available_charge_capacity())
                    bess.optimized_charge_power = charge
                    bess.optimized_discharge_power = 0
                    bess.mode = 'charging'
                    source.optimized_active_load -= charge if source.optimized_active_load > charge else 0
        
        if not self.grid_connected:
            self.handle_off_grid_operation(available_sources)

    def handle_off_grid_operation(self, sources):
        print("\nOff-grid operation mode activated")
        
        sources = [s for s in sources if s.source_type != 'grid']
        
        total_generation = sum(s.optimized_active_load for s in sources)
        total_demand = self.total_load_demand
        
        for bess in self.bess_systems:
            if bess.mode == 'discharging':
                total_generation += bess.optimized_discharge_power
        
        if total_generation < total_demand:
            deficit = total_demand - total_generation
            print(f"Generation deficit in off-grid mode: {deficit:.2f} kW")
            
            for source in sources:
                if source.optimized_active_load < source.max_capacity:
                    additional_capacity = min(deficit, 
                                              source.max_capacity - source.optimized_active_load)
                    source.optimized_active_load += additional_capacity
                    deficit -= additional_capacity
                    print(f"Increased {source.name} output by {additional_capacity:.2f} kW")
                    
                    if deficit <= 0:
                        break
            
            if deficit > 0:
                print(f"Load shedding required: {deficit:.2f} kW")

    def allocate_active_power(self, sources, total_load):
        remaining_load = total_load
        print('The sources are:')
        for source in sources:
            print(f" - {source.name} (Max: {source.max_capacity} kW, Min: {source.min_capacity} kW)")
        print(f"Total load: {total_load} kW")
        print(f"Remaining load: {remaining_load} kW")

        solar_sources = [s for s in sources if s.name.lower() == 'solar']
        non_renewable_sources = [s for s in sources if s.name.lower() not in ['solar', 'wind', 'bess']]
        
        if solar_sources and total_load <= sum(s.max_capacity for s in solar_sources):
            print("Applying solar-first optimization strategy")
            
            for source in non_renewable_sources:
                if source.name.lower() != 'grid' and source.available:
                    if remaining_load > source.min_capacity:
                        min_allocation = source.min_capacity
                        source.optimized_active_load = min_allocation
                        remaining_load -= min_allocation
                        print(f"Minimum allocation to {source.name}: {min_allocation:.2f} kW")
                elif source.name.lower() == 'grid' and self.grid_connected:
                    min_allocation = 10
                    source.optimized_active_load = min_allocation
                    remaining_load -= min_allocation
                    print(f"Minimum allocation to {source.name}: {min_allocation:.2f} kW")
            
            for source in solar_sources:
                if remaining_load > 0:
                    allocation = min(remaining_load, source.max_capacity)
                    source.optimized_active_load = allocation
                    remaining_load -= allocation
                    print(f"Solar allocation to {source.name}: {allocation:.2f} kW")
        
        else:
            for source in sources:
                if remaining_load <= 0:
                    break
                
                max_possible = min(source.max_capacity, remaining_load)
                min_required = source.min_capacity
                
                if max_possible >= min_required:
                    allocation = max(min_required, max_possible)
                    source.optimized_active_load = allocation
                    remaining_load -= allocation
                    print(f"Optimized allocation to {source.name}: {allocation:.2f} kW")
                else:
                    source.optimized_active_load = 0

        return remaining_load
    
    def allocate_reactive_power(self, sources, total_reactive_load):
        remaining_reactive_load = total_reactive_load
        
        for source in sources:
            if remaining_reactive_load <= 0:
                break
            
            if source.name != 'Solar':
                max_reactive = source.optimized_active_load * 0.5
                allocation = min(max_reactive, remaining_reactive_load)
                source.optimized_reactive_load = allocation
                remaining_reactive_load -= allocation
            else:
                allocation = 0
                source.optimized_reactive_load = allocation
                remaining_reactive_load -= allocation

            print(f"Reactive power allocation to {source.name}: {allocation:.2f} kVAR")
        
        return remaining_reactive_load
    
    def apply_load_sharing(self, sources):
        source_groups = {}
        for source in sources:
            source_type = source.name.split('_')[0] if '_' in source.name else source.name
            if source_type not in source_groups:
                source_groups[source_type] = []
            source_groups[source_type].append(source)
        
        for group_name, group_sources in source_groups.items():
            if len(group_sources) > 1:
                total_group_load = sum(s.optimized_active_load for s in group_sources)
                if total_group_load > 0:
                    if all(s.max_capacity == group_sources[0].max_capacity for s in group_sources):
                        avg_load = total_group_load / len(group_sources)
                        for s in group_sources:
                            s.optimized_active_load = avg_load
                    else:
                        total_max = sum(s.max_capacity for s in group_sources)
                        percentage = (total_group_load / total_max) * 100
                        for s in group_sources:
                            s.optimized_active_load = (percentage / 100) * s.max_capacity
                   
                    print(f"Load sharing applied to {group_name} sources")
    
    def generate_results(self):
        if not self.sources:
            return pd.DataFrame(), 0, 0
        
        results = []
        total_current_load = 0
        total_optimized_load = 0
        total_current_cost = 0
        total_optimized_cost = 0
        
        # Calculate total BESS contribution
        total_bess_contribution = sum(b.optimized_discharge_power for b in self.bess_systems)
        
        for source in self.sources:
            current_load = source.current_active_load
            optimized_load = source.optimized_active_load
            cost_per_kwh = source.cost
            
            current_cost_hr = current_load * cost_per_kwh
            optimized_cost_hr = optimized_load * cost_per_kwh
            savings_hr = current_cost_hr - optimized_cost_hr
            
            row = {
                'ENERGY SOURCE': source.name,
                'CURRENT LOAD (kW)': round(current_load, 2),
                'OPTIMIZED LOAD (kW)': round(optimized_load, 2),
                'COST PER kWH ($)': round(cost_per_kwh, 2),
                'CURRENT COST/HR': round(current_cost_hr, 2),
                'OPTIMIZED COST/HR': round(optimized_cost_hr, 2),
                'SAVINGS/HR': round(savings_hr, 2),
                'OPTIMIZED CHARGE kW': 0.0,
                'OPTIMIZED DISCHARGE kW': 0.0
            }
            
            if 'BESS' in source.name.upper():
                bess = next((b for b in self.bess_systems if b.name == source.name), None)
                if bess:
                    row['OPTIMIZED CHARGE kW'] = round(bess.optimized_charge_power, 2)
                    row['OPTIMIZED DISCHARGE kW'] = round(bess.optimized_discharge_power, 2)
            
            results.append(row)
            
            total_current_load += current_load
            total_optimized_load += optimized_load
            total_current_cost += current_cost_hr
            total_optimized_cost += optimized_cost_hr
        
        # Add BESS Contribution row
        if total_bess_contribution > 0:
            results.append({
                'ENERGY SOURCE': 'BESS Contribution',
                'CURRENT LOAD (kW)': 0.0,
                'OPTIMIZED LOAD (kW)': round(total_bess_contribution, 2),
                'COST PER kWH ($)': '',
                'CURRENT COST/HR': 0.0,
                'OPTIMIZED COST/HR': 0.0,
                'SAVINGS/HR': 0.0,
                'OPTIMIZED CHARGE kW': round(sum(b.optimized_charge_power for b in self.bess_systems), 2),
                'OPTIMIZED DISCHARGE kW': round(total_bess_contribution, 2)
            })
        
        # Add total row
        total_savings_hr = total_current_cost - total_optimized_cost
        results.append({
            'ENERGY SOURCE': 'TOTAL',
            'CURRENT LOAD (kW)': round(total_current_load, 2),
            'OPTIMIZED LOAD (kW)': round(total_optimized_load + total_bess_contribution, 2),
            'COST PER kWH ($)': '',
            'CURRENT COST/HR': round(total_current_cost, 2),
            'OPTIMIZED COST/HR': round(total_optimized_cost, 2),
            'SAVINGS/HR': round(total_savings_hr, 2),
            'OPTIMIZED CHARGE kW': '',
            'OPTIMIZED DISCHARGE kW': ''
        })
        
        return pd.DataFrame(results), total_current_cost, total_savings_hr
    
    def generate_recommendations(self, results_df):
        recommendations = []
        
        for _, row in results_df.iterrows():
            source = row['ENERGY SOURCE']
            if source in ['TOTAL', 'BESS Contribution']:
                continue
            
            current_kw = row['CURRENT LOAD (kW)']
            optimized_kw = row['OPTIMIZED LOAD (kW)']
            
            rec = f"**{source}**:\n"
            
            if optimized_kw > current_kw:
                rec += f"• Increase load from {current_kw} kW to {optimized_kw} kW\n"
            elif optimized_kw < current_kw:
                rec += f"• Reduce load from {current_kw} kW to {optimized_kw} kW\n"
            else:
                rec += f"• Maintain current load of {current_kw} kW\n"
            
            recommendations.append(rec)
        
        total_row = results_df[results_df['ENERGY SOURCE'] == 'TOTAL'].iloc[0]
        
        summary = f"""
**OPTIMIZATION SUMMARY**:
• Total current load: {total_row['CURRENT LOAD (kW)']} kW
• Total optimized load: {total_row['OPTIMIZED LOAD (kW)']} kW
• Hourly cost savings: ${total_row['SAVINGS/HR']}
• Optimization prioritizes efficiency, cost, and reliability
        """
        
        recommendations.append(summary)
        return '\n\n'.join(recommendations)