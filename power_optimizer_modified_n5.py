import pandas as pd
import numpy as np
import json
import os
import mysql.connector


class BatteryEnergyStorageSystem:
    """Battery Energy Storage System class for managing BESS operations and optimization."""
    def __init__(self, name, capacity_kwh, power_rating_kw, charge_efficiency=0.95, 
                 discharge_efficiency=0.95, min_soc=20, max_soc=95, current_soc=50,
                 production_cost=2.0, carbon_emission=0.1, total_cost=None, reliability_score=9.0):
        self.name = name
        self.capacity_kwh = capacity_kwh
        self.power_rating_kw = power_rating_kw
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.current_soc = current_soc
        self.discharge_threshold = 50
        self.charge_threshold = 85
        self.available = True
        
        # Cost parameters
        self.production_cost = production_cost
        self.carbon_emission = carbon_emission
        self.total_cost = total_cost if total_cost else production_cost
        
        self.mode = 'standby'
        self.grid_synced = True
        
        # Current operation
        self.current_power_input = 0  # Positive for charging, negative for discharging
        self.current_charge_power = 0
        self.current_discharge_power = 0
        
        # Optimized operation
        self.optimized_charge_power = 0
        self.optimized_discharge_power = 0
        
        # Grid interaction
        self.grid_feed_power = 0
        self.grid_import_power = 0

        self.reliability_score = reliability_score
        
    def get_available_charge_capacity(self):
        """Get available charging capacity in kWh"""
        available_soc = self.max_soc - self.current_soc
        return (available_soc / 100) * self.capacity_kwh
    
    def get_available_discharge_capacity(self):
        """Get available discharging capacity in kWh"""
        available_soc = self.current_soc - self.min_soc
        return (available_soc / 100) * self.capacity_kwh
    
    def can_charge(self, power_kw):
        """Check if BESS can charge at given power"""
        if self.current_soc >= self.max_soc:
            return False
        max_charge_power = min(self.power_rating_kw, 
                              self.get_available_charge_capacity())
        return power_kw <= max_charge_power
    
    def can_discharge(self, power_kw):
        """Check if BESS can discharge at given power"""
        if self.current_soc <= self.min_soc:
            return False
        max_discharge_power = min(self.power_rating_kw, 
                                 self.get_available_discharge_capacity())
        return power_kw <= max_discharge_power
    
    def should_discharge(self):
        """Determine if BESS should discharge based on SOC"""
        return self.current_soc >= self.discharge_threshold
    
    def should_charge(self):
        """Determine if BESS should charge based on SOC"""
        return self.current_soc <= self.charge_threshold
    
    def get_current_operating_cost(self):
        """Calculate current operating cost"""
        if self.current_power_input > 0:  # Charging
            return abs(self.current_power_input) * self.total_cost
        elif self.current_power_input < 0:  # Discharging (providing power)
            return abs(self.current_power_input) * self.total_cost
        return 0
    
    def get_optimized_operating_cost(self):
        """Calculate optimized operating cost"""
        if self.optimized_charge_power > 0:
            return self.optimized_charge_power * self.total_cost
        elif self.optimized_discharge_power > 0:
            return self.optimized_discharge_power * self.total_cost
        return 0

class PowerSource:
    """Power source class with optimization capabilities"""
    def __init__(self, name, production_cost, carbon_emission, total_cost, min_capacity, max_capacity, 
                 power_reading=0, reactive_power_reading=0, source_type='conventional', reliability_score=10.0):
        self.name = name
        self.production_cost = production_cost
        self.carbon_emission = carbon_emission
        self.total_cost = total_cost
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.power_reading = power_reading
        self.reactive_power_reading = reactive_power_reading
        self.source_type = source_type
        self.available = True
        self.reliability_score = reliability_score
        
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
        self.carbon_cost_pkr_kg = 50.0  # Standard carbon cost for Pakistan
        
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

    def connect_database(self):
        """Establish database connection"""
        try:
            return mysql.connector.connect(**self.db_config)
        except mysql.connector.Error as err:
            print(f"Database connection error: {err}")
            return None
    
    def extract_data(self, query):
        """Extract data from database"""
        connection = self.connect_database()
        if not connection:
            return pd.DataFrame()
        
        try:
            df = pd.read_sql(query, connection)
            return df
        except Exception as e:
            print(f"Data extraction error: {e}")
            return pd.DataFrame()
        finally:
            connection.close()

    def calculate_grid_reliability(self, device_id):
        """Calculate grid reliability score based on voltage and frequency stability"""
        query = f"""
        SELECT a_u, b_u, c_u, grid_frequency 
        FROM smart_meter 
        WHERE device_id = {device_id} 
        AND datetime >= NOW() - INTERVAL 1 MONTH
        """
        
        df = self.extract_data(query)
        if df.empty:
            return 5.0
        
        nom_voltage = 400
        voltage_min = nom_voltage * 0.9
        voltage_max = nom_voltage * 1.1
        freq_min = 49.5
        freq_max = 50.5
        
        voltages = df[['a_u', 'b_u', 'c_u']].values.flatten()
        voltage_std = np.std(voltages)
        voltage_in_range = np.mean((voltages >= voltage_min) & (voltages <= voltage_max)) * 10
        
        freq_std = df['grid_frequency'].std()
        freq_in_range = np.mean((df['grid_frequency'] >= freq_min) & (df['grid_frequency'] <= freq_max)) * 10
        
        voltage_score = max(0, 10 - (voltage_std / 10)) * 0.6 + voltage_in_range * 0.4
        freq_score = max(0, 10 - (freq_std * 10)) * 0.6 + freq_in_range * 0.4
        
        return (voltage_score + freq_score) / 2

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
                
            # Calculate total cost including carbon
            total_cost = config['production_cost'] + (config['carbon_emission'] * self.carbon_cost_pkr_kg)
                
            reliability_score = float(config.get('reliability_score', 10.0))
            
            source = PowerSource(
                name=config['name'],
                production_cost=float(config['production_cost']),
                carbon_emission=float(config['carbon_emission']),
                total_cost=total_cost,
                min_capacity=float(config['min_capacity']),
                max_capacity=float(config['max_capacity']),
                source_type=source_type,
                reliability_score=reliability_score
            )
            
            if source.name.lower() == 'grid':
                device_id = config.get('device_id', 0)
                source.reliability_score = self.calculate_grid_reliability(device_id)
                print('From the database record The reliability score for the grid source is:', source.reliability_score)

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
                print(f"Production Cost: {source.production_cost} PKR/kWh")
                print(f"Carbon Emission: {source.carbon_emission} kg CO2/kWh")
                print(f"Total Cost: {source.total_cost:.2f} PKR/kWh")
                print(f"Reliability Score: {source.reliability_score:.2f}")
                print('-----------------------------------------')
            
            source.available = True
            self.sources.append(source)
        
        # Initialize BESS systems
        bess_config = self.load_bess_config(site_id)
        self.bess_systems = []
        
        for config in bess_config:
            # Calculate total cost for BESS
            total_cost = config['production_cost'] + (config['carbon_emission'] * self.carbon_cost_pkr_kg)
            
            reliability_score = float(config.get('reliability_score', 9.0))
            
            bess = BatteryEnergyStorageSystem(
                name=config['name'],
                capacity_kwh=float(config['capacity_kwh']),
                power_rating_kw=float(config['power_rating_kw']),
                current_soc=float(config.get('current_soc', 50)),
                production_cost=float(config['production_cost']),
                carbon_emission=float(config['carbon_emission']),
                total_cost=total_cost,
                reliability_score=reliability_score
            )
            bess.discharge_threshold = float(config.get('discharge_threshold', 50))
            bess.charge_threshold = float(config.get('charge_threshold', 85))
            bess.current_power_input = float(config.get('power_input', 0))
            bess.grid_synced = self.grid_connected
            
            # Set current operation based on power input
            if bess.current_power_input > 0:
                bess.current_charge_power = bess.current_power_input
                bess.current_discharge_power = 0
                bess.mode = 'charging'
            elif bess.current_power_input < 0:
                bess.current_charge_power = 0
                bess.current_discharge_power = abs(bess.current_power_input)
                bess.mode = 'discharging'
            else:
                bess.current_charge_power = 0
                bess.current_discharge_power = 0
                bess.mode = 'standby'
            
            self.bess_systems.append(bess)
            
            print('-----------------------------------------')
            print(f"BESS: {bess.name}")
            print(f"Capacity: {bess.capacity_kwh} kWh")
            print(f"Power Rating: {bess.power_rating_kw} kW")
            print(f"Current SOC: {bess.current_soc}%")
            print(f"Current Mode: {bess.mode}")
            print(f"Power Input: {bess.current_power_input} kW")
            print(f"Production Cost: {bess.production_cost} PKR/kWh")
            print(f"Total Cost: {bess.total_cost:.2f} PKR/kWh")
            print(f"Reliability Score: {bess.reliability_score:.2f}")
            print('-----------------------------------------')

    def calculate_efficiency_score(self, source):
        """Calculate efficiency score as sum of total cost and inverted reliability"""
        inverted_reliability = 11 - source.reliability_score
        return source.total_cost + inverted_reliability

    def optimize_power_allocation(self):
        """Main optimization algorithm"""
        available_sources = [s for s in self.sources if s.available]
        
        if not available_sources:
            print("No available power sources")
            return
        
        total_active_load = sum(s.current_active_load for s in available_sources)
        total_reactive_load = sum(s.current_reactive_load for s in available_sources)
        
        # Add BESS current contribution to total load
        total_bess_discharge = sum(abs(b.current_power_input) for b in self.bess_systems if b.current_power_input < 0)
        total_active_load += total_bess_discharge
        
        print(f"Total active load (including BESS): {total_active_load:.2f} kW")
        print(f"Total reactive load: {total_reactive_load:.2f} kVAR")
        
        # Calculate efficiency scores for optimization
        for source in available_sources:
            source.efficiency_score = self.calculate_efficiency_score(source)
        
        # Sort sources by efficiency (lowest cost first)
        available_sources.sort(key=lambda x: x.efficiency_score)
        
        # Reset optimized loads
        for source in available_sources:
            source.optimized_active_load = 0
            source.optimized_reactive_load = 0
        
        # Optimize BESS operation first
        self.optimize_bess_operation()
        
        # Calculate remaining load after BESS optimization
        total_bess_optimized_discharge = sum(b.optimized_discharge_power for b in self.bess_systems)
        remaining_active_load = total_active_load - total_bess_optimized_discharge
        
        # Allocate remaining load to sources
        self.allocate_active_power(available_sources, remaining_active_load)

        
        # self.apply_load_sharing(kvar_sources)
        self.apply_load_sharing(available_sources)
        
        remaining_reactive_load = total_reactive_load
        kvar_sources = [s for s in self.sources]
        # in kvar_sources add BESS
        kvar_sources.extend(self.bess_systems)
        print('The KVAR sources are:')
        for kvar in kvar_sources:
            print(f" - {kvar.name}")

        self.allocate_reactive_power(kvar_sources, remaining_reactive_load)

        # self.allocate_reactive_power(available_sources, remaining_reactive_load)

        
        # self.apply_load_sharing(kvar_sources)
        # self.apply_load_sharing(available_sources)

        if not self.grid_connected:
            self.handle_off_grid_operation(available_sources)
            # self.handle_off_grid_operation(kvar_sources)

    def optimize_bess_operation(self):
        """Optimize BESS charging/discharging strategy"""
        for bess in self.bess_systems:
            print(f"\nOptimizing BESS: {bess.name}")
            print(f"Current SOC: {bess.current_soc}%")
            print(f"Discharge Threshold: {bess.discharge_threshold}%")
            print(f"Charge Threshold: {bess.charge_threshold}%")
            
            # Reset optimized values
            bess.optimized_charge_power = 0
            bess.optimized_discharge_power = 0
            
            # Determine optimal operation mode
            if bess.current_soc >= bess.discharge_threshold:
                # Should discharge to provide power
                max_discharge = min(
                    bess.power_rating_kw,
                    bess.get_available_discharge_capacity()
                )
                bess.optimized_discharge_power = max_discharge
                bess.mode = 'discharging'
                print(f"BESS {bess.name} optimized for discharging: {max_discharge:.2f} kW")
                
            elif bess.current_soc <= bess.charge_threshold:
                # Should charge when excess power available
                max_charge = min(
                    bess.power_rating_kw,
                    bess.get_available_charge_capacity()
                )
                bess.optimized_charge_power = max_charge
                bess.mode = 'charging'
                print(f"BESS {bess.name} optimized for charging: {max_charge:.2f} kW")
                
            else:
                # Maintain current operation or standby
                bess.mode = 'standby'
                print(f"BESS {bess.name} in standby mode")

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
        bess_discharging = any(b.optimized_discharge_power > 0 for b in self.bess_systems)
        # if solar_sources and total_load <= sum(s.max_capacity for s in solar_sources):
        # also add condition if self.optimized_discharge_power ==0
        # if solar_sources and total_load <= sum(s.max_capacity for s in solar_sources) and self.optimized_discharge_power == 0:
        if solar_sources and total_load <= sum(s.max_capacity for s in solar_sources) and not bess_discharging:
            print("Applying solar-first optimization strategy")
            
            for source in non_renewable_sources:
                if source.name.lower() != 'grid' and source.available:
                    if remaining_load > source.min_capacity:
                        min_allocation = source.min_capacity
                        source.optimized_active_load = min_allocation
                        remaining_load -= min_allocation
                        print(f"Minimum allocation to {source.name}: {min_allocation:.2f} kW")
                        break
                # elif source.name.lower() == 'grid' and self.grid_connected or source.name.lower() == 'bess' :
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
        """Allocate reactive power among sources"""
        remaining_reactive_load = total_reactive_load
        
        for source in sources:
            print(f'The source name is {source.name}')
            if remaining_reactive_load <= 0:
                break

            if source.name.lower() == 'bess':
                print('The name of source is BESS')
                max_reactive = source.optimized_discharge_power * 0.8  # Higher power factor for BESS
                # max_reactive = source.power_rating_kw * 0.8
                allocation = min(max_reactive, remaining_reactive_load)
                source.optimized_reactive_load = allocation
                remaining_reactive_load -= allocation
            
            # Reactive power capacity based on active power allocation
            elif source.name.lower() != 'solar':
                max_reactive = source.optimized_active_load * 0.6  # Power factor consideration
                allocation = min(max_reactive, remaining_reactive_load)
                source.optimized_reactive_load = allocation
                remaining_reactive_load -= allocation

            

            else:
                # Solar typically doesn't provide significant reactive power
                allocation = 0
                source.optimized_reactive_load = allocation

            print(f"Reactive power allocation to {source.name}: {allocation:.2f} kVAR")
        
        return remaining_reactive_load
    
    def apply_load_sharing(self, sources):
        """Apply load sharing among similar sources"""
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
                    # Equal load sharing for similar capacity units
                    if all(abs(s.max_capacity - group_sources[0].max_capacity) < 50 for s in group_sources):
                        # Try load sharing with all machines first
                        active_sources = group_sources.copy()
                        
                        while len(active_sources) > 1:
                            avg_load = total_group_load / len(active_sources)
                            min_capacity_required = min(s.min_capacity for s in active_sources)
                            
                            if avg_load >= min_capacity_required:
                                # All machines can handle the average load
                                for s in group_sources:
                                    if s in active_sources:
                                        s.optimized_active_load = avg_load
                                    else:
                                        s.optimized_active_load = 0  # Shut off
                                break
                            else:
                                # Average load is below minimum capacity, shut off one machine
                                # Find machine with highest minimum capacity to shut off first
                                machine_to_shutdown = max(active_sources, key=lambda x: x.min_capacity)
                                active_sources.remove(machine_to_shutdown)
                                machine_to_shutdown.optimized_active_load = 0
                        
                        # If only one machine left, it takes all the load
                        if len(active_sources) == 1:
                            remaining_machine = active_sources[0]
                            if total_group_load >= remaining_machine.min_capacity:
                                remaining_machine.optimized_active_load = total_group_load
                            else:
                                # Even single machine can't handle the load, set to minimum
                                remaining_machine.optimized_active_load = remaining_machine.min_capacity
                            
                            # Shut off all other machines
                            for s in group_sources:
                                if s != remaining_machine:
                                    s.optimized_active_load = 0
                                    
                    else:
                        # Proportional load sharing based on capacity
                        total_max = sum(s.max_capacity for s in group_sources)
                        
                        # First try proportional distribution
                        temp_loads = {}
                        for s in group_sources:
                            proportion = s.max_capacity / total_max
                            temp_loads[s] = proportion * total_group_load
                        
                        # Check if any machine gets load below its minimum capacity
                        machines_to_shutdown = []
                        for s, load in temp_loads.items():
                            if load < s.min_capacity:
                                machines_to_shutdown.append(s)
                        
                        if machines_to_shutdown:
                            # Shut down machines that can't meet minimum and redistribute
                            active_sources = [s for s in group_sources if s not in machines_to_shutdown]
                            
                            if active_sources:
                                # Recalculate proportional sharing among remaining machines
                                remaining_total_max = sum(s.max_capacity for s in active_sources)
                                for s in group_sources:
                                    if s in active_sources:
                                        proportion = s.max_capacity / remaining_total_max
                                        new_load = proportion * total_group_load
                                        # Ensure the new load meets minimum capacity
                                        s.optimized_active_load = max(new_load, s.min_capacity)
                                    else:
                                        s.optimized_active_load = 0
                            else:
                                # All machines would be below minimum, keep one with lowest min_capacity
                                best_machine = min(group_sources, key=lambda x: x.min_capacity)
                                best_machine.optimized_active_load = max(total_group_load, best_machine.min_capacity)
                                for s in group_sources:
                                    if s != best_machine:
                                        s.optimized_active_load = 0
                        else:
                            # All machines can handle their proportional load
                            for s in group_sources:
                                s.optimized_active_load = temp_loads[s]
                                
                print(f"Load sharing applied to {group_name} sources")
    
    # def apply_load_sharing(self, sources):
    #     """Apply load sharing among similar sources"""
    #     source_groups = {}
    #     for source in sources:
    #         source_type = source.name.split('_')[0] if '_' in source.name else source.name
    #         if source_type not in source_groups:
    #             source_groups[source_type] = []
    #         source_groups[source_type].append(source)
        
    #     for group_name, group_sources in source_groups.items():
    #         if len(group_sources) > 1:
    #             total_group_load = sum(s.optimized_active_load for s in group_sources)
    #             if total_group_load > 0:
    #                 # Equal load sharing for similar capacity units
    #                 if all(abs(s.max_capacity - group_sources[0].max_capacity) < 50 for s in group_sources):
    #                     avg_load = total_group_load / len(group_sources)
    #                     # min_c = min(s.min_capacity for s in group_sources)
    #                     for s in group_sources:
    #                         min_c = s.min_capacity
    #                         if avg_load < min_c:
    #                             # s.optimized_active_load = min_c
    #                             break
    #                         else:
    #                             s.optimized_active_load = avg_load
    #             #     # make new source_group and remove 1 source form source_groups
    #             # new_source_group = group_sources[0]
    #             # source_groups[group_name] = group_sources[1:]
    #             # source_groups[new_source_group.name] = [new_source_group]
    #             # new_total_group_load = sum(s.optimized_active_load for s in source_groups[new_source_group.name])
    #             # print(f"New total group load for {new_source_group.name}: {new_total_group_load}")
    #             # if new_total_group_load > 0:


    #                 else:
    #                     # Proportional load sharing based on capacity
    #                     total_max = sum(s.max_capacity for s in group_sources)
    #                     for s in group_sources:
    #                         proportion = s.max_capacity / total_max
    #                         s.optimized_active_load = proportion * total_group_load
                   
    #                 print(f"Load sharing applied to {group_name} sources")
    
    def generate_results(self):
        """Generate comprehensive results including all sources and BESS"""
        results = []
        total_current_load = 0
        total_optimized_load = 0
        total_current_cost = 0
        total_optimized_cost = 0
        total_current_kvar = 0
        total_optimized_kvar = 0

        # Process all power sources
        for source in self.sources:
            current_load = source.current_active_load
            optimized_load = source.optimized_active_load
            cost_per_kwh = source.total_cost

            current_kvar = source.current_reactive_load
            optimized_kvar = source.optimized_reactive_load

            current_cost_hr = current_load * cost_per_kwh
            optimized_cost_hr = optimized_load * cost_per_kwh
            # assign priority number as per the available sources order
            # source.priority_number = self.sources.index(source) + 1

            row = {
                'ENERGY SOURCE': source.name,
                'CURRENT LOAD (kW)': round(current_load, 2),
                'OPTIMIZED LOAD (kW)': round(optimized_load, 2),
                'CURRENT KVAR (kVAR)': round(current_kvar, 2),
                'OPTIMIZED KVAR (kVAR)': round(optimized_kvar, 2),
                'TOTAL COST (PKR/kWh)': round(cost_per_kwh, 2),
                'PRODUCTION COST (PKR/kWh)': round(source.production_cost, 2),
                'CARBON COST (PKR/kWh)': round(source.carbon_emission * self.carbon_cost_pkr_kg, 2),
                'CURRENT COST/HR': round(current_cost_hr, 2),
                'OPTIMIZED COST/HR': round(optimized_cost_hr, 2),
                'OPTIMIZED CHARGE (kW)': 0.0,
                'OPTIMIZED DISCHARGE (kW)': 0.0,
                'RELIABILITY SCORE': round(source.reliability_score, 2),
                'EFFICIENCY SCORE': round(source.efficiency_score, 2),
                # 'PRIORITY NUMBER': round(source.priority_number, 2),
                'STATUS': 'Active' if source.available else 'Inactive'
            }
            
            results.append(row)
            
            total_current_load += current_load
            total_optimized_load += optimized_load
            total_current_cost += current_cost_hr
            total_optimized_cost += optimized_cost_hr
            total_current_kvar += current_kvar
            total_optimized_kvar += optimized_kvar
        
        # Process all BESS systems
        for bess in self.bess_systems:
            # Current BESS operation
            current_bess_load = abs(bess.current_power_input) if bess.current_power_input < 0 else 0  # Only count discharge as load
            current_bess_charge = bess.current_power_input if bess.current_power_input > 0 else 0
            current_bess_discharge = abs(bess.current_power_input) if bess.current_power_input < 0 else 0

            # bess.current_reactive_load = bess.get_current_reactive_load()
            try:
                current_kvar = bess.current_reactive_load
                
            except AttributeError:
                current_kvar = 0

            try:
                optimized_kvar = bess.optimized_reactive_load
            except AttributeError:
                optimized_kvar = 0

            # Optimized BESS operation
            optimized_bess_load = bess.optimized_discharge_power

            # bess.efficiency_score = bess.get_efficiency_score()

            # Cost calculations
            current_cost_hr = bess.get_current_operating_cost()
            optimized_cost_hr = bess.get_optimized_operating_cost()
            
            # Determine status
            if bess.mode == 'charging':
                status = f'Charging (SOC: {bess.current_soc}%)'
            elif bess.mode == 'discharging':
                status = f'Discharging (SOC: {bess.current_soc}%)'
            else:
                status = f'Standby (SOC: {bess.current_soc}%)'
            
            row = {
                'ENERGY SOURCE': bess.name,
                'CURRENT LOAD (kW)': round(current_bess_discharge, 2),
                'OPTIMIZED LOAD (kW)': round(optimized_bess_load, 2),
                'CURRENT KVAR (kVAR)': round(current_kvar, 2),
                'OPTIMIZED KVAR (kVAR)': round(optimized_kvar, 2),
                'TOTAL COST (PKR/kWh)': round(bess.total_cost, 2),
                'PRODUCTION COST (PKR/kWh)': round(bess.production_cost, 2),
                'CARBON COST (PKR/kWh)': round(bess.carbon_emission * self.carbon_cost_pkr_kg, 2),
                'CURRENT COST/HR': round(current_cost_hr, 2),
                'OPTIMIZED COST/HR': round(optimized_cost_hr, 2),
                'OPTIMIZED CHARGE (kW)': round(bess.optimized_charge_power, 2),
                'OPTIMIZED DISCHARGE (kW)': round(bess.optimized_discharge_power, 2),
                'RELIABILITY SCORE': round(bess.reliability_score, 2),
                # 'EFFICIENCY SCORE': round(bess.efficiency_score, 2),
                # 'PRIORITY NUMBER': round(bess.priority_number, 2),
                'STATUS': status
            }
            
            results.append(row)
            
            # Add to totals (discharge contributes to load, charge is consumption)
            total_current_load += current_bess_discharge
            total_optimized_load += optimized_bess_load
            total_current_cost += current_cost_hr
            total_optimized_cost += optimized_cost_hr
            total_current_kvar += current_kvar
            total_optimized_kvar += optimized_kvar

        # Add total row
        total_savings_hr = total_current_cost - total_optimized_cost
        results.append({
            'ENERGY SOURCE': 'TOTAL',
            'CURRENT LOAD (kW)': round(total_current_load, 2),
            'OPTIMIZED LOAD (kW)': round(total_optimized_load, 2),
            'CURRENT KVAR (kVAR)': round(total_current_kvar, 2),
            'OPTIMIZED KVAR (kVAR)': round(total_optimized_kvar, 2),
            'TOTAL COST (PKR/kWh)': '',
            'PRODUCTION COST (PKR/kWh)': '',
            'CARBON COST (PKR/kWh)': '',
            'CURRENT COST/HR': round(total_current_cost, 2),
            'OPTIMIZED COST/HR': round(total_optimized_cost, 2),
            'OPTIMIZED CHARGE (kW)': round(sum(b.optimized_charge_power for b in self.bess_systems), 2),
            'OPTIMIZED DISCHARGE (kW)': round(sum(b.optimized_discharge_power for b in self.bess_systems), 2),
            'RELIABILITY SCORE': '',
            'STATUS': f'Savings: PKR {total_savings_hr:.2f}/hr'
        })

        df_r = pd.DataFrame(results)
        # assign priority score to each source based on efficiency score lowest efficecny gets 1st next gets 2nd and so on
        # df_r['priority'] = df_r['EFFICIENCY SCORE'].rank(method='min').astype(int)

        df_r['priority'] = (
                df_r['EFFICIENCY SCORE']
                .rank(method='min')
                .fillna(0)    # replace NaN ranks with 0 (or any default)
                .astype(int)
            )

        # return df_r, total_current_cost, total_savings_hr
        return df_r, total_current_cost, total_savings_hr

    def generate_recommendations(self, results_df):
        """Generate optimization recommendations"""
        recommendations = []
        
        # Source-specific recommendations
        for _, row in results_df.iterrows():
            source = row['ENERGY SOURCE']
            if source == 'TOTAL':
                continue
            
            current_kw = row['CURRENT LOAD (kW)']
            optimized_kw = row['OPTIMIZED LOAD (kW)']
            
            rec = f"**{source}**:\n"
            
            if 'BESS' in source:
                charge_kw = row.get('OPTIMIZED CHARGE (kW)', 0)
                discharge_kw = row.get('OPTIMIZED DISCHARGE (kW)', 0)
                
                if charge_kw > 0:
                    rec += f"• Charge at {charge_kw:.1f} kW\n"
                elif discharge_kw > 0:
                    rec += f"• Discharge at {discharge_kw:.1f} kW to provide {discharge_kw:.1f} kW to load\n"
                else:
                    rec += f"• Maintain standby mode\n"
                
                rec += f"• Current status: {row['STATUS']}\n"
                rec += f"• Reliability score: {row['RELIABILITY SCORE']}\n"
            else:
                if optimized_kw > current_kw:
                    rec += f"• Increase load from {current_kw:.1f} kW to {optimized_kw:.1f} kW\n"
                elif optimized_kw < current_kw:
                    rec += f"• Reduce load from {current_kw:.1f} kW to {optimized_kw:.1f} kW\n"
                else:
                    rec += f"• Maintain current load of {current_kw:.1f} kW\n"
                rec += f"• Reliability score: {row['RELIABILITY SCORE']}\n"
            
            recommendations.append(rec)
        
        # Overall summary
        total_row = results_df[results_df['ENERGY SOURCE'] == 'TOTAL'].iloc[0]
        
        summary = f"""
**OPTIMIZATION SUMMARY**:
• Total current load: {total_row['CURRENT LOAD (kW)']} kW
• Total optimized load: {total_row['OPTIMIZED LOAD (kW)']} kW
• Current hourly cost: PKR {total_row['CURRENT COST/HR']:,.2f}
• Optimized hourly cost: PKR {total_row['OPTIMIZED COST/HR']:,.2f}
• Hourly cost savings: PKR {total_row['CURRENT COST/HR'] - total_row['OPTIMIZED COST/HR']:,.2f}
• Total BESS charge power: {total_row['OPTIMIZED CHARGE (kW)']} kW
• Total BESS discharge power: {total_row['OPTIMIZED DISCHARGE (kW)']} kW
• Optimization prioritizes lowest total cost (production + carbon) with reliability
        """
        
        recommendations.append(summary)
        return '\n\n'.join(recommendations)