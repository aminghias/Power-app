import pandas as pd
import numpy as np
import json
import os
import mysql.connector

class BatteryEnergyStorageSystem:
    """Battery Energy Storage System class"""
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
        self.available = False  # Default to False, will be set based on SOC
        
        # Cost parameters
        self.production_cost = production_cost
        self.carbon_emission = carbon_emission
        self.total_cost = total_cost if total_cost else production_cost
        
        self.mode = 'standby'
        self.grid_synced = True
        
        # Current operation
        self.current_power_input = 0
        self.current_charge_power = 0
        self.current_discharge_power = 0
        
        # Optimized operation
        self.optimized_charge_power = 0
        self.optimized_discharge_power = 0
        
        # Grid interaction
        self.grid_feed_power = 0
        self.grid_import_power = 0

        self.reliability_score = reliability_score
        
    def check_availability(self):
        """Check if BESS is available based on SOC"""
        self.available = self.min_soc < self.current_soc < self.max_soc
        return self.available
    
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
        elif self.current_power_input < 0:  # Discharging
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
                 power_reading=0, reactive_power_reading=0, source_type='conventional', reliability_score=10.0,
                 wind_speed=None, fuel_pressure=None, fuel_level=None, gas_pressure=None, ghi=None):
        self.name = name
        self.production_cost = production_cost
        self.carbon_emission = carbon_emission
        self.total_cost = total_cost
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.power_reading = power_reading
        self.reactive_power_reading = reactive_power_reading
        self.source_type = source_type
        self.available = False  # Default to False, will be set based on parameters
        self.reliability_score = reliability_score
        self.wind_speed = wind_speed
        self.fuel_pressure = fuel_pressure
        self.fuel_level = fuel_level
        self.gas_pressure = gas_pressure
        self.ghi = ghi
        
        self.current_active_load = power_reading
        self.current_reactive_load = reactive_power_reading
        
        self.optimized_active_load = 0
        self.optimized_reactive_load = 0
        
        self.efficiency_score = 0
    
    def check_availability(self):
        """Check if source is available based on specific parameters"""
        if self.source_type == 'renewable' and self.name.lower().startswith('wind'):
            # Wind turbine: operational between 3.5 and 25 m/s
            self.available = self.wind_speed is not None and 3.5 <= self.wind_speed <= 25
        elif self.source_type == 'conventional' and self.name.lower().startswith('diesel'):
            # Diesel generator: fuel pressure > 2 bar, fuel level > 10%
            self.available = (self.fuel_pressure is not None and self.fuel_pressure > 2 and
                              self.fuel_level is not None and self.fuel_level > 10)
        elif self.source_type == 'conventional' and self.name.lower().startswith('gas'):
            # Gas generator: gas pressure > 1 bar
            self.available = self.gas_pressure is not None and self.gas_pressure > 1
        elif self.source_type == 'renewable' and self.name.lower().startswith('solar'):
            # Solar: GHI > 100 W/m²
            self.available = self.ghi is not None and self.ghi > 100
        elif self.source_type == 'grid':
            # Grid: assumed always available unless explicitly disabled
            self.available = True
        else:
            self.available = False
        return self.available

class PowerOptimizer:
    """Main power optimization system"""

    def __init__(self, db_config):
        self.db_config = db_config
        self.sources = []
        self.bess_systems = []
        self.carbon_cost_pkr_kg = 50.0
        self.global_params = {
            'wind_speed': None,
            'fuel_pressure': None,
            'fuel_level': None,
            'gas_pressure': None,
            'ghi': None
        }
        self.reserve_percent = 0.25  # Default 25%
        self.plant_max_load = 5000.0  # Default 5 MW
        self.power_failure_cost = 1000.0  # Default PKR/min
        self.reserve_margin = self.reserve_percent
        
        self.grid_connected = True
        self.allow_grid_feed = True
        self.grid_feed_limit = 1000
        
        self.total_load_demand = 0
        self.total_reactive_demand = 0
    
    def set_reserve_params(self, reserve_percent=25.0, plant_max_load=5000.0, power_failure_cost=1000.0):
        """Set reserve and reliability parameters"""
        self.reserve_percent = reserve_percent / 100
        self.plant_max_load = plant_max_load
        self.power_failure_cost = power_failure_cost
        self.reserve_margin = self.reserve_percent
        if self.power_failure_cost > 5000:
            self.reserve_margin = max(self.reserve_margin, 0.5)  # Increase reserve for high cost
    
    def set_global_params(self, wind_speed=None, fuel_pressure=None, fuel_level=None, gas_pressure=None, ghi=None):
        """Set global operational parameters"""
        self.global_params.update({
            'wind_speed': wind_speed,
            'fuel_pressure': fuel_pressure,
            'fuel_level': fuel_level,
            'gas_pressure': gas_pressure,
            'ghi': ghi
        })
        
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
                reliability_score=reliability_score,
                wind_speed=self.global_params['wind_speed'] if config['name'].lower().startswith('wind') else None,
                fuel_pressure=self.global_params['fuel_pressure'] if config['name'].lower().startswith('diesel') else None,
                fuel_level=self.global_params['fuel_level'] if config['name'].lower().startswith('diesel') else None,
                gas_pressure=self.global_params['gas_pressure'] if config['name'].lower().startswith('gas') else None,
                ghi=self.global_params['ghi'] if config['name'].lower().startswith('solar') else None
            )
            
            if source.name.lower() == 'grid':
                device_id = config.get('device_id', 0)
                source.reliability_score = self.calculate_grid_reliability(device_id)
                print('From the database record The reliability score for the grid source is:', source.reliability_score)
            
            # Check availability based on parameters
            source.check_availability()
            
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
                print(f"Available: {source.available}")
                if source.name.lower().startswith('wind'):
                    print(f"Wind Speed: {source.wind_speed} m/s")
                elif source.name.lower().startswith('diesel'):
                    print(f"Fuel Pressure: {source.fuel_pressure} bar")
                    print(f"Fuel Level: {source.fuel_level}%")
                elif source.name.lower().startswith('gas'):
                    print(f"Gas Pressure: {source.gas_pressure} bar")
                elif source.name.lower().startswith('solar'):
                    print(f"GHI: {source.ghi} W/m²")
                print('-----------------------------------------')
            
            self.sources.append(source)
        
        self.validate_redundancy()
        
        bess_config = self.load_bess_config(site_id)
        self.bess_systems = []
        
        for config in bess_config:
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
            
            bess.check_availability()
            
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
            print(f"Available: {bess.available}")
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
        
        total_bess_discharge = sum(abs(b.current_power_input) for b in self.bess_systems if b.current_power_input < 0 and b.available)
        total_active_load += total_bess_discharge
        
        self.total_load_demand = total_active_load  # Set for validation
        
        print(f"Total active load (including BESS): {total_active_load:.2f} kW")
        print(f"Total reactive load: {total_reactive_load:.2f} kVAR")
        
        for source in available_sources:
            source.efficiency_score = self.calculate_efficiency_score(source)
        
        available_sources.sort(key=lambda x: x.efficiency_score)
        
        for source in available_sources:
            source.optimized_active_load = 0
            source.optimized_reactive_load = 0
        
        self.optimize_bess_operation()
        
        total_bess_optimized_discharge = sum(b.optimized_discharge_power for b in self.bess_systems if b.available)
        remaining_active_load = total_active_load - total_bess_optimized_discharge
        
        self.allocate_active_power(available_sources, remaining_active_load)
        
        remaining_reactive_load = total_reactive_load
        self.allocate_reactive_power(available_sources, remaining_reactive_load)
        
        self.apply_load_sharing(available_sources)
        
        if not self.grid_connected:
            self.handle_off_grid_operation(available_sources)

    def simulate_source_trip(self, tripped_source_name):
        """Simulate a source trip and re-optimize"""
        tripped_source = next((s for s in self.sources if s.name == tripped_source_name), None)
        if tripped_source:
            original_available = tripped_source.available
            tripped_source.available = False
            print(f"Simulating trip of {tripped_source_name}. Re-optimizing...")
            self.optimize_power_allocation()
            tripped_source.available = original_available  # Restore for normal operation
        else:
            print(f"Source {tripped_source_name} not found.")

    def allocate_active_power(self, sources, total_load):
        """Allocate active power among available sources with reserve margin"""
        remaining_load = total_load * (1 + self.reserve_margin)  # Allocate as if load is higher for cushion
        print('\nAllocating power among sources with reliability reserves (effective load for allocation: {remaining_load:.2f} kW):')
        for source in sources:
            print(f" - {source.name} (Max: {source.max_capacity} kW, Min: {source.min_capacity} kW, Cost: {source.total_cost:.2f} PKR/kWh, Reliability: {source.reliability_score:.2f})")
        print(f"Total load to allocate: {total_load:.2f} kW")

        renewable_sources = [s for s in sources if s.source_type == 'renewable']
        conventional_sources = [s for s in sources if s.source_type not in ['renewable', 'grid']]
        grid_sources = [s for s in sources if s.source_type == 'grid']
        
        # Apply reserve margin to non-backup sources
        for source in renewable_sources + conventional_sources:
            effective_max = source.max_capacity * (1 - self.reserve_margin)
            source.max_capacity = effective_max  # Temporary for allocation
        
        solar_sources = [s for s in renewable_sources if s.name.lower().startswith('solar')]
        bess_discharging = any(b.optimized_discharge_power > 0 for b in self.bess_systems if b.available)
        
        if solar_sources and total_load <= sum(s.max_capacity for s in solar_sources) and not bess_discharging:
            print("Applying solar-first optimization strategy")
            
            # Run N+1 for conventional if no BESS and high failure cost
            if self.power_failure_cost > 5000 and not self.bess_systems:
                num_conventional = len(conventional_sources)
                if num_conventional > 1:
                    print("High failure cost, no BESS: Running N+1 conventional sources for redundancy.")
                    for source in conventional_sources:
                        min_allocation = min(source.min_capacity, remaining_load / num_conventional)
                        source.optimized_active_load = min_allocation
                        remaining_load -= min_allocation
                        print(f"Redundant allocation to {source.name}: {min_allocation:.2f} kW")
            
            for source in conventional_sources:
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
            for source in renewable_sources:
                if remaining_load <= 0:
                    break
                allocation = min(remaining_load, source.max_capacity)
                source.optimized_active_load = allocation
                remaining_load -= allocation
                print(f"Renewable allocation to {source.name}: {allocation:.2f} kW")
            
            for source in conventional_sources:
                if remaining_load <= 0:
                    break
                min_required = source.min_capacity
                max_possible = min(source.max_capacity, remaining_load + min_required)
                if max_possible >= min_required:
                    allocation = min(max_possible, remaining_load + min_required)
                    source.optimized_active_load = allocation
                    remaining_load -= allocation
                    print(f"Conventional allocation to {source.name}: {allocation:.2f} kW")
            
            for source in grid_sources:
                if remaining_load > 0:
                    allocation = min(remaining_load, source.max_capacity)
                    source.optimized_active_load = allocation
                    remaining_load -= allocation
                    print(f"Grid allocation to {source.name}: {allocation:.2f} kW")

        # Restore original max_capacity and decrease allocations to create cushion
        for source in renewable_sources + conventional_sources:
            source.max_capacity /= (1 - self.reserve_margin)
            source.optimized_active_load /= (1 + self.reserve_margin)  # Scale down to actual load
        
        return remaining_load

    def allocate_reactive_power(self, sources, total_reactive_load):
        """Allocate reactive power among sources"""
        remaining_reactive_load = total_reactive_load
        
        for source in sources:
            if remaining_reactive_load <= 0:
                break
            
            if source.name.lower() != 'solar':
                max_reactive = source.optimized_active_load * 0.6
                allocation = min(max_reactive, remaining_reactive_load)
                source.optimized_reactive_load = allocation
                remaining_reactive_load -= allocation
            else:
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
                    if all(abs(s.max_capacity - group_sources[0].max_capacity) < 50 for s in group_sources):
                        avg_load = total_group_load / len(group_sources)
                        for s in group_sources:
                            s.optimized_active_load = avg_load
                    else:
                        total_max = sum(s.max_capacity for s in group_sources)
                        for s in group_sources:
                            proportion = s.max_capacity / total_max
                            s.optimized_active_load = proportion * total_group_load
                   
                    print(f"Load sharing applied to {group_name} sources")
    
    def generate_results(self):
        """Generate comprehensive results including all sources and BESS with priority"""
        results = []
        total_current_load = 0
        total_optimized_load = 0
        total_current_cost = 0
        total_optimized_cost = 0
        
        all_sources = []
        for source in self.sources:
            all_sources.append({
                'name': source.name,
                'current_load': source.current_active_load,
                'optimized_load': source.optimized_active_load,
                'total_cost': source.total_cost,
                'production_cost': source.production_cost,
                'carbon_cost': source.carbon_emission * self.carbon_cost_pkr_kg,
                'reliability_score': source.reliability_score,
                'is_bess': False,
                'charge_power': 0.0,
                'discharge_power': 0.0,
                'status': 'Active' if source.available else 'Inactive',
                'efficiency_score': self.calculate_efficiency_score(source)
            })
        
        for bess in self.bess_systems:
            current_bess_load = abs(bess.current_power_input) if bess.current_power_input < 0 else 0
            current_bess_charge = bess.current_power_input if bess.current_power_input > 0 else 0
            current_bess_discharge = abs(bess.current_power_input) if bess.current_power_input < 0 else 0
            optimized_bess_load = bess.optimized_discharge_power
            
            if bess.mode == 'charging':
                status = f'Charging (SOC: {bess.current_soc}%)'
            elif bess.mode == 'discharging':
                status = f'Discharging (SOC: {bess.current_soc}%)'
            else:
                status = f'Standby (SOC: {bess.current_soc}%)'
            
            all_sources.append({
                'name': bess.name,
                'current_load': current_bess_discharge,
                'optimized_load': optimized_bess_load,
                'total_cost': bess.total_cost,
                'production_cost': bess.production_cost,
                'carbon_cost': bess.carbon_emission * self.carbon_cost_pkr_kg,
                'reliability_score': bess.reliability_score,
                'is_bess': True,
                'charge_power': bess.optimized_charge_power,
                'discharge_power': bess.optimized_discharge_power,
                'status': status,
                'efficiency_score': self.calculate_efficiency_score(bess)
            })
        
        all_sources.sort(key=lambda x: x['efficiency_score'])
        
        for idx, source in enumerate(all_sources, 1):
            source['priority'] = idx
            
            row = {
                'ENERGY SOURCE': source['name'],
                'CURRENT LOAD (kW)': round(source['current_load'], 2),
                'OPTIMIZED LOAD (kW)': round(source['optimized_load'], 2),
                'TOTAL COST (PKR/kWh)': round(source['total_cost'], 2),
                'PRODUCTION COST (PKR/kWh)': round(source['production_cost'], 2),
                'CARBON COST (PKR/kWh)': round(source['carbon_cost'], 2),
                'CURRENT COST/HR': round(source['current_load'] * source['total_cost'], 2),
                'OPTIMIZED COST/HR': round(source['optimized_load'] * source['total_cost'], 2),
                'OPTIMIZED CHARGE (kW)': round(source['charge_power'], 2) if source['is_bess'] else 0.0,
                'OPTIMIZED DISCHARGE (kW)': round(source['discharge_power'], 2) if source['is_bess'] else 0.0,
                'RELIABILITY SCORE': round(source['reliability_score'], 2),
                'TOTAL SCORE': round(source['efficiency_score'], 2),
                'PRIORITY': source['priority'],
                'STATUS': source['status']
            }
            
            results.append(row)
            
            total_current_load += source['current_load']
            total_optimized_load += source['optimized_load']
            total_current_cost += source['current_load'] * source['total_cost']
            total_optimized_cost += source['optimized_load'] * source['total_cost']
        
        total_savings_hr = total_current_cost - total_optimized_cost
        results.append({
            'ENERGY SOURCE': 'TOTAL',
            'CURRENT LOAD (kW)': round(total_current_load, 2),
            'OPTIMIZED LOAD (kW)': round(total_optimized_load, 2),
            'TOTAL COST (PKR/kWh)': '',
            'PRODUCTION COST (PKR/kWh)': '',
            'CARBON COST (PKR/kWh)': '',
            'CURRENT COST/HR': round(total_current_cost, 2),
            'OPTIMIZED COST/HR': round(total_optimized_cost, 2),
            'OPTIMIZED CHARGE (kW)': round(sum(b.optimized_charge_power for b in self.bess_systems if b.available), 2),
            'OPTIMIZED DISCHARGE (kW)': round(sum(b.optimized_discharge_power for b in self.bess_systems if b.available), 2),
            'RELIABILITY SCORE': '',
            'TOTAL SCORE': '',
            'PRIORITY': '',
            'STATUS': f'Savings: PKR {total_savings_hr:.2f}/hr'
        })
        
        return pd.DataFrame(results), total_current_cost, total_savings_hr
    
    def generate_recommendations(self, results_df):
        """Generate optimization recommendations"""
        recommendations = []
        
        for _, row in results_df.iterrows():
            source = row['ENERGY SOURCE']
            if source == 'TOTAL':
                continue
            
            current_kw = row['CURRENT LOAD (kW)']
            optimized_kw = row['OPTIMIZED LOAD (kW)']
            
            rec = f"**{source} (Priority: {row['PRIORITY']})**:\n"
            
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
                rec += f"• Total score: {row['TOTAL SCORE']}\n"
            else:
                if optimized_kw > current_kw:
                    rec += f"• Increase load from {current_kw:.1f} kW to {optimized_kw:.1f} kW\n"
                elif optimized_kw < current_kw:
                    rec += f"• Reduce load from {current_kw:.1f} kW to {optimized_kw:.1f} kW\n"
                else:
                    rec += f"• Maintain current load of {current_kw:.1f} kW\n"
                rec += f"• Reliability score: {row['RELIABILITY SCORE']}\n"
                rec += f"• Total score: {row['TOTAL SCORE']}\n"
            
            recommendations.append(rec)
        
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
• Optimization prioritizes lowest total score (cost + inverted reliability)
        """
        
        recommendations.append(summary)
        return '\n\n'.join(recommendations)