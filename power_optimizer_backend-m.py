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
        
        # Reliability optimized operation
        self.reliability_optimized_charge_power = 0
        self.reliability_optimized_discharge_power = 0
        
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
    
    def get_reliability_optimized_operating_cost(self):
        """Calculate reliability optimized operating cost"""
        if self.reliability_optimized_charge_power > 0:
            return self.reliability_optimized_charge_power * self.total_cost
        elif self.reliability_optimized_discharge_power > 0:
            return self.reliability_optimized_discharge_power * self.total_cost
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
        
        # Reliability optimized loads
        self.reliability_optimized_active_load = 0
        self.reliability_optimized_reactive_load = 0
        
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
            'ghi': None,
            
        }
        
        # Reliability parameters
        self.max_peak_running_load = 1000.0
        self.total_critical_load = 500.0
        self.loss_tripping_cost_pkr = 50000.0
        self.loss_production_cost_pkr_per_hour = 10000.0
        
        self.grid_connected = True
        self.allow_grid_feed = True
        self.grid_feed_limit = 1000
        
        self.total_load_demand = 0
        self.total_reactive_demand = 0
        
        # Optimization results storage
        self.cost_optimized_results = None
        self.reliability_optimized_results = None
        self.selected_optimization = 'cost'  # 'cost' or 'reliability'
    
    def set_global_params(self, wind_speed=None, fuel_pressure=None, fuel_level=None, gas_pressure=None, ghi=None):
        """Set global operational parameters"""
        self.global_params.update({
            'wind_speed': wind_speed,
            'fuel_pressure': fuel_pressure,
            'fuel_level': fuel_level,
            'gas_pressure': gas_pressure,
            'ghi': ghi
        })
    
    def set_reliability_params(self, max_peak_running_load=None, total_critical_load=None, 
                              loss_tripping_cost_pkr=None, loss_production_cost_pkr_per_hour=None):
        """Set reliability optimization parameters"""
        if max_peak_running_load is not None:
            self.max_peak_running_load = max_peak_running_load
        if total_critical_load is not None:
            self.total_critical_load = total_critical_load
        if loss_tripping_cost_pkr is not None:
            self.loss_tripping_cost_pkr = loss_tripping_cost_pkr
        if loss_production_cost_pkr_per_hour is not None:
            self.loss_production_cost_pkr_per_hour = loss_production_cost_pkr_per_hour
        
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
            
            # Check BESS availability
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

    def calculate_total_loss_cost(self):
        """Calculate total loss cost for reliability analysis"""
        return self.loss_tripping_cost_pkr + self.loss_production_cost_pkr_per_hour
    
    def determine_redundancy_requirement(self):
        """Determine required redundancy (n+1 or n+2) based on loss cost"""
        total_loss = self.calculate_total_loss_cost()
        
        if total_loss >= 100000:  # Very high loss
            return "n+2"  # BESS + base load
        elif total_loss >= 50000:  # High loss
            return "n+1"  # Grid/Gas/Diesel base load
        else:
            return "n+0"  # Normal operation

    def calculate_renewable_reduction_factors(self, sources):
        """Calculate reduction factors for renewable sources"""
        total_engine_max = sum(s.max_capacity for s in sources 
                              if s.source_type == 'conventional' and s.name.lower() != 'grid')
        
        solar_running = sum(s.optimized_active_load for s in sources 
                           if s.name.lower().startswith('solar'))
        wind_running = sum(s.optimized_active_load for s in sources 
                          if s.name.lower().startswith('wind'))
        
        # 50% solar reduction factor
        x = 0.5 * solar_running
        sp = x / total_engine_max if total_engine_max > 0 else 0
        
        # 70% wind reduction factor
        y = 0.7 * wind_running
        wp = y / total_engine_max if total_engine_max > 0 else 0
        
        return sp, wp

    def apply_reliability_constraints(self, sources):
        """Apply reliability constraints to engine max loads"""
        # Calculate renewable reduction factors
        sp, wp = self.calculate_renewable_reduction_factors(sources)
        
        # Reserve factor (10% reserve)
        reserve_factor = 0.1
        
        # Calculate total reduction factor
        total_reduction = 1 - reserve_factor - sp - wp
        
        print(f"\nReliability Analysis:")
        print(f"Solar reduction factor (sp): {sp:.3f}")
        print(f"Wind reduction factor (wp): {wp:.3f}")
        print(f"Reserve factor: {reserve_factor:.3f}")
        print(f"Total reduction factor: {total_reduction:.3f}")
        
        # Apply constraints to engine sources
        for source in sources:
            if (source.source_type == 'conventional' and 
                source.name.lower() not in ['grid', 'bess']):
                
                original_max = source.max_capacity
                source.reliability_max_capacity = max(0, original_max * total_reduction)
                
                print(f"{source.name}: Original max {original_max} kW -> "
                      f"Reliability max {source.reliability_max_capacity:.2f} kW")

    def optimize_power_allocation(self):
        """Main optimization algorithm with both cost and reliability optimization"""
        available_sources = [s for s in self.sources if s.available]
        
        if not available_sources:
            print("No available power sources")
            return
        
        total_active_load = sum(s.current_active_load for s in self.sources)
        total_reactive_load = sum(s.current_reactive_load for s in self.sources)

        # Add BESS current contribution to total load
        total_bess_discharge = sum(abs(b.current_power_input) for b in self.bess_systems if b.current_power_input < 0)
        total_active_load += total_bess_discharge
        
        print(f"Total active load (including BESS): {total_active_load:.2f} kW")
        print(f"Total reactive load: {total_reactive_load:.2f} kVAR")
        
        # First, run cost optimization
        print("\n=== COST OPTIMIZATION ===")
        self.run_cost_optimization(available_sources, total_active_load, total_reactive_load)
        
        # Then, run reliability optimization
        print("\n=== RELIABILITY OPTIMIZATION ===")
        self.run_reliability_optimization(available_sources, total_active_load, total_reactive_load)
        
        # Compare and select best approach
        self.select_optimal_strategy()

    def run_cost_optimization(self, available_sources, total_active_load, total_reactive_load):
        """Run cost-based optimization (original algorithm)"""
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
        self.apply_load_sharing(available_sources)
        
        remaining_reactive_load = total_reactive_load
        kvar_sources = [s for s in self.sources]
        kvar_sources.extend(self.bess_systems)
        self.allocate_reactive_power(kvar_sources, remaining_reactive_load)

        if not self.grid_connected:
            self.handle_off_grid_operation(available_sources)

    def run_reliability_optimization(self, available_sources, total_active_load, total_reactive_load):
        """Run reliability-based optimization"""
        
        # Determine redundancy requirement
        redundancy = self.determine_redundancy_requirement()
        print(f"Required redundancy: {redundancy}")
        
        # Apply reliability constraints
        self.apply_reliability_constraints(available_sources)
        
        # Reset reliability optimized loads
        for source in available_sources:
            source.reliability_optimized_active_load = 0
            source.reliability_optimized_reactive_load = 0
        
        # Reset BESS reliability optimization
        for bess in self.bess_systems:
            bess.reliability_optimized_charge_power = 0
            bess.reliability_optimized_discharge_power = 0
        
        # For reliability optimization, prioritize by reliability score instead of cost
        reliability_sources = available_sources.copy()
        reliability_sources.sort(key=lambda x: -x.reliability_score)  # Higher reliability first
        
        # BESS optimization for reliability
        self.optimize_bess_reliability_operation(redundancy)
        
        # Calculate remaining load after BESS optimization
        total_bess_reliability_discharge = sum(b.reliability_optimized_discharge_power for b in self.bess_systems)
        remaining_active_load = total_active_load - total_bess_reliability_discharge
        
        # Allocate load with reliability constraints
        self.allocate_reliability_active_power(reliability_sources, remaining_active_load, redundancy)
        self.apply_reliability_load_sharing(reliability_sources)
        
        # Reactive power allocation for reliability
        remaining_reactive_load = total_reactive_load
        kvar_sources = [s for s in self.sources]
        kvar_sources.extend(self.bess_systems)
        self.allocate_reliability_reactive_power(kvar_sources, remaining_reactive_load)

    def optimize_bess_reliability_operation(self, redundancy):
        """Optimize BESS for reliability requirements"""
        for bess in self.bess_systems:
            print(f"\nReliability optimizing BESS: {bess.name}")
            
            if redundancy == "n+2":
                # Very high loss scenario - maximize BESS usage
                if bess.current_soc >= bess.discharge_threshold:
                    max_discharge = min(
                        bess.power_rating_kw,
                        bess.get_available_discharge_capacity()
                    )
                    bess.reliability_optimized_discharge_power = max_discharge
                    print(f"BESS {bess.name} reliability optimized for maximum discharge: {max_discharge:.2f} kW")
                elif bess.current_soc <= bess.charge_threshold:
                    # Keep ready for discharge
                    bess.reliability_optimized_charge_power = 0
                    print(f"BESS {bess.name} kept ready for discharge")
            
            elif redundancy == "n+1":
                # High loss scenario - moderate BESS usage
                if bess.current_soc >= bess.discharge_threshold:
                    max_discharge = min(
                        bess.power_rating_kw * 0.8,  # 80% for reliability margin
                        bess.get_available_discharge_capacity()
                    )
                    bess.reliability_optimized_discharge_power = max_discharge
                    print(f"BESS {bess.name} reliability optimized for moderate discharge: {max_discharge:.2f} kW")
            
            else:
                # Normal operation - same as cost optimization
                bess.reliability_optimized_discharge_power = bess.optimized_discharge_power
                bess.reliability_optimized_charge_power = bess.optimized_charge_power

    def allocate_reliability_active_power(self, sources, total_load, redundancy):
        """Allocate active power with reliability constraints"""
        remaining_load = total_load
        
        print(f"\nReliability power allocation:")
        print(f"Redundancy requirement: {redundancy}")
        print(f"Total load: {total_load} kW")
        
        # Separate sources by type
        conventional_sources = [s for s in sources if s.source_type == 'conventional' and s.name.lower() != 'grid']
        renewable_sources = [s for s in sources if s.source_type == 'renewable']
        grid_sources = [s for s in sources if s.name.lower() == 'grid']
        
        if redundancy in ["n+1", "n+2"]:
            # Base load strategy - ensure minimum reliable generation
            for source in conventional_sources + grid_sources:
                if remaining_load <= 0:
                    break
                
                # Use reliability max capacity instead of normal max capacity
                max_capacity = getattr(source, 'reliability_max_capacity', source.max_capacity)
                min_capacity = source.min_capacity
                
                if max_capacity >= min_capacity and remaining_load > min_capacity:
                    allocation = min(max_capacity, remaining_load)
                    # Ensure minimum allocation for base load
                    allocation = max(allocation, min_capacity)
                    source.reliability_optimized_active_load = allocation
                    remaining_load -= allocation
                    print(f"Base load allocation to {source.name}: {allocation:.2f} kW (Max: {max_capacity:.2f} kW)")
            
            # Then allocate renewables with their reduced capacity
            for source in renewable_sources:
                if remaining_load <= 0:
                    break
                
                allocation = min(source.max_capacity, remaining_load)
                source.reliability_optimized_active_load = allocation
                remaining_load -= allocation
                print(f"Renewable allocation to {source.name}: {allocation:.2f} kW")
        
        else:
            # Normal allocation
            for source in sources:
                if remaining_load <= 0:
                    break
                
                max_possible = min(source.max_capacity, remaining_load)
                min_required = source.min_capacity
                
                if max_possible >= min_required:
                    allocation = max(min_required, max_possible)
                    source.reliability_optimized_active_load = allocation
                    remaining_load -= allocation
                    print(f"Normal allocation to {source.name}: {allocation:.2f} kW")

        return remaining_load

    def apply_reliability_load_sharing(self, sources):
        """Apply load sharing for reliability optimization"""
        source_groups = {}
        for source in sources:
            source_type = source.name.split('_')[0] if '_' in source.name else source.name
            if source_type not in source_groups:
                source_groups[source_type] = []
            source_groups[source_type].append(source)
            
        for group_name, group_sources in source_groups.items():
            if len(group_sources) > 1:
                total_group_load = sum(s.reliability_optimized_active_load for s in group_sources)
                if total_group_load > 0:
                    # For reliability, prefer keeping more machines online at lower loads
                    active_sources = group_sources.copy()
                    
                    while len(active_sources) > 0:
                        avg_load = total_group_load / len(active_sources)
                        min_capacity_required = min(s.min_capacity for s in active_sources)
                        
                        if avg_load >= min_capacity_required:
                            # All machines can handle the average load
                            for s in group_sources:
                                if s in active_sources:
                                    s.reliability_optimized_active_load = avg_load
                                else:
                                    s.reliability_optimized_active_load = 0
                            break
                        else:
                            # Remove one machine and redistribute
                            machine_to_shutdown = max(active_sources, key=lambda x: x.min_capacity)
                            active_sources.remove(machine_to_shutdown)
                            machine_to_shutdown.reliability_optimized_active_load = 0
                    
                    print(f"Reliability load sharing applied to {group_name} sources")

    def allocate_reliability_reactive_power(self, sources, total_reactive_load):
        """Allocate reactive power for reliability optimization"""
        remaining_reactive_load = total_reactive_load
        
        for source in sources:
            if remaining_reactive_load <= 0:
                break

            if hasattr(source, 'reliability_optimized_active_load'):
                # For power sources
                if source.name.lower() != 'solar':
                    max_reactive = source.reliability_optimized_active_load * 0.6
                    allocation = min(max_reactive, remaining_reactive_load)
                    source.reliability_optimized_reactive_load = allocation
                    remaining_reactive_load -= allocation
                else:
                    source.reliability_optimized_reactive_load = 0
            else:
                # For BESS
                if hasattr(source, 'reliability_optimized_discharge_power'):
                    max_reactive = source.reliability_optimized_discharge_power * 0.8
                    allocation = min(max_reactive, remaining_reactive_load)
                    try:
                        source.reliability_optimized_reactive_load = allocation
                    except AttributeError:
                        pass
                    remaining_reactive_load -= allocation

            print(f"Reliability reactive power allocation to {source.name}: {allocation:.2f} kVAR")

        return remaining_reactive_load

    def select_optimal_strategy(self):
        """Compare cost vs reliability optimization and select best strategy"""
        print("\n=== STRATEGY COMPARISON ===")
        
        # Calculate costs for both strategies
        cost_optimized_cost = self.calculate_total_operational_cost('cost')
        reliability_optimized_cost = self.calculate_total_operational_cost('reliability')
        
        # Calculate total loss cost
        total_loss_cost = self.calculate_total_loss_cost()
        
        # Cost difference
        cost_difference = reliability_optimized_cost - cost_optimized_cost
        
        print(f"Cost-optimized operational cost: PKR {cost_optimized_cost:.2f}/hr")
        print(f"Reliability-optimized operational cost: PKR {reliability_optimized_cost:.2f}/hr")
        print(f"Additional cost for reliability: PKR {cost_difference:.2f}/hr")
        print(f"Total loss cost (if failure): PKR {total_loss_cost:.2f}")
        
        # Decision logic
        if cost_difference <= 0:
            # Reliability optimization is cheaper or same cost
            self.selected_optimization = 'reliability'
            decision_reason = "Reliability optimization provides better or equal cost"
        elif cost_difference < (total_loss_cost * 0.01):  # 1% of loss cost as threshold
            # Small additional cost for much better reliability
            self.selected_optimization = 'reliability'
            decision_reason = f"Small additional cost ({cost_difference:.2f} PKR/hr) justified by high loss risk"
        else:
            # Cost optimization is significantly cheaper
            self.selected_optimization = 'cost'
            decision_reason = f"Cost difference ({cost_difference:.2f} PKR/hr) exceeds acceptable threshold"
        
        print(f"\nSelected strategy: {self.selected_optimization.upper()}")
        print(f"Reason: {decision_reason}")

    def calculate_total_operational_cost(self, optimization_type):
        """Calculate total operational cost for given optimization type"""
        total_cost = 0
        
        if optimization_type == 'cost':
            # Cost-optimized calculation
            for source in self.sources:
                if source.available:
                    total_cost += source.optimized_active_load * source.total_cost
            
            for bess in self.bess_systems:
                total_cost += bess.get_optimized_operating_cost()
        
        else:  # reliability
            # Reliability-optimized calculation
            for source in self.sources:
                if source.available:
                    load = getattr(source, 'reliability_optimized_active_load', 0)
                    total_cost += load * source.total_cost
            
            for bess in self.bess_systems:
                total_cost += bess.get_reliability_optimized_operating_cost()
        
        return total_cost

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
        
        if self.selected_optimization == 'cost':
            total_generation = sum(s.optimized_active_load for s in sources)
        else:
            total_generation = sum(getattr(s, 'reliability_optimized_active_load', 0) for s in sources)
            
        total_demand = self.total_load_demand
        
        for bess in self.bess_systems:
            if self.selected_optimization == 'cost':
                if bess.mode == 'discharging':
                    total_generation += bess.optimized_discharge_power
            else:
                total_generation += bess.reliability_optimized_discharge_power
        
        if total_generation < total_demand:
            deficit = total_demand - total_generation
            print(f"Generation deficit in off-grid mode: {deficit:.2f} kW")
            
            for source in sources:
                max_capacity = source.max_capacity
                current_load = (source.optimized_active_load if self.selected_optimization == 'cost' 
                              else getattr(source, 'reliability_optimized_active_load', 0))
                
                if current_load < max_capacity:
                    additional_capacity = min(deficit, max_capacity - current_load)
                    
                    if self.selected_optimization == 'cost':
                        source.optimized_active_load += additional_capacity
                    else:
                        source.reliability_optimized_active_load += additional_capacity
                    
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

        if remaining_reactive_load > 0:
            print(f"Unallocated reactive power: {remaining_reactive_load:.2f} kVAR")
            # allocate this reactive load to grid
            grid_source = next((s for s in sources if s.name.lower() == 'grid'), None)
            if grid_source:
                grid_source.optimized_reactive_load += remaining_reactive_load
                print(f"Allocated {remaining_reactive_load:.2f} kVAR to grid")

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
    
    def generate_results(self):
        """Generate comprehensive results including all sources and BESS with both optimizations"""
        results = []
        total_current_load = 0
        total_cost_optimized_load = 0
        total_reliability_optimized_load = 0
        total_current_cost = 0
        total_cost_optimized_cost = 0
        total_reliability_optimized_cost = 0
        total_current_kvar = 0
        total_cost_optimized_kvar = 0
        total_reliability_optimized_kvar = 0

        # Process all power sources
        for source in self.sources:
            current_load = source.current_active_load
            cost_optimized_load = source.optimized_active_load
            reliability_optimized_load = getattr(source, 'reliability_optimized_active_load', 0)
            cost_per_kwh = source.total_cost

            current_kvar = source.current_reactive_load
            cost_optimized_kvar = source.optimized_reactive_load
            reliability_optimized_kvar = getattr(source, 'reliability_optimized_reactive_load', 0)

            current_cost_hr = current_load * cost_per_kwh
            cost_optimized_cost_hr = cost_optimized_load * cost_per_kwh
            reliability_optimized_cost_hr = reliability_optimized_load * cost_per_kwh

            row = {
                'ENERGY SOURCE': source.name,
                'CURRENT LOAD (kW)': round(current_load, 2),
                'COST OPTIMIZED LOAD (kW)': round(cost_optimized_load, 2),
                'RELIABILITY OPTIMIZED LOAD (kW)': round(reliability_optimized_load, 2),
                'CURRENT KVAR (kVAR)': round(current_kvar, 2),
                'COST OPTIMIZED KVAR (kVAR)': round(cost_optimized_kvar, 2),
                'RELIABILITY OPTIMIZED KVAR (kVAR)': round(reliability_optimized_kvar, 2),
                'TOTAL COST (PKR/kWh)': round(cost_per_kwh, 2),
                'PRODUCTION COST (PKR/kWh)': round(source.production_cost, 2),
                'CARBON COST (PKR/kWh)': round(source.carbon_emission * self.carbon_cost_pkr_kg, 2),
                'CURRENT COST/HR': round(current_cost_hr, 2),
                'COST OPTIMIZED COST/HR': round(cost_optimized_cost_hr, 2),
                'RELIABILITY OPTIMIZED COST/HR': round(reliability_optimized_cost_hr, 2),
                'OPTIMIZED CHARGE (kW)': 0.0,
                'OPTIMIZED DISCHARGE (kW)': 0.0,
                'RELIABILITY OPTIMIZED CHARGE (kW)': 0.0,
                'RELIABILITY OPTIMIZED DISCHARGE (kW)': 0.0,
                'RELIABILITY SCORE': round(source.reliability_score, 2),
                'EFFICIENCY SCORE': round(source.efficiency_score, 2),
                'RELIABILITY MAX CAPACITY (kW)': round(getattr(source, 'reliability_max_capacity', source.max_capacity), 2),
                'STATUS': 'Active' if source.available else 'Inactive',
                'SELECTED OPTIMIZATION': self.selected_optimization.upper()
            }
            
            results.append(row)
            
            total_current_load += current_load
            total_cost_optimized_load += cost_optimized_load
            total_reliability_optimized_load += reliability_optimized_load
            total_current_cost += current_cost_hr
            total_cost_optimized_cost += cost_optimized_cost_hr
            total_reliability_optimized_cost += reliability_optimized_cost_hr
            total_current_kvar += current_kvar
            total_cost_optimized_kvar += cost_optimized_kvar
            total_reliability_optimized_kvar += reliability_optimized_kvar
        
        # Process all BESS systems
        for bess in self.bess_systems:
            # Current BESS operation
            current_bess_load = abs(bess.current_power_input) if bess.current_power_input < 0 else 0
            current_bess_charge = bess.current_power_input if bess.current_power_input > 0 else 0
            current_bess_discharge = abs(bess.current_power_input) if bess.current_power_input < 0 else 0

            try:
                current_kvar = bess.current_reactive_load
            except AttributeError:
                current_kvar = 0

            try:
                cost_optimized_kvar = bess.optimized_reactive_load
            except AttributeError:
                cost_optimized_kvar = 0
            
            try:
                reliability_optimized_kvar = bess.reliability_optimized_reactive_load
            except AttributeError:
                reliability_optimized_kvar = 0

            # Optimized BESS operation
            cost_optimized_bess_load = bess.optimized_discharge_power
            reliability_optimized_bess_load = bess.reliability_optimized_discharge_power

            # Cost calculations
            current_cost_hr = bess.get_current_operating_cost()
            cost_optimized_cost_hr = bess.get_optimized_operating_cost()
            reliability_optimized_cost_hr = bess.get_reliability_optimized_operating_cost()
            
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
                'COST OPTIMIZED LOAD (kW)': round(cost_optimized_bess_load, 2),
                'RELIABILITY OPTIMIZED LOAD (kW)': round(reliability_optimized_bess_load, 2),
                'CURRENT KVAR (kVAR)': round(current_kvar, 2),
                'COST OPTIMIZED KVAR (kVAR)': round(cost_optimized_kvar, 2),
                'RELIABILITY OPTIMIZED KVAR (kVAR)': round(reliability_optimized_kvar, 2),
                'TOTAL COST (PKR/kWh)': round(bess.total_cost, 2),
                'PRODUCTION COST (PKR/kWh)': round(bess.production_cost, 2),
                'CARBON COST (PKR/kWh)': round(bess.carbon_emission * self.carbon_cost_pkr_kg, 2),
                'CURRENT COST/HR': round(current_cost_hr, 2),
                'COST OPTIMIZED COST/HR': round(cost_optimized_cost_hr, 2),
                'RELIABILITY OPTIMIZED COST/HR': round(reliability_optimized_cost_hr, 2),
                'OPTIMIZED CHARGE (kW)': round(bess.optimized_charge_power, 2),
                'OPTIMIZED DISCHARGE (kW)': round(bess.optimized_discharge_power, 2),
                'RELIABILITY OPTIMIZED CHARGE (kW)': round(bess.reliability_optimized_charge_power, 2),
                'RELIABILITY OPTIMIZED DISCHARGE (kW)': round(bess.reliability_optimized_discharge_power, 2),
                'RELIABILITY SCORE': round(bess.reliability_score, 2),
                'RELIABILITY MAX CAPACITY (kW)': round(bess.power_rating_kw, 2),
                'STATUS': status,
                'SELECTED OPTIMIZATION': self.selected_optimization.upper()
            }
            
            results.append(row)
            
            # Add to totals
            total_current_load += current_bess_discharge
            total_cost_optimized_load += cost_optimized_bess_load
            total_reliability_optimized_load += reliability_optimized_bess_load
            total_current_cost += current_cost_hr
            total_cost_optimized_cost += cost_optimized_cost_hr
            total_reliability_optimized_cost += reliability_optimized_cost_hr
            total_current_kvar += current_kvar
            total_cost_optimized_kvar += cost_optimized_kvar
            total_reliability_optimized_kvar += reliability_optimized_kvar

        # Add total row
        cost_savings_hr = total_current_cost - total_cost_optimized_cost
        reliability_cost_difference = total_reliability_optimized_cost - total_cost_optimized_cost
        
        total_production_cost = sum(
            (source.current_active_load * source.production_cost for source in self.sources if source.available)
        ) + sum(
            (bess.get_current_production_cost() for bess in self.bess_systems)
        )

        total_optimized_charge= sum(
            (bess.optimized_charge_power for bess in self.bess_systems)
        )

        total_optimized_discharge= sum(
            (bess.optimized_discharge_power for bess in self.bess_systems)
        )

        total_reliability_optimized_charge= sum(
            (bess.reliability_optimized_charge_power for bess in self.bess_systems)
        )

        total_reliability_optimized_discharge= sum(
            (bess.reliability_optimized_discharge_power for bess in self.bess_systems)
        )

        total_carbon_cost = sum(
            (source.current_active_load * source.carbon_emission * self.carbon_cost_pkr_kg for source in self.sources if source.available)
        ) + sum(
            (bess.get_current_carbon_cost(self.carbon_cost_pkr_kg) for bess in self.bess_systems)
        )

        total_reliability_score = np.mean([s.reliability_score for s in self.sources + self.bess_systems if s.available])
        total_reliability_max_capacity = sum([getattr(s, 'reliability_max_capacity', s.max_capacity) for s in self.sources + self.bess_systems if s.available])



        results.append({
            'ENERGY SOURCE': 'TOTAL',
            'CURRENT LOAD (kW)': round(total_current_load, 2),
            'COST OPTIMIZED LOAD (KW)': round(total_cost_optimized_load, 2),
            'RELIABILITY OPTIMIZED LOAD (KW)': round(total_reliability_optimized_load, 2),
            'CURRENT KVAR (kVAR)': round(total_current_kvar, 2),
            'COST OPTIMIZED KVAR (kVAR)': round(total_cost_optimized_kvar, 2),
            'RELIABILITY OPTIMIZED KVAR (kVAR)': round(total_reliability_optimized_kvar, 2),
            'TOTAL COST (PKR/kWh)': round(total_current_cost, 2),
            'PRODUCTION COST (PKR/kWh)': round(total_production_cost, 2),
            'CARBON COST (PKR/kWh)': round(total_carbon_cost, 2),
            'CURRENT COST/HR': round(total_current_cost, 2),
            'COST OPTIMIZED COST/HR': round(total_cost_optimized_cost, 2),
            'RELIABILITY OPTIMIZED COST/HR': round(total_reliability_optimized_cost, 2),
            'OPTIMIZED CHARGE (kW)': round(total_optimized_charge, 2),
            'OPTIMIZED DISCHARGE (kW)': round(total_optimized_discharge, 2),
            'RELIABILITY OPTIMIZED CHARGE (kW)': round(total_reliability_optimized_charge, 2),
            'RELIABILITY OPTIMIZED DISCHARGE (kW)': round(total_reliability_optimized_discharge, 2),
            'RELIABILITY SCORE': round(total_reliability_score, 2),
            'RELIABILITY MAX CAPACITY (kW)': round(total_reliability_max_capacity, 2),
            'STATUS': 'TOTAL',
            'SELECTED OPTIMIZATION': self.selected_optimization.upper()
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