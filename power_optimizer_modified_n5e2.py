import pandas as pd
import numpy as np
import json
import os
import mysql.connector
import uuid

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
        
        # Store both cost and reliability optimized values
        self.cost_optimized_charge_power = 0
        self.cost_optimized_discharge_power = 0
        self.rel_optimized_charge_power = 0
        self.rel_optimized_discharge_power = 0
        
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
    
    def get_optimized_operating_cost(self, mode='cost'):
        """Calculate optimized operating cost for specified mode"""
        if mode == 'cost':
            if self.cost_optimized_charge_power > 0:
                return self.cost_optimized_charge_power * self.total_cost
            elif self.cost_optimized_discharge_power > 0:
                return self.cost_optimized_discharge_power * self.total_cost
        else:  # reliability
            if self.rel_optimized_charge_power > 0:
                return self.rel_optimized_charge_power * self.total_cost
            elif self.rel_optimized_discharge_power > 0:
                return self.rel_optimized_discharge_power * self.total_cost
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
        self.effective_max = max_capacity  # For reliability mode
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
        
        self.cost_optimized_active_load = 0
        self.cost_optimized_reactive_load = 0
        self.rel_optimized_active_load = 0
        self.rel_optimized_reactive_load = 0
        
        self.grid_feed_power = 0 if source_type == 'grid' else 0
        self.cost_grid_feed_power = 0
        self.rel_grid_feed_power = 0
        
        self.efficiency_score = 0
    
    def check_availability(self):
        """Check if source is available based on specific parameters"""
        if self.source_type == 'renewable' and self.name.lower().startswith('wind'):
            self.available = self.wind_speed is not None and 3.5 <= self.wind_speed <= 25
        elif self.source_type == 'conventional' and self.name.lower().startswith('diesel'):
            self.available = (self.fuel_pressure is not None and self.fuel_pressure > 2 and
                            self.fuel_level is not None and self.fuel_level > 10)
        elif self.source_type == 'conventional' and self.name.lower().startswith('gas'):
            self.available = self.gas_pressure is not None and self.gas_pressure > 1
        elif self.source_type == 'renewable' and self.name.lower().startswith('solar'):
            self.available = self.ghi is not None and self.ghi > 100
        elif self.source_type == 'grid':
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
        self.bess_systems_dict = {}  # Dictionary for BESS systems
        self.power_sources = {}  # Dictionary for power sources
        self.carbon_cost_pkr_kg = 50.0
        self.global_params = {
            'wind_speed': None,
            'fuel_pressure': None,
            'fuel_level': None,
            'gas_pressure': None,
            'ghi': None
        }
        
        self.grid_connected = True
        self.allow_grid_feed = True
        self.grid_feed_limit = 1000
        
        self.total_load_demand = 0
        self.total_reactive_demand = 0
        
        self.max_peak_load = None
        self.critical_load = None
        self.tripping_cost = None
        self.production_loss_hourly = None
        
        self.very_high_threshold = 1000000
        self.high_threshold = 100000
        self.redundancy_level = 0
        self.prefer_bess = False
        
        self.grid_feed = 0
        self.cost_grid_feed = 0
        self.rel_grid_feed = 0

        self.optimized_mode = None
        self.optimized_loads = {}  
        self.optimized_charges = {}  
        self.optimized_discharges = {}  
        self.optimized_grid_feeds = {}  
        self.cost_optimized_loads = {}  
        self.cost_optimized_charges = {}  
        self.cost_optimized_discharges = {}  
        self.cost_optimized_grid_feeds = {}  
        self.reliability_optimized_loads = {}  
        self.reliability_optimized_charges = {}  
        self.reliability_optimized_discharges = {}  
        self.reliability_optimized_grid_feeds = {}
    
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
        """Load power sources configuration from JSON file"""
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
        """Load BESS configuration from JSON file"""
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
        """Initialize power sources and BESS systems from configuration"""
        sources_config = self.load_sources_config(site_id)
        if not sources_config:
            print("No sources configuration found")
            return
        
        self.sources = []
        self.power_sources = {}
        
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
                print(f"From the database record The reliability score for the grid source is: {source.reliability_score}")
            
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
            self.power_sources[source.name] = source
        
        bess_config = self.load_bess_config(site_id)
        self.bess_systems = []
        self.bess_systems_dict = {}
        
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
            self.bess_systems_dict[bess.name] = bess
            
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
    
    def calculate_current_cost(self):
        """Calculate the current total cost per hour based on active power loads and production costs"""
        total_cost = 0.0
        
        for source in self.power_sources.values():
            active_power = source.current_active_load if source.available else 0.0
            production_cost = source.total_cost
            total_cost += active_power * production_cost
        
        for bess in self.bess_systems_dict.values():
            power_input = bess.current_power_input
            production_cost = bess.total_cost
            if power_input != 0 and bess.available:
                total_cost += abs(power_input) * production_cost
        
        return total_cost
    
    def calculate_optimized_cost(self, mode='cost'):
        """Calculate the optimized total cost per hour based on optimized active power loads and production costs"""
        total_cost = 0.0
        
        for source in self.power_sources.values():
            if mode == 'cost':
                active_power = source.cost_optimized_active_load if source.available else 0.0
            else:
                active_power = source.rel_optimized_active_load if source.available else 0.0
            production_cost = source.total_cost
            total_cost += active_power * production_cost
        
        for bess in self.bess_systems_dict.values():
            if mode == 'cost':
                power = (bess.cost_optimized_discharge_power if bess.cost_optimized_discharge_power > 0 
                         else bess.cost_optimized_charge_power)
            else:
                power = (bess.rel_optimized_discharge_power if bess.rel_optimized_discharge_power > 0 
                         else bess.rel_optimized_charge_power)
            production_cost = bess.total_cost
            if power != 0 and bess.available:
                total_cost += abs(power) * production_cost
                
        return total_cost
    
    def calculate_total_loss(self):
        """Calculate total loss from tripping cost and production loss"""
        return (self.tripping_cost or 0) + (self.production_loss_hourly or 0)

    def optimize_power_allocation_cost(self):
        """Optimize power allocation to minimize cost"""
        self.cost_optimized_loads = {}
        self.cost_optimized_charges = {}
        self.cost_optimized_discharges = {}
        self.cost_optimized_grid_feeds = {}
        
        total_demand = self.max_peak_load or self.total_load_demand
        available_capacity = 0.0
        
        for source in self.power_sources.values():
            if source.available:
                available_capacity += min(source.current_active_load, source.max_capacity)
        
        remaining_demand = total_demand
        for source in sorted(self.power_sources.values(), key=lambda x: x.total_cost):
            if source.available and remaining_demand > 0:
                load = min(remaining_demand, source.current_active_load)
                self.cost_optimized_loads[source.name] = load
                remaining_demand -= load
        
        for bess in self.bess_systems_dict.values():
            if bess.available:
                if remaining_demand > 0 and bess.current_soc > bess.discharge_threshold:
                    discharge = min(remaining_demand, bess.power_rating_kw)
                    self.cost_optimized_discharges[bess.name] = discharge
                    remaining_demand -= discharge
                elif remaining_demand < 0 and bess.current_soc < bess.charge_threshold:
                    charge = min(abs(remaining_demand), bess.power_rating_kw)
                    self.cost_optimized_charges[bess.name] = charge
                    remaining_demand += charge
        
        if remaining_demand > 0:
            self.cost_optimized_grid_feeds['grid'] = remaining_demand

    def optimize_power_allocation_reliability(self):
        """Optimize power allocation to maximize reliability"""
        self.reliability_optimized_loads = {}
        self.reliability_optimized_charges = {}
        self.reliability_optimized_discharges = {}
        self.reliability_optimized_grid_feeds = {}
        
        total_demand = self.max_peak_load or self.total_load_demand
        remaining_demand = total_demand
        
        for source in sorted(self.power_sources.values(), key=lambda x: x.reliability_score, reverse=True):
            if source.available and remaining_demand > 0:
                load = min(remaining_demand, source.current_active_load)
                self.reliability_optimized_loads[source.name] = load
                remaining_demand -= load
        
        for bess in self.bess_systems_dict.values():
            if bess.available and remaining_demand > 0 and bess.current_soc > bess.discharge_threshold:
                discharge = min(remaining_demand, bess.power_rating_kw)
                self.reliability_optimized_discharges[bess.name] = discharge
                remaining_demand -= discharge
        
        if remaining_demand > 0:
            self.reliability_optimized_grid_feeds['grid'] = remaining_demand

    def optimize_power_allocation(self):
        """Optimize power allocation based on cost or reliability criteria"""
        current_cost = self.calculate_current_cost()
        self.optimize_power_allocation_cost()
        cost_optimized_cost = self.calculate_optimized_cost(mode='cost')
        cost_diff = current_cost - cost_optimized_cost

        self.optimize_power_allocation_reliability()
        rel_optimized_cost = self.calculate_optimized_cost(mode='reliability')
        total_loss = self.calculate_total_loss()

        if cost_diff > total_loss:
            self.optimized_mode = 'cost'
            self.optimized_loads = self.cost_optimized_loads.copy()
            self.optimized_charges = self.cost_optimized_charges.copy()
            self.optimized_discharges = self.cost_optimized_discharges.copy()
            self.optimized_grid_feeds = self.cost_optimized_grid_feeds.copy()
        else:
            self.optimized_mode = 'reliability'
            self.optimized_loads = self.reliability_optimized_loads.copy()
            self.optimized_charges = self.reliability_optimized_charges.copy()
            self.optimized_discharges = self.reliability_optimized_discharges.copy()
            self.optimized_grid_feeds = self.reliability_optimized_grid_feeds.copy()

        self.update_optimized_results()

    def update_optimized_results(self):
        """Update internal state with chosen optimization results"""
        for source in self.sources:
            source.optimized_active_load = (source.cost_optimized_active_load 
                                          if self.optimized_mode == 'cost' 
                                          else source.rel_optimized_active_load)
            source.optimized_reactive_load = (source.cost_optimized_reactive_load 
                                            if self.optimized_mode == 'cost' 
                                            else source.rel_optimized_reactive_load)
            source.grid_feed_power = (source.cost_grid_feed_power 
                                    if self.optimized_mode == 'cost' 
                                    else source.rel_grid_feed_power)
        for bess in self.bess_systems:
            bess.optimized_charge_power = (bess.cost_optimized_charge_power 
                                         if self.optimized_mode == 'cost' 
                                         else bess.rel_optimized_charge_power)
            bess.optimized_discharge_power = (bess.cost_optimized_discharge_power 
                                            if self.optimized_mode == 'cost' 
                                            else bess.rel_optimized_discharge_power)
        self.grid_feed = self.cost_grid_feed if self.optimized_mode == 'cost' else self.rel_grid_feed

    def optimize_bess_operation(self, mode='cost'):
        """Optimize BESS charging/discharging strategy"""
        for bess in self.bess_systems:
            print(f"\nOptimizing BESS: {bess.name} for {mode} mode")
            print(f"Current SOC: {bess.current_soc}%")
            print(f"Discharge Threshold: {bess.discharge_threshold}%")
            print(f"Charge Threshold: {bess.charge_threshold}%")
            
            if mode == 'cost':
                bess.cost_optimized_charge_power = 0
                bess.cost_optimized_discharge_power = 0
            else:
                bess.rel_optimized_charge_power = 0
                bess.rel_optimized_discharge_power = 0
            
            if bess.current_soc >= bess.discharge_threshold:
                max_discharge = min(
                    bess.power_rating_kw,
                    bess.get_available_discharge_capacity()
                )
                if mode == 'cost':
                    bess.cost_optimized_discharge_power = max_discharge
                else:
                    bess.rel_optimized_discharge_power = max_discharge
                bess.mode = 'discharging'
                print(f"BESS {bess.name} optimized for discharging: {max_discharge:.2f} kW")
                
            elif bess.current_soc <= bess.charge_threshold:
                max_charge = min(
                    bess.power_rating_kw,
                    bess.get_available_charge_capacity()
                )
                if mode == 'cost':
                    bess.cost_optimized_charge_power = max_charge
                else:
                    bess.rel_optimized_charge_power = max_charge
                bess.mode = 'charging'
                print(f"BESS {bess.name} optimized for charging: {max_charge:.2f} kW")
                
            else:
                bess.mode = 'standby'
                print(f"BESS {bess.name} in standby mode")

    def handle_off_grid_operation(self, sources):
        """Handle off-grid operation mode"""
        print("\nOff-grid operation mode activated")
        
        sources = [s for s in sources if s.source_type != 'grid']
        
        total_generation = sum(s.cost_optimized_active_load for s in sources)
        total_demand = self.total_load_demand
        
        for bess in self.bess_systems:
            if bess.mode == 'discharging':
                total_generation += bess.cost_optimized_discharge_power
        
        if total_generation < total_demand:
            deficit = total_demand - total_generation
            print(f"Generation deficit in off-grid mode: {deficit:.2f} kW")
            
            for source in sources:
                if source.cost_optimized_active_load < source.effective_max:
                    additional_capacity = min(deficit, 
                                            source.effective_max - source.cost_optimized_active_load)
                    source.cost_optimized_active_load += additional_capacity
                    deficit -= additional_capacity
                    print(f"Increased {source.name} output by {additional_capacity:.2f} kW")
                    
                    if deficit <= 0:
                        break
            
            if deficit > 0:
                print(f"Load shedding required: {deficit:.2f} kW")

    def allocate_active_power(self, sources, total_load, use_effective=False, mode='cost'):
        """Allocate active power among sources"""
        remaining_load = total_load
        print('The sources are:')
        for source in sources:
            print(f" - {source.name} (Max: {source.max_capacity} kW, Min: {source.min_capacity} kW)")
        print(f"Total load: {total_load} kW")
        print(f"Remaining load: {remaining_load} kW")

        solar_sources = [s for s in sources if s.name.lower().startswith('solar')]
        non_renewable_sources = [s for s in sources if not s.name.lower().startswith('solar') and not s.name.lower().startswith('wind') and not s.name.lower().startswith('bess')]
        bess_discharging = any(b.cost_optimized_discharge_power > 0 for b in self.bess_systems) if mode == 'cost' else any(b.rel_optimized_discharge_power > 0 for b in self.bess_systems)
        if solar_sources and total_load <= sum(s.effective_max if use_effective else s.max_capacity for s in solar_sources) and not bess_discharging:
            print("Applying solar-first optimization strategy")
            
            for source in non_renewable_sources:
                if source.name.lower() != 'grid' and source.available:
                    if remaining_load > source.min_capacity:
                        min_allocation = source.min_capacity
                        if mode == 'cost':
                            source.cost_optimized_active_load = min_allocation
                        else:
                            source.rel_optimized_active_load = min_allocation
                        remaining_load -= min_allocation
                        print(f"Minimum allocation to {source.name}: {min_allocation:.2f} kW")
                        break
                elif source.name.lower() == 'grid' and self.grid_connected:
                    min_allocation = 10
                    if mode == 'cost':
                        source.cost_optimized_active_load = min_allocation
                    else:
                        source.rel_optimized_active_load = min_allocation
                    remaining_load -= min_allocation
                    print(f"Minimum allocation to {source.name}: {min_allocation:.2f} kW")
            
            for source in solar_sources:
                if remaining_load > 0:
                    max_cap = source.effective_max if use_effective else source.max_capacity
                    allocation = min(remaining_load, max_cap)
                    if mode == 'cost':
                        source.cost_optimized_active_load = allocation
                    else:
                        source.rel_optimized_active_load = allocation
                    remaining_load -= allocation
                    print(f"Solar allocation to {source.name}: {allocation:.2f} kW")
        
        else:
            for source in sources:
                if remaining_load <= 0:
                    break
                
                max_cap = source.effective_max if use_effective else source.max_capacity
                max_possible = min(max_cap, remaining_load)
                min_required = source.min_capacity
                
                if max_possible >= min_required:
                    allocation = max_possible
                    if mode == 'cost':
                        source.cost_optimized_active_load = allocation
                    else:
                        source.rel_optimized_active_load = allocation
                    remaining_load -= allocation
                    print(f"Optimized allocation to {source.name}: {allocation:.2f} kW")
                else:
                    if mode == 'cost':
                        source.cost_optimized_active_load = 0
                    else:
                        source.rel_optimized_active_load = 0

        return remaining_load
    
    def allocate_reactive_power(self, sources, total_reactive_load, mode='cost'):
        """Allocate reactive power among sources"""
        remaining_reactive_load = total_reactive_load
        
        for source in sources:
            if remaining_reactive_load <= 0:
                break

            if hasattr(source, 'power_rating_kw'):  # BESS
                max_reactive = source.power_rating_kw * 0.8
                allocation = min(max_reactive, remaining_reactive_load)
                if mode == 'cost':
                    source.cost_optimized_reactive_load = allocation
                else:
                    source.rel_optimized_reactive_load = allocation
                remaining_reactive_load -= allocation
            
            elif source.name.lower() != 'solar':
                max_reactive = (source.cost_optimized_active_load if mode == 'cost' else source.rel_optimized_active_load) * 0.6
                allocation = min(max_reactive, remaining_reactive_load)
                if mode == 'cost':
                    source.cost_optimized_reactive_load = allocation
                else:
                    source.rel_optimized_reactive_load = allocation
                remaining_reactive_load -= allocation

            else:
                allocation = 0
                if mode == 'cost':
                    source.cost_optimized_reactive_load = allocation
                else:
                    source.rel_optimized_reactive_load = allocation

            print(f"Reactive power allocation to {source.name}: {allocation:.2f} kVAR")

        if remaining_reactive_load > 0:
            print(f"Unallocated reactive power: {remaining_reactive_load:.2f} kVAR")
            grid_source = next((s for s in sources if hasattr(s, 'name') and s.name.lower() == 'grid'), None)
            if grid_source:
                if mode == 'cost':
                    grid_source.cost_optimized_reactive_load += remaining_reactive_load
                    print(f"Allocated {remaining_reactive_load:.2f} kVAR to grid")
                else:
                    grid_source.rel_optimized_reactive_load += remaining_reactive_load
                    print(f"Allocated {remaining_reactive_load:.2f} kVAR to grid")

        return remaining_reactive_load
    
    def apply_load_sharing(self, sources, redundancy_level=0, use_effective=False, mode='cost'):
        """Apply load sharing among similar sources with optional redundancy"""
        source_groups = {}
        for source in sources:
            source_type = source.name.split('_')[0] if '_' in source.name else source.name
            if source_type not in source_groups:
                source_groups[source_type] = []
            source_groups[source_type].append(source)
            
        for group_name, group_sources in source_groups.items():
            if len(group_sources) > 1:
                total_group_load = sum(s.cost_optimized_active_load if mode == 'cost' else s.rel_optimized_active_load for s in group_sources)
                if total_group_load > 0:
                    if all(abs(s.max_capacity - group_sources[0].max_capacity) < 50 for s in group_sources):
                        active_sources = group_sources.copy()
                        
                        while len(active_sources) > 1:
                            avg_load = total_group_load / len(active_sources)
                            min_c = min(s.min_capacity for s in active_sources)
                            
                            if avg_load >= min_c:
                                for s in group_sources:
                                    if s in active_sources:
                                        if mode == 'cost':
                                            s.cost_optimized_active_load = avg_load
                                        else:
                                            s.rel_optimized_active_load = avg_load
                                    else:
                                        if mode == 'cost':
                                            s.cost_optimized_active_load = 0
                                        else:
                                            s.rel_optimized_active_load = 0
                                break
                            else:
                                machine_to_shutdown = max(active_sources, key=lambda x: x.min_capacity)
                                active_sources.remove(machine_to_shutdown)
                                if mode == 'cost':
                                    machine_to_shutdown.cost_optimized_active_load = 0
                                else:
                                    machine_to_shutdown.rel_optimized_active_load = 0
                        
                        if len(active_sources) == 1:
                            remaining_machine = active_sources[0]
                            if total_group_load >= remaining_machine.min_capacity:
                                if mode == 'cost':
                                    remaining_machine.cost_optimized_active_load = total_group_load
                                else:
                                    remaining_machine.rel_optimized_active_load = total_group_load
                            else:
                                if mode == 'cost':
                                    remaining_machine.cost_optimized_active_load = remaining_machine.min_capacity
                                else:
                                    remaining_machine.rel_optimized_active_load = remaining_machine.min_capacity
                            
                            for s in group_sources:
                                if s != remaining_machine:
                                    if mode == 'cost':
                                        s.cost_optimized_active_load = 0
                                    else:
                                        s.rel_optimized_active_load = 0
                                    
                    else:
                        total_max = sum(s.effective_max if use_effective else s.max_capacity for s in group_sources)
                        
                        temp_loads = {}
                        for s in group_sources:
                            proportion = (s.effective_max if use_effective else s.max_capacity) / total_max
                            temp_loads[s] = proportion * total_group_load
                        
                        machines_to_shutdown = []
                        for s, load in temp_loads.items():
                            if load < s.min_capacity:
                                machines_to_shutdown.append(s)
                        
                        if machines_to_shutdown:
                            active_sources = [s for s in group_sources if s not in machines_to_shutdown]
                            
                            if active_sources:
                                remaining_total_max = sum(s.effective_max if use_effective else s.max_capacity for s in active_sources)
                                for s in group_sources:
                                    if s in active_sources:
                                        proportion = (s.effective_max if use_effective else s.max_capacity) / remaining_total_max
                                        new_load = proportion * total_group_load
                                        if mode == 'cost':
                                            s.cost_optimized_active_load = max(new_load, s.min_capacity)
                                        else:
                                            s.rel_optimized_active_load = max(new_load, s.min_capacity)
                                    else:
                                        if mode == 'cost':
                                            s.cost_optimized_active_load = 0
                                        else:
                                            s.rel_optimized_active_load = 0
                            else:
                                best_machine = min(group_sources, key=lambda x: x.min_capacity)
                                if mode == 'cost':
                                    best_machine.cost_optimized_active_load = max(total_group_load, best_machine.min_capacity)
                                else:
                                    best_machine.rel_optimized_active_load = max(total_group_load, best_machine.min_capacity)
                                for s in group_sources:
                                    if s != best_machine:
                                        if mode == 'cost':
                                            s.cost_optimized_active_load = 0
                                        else:
                                            s.rel_optimized_active_load = 0
                        else:
                            for s in group_sources:
                                if mode == 'cost':
                                    s.cost_optimized_active_load = temp_loads[s]
                                else:
                                    s.rel_optimized_active_load = temp_loads[s]
                    
                    active_sources = [s for s in group_sources if (s.cost_optimized_active_load if mode == 'cost' else s.rel_optimized_active_load) > 0]
                    number_needed = len(active_sources)
                    desired = number_needed + redundancy_level
                    if desired > len(group_sources):
                        desired = len(group_sources)
                    additional = desired - number_needed
                    if additional > 0:
                        shut_off = [s for s in group_sources if (s.cost_optimized_active_load if mode == 'cost' else s.rel_optimized_active_load) == 0]
                        additional = min(additional, len(shut_off))
                        for i in range(additional):
                            extra = shut_off[i]
                            if mode == 'cost':
                                extra.cost_optimized_active_load = extra.min_capacity
                            else:
                                extra.rel_optimized_active_load = extra.min_capacity
                        print(f"Added {additional} redundant units for {group_name} at min capacity")
                
                print(f"Load sharing applied to {group_name} sources")
    
    def generate_cost_results(self):
        """Generate results for cost optimization"""
        results = []
        total_current_load = 0
        total_optimized_load = 0
        total_current_cost = 0
        total_optimized_cost = 0
        total_current_kvar = 0
        total_optimized_kvar = 0
        total_grid_feed = 0

        for source in self.sources:
            current_load = source.current_active_load
            optimized_load = source.cost_optimized_active_load
            cost_per_kwh = source.total_cost

            current_kvar = source.current_reactive_load
            optimized_kvar = source.cost_optimized_reactive_load

            current_cost_hr = current_load * cost_per_kwh
            
            if source.name.lower() == 'grid':
                effective_optimized = optimized_load - source.cost_grid_feed_power
                optimized_cost_hr = max(0, effective_optimized) * cost_per_kwh
                total_grid_feed += source.cost_grid_feed_power
            else:
                effective_optimized = optimized_load
                optimized_cost_hr = optimized_load * cost_per_kwh

            row = {
                'ENERGY SOURCE': source.name,
                'CURRENT LOAD (kW)': round(current_load, 2),
                'OPTIMIZED LOAD (kW)': round(effective_optimized, 2),
                'CURRENT KVAR (kVAR)': round(current_kvar, 2),
                'OPTIMIZED KVAR (kVAR)': round(optimized_kvar, 2),
                'TOTAL COST (PKR/kWh)': round(cost_per_kwh, 2),
                'PRODUCTION COST (PKR/kWh)': round(source.production_cost, 2),
                'CARBON COST (PKR/kWh)': round(source.carbon_emission * self.carbon_cost_pkr_kg, 2),
                'CURRENT COST/HR': round(current_cost_hr, 2),
                'OPTIMIZED COST/HR': round(optimized_cost_hr, 2),
                'OPTIMIZED CHARGE (kW)': 0.0,
                'OPTIMIZED DISCHARGE (kW)': 0.0,
                'GRID FEED (kW)': round(source.cost_grid_feed_power, 2),
                'RELIABILITY SCORE': round(source.reliability_score, 2),
                'EFFICIENCY SCORE': round(self.calculate_efficiency_score(source), 2),
                'STATUS': 'Active' if source.available else 'Inactive'
            }
            
            results.append(row)
            
            total_current_load += current_load
            total_optimized_load += effective_optimized
            total_current_cost += current_cost_hr
            total_optimized_cost += optimized_cost_hr
            total_current_kvar += current_kvar
            total_optimized_kvar += optimized_kvar
            
        for bess in self.bess_systems:
            current_bess_discharge = abs(bess.current_power_input) if bess.current_power_input < 0 else 0
            optimized_bess_load = bess.cost_optimized_discharge_power

            try:
                current_kvar = bess.current_reactive_load
            except AttributeError:
                current_kvar = 0

            try:
                optimized_kvar = bess.cost_optimized_reactive_load
            except AttributeError:
                optimized_kvar = 0

            current_cost_hr = bess.get_current_operating_cost()
            optimized_cost_hr = bess.get_optimized_operating_cost(mode='cost')
            
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
                'OPTIMIZED CHARGE (kW)': round(bess.cost_optimized_charge_power, 2),
                'OPTIMIZED DISCHARGE (kW)': round(bess.cost_optimized_discharge_power, 2),
                'GRID FEED (kW)': 0.0,
                'RELIABILITY SCORE': round(bess.reliability_score, 2),
                'EFFICIENCY SCORE': round(0, 2),  # Placeholder
                'STATUS': status
            }
            
            results.append(row)
            
            total_current_load += current_bess_discharge
            total_optimized_load += optimized_bess_load
            total_current_cost += current_cost_hr
            total_optimized_cost += optimized_cost_hr
            total_current_kvar += current_kvar
            total_optimized_kvar += optimized_kvar

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
            'OPTIMIZED CHARGE (kW)': round(sum(b.cost_optimized_charge_power for b in self.bess_systems), 2),
            'OPTIMIZED DISCHARGE (kW)': round(sum(b.cost_optimized_discharge_power for b in self.bess_systems), 2),
            'GRID FEED (kW)': round(total_grid_feed, 2),
            'RELIABILITY SCORE': '',
            'EFFICIENCY SCORE': '',
            'STATUS': f'Cost Savings: PKR {total_savings_hr:.2f}/hr'
        })

        df = pd.DataFrame(results)
        df['EFFICIENCY SCORE'] = pd.to_numeric(df['EFFICIENCY SCORE'], errors='coerce')
        df['priority'] = df['EFFICIENCY SCORE'].rank(method='min').fillna(0).astype(int)

        return df, total_current_cost, total_savings_hr

    def generate_reliability_results(self):
        """Generate results for reliability optimization"""
        results = []
        total_current_load = 0
        total_optimized_load = 0
        total_current_cost = 0
        total_optimized_cost = 0
        total_current_kvar = 0
        total_optimized_kvar = 0
        total_grid_feed = 0

        for source in self.sources:
            current_load = source.current_active_load
            optimized_load = source.rel_optimized_active_load
            cost_per_kwh = source.total_cost

            current_kvar = source.current_reactive_load
            optimized_kvar = source.rel_optimized_reactive_load

            current_cost_hr = current_load * cost_per_kwh
            
            if source.name.lower() == 'grid':
                effective_optimized = optimized_load - source.rel_grid_feed_power
                optimized_cost_hr = max(0, effective_optimized) * cost_per_kwh
                total_grid_feed += source.rel_grid_feed_power
            else:
                effective_optimized = optimized_load
                optimized_cost_hr = optimized_load * cost_per_kwh

            row = {
                'ENERGY SOURCE': source.name,
                'CURRENT LOAD (kW)': round(current_load, 2),
                'OPTIMIZED LOAD (kW)': round(effective_optimized, 2),
                'CURRENT KVAR (kVAR)': round(current_kvar, 2),
                'OPTIMIZED KVAR (kVAR)': round(optimized_kvar, 2),
                'TOTAL COST (PKR/kWh)': round(cost_per_kwh, 2),
                'PRODUCTION COST (PKR/kWh)': round(source.production_cost, 2),
                'CARBON COST (PKR/kWh)': round(source.carbon_emission * self.carbon_cost_pkr_kg, 2),
                'CURRENT COST/HR': round(current_cost_hr, 2),
                'OPTIMIZED COST/HR': round(optimized_cost_hr, 2),
                'OPTIMIZED CHARGE (kW)': 0.0,
                'OPTIMIZED DISCHARGE (kW)': 0.0,
                'GRID FEED (kW)': round(source.rel_grid_feed_power, 2),
                'RELIABILITY SCORE': round(source.reliability_score, 2),
                'EFFICIENCY SCORE': round(self.calculate_efficiency_score(source), 2),
                'STATUS': 'Active' if source.available else 'Inactive'
            }
            
            results.append(row)
            
            total_current_load += current_load
            total_optimized_load += effective_optimized
            total_current_cost += current_cost_hr
            total_optimized_cost += optimized_cost_hr
            total_current_kvar += current_kvar
            total_optimized_kvar += optimized_kvar
            
        for bess in self.bess_systems:
            current_bess_discharge = abs(bess.current_power_input) if bess.current_power_input < 0 else 0
            optimized_bess_load = bess.rel_optimized_discharge_power

            try:
                current_kvar = bess.current_reactive_load
            except AttributeError:
                current_kvar = 0

            try:
                optimized_kvar = bess.rel_optimized_reactive_load
            except AttributeError:
                optimized_kvar = 0

            current_cost_hr = bess.get_current_operating_cost()
            optimized_cost_hr = bess.get_optimized_operating_cost(mode='reliability')
            
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
                'OPTIMIZED CHARGE (kW)': round(bess.rel_optimized_charge_power, 2),
                'OPTIMIZED DISCHARGE (kW)': round(bess.rel_optimized_discharge_power, 2),
                'GRID FEED (kW)': 0.0,
                'RELIABILITY SCORE': round(bess.reliability_score, 2),
                'EFFICIENCY SCORE': round(0, 2),  # Placeholder
                'STATUS': status
            }
            
            results.append(row)
            
            total_current_load += current_bess_discharge
            total_optimized_load += optimized_bess_load
            total_current_cost += current_cost_hr
            total_optimized_cost += optimized_cost_hr
            total_current_kvar += current_kvar
            total_optimized_kvar += optimized_kvar

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
            'OPTIMIZED CHARGE (kW)': round(sum(b.rel_optimized_charge_power for b in self.bess_systems), 2),
            'OPTIMIZED DISCHARGE (kW)': round(sum(b.rel_optimized_discharge_power for b in self.bess_systems), 2),
            'GRID FEED (kW)': round(total_grid_feed, 2),
            'RELIABILITY SCORE': '',
            'EFFICIENCY SCORE': '',
            'STATUS': f'Reliability Savings: PKR {total_savings_hr:.2f}/hr'
        })

        df = pd.DataFrame(results)
        df['EFFICIENCY SCORE'] = pd.to_numeric(df['EFFICIENCY SCORE'], errors='coerce')
        df['priority'] = df['EFFICIENCY SCORE'].rank(method='min').fillna(0).astype(int)

        return df, total_current_cost, total_savings_hr

    def generate_results(self):
        """Generate comprehensive results including both cost and reliability optimizations"""
        self.optimize_power_allocation_cost()
        cost_results_df, cost_total_current_cost, cost_total_savings_hr = self.generate_cost_results()
        
        self.optimize_power_allocation_reliability()
        rel_results_df, rel_total_current_cost, rel_total_savings_hr = self.generate_reliability_results()
        
        results = []
        total_current_load = 0
        total_cost_optimized_load = 0
        total_rel_optimized_load = 0
        total_current_cost = 0
        total_cost_optimized_cost = 0
        total_rel_optimized_cost = 0
        total_current_kvar = 0
        total_cost_optimized_kvar = 0
        total_rel_optimized_kvar = 0
        total_cost_grid_feed = 0
        total_rel_grid_feed = 0
        total_cost_bess_charge = 0
        total_cost_bess_discharge = 0
        total_rel_bess_charge = 0
        total_rel_bess_discharge = 0

        for source in self.sources:
            current_load = source.current_active_load
            cost_optimized_load = source.cost_optimized_active_load
            rel_optimized_load = source.rel_optimized_active_load
            cost_per_kwh = source.total_cost

            current_kvar = source.current_reactive_load
            cost_optimized_kvar = source.cost_optimized_reactive_load
            rel_optimized_kvar = source.rel_optimized_reactive_load

            current_cost_hr = current_load * cost_per_kwh
            
            if source.name.lower() == 'grid':
                cost_effective_optimized = cost_optimized_load - source.cost_grid_feed_power
                rel_effective_optimized = rel_optimized_load - source.rel_grid_feed_power
                cost_optimized_cost_hr = max(0, cost_effective_optimized) * cost_per_kwh
                rel_optimized_cost_hr = max(0, rel_effective_optimized) * cost_per_kwh
                total_cost_grid_feed += source.cost_grid_feed_power
                total_rel_grid_feed += source.rel_grid_feed_power
            else:
                cost_effective_optimized = cost_optimized_load
                rel_effective_optimized = rel_optimized_load
                cost_optimized_cost_hr = cost_optimized_load * cost_per_kwh
                rel_optimized_cost_hr = rel_optimized_load * cost_per_kwh

            row = {
                'ENERGY SOURCE': source.name,
                'CURRENT LOAD (kW)': round(current_load, 2),
                'COST OPT LOAD (kW)': round(cost_effective_optimized, 2),
                'REL OPT LOAD (kW)': round(rel_effective_optimized, 2),
                'CURRENT KVAR (kVAR)': round(current_kvar, 2),
                'COST OPT KVAR (kVAR)': round(cost_optimized_kvar, 2),
                'REL OPT KVAR (kVAR)': round(rel_optimized_kvar, 2),
                'TOTAL COST (PKR/kWh)': round(cost_per_kwh, 2),
                'PRODUCTION COST (PKR/kWh)': round(source.production_cost, 2),
                'CARBON COST (PKR/kWh)': round(source.carbon_emission * self.carbon_cost_pkr_kg, 2),
                'CURRENT COST/HR': round(current_cost_hr, 2),
                'COST OPT COST/HR': round(cost_optimized_cost_hr, 2),
                'REL OPT COST/HR': round(rel_optimized_cost_hr, 2),
                'COST OPT CHARGE (kW)': 0.0,
                'COST OPT DISCHARGE (kW)': 0.0,
                'REL OPT CHARGE (kW)': 0.0,
                'REL OPT DISCHARGE (kW)': 0.0,
                'COST GRID FEED (kW)': round(source.cost_grid_feed_power, 2),
                'REL GRID FEED (kW)': round(source.rel_grid_feed_power, 2),
                'RELIABILITY SCORE': round(source.reliability_score, 2),
                'EFFICIENCY SCORE': round(self.calculate_efficiency_score(source), 2),
                'STATUS': 'Active' if source.available else 'Inactive'
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
        
        for bess in self.bess_systems:
            current_bess_discharge = abs(bess.current_power_input) if bess.current_power_input < 0 else 0
            cost_optimized_bess_load = bess.cost_optimized_discharge_power
            rel_optimized_bess_load = bess.rel_optimized_discharge_power

            try:
                current_kvar = bess.current_reactive_load
            except AttributeError:
                current_kvar = 0

            try:
                cost_optimized_kvar = bess.cost_optimized_reactive_load
            except AttributeError:
                cost_optimized_kvar = 0

            try:
                rel_optimized_kvar = bess.rel_optimized_reactive_load
            except AttributeError:
                rel_optimized_kvar = 0

            current_cost_hr = bess.get_current_operating_cost()
            cost_optimized_cost_hr = bess.get_optimized_operating_cost(mode='cost')
            rel_optimized_cost_hr = bess.get_optimized_operating_cost(mode='reliability')
            
            if bess.mode == 'charging':
                status = f'Charging (SOC: {bess.current_soc}%)'
            elif bess.mode == 'discharging':
                status = f'Discharging (SOC: {bess.current_soc}%)'
            else:
                status = f'Standby (SOC: {bess.current_soc}%)'
            
            row = {
                'ENERGY SOURCE': bess.name,
                'CURRENT LOAD (kW)': round(current_bess_discharge, 2),
                'COST OPT LOAD (kW)': round(cost_optimized_bess_load, 2),
                'REL OPT LOAD (kW)': round(rel_optimized_bess_load, 2),
                'CURRENT KVAR (kVAR)': round(current_kvar, 2),
                'COST OPT KVAR (kVAR)': round(cost_optimized_kvar, 2),
                'REL OPT KVAR (kVAR)': round(rel_optimized_kvar, 2),
                'TOTAL COST (PKR/kWh)': round(bess.total_cost, 2),
                'PRODUCTION COST (PKR/kWh)': round(bess.production_cost, 2),
                'CARBON COST (PKR/kWh)': round(bess.carbon_emission * self.carbon_cost_pkr_kg, 2),
                'CURRENT COST/HR': round(current_cost_hr, 2),
                'COST OPT COST/HR': round(cost_optimized_cost_hr, 2),
                'REL OPT COST/HR': round(rel_optimized_cost_hr, 2),
                'COST OPT CHARGE (kW)': round(bess.cost_optimized_charge_power, 2),
                'COST OPT DISCHARGE (kW)': round(bess.cost_optimized_discharge_power, 2),
                'REL OPT CHARGE (kW)': round(bess.rel_optimized_charge_power, 2),
                'REL OPT DISCHARGE (kW)': round(bess.rel_optimized_discharge_power, 2),
                'COST GRID FEED (kW)': 0.0,
                'REL GRID FEED (kW)': 0.0,
                'RELIABILITY SCORE': round(bess.reliability_score, 2),
                'EFFICIENCY SCORE': round(0, 2),  # Placeholder
                'STATUS': status
            }
            
            results.append(row)
            
            total_current_load += current_bess_discharge
            total_cost_optimized_load += cost_optimized_bess_load
            total_rel_optimized_load += rel_optimized_bess_load
            total_current_cost += current_cost_hr
            total_cost_optimized_cost += cost_optimized_cost_hr
            total_rel_optimized_cost += rel_optimized_cost_hr
            total_current_kvar += current_kvar
            total_cost_optimized_kvar += cost_optimized_kvar
            total_rel_optimized_kvar += rel_optimized_kvar
            total_cost_bess_charge += bess.cost_optimized_charge_power
            total_cost_bess_discharge += bess.cost_optimized_discharge_power
            total_rel_bess_charge += bess.rel_optimized_charge_power
            total_rel_bess_discharge += bess.rel_optimized_discharge_power

        results.append({
            'ENERGY SOURCE': 'TOTAL',
            'CURRENT LOAD (kW)': round(total_current_load, 2),
            'COST OPT LOAD (kW)': round(total_cost_optimized_load, 2),
            'REL OPT LOAD (kW)': round(total_rel_optimized_load, 2),
            'CURRENT KVAR (kVAR)': round(total_current_kvar, 2),
            'COST OPT KVAR (kVAR)': round(total_cost_optimized_kvar, 2),
            'REL OPT KVAR (kVAR)': round(total_rel_optimized_kvar, 2),
            'TOTAL COST (PKR/kWh)': '',
            'PRODUCTION COST (PKR/kWh)': '',
            'CARBON COST (PKR/kWh)': '',
            'CURRENT COST/HR': round(total_current_cost, 2),
            'COST OPT COST/HR': round(total_cost_optimized_cost, 2),
            'REL OPT COST/HR': round(total_rel_optimized_cost, 2),
            'COST OPT CHARGE (kW)': round(total_cost_bess_charge, 2),
            'COST OPT DISCHARGE (kW)': round(total_cost_bess_discharge, 2),
            'REL OPT CHARGE (kW)': round(total_rel_bess_charge, 2),
            'REL OPT DISCHARGE (kW)': round(total_rel_bess_discharge, 2),
            'COST GRID FEED (kW)': round(total_cost_grid_feed, 2),
            'REL GRID FEED (kW)': round(total_rel_grid_feed, 2),
            'RELIABILITY SCORE': '',
            'EFFICIENCY SCORE': '',
            'STATUS': f'Cost Savings: PKR {cost_total_savings_hr:.2f}/hr, Rel Savings: PKR {rel_total_savings_hr:.2f}/hr'
        })

        df_r = pd.DataFrame(results)
        df_r['EFFICIENCY SCORE'] = pd.to_numeric(df_r['EFFICIENCY SCORE'], errors='coerce')
        df_r['priority'] = df_r['EFFICIENCY SCORE'].rank(method='min').fillna(0).astype(int)

        cost_opt_cost = df_r[df_r['ENERGY SOURCE'] == 'TOTAL']['COST OPT COST/HR'].values[0]
        rel_opt_cost = df_r[df_r['ENERGY SOURCE'] == 'TOTAL']['REL OPT COST/HR'].values[0]
        cost_diff = rel_opt_cost - cost_opt_cost
        total_loss = self.calculate_total_loss()
        
        if total_loss > cost_diff:
            self.optimized_mode = 'reliability'
            for source in self.sources:
                source.optimized_active_load = source.rel_optimized_active_load
                source.optimized_reactive_load = source.rel_optimized_reactive_load
                source.grid_feed_power = source.rel_grid_feed_power
            for bess in self.bess_systems:
                bess.optimized_charge_power = bess.rel_optimized_charge_power
                bess.optimized_discharge_power = bess.rel_optimized_discharge_power
            self.grid_feed = self.rel_grid_feed
            total_optimized_cost = total_rel_optimized_cost
            total_savings_hr = rel_total_savings_hr
        else:
            self.optimized_mode = 'cost'
            for source in self.sources:
                source.optimized_active_load = source.cost_optimized_active_load
                source.optimized_reactive_load = source.cost_optimized_reactive_load
                source.grid_feed_power = source.cost_grid_feed_power
            for bess in self.bess_systems:
                bess.optimized_charge_power = bess.cost_optimized_charge_power
                bess.optimized_discharge_power = bess.cost_optimized_discharge_power
            self.grid_feed = self.cost_grid_feed
            total_optimized_cost = total_cost_optimized_cost
            total_savings_hr = cost_total_savings_hr

        return df_r, total_current_cost, total_savings_hr