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
        # self.optimized_charge_power = 0
        # self.optimized_discharge_power = 0

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
    
    # def get_optimized_operating_cost(self):
    #     """Calculate optimized operating cost"""
    #     if self.optimized_charge_power > 0:
    #         return self.optimized_charge_power * self.total_cost
    #     elif self.optimized_discharge_power > 0:
    #         return self.optimized_discharge_power * self.total_cost
    #     return 0

    def get_cost_optimized_operating_cost(self):
        """Calculate optimized operating cost"""
        if self.cost_optimized_charge_power > 0:
            return self.cost_optimized_charge_power * self.total_cost
        elif self.cost_optimized_discharge_power > 0:
            return self.cost_optimized_discharge_power * self.total_cost
        return 0
    
    def get_rel_optimized_operating_cost(self):
        """Calculate reliability optimized operating cost"""
        if self.rel_optimized_charge_power > 0:
            return self.rel_optimized_charge_power * self.total_cost
        elif self.rel_optimized_discharge_power > 0:
            return self.rel_optimized_discharge_power * self.total_cost
        return 0




class PowerSource:
    """Power source class with optimization capabilities"""
    def __init__(self, name, production_cost, carbon_emission, total_cost, min_capacity, max_capacity, 
                 power_reading=0, reactive_power_reading=0, availability_grid=True, source_type='conventional', reliability_score=10.0,
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
        self.grid_available = availability_grid

        self.current_active_load = power_reading
        self.current_reactive_load = reactive_power_reading
        
        self.optimized_cost_active_load = 0
        self.optimized_cost_reactive_load = 0

        self.optimized_rel_active_load = 0
        self.optimized_rel_reactive_load = 0
        
        self.grid_feed_power = 0 if source_type == 'grid' else 0
        
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
            # self.available = True
            self.available = self.grid_available
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
            'grid_available': None
        }
        
        self.grid_connected = True
        # self.allow_grid_feed = True
        self.allow_grid_feed = False
        self.grid_feed_limit = 1000
        
        self.total_load_demand = 0
        self.total_reactive_demand = 0
        
        self.max_peak_load = None
        self.critical_load = None
        self.tripping_cost = None
        self.production_loss_hourly = None
        
        self.very_high_threshold = 1000000
        self.high_threshold = 100000
        self.critical_threshold = 500000  # 5 lacs threshold for enhanced reliability mode
        self.redundancy_level = 0
        self.prefer_bess = False
        self.enhanced_reliability_mode = False  # New flag for enhanced reliability mode

        self.total_generation = 0
        self.total_load = 0
        self.total_charging = 0
        
        self.grid_feed = 0

    def set_global_params(self, grid_available=True, wind_speed=None, fuel_pressure=None, fuel_level=None, gas_pressure=None, ghi=None):
        """Set global operational parameters"""
        self.global_params.update({
            'wind_speed': wind_speed,
            'fuel_pressure': fuel_pressure,
            'fuel_level': fuel_level,
            'gas_pressure': gas_pressure,
            'ghi': ghi,
            'grid_available': grid_available
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
                ghi=self.global_params['ghi'] if config['name'].lower().startswith('solar') else None,
                availability_grid=self.global_params['grid_available'] if config['name'].lower() == 'grid' else True
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
            print('BESS Discharge Threshold:', bess.discharge_threshold)
            bess.current_power_input = float(config.get('power_input', 0))
            print('BESS Current Power Input:', bess.current_power_input)
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

    def optimize_power_allocation_cost(self):
        """Cost-optimized power allocation (original)"""
        available_sources = [s for s in self.sources if s.available]
        
        if not available_sources:
            print("No available power sources")
            return
        
        total_active_load = sum(s.current_active_load for s in self.sources)
        total_reactive_load = sum(s.current_reactive_load for s in self.sources)

        # Add BESS current contribution to total load
        total_bess_discharge = sum(abs(b.current_power_input) for b in self.bess_systems if b.current_power_input < 0)
        total_active_load += total_bess_discharge

        total_bess_charge = sum(b.current_power_input for b in self.bess_systems if b.current_power_input > 0)
        total_active_load -= total_bess_charge

        # print charge and discharge
        print(f"Total BESS Discharge Contribution: {total_bess_discharge:.2f} kW")
        print(f"Total BESS Charge Contribution: {total_bess_charge:.2f} kW")
        
        print(f"Total active load (including BESS): {total_active_load:.2f} kW")
        print(f"Total reactive load: {total_reactive_load:.2f} kVAR")
        
        # Calculate efficiency scores for optimization
        for source in available_sources:
            source.efficiency_score = self.calculate_efficiency_score(source)
        
        # Sort sources by efficiency (lowest cost first)
        available_sources.sort(key=lambda x: x.efficiency_score)
        
        # Reset optimized loads
        for source in available_sources:
            source.optimized_cost_active_load = 0
            source.optimized_cost_reactive_load = 0
        
        # Optimize BESS operation first
        self.optimize_bess_operation_cost()


        charge_power = sum(b.cost_optimized_charge_power for b in self.bess_systems)

        # solar_power_max= sum(s.effective_max for s in self.sources if s.name.lower().startswith('solar'))
        # wind_power_max= sum(s.effective_max for s in self.sources if s.name.lower().startswith('wind'))
        # renewable_power= solar_power_max + wind_power_max

        # if charge_power > renewable_power:
        #     for bess in self.bess_systems:
        #         bess.cost_optimized_charge_power = 0.4*renewable_power/len(self.bess_systems)

        # # now accomadate the cost_optimzed_charge in the load
        # total_active_load += sum(b.cost_optimized_charge_power for b in self.bess_systems)
        # remaining_active_load = total_active_load


        solar_power_max= sum(s.effective_max for s in self.sources if s.name.lower().startswith('solar'))
        
        if charge_power > solar_power_max:
            to_charge = 0.5*solar_power_max
            to_charge = max(to_charge, max(b.current_power_input for b in self.bess_systems))
            for bess in self.bess_systems:
                bess.cost_optimized_charge_power = to_charge/len(self.bess_systems)
     

        else:
            to_charge= sum(b.cost_optimized_charge_power for b in self.bess_systems)
        # total_active_load += sum(b.rel_optimized_charge_power for b in self.bess_systems)
        total_active_load += to_charge
        

       


        # Calculate remaining load after BESS optimization
        total_bess_optimized_discharge = sum(b.cost_optimized_discharge_power for b in self.bess_systems)
        remaining_active_load = total_active_load - total_bess_optimized_discharge
        
        # Allocate remaining load to sources
        unallocated_load = self.allocate_cost_active_power(available_sources, remaining_active_load, use_effective=False)
        
        # Apply load sharing
        self.apply_load_sharing_cost(available_sources, redundancy_level=0, use_effective=False)
        
        # Validate and balance final allocation
        self.validate_and_balance_cost_allocation(available_sources, remaining_active_load)
        
        kvar_sources = [s for s in self.sources]
        kvar_sources.extend(self.bess_systems)
        print('The KVAR sources are:')
        for kvar in kvar_sources:
            print(f" - {kvar.name}")

        self.allocate_cost_reactive_power(kvar_sources, total_reactive_load)
        
        if not self.grid_connected:
            self.handle_off_grid_operation_cost(available_sources)

    def validate_and_balance_cost_allocation(self, sources, target_load):
        """Validate that allocated load matches target and balance if needed"""
        total_allocated = sum(s.optimized_cost_active_load for s in sources)
        difference = target_load - total_allocated
        
        print(f"Target load: {target_load:.2f} kW")
        print(f"Total allocated: {total_allocated:.2f} kW")
        print(f"Difference: {difference:.2f} kW")
        
        if abs(difference) > 0.01:  # Tolerance of 0.01 kW
            if difference > 0:  # Need to allocate more
                print(f"Allocating additional {difference:.2f} kW")
                self.allocate_additional_load_cost(sources, difference)
            else:  # Need to reduce allocation
                print(f"Reducing allocation by {abs(difference):.2f} kW")
                self.reduce_allocation_cost(sources, abs(difference))

    def allocate_additional_load_cost(self, sources, additional_load):
        """Allocate additional load to available sources"""
        remaining = additional_load
        
        # Try to allocate to sources with available capacity
        for source in sources:
            if remaining <= 0:
                break
            available_capacity = source.max_capacity - source.optimized_cost_active_load
            if available_capacity > 0:
                allocation = min(remaining, available_capacity)
                source.optimized_cost_active_load += allocation
                remaining -= allocation
                print(f"Added {allocation:.2f} kW to {source.name}")
        
        if remaining > 0:
            print(f"WARNING: Could not allocate {remaining:.2f} kW - insufficient capacity")

    def reduce_allocation_cost(self, sources, excess_load):
        """Reduce allocation from sources while respecting minimum capacity"""
        remaining = excess_load
        
        # Reduce from sources with load above minimum capacity
        sources_sorted = sorted(sources, key=lambda x: x.optimized_cost_active_load - x.min_capacity, reverse=True)
        
        for source in sources_sorted:
            if remaining <= 0:
                break
            reducible = source.optimized_cost_active_load - source.min_capacity
            if reducible > 0:
                reduction = min(remaining, reducible)
                source.optimized_cost_active_load -= reduction
                remaining -= reduction
                print(f"Reduced {reduction:.2f} kW from {source.name}")
        
        if remaining > 0:
            print(f"WARNING: Could not reduce {remaining:.2f} kW - minimum capacity constraints")

    # def optimize_bess_operation(self):
    #     """Optimize BESS charging/discharging strategy"""
    #     for bess in self.bess_systems:
    #         print(f"\nOptimizing BESS: {bess.name}")
    #         print(f"Current SOC: {bess.current_soc}%")
    #         print(f"Discharge Threshold: {bess.discharge_threshold}%")
    #         print(f"Charge Threshold: {bess.charge_threshold}%")
            
    #         # Reset optimized values
    #         # bess.optimized_charge_power = 0
    #         # bess.optimized_discharge_power = 0

    #         bess.optimized_charge_power = 0
    #         bess.optimized_discharge_power = 0
            
    #         # Determine optimal operation mode
    #         if bess.current_soc >= bess.discharge_threshold:
    #             # Should discharge to provide power
    #             max_discharge = min(
    #                 bess.power_rating_kw,
    #                 bess.get_available_discharge_capacity()
    #             )
    #             bess.optimized_discharge_power = max_discharge
    #             bess.mode = 'discharging'
    #             print(f"BESS {bess.name} optimized for discharging: {max_discharge:.2f} kW")
                
    #         elif bess.current_soc <= bess.charge_threshold:
    #             # Should charge when excess power available
    #             max_charge = min(
    #                 bess.power_rating_kw,
    #                 bess.get_available_charge_capacity()
    #             )
    #             print(f"For BESS {bess.name}, the maximum charge power is: {max_charge:.2f} kW")
    #             print(f"For BESS {bess.name}, the available charge capacity is: {bess.get_available_charge_capacity():.2f} kWh")
    #             print(f"For BESS {bess.name}, the current power input is: {bess.current_power_input:.2f} kW")
    #             print(f"For BESS {bess.name}, the power rating is: {bess.power_rating_kw:.2f} kW")
    #             bess.optimized_charge_power = max_charge
    #             bess.mode = 'charging'
    #             print(f"BESS {bess.name} optimized for charging: {max_charge:.2f} kW")
                
    #         else:
    #             # Maintain current operation or standby
    #             bess.mode = 'standby'
    #             print(f"BESS {bess.name} in standby mode")

    def optimize_bess_operation_rel(self):
        """Optimize BESS charging/discharging strategy"""
        for bess in self.bess_systems:
            print(f"\nOptimizing BESS: {bess.name}")
            print(f"Current SOC: {bess.current_soc}%")
            print(f"Discharge Threshold: {bess.discharge_threshold}%")
            print(f"Charge Threshold: {bess.charge_threshold}%")
            
            # Reset optimized values
            # bess.optimized_charge_power = 0
            # bess.optimized_discharge_power = 0

            bess.rel_optimized_charge_power = 0
            bess.rel_optimized_discharge_power = 0

            # Determine optimal operation mode
            if bess.current_soc >= bess.discharge_threshold:
                # Should discharge to provide power
                max_discharge = min(
                    bess.power_rating_kw,
                    bess.get_available_discharge_capacity()
                )
                bess.rel_optimized_discharge_power = max_discharge
                bess.mode = 'discharging'
                print(f"BESS {bess.name} optimized for discharging: {max_discharge:.2f} kW")
                
            elif bess.current_soc <= bess.charge_threshold:
                # Should charge when excess power available
                max_charge = min(
                    bess.power_rating_kw,
                    bess.get_available_charge_capacity()
                )
                print(f"For BESS {bess.name}, the maximum charge power is: {max_charge:.2f} kW")
                print(f"For BESS {bess.name}, the available charge capacity is: {bess.get_available_charge_capacity():.2f} kWh")
                print(f"For BESS {bess.name}, the current power input is: {bess.current_power_input:.2f} kW")
                print(f"For BESS {bess.name}, the power rating is: {bess.power_rating_kw:.2f} kW")
                bess.rel_optimized_charge_power = max_charge
                bess.mode = 'charging'
                print(f"BESS {bess.name} optimized for charging: {max_charge:.2f} kW")
                
            else:
                # Maintain current operation or standby
                bess.mode = 'standby'
                print(f"BESS {bess.name} in standby mode")

    def optimize_bess_operation_cost(self):
        """Optimize BESS charging/discharging strategy"""
        for bess in self.bess_systems:
            print(f"\nOptimizing BESS: {bess.name}")
            print(f"Current SOC: {bess.current_soc}%")
            print(f"Discharge Threshold: {bess.discharge_threshold}%")
            print(f"Charge Threshold: {bess.charge_threshold}%")
            
            # Reset optimized values
            # bess.optimized_charge_power = 0
            # bess.optimized_discharge_power = 0

            bess.cost_optimized_charge_power = 0
            bess.cost_optimized_discharge_power = 0
            
            # Determine optimal operation mode
            if bess.current_soc >= bess.discharge_threshold:
                # Should discharge to provide power
                max_discharge = min(
                    bess.power_rating_kw,
                    bess.get_available_discharge_capacity()
                )
                bess.cost_optimized_discharge_power = max_discharge
                bess.mode = 'discharging'
                print(f"BESS {bess.name} optimized for discharging: {max_discharge:.2f} kW")
                
            elif bess.current_soc <= bess.charge_threshold:
                # Should charge when excess power available
                max_charge = min(
                    bess.power_rating_kw,
                    bess.get_available_charge_capacity()
                )
                print(f"For BESS {bess.name}, the maximum charge power is: {max_charge:.2f} kW")
                print(f"For BESS {bess.name}, the available charge capacity is: {bess.get_available_charge_capacity():.2f} kWh")
                print(f"For BESS {bess.name}, the current power input is: {bess.current_power_input:.2f} kW")
                print(f"For BESS {bess.name}, the power rating is: {bess.power_rating_kw:.2f} kW")
                bess.cost_optimized_charge_power = max_charge
                bess.mode = 'charging'
                print(f"BESS {bess.name} optimized for charging: {max_charge:.2f} kW")
                
            else:
                # Maintain current operation or standby
                bess.mode = 'standby'
                print(f"BESS {bess.name} in standby mode")

    # def optimize_bess_for_standby(self):
    #     """Keep BESS in standby mode for enhanced reliability"""
    #     for bess in self.bess_systems:
    #         print(f"\nKeeping BESS {bess.name} in standby for enhanced reliability")
    #         print(f"Current SOC: {bess.current_soc}%")
            
    #         # Reset optimized values - keep in standby
    #         bess.optimized_charge_power = 0
    #         bess.optimized_discharge_power = 0
    #         bess.mode = 'standby'
    #         print(f"BESS {bess.name} kept in standby mode for emergency availability")

    def optimize_bess_for_standby_rel(self):
        """Keep BESS in standby mode for enhanced reliability"""
        for bess in self.bess_systems:
            print(f"\nKeeping BESS {bess.name} in standby for enhanced reliability")
            print(f"Current SOC: {bess.current_soc}%")
            self.optimize_bess_operation_rel()
            # Reset optimized values - keep in standby
            # bess.rel_optimized_charge_power = 0
            bess.rel_optimized_discharge_power = 0
            bess.mode = 'standby'
            print(f"BESS {bess.name} kept in standby mode for emergency availability")

    def handle_off_grid_operation_cost(self, sources):
        print("\nOff-grid operation mode activated")
        
        sources = [s for s in sources if s.source_type != 'grid']
        
        total_generation = sum(s.optimized_cost_active_load for s in sources)
        total_demand = self.total_load_demand
        
        for bess in self.bess_systems:
            if bess.mode == 'discharging':
                total_generation += bess.cost_optimized_discharge_power
        
        if total_generation < total_demand:
            deficit = total_demand - total_generation
            print(f"Generation deficit in off-grid mode: {deficit:.2f} kW")
            
            for source in sources:
                if source.optimized_cost_active_load < source.effective_max:
                    additional_capacity = min(deficit, 
                                            source.effective_max - source.optimized_cost_active_load)
                    source.optimized_cost_active_load += additional_capacity
                    deficit -= additional_capacity
                    print(f"Increased {source.name} output by {additional_capacity:.2f} kW")
                    
                    if deficit <= 0:
                        break
            
            if deficit > 0:
                print(f"Load shedding required: {deficit:.2f} kW")
                for source in sources:
                    if source.optimized_cost_active_load > 0:
                        reduction = min(deficit, source.optimized_cost_active_load)
                        source.optimized_cost_active_load -= reduction
                        deficit -= reduction
                        print(f"Decreased {source.name} output by {reduction:.2f} kW")
                        
                        if deficit <= 0:
                            break

    def handle_off_grid_operation_rel(self, sources):
        print("\nOff-grid operation mode activated")
        
        sources = [s for s in sources if s.source_type != 'grid']
        
        total_generation = sum(s.optimized_rel_active_load for s in sources)
        total_demand = self.total_load_demand
        
        for bess in self.bess_systems:
            if bess.mode == 'discharging':
                total_generation += bess.rel_optimized_discharge_power
        
        if total_generation < total_demand:
            deficit = total_demand - total_generation
            print(f"Generation deficit in off-grid mode: {deficit:.2f} kW")
            
            for source in sources:
                if source.optimized_rel_active_load < source.effective_max:
                    additional_capacity = min(deficit, 
                                            source.effective_max - source.optimized_rel_active_load)
                    source.optimized_rel_active_load += additional_capacity
                    deficit -= additional_capacity
                    print(f"Increased {source.name} output by {additional_capacity:.2f} kW")
                    
                    if deficit <= 0:
                        break
            
            if deficit > 0:
                print(f"Load shedding required: {deficit:.2f} kW")

    def allocate_cost_active_power(self, sources, total_load, use_effective=False):
        remaining_load = total_load
        print('The sources are:')
        for source in sources:
            print(f" - {source.name} (Max: {source.max_capacity} kW, Min: {source.min_capacity} kW)")
        print(f"Total load: {total_load} kW")
        print(f"Remaining load: {remaining_load} kW")

        solar_sources = [s for s in sources if s.name.lower().startswith('solar')]
        non_renewable_sources = [s for s in sources if not s.name.lower().startswith('solar') and not s.name.lower().startswith('wind') and not s.name.lower().startswith('bess')]
        bess_discharging = any(b.cost_optimized_discharge_power > 0 for b in self.bess_systems)
        
        if solar_sources and total_load <= sum(s.effective_max if use_effective else s.max_capacity for s in solar_sources) and not bess_discharging:
            print("Applying solar-first optimization strategy")
            
            for source in non_renewable_sources:
                if source.name.lower() != 'grid' and source.available:
                    if remaining_load > source.min_capacity:
                        min_allocation = source.min_capacity
                        source.optimized_cost_active_load = min_allocation
                        remaining_load -= min_allocation
                        print(f"Minimum allocation to {source.name}: {min_allocation:.2f} kW")
                        break
                elif source.name.lower() == 'grid' and self.grid_connected:
                    min_allocation = 10
                    source.optimized_cost_active_load = min_allocation
                    remaining_load -= min_allocation
                    print(f"Minimum allocation to {source.name}: {min_allocation:.2f} kW")
            
            for source in solar_sources:
                if remaining_load > 0:
                    max_cap = source.effective_max if use_effective else source.max_capacity
                    allocation = min(remaining_load, max_cap)
                    source.optimized_cost_active_load = allocation
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
                    source.optimized_cost_active_load = allocation
                    remaining_load -= allocation
                    print(f"Optimized allocation to {source.name}: {allocation:.2f} kW")
                else:
                    source.optimized_cost_active_load = 0

        # Ensure any remaining load gets allocated
        if remaining_load > 0.01:
            print(f"Allocating remaining load: {remaining_load:.2f} kW")
            for source in sources:
                if remaining_load <= 0:
                    break
                available_capacity = (source.effective_max if use_effective else source.max_capacity) - source.optimized_cost_active_load
                if available_capacity > 0:
                    additional = min(remaining_load, available_capacity)
                    source.optimized_cost_active_load += additional
                    remaining_load -= additional
                    print(f"Additional allocation to {source.name}: {additional:.2f} kW")

        return remaining_load

    def allocate_rel_active_power(self, sources, total_load, use_effective=False):
        remaining_load = total_load
        print('The sources are:')
        for source in sources:
            print(f" - {source.name} (Max: {source.max_capacity} kW, Min: {source.min_capacity} kW)")
        print(f"Total load: {total_load} kW")
        print(f"Remaining load: {remaining_load} kW")

        # Enhanced reliability mode: keep grid and BESS as standby if loss > 5 lacs
        if self.enhanced_reliability_mode:
            print("Enhanced reliability mode: Prioritizing renewable sources and keeping dispatchable sources as standby")
            
            # First prioritize renewable sources
            renewable_sources = [s for s in sources if s.source_type == 'renewable' and s.available]
            conventional_sources = [s for s in sources if s.source_type == 'conventional' and s.name.lower() != 'grid' and s.available]
            grid_source = [s for s in sources if s.name.lower() == 'grid' and s.available]
            
            # Allocate to renewables first
            for source in renewable_sources:
                if remaining_load <= 0:
                    break
                max_cap = source.effective_max if use_effective else source.max_capacity
                allocation = min(remaining_load, max_cap)
                source.optimized_rel_active_load = allocation
                remaining_load -= allocation
                print(f"Renewable allocation to {source.name}: {allocation:.2f} kW")
            
            # Then allocate minimum required to conventional sources to cover remaining load
            if remaining_load > 0:
                for source in conventional_sources:
                    if remaining_load <= 0:
                        break
                    max_cap = source.effective_max if use_effective else source.max_capacity
                    min_required = source.min_capacity
                    
                    if remaining_load >= min_required:
                        allocation = min(remaining_load, max_cap)
                        source.optimized_rel_active_load = allocation
                        remaining_load -= allocation
                        print(f"Conventional allocation to {source.name}: {allocation:.2f} kW")
                    else:
                        source.optimized_rel_active_load = 0
            
            # Keep grid as minimal standby allocation (emergency backup)
            if grid_source and remaining_load > 0:
                grid_standby = min(10, remaining_load)  # Minimal grid connection for standby
                grid_source[0].optimized_rel_active_load = grid_standby
                remaining_load -= grid_standby
                print(f"Grid standby allocation: {grid_standby:.2f} kW")
            elif grid_source:
                # Even if no remaining load, keep minimal grid standby
                grid_source[0].optimized_rel_active_load = 5
                print("Grid kept at minimal standby: 5.0 kW")
        
        else:
            # Normal reliability mode logic (same as before)
            solar_sources = [s for s in sources if s.name.lower().startswith('solar')]
            non_renewable_sources = [s for s in sources if not s.name.lower().startswith('solar') and not s.name.lower().startswith('wind') and not s.name.lower().startswith('bess')]
            bess_discharging = any(b.rel_optimized_discharge_power > 0 for b in self.bess_systems)
            
            if solar_sources and total_load <= sum(s.effective_max if use_effective else s.max_capacity for s in solar_sources) and not bess_discharging:
                print("Applying solar-first optimization strategy")
                
                for source in non_renewable_sources:
                    if source.name.lower() != 'grid' and source.available:
                        if remaining_load > source.min_capacity:
                            min_allocation = source.min_capacity
                            source.optimized_rel_active_load = min_allocation
                            remaining_load -= min_allocation
                            print(f"Minimum allocation to {source.name}: {min_allocation:.2f} kW")
                            break
                    elif source.name.lower() == 'grid' and self.grid_connected:
                        min_allocation = 10
                        source.optimized_rel_active_load = min_allocation
                        remaining_load -= min_allocation
                        print(f"Minimum allocation to {source.name}: {min_allocation:.2f} kW")
                
                for source in solar_sources:
                    if remaining_load > 0:
                        max_cap = source.effective_max if use_effective else source.max_capacity
                        allocation = min(remaining_load, max_cap) 
                        source.optimized_rel_active_load = allocation
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
                        source.optimized_rel_active_load = allocation
                        remaining_load -= allocation
                        print(f"Optimized allocation to {source.name}: {allocation:.2f} kW")
                    else:
                        source.optimized_rel_active_load = 0

            # Ensure any remaining load gets allocated
            if remaining_load > 0.01:
                print(f"Allocating remaining load: {remaining_load:.2f} kW")
                for source in sources:
                    if remaining_load <= 0:
                        break
                    available_capacity = (source.effective_max if use_effective else source.max_capacity) - source.optimized_rel_active_load
                    if available_capacity > 0:
                        additional = min(remaining_load, available_capacity)
                        source.optimized_rel_active_load += additional
                        remaining_load -= additional
                        print(f"Additional allocation to {source.name}: {additional:.2f} kW")

        return remaining_load

    def optimize_power_allocation_reliability(self):
        """Reliability-optimized power allocation with enhanced mode for high losses"""
        # Reset optimized
        for source in self.sources:
            source.optimized_rel_active_load = 0
            source.optimized_rel_reactive_load = 0
            source.effective_max = source.max_capacity
            source.grid_feed_power = 0
        
        for bess in self.bess_systems:
        #     bess.optimized_charge_power = 0
        #     bess.optimized_discharge_power = 0

            bess.rel_optimized_charge_power = 0
            bess.rel_optimized_discharge_power = 0

        self.grid_feed = 0
        
        # Determine reliability mode and parameters
        total_loss = self.tripping_cost + self.production_loss_hourly
        
        # Enhanced reliability mode for losses > 5 lacs (500,000 PKR)
        if total_loss > self.critical_threshold:
            self.enhanced_reliability_mode = True
            self.redundancy_level = 2
            self.prefer_bess = False  # Keep BESS as standby
            print(f"ENHANCED RELIABILITY MODE ACTIVATED - Total Loss: PKR {total_loss:,.2f} > PKR {self.critical_threshold:,.2f}")
            print("Keeping dispatchable sources (Grid & BESS) as standby for emergency availability")
        elif total_loss > self.very_high_threshold:
            self.enhanced_reliability_mode = False
            self.redundancy_level = 2
            self.prefer_bess = True
        elif total_loss > self.high_threshold:
            self.enhanced_reliability_mode = False
            self.redundancy_level = 1
            self.prefer_bess = False
        else:
            self.enhanced_reliability_mode = False
            self.redundancy_level = 0
            self.prefer_bess = False
        
        available_sources = [s for s in self.sources if s.available]
        
        if not available_sources:
            print("No available power sources")
            return
        
        total_active_load = sum(s.current_active_load for s in self.sources)
        total_reactive_load = sum(s.current_reactive_load for s in self.sources)

        # # Add BESS current contribution to total load
        # total_bess_discharge = sum(abs(b.current_power_input) for b in self.bess_systems if b.current_power_input < 0)
        # total_active_load += total_bess_discharge

        total_bess_discharge = sum(abs(b.current_power_input) for b in self.bess_systems if b.current_power_input < 0)
        total_active_load += total_bess_discharge

        total_bess_charge = sum(b.current_power_input for b in self.bess_systems if b.current_power_input > 0)
        total_active_load -= total_bess_charge

        print(f"Total BESS Discharge Contribution: {total_bess_discharge:.2f} kW")
        print(f"Total BESS Charge Contribution: {total_bess_charge:.2f} kW")
        print(f"Total active load (including BESS): {total_active_load:.2f} kW")
        print(f"Total reactive load: {total_reactive_load:.2f} kVAR")
        
        # Calculate efficiency scores
        for source in available_sources:
            source.efficiency_score = self.calculate_efficiency_score(source)
        
        # Sort by reliability descending, then total cost ascending
        available_sources.sort(key=lambda x: x.efficiency_score)
        
        # Apply reserve for renewable variability on engines
        solar_running = sum(s.current_active_load for s in self.sources if s.name.lower().startswith('solar'))
        wind_running = sum(s.current_active_load for s in self.sources if s.name.lower().startswith('wind'))
        sum_engine_max = sum(s.max_capacity for s in self.sources if s.name.lower().startswith('diesel') or s.name.lower().startswith('gas'))
        
        sp = (0.5 * solar_running) / sum_engine_max if sum_engine_max > 0 else 0
        wp = (0.7 * wind_running) / sum_engine_max if sum_engine_max > 0 else 0
        reserve_factor = max(0.2, 1 - 0.1 - sp - wp)
        
        for source in self.sources:
            if source.name.lower().startswith('diesel') or source.name.lower().startswith('gas'):
                source.effective_max = source.max_capacity * reserve_factor
        
        # Optimize BESS based on mode
        if self.enhanced_reliability_mode:
            # Keep BESS in standby for enhanced reliability
            self.optimize_bess_for_standby_rel()
        elif self.prefer_bess:
            for bess in self.bess_systems:
                bess.check_availability()
                if bess.available:
                    max_discharge = bess.get_available_discharge_capacity()
                    bess.rel_optimized_discharge_power = min(bess.power_rating_kw, max_discharge)
                    bess.mode = 'discharging'
        else:
            self.optimize_bess_operation_rel()

        # now accomadate the cost_optimzed_charge in the load

        # charge_power = sum(b.rel_optimized_charge_power for b in self.bess_systems)

        # # if charge power is less than the solar + wind power, then we can accommodate it
        # renewable_power = sum(s.current_active_load for s in self.sources if s.source_type == 'renewable')

        # solar_power = sum(s.current_active_load for s in self.sources if s.name.lower().startswith('solar'))
        # # wind_power = sum(s.current_active_load for s in self.sources if s.name.lower().startswith('wind'))
        # solar_power_max= sum(s.effective_max for s in self.sources if s.name.lower().startswith('solar'))
        
        # if charge_power > solar_power_max:
        #     for bess in self.bess_systems:
        #         bess.rel_optimized_charge_power = 0.5*solar_power_max/len(self.bess_systems)
        # # if charge_power > renewable_power:
        #     print(f"Reducing BESS charge power from {charge_power:.2f} kW to match renewable generation of {renewable_power:.2f} kW")
        #     # Reduce charge power proportionally
        #     for bess in self.bess_systems:
        #         if bess.rel_optimized_charge_power > 0:
        #             reduction = (bess.rel_optimized_charge_power / charge_power) * (charge_power - renewable_power)
        #             bess.rel_optimized_charge_power -= reduction
        #             if bess.rel_optimized_charge_power < 0:
        #                 bess.rel_optimized_charge_power = 0
        #             print(f"BESS {bess.name} new charge power: {bess.rel_optimized_charge_power:.2f} kW")

        
        charge_power = sum(b.cost_optimized_charge_power for b in self.bess_systems)

        solar_power_max= sum(s.effective_max for s in self.sources if s.name.lower().startswith('solar'))
        # wind_power_max= sum(s.effective_max for s in self.sources if s.name.lower().startswith('wind'))
        # renewable_power= solar_power_max + wind_power_max

        if charge_power > solar_power_max:
            to_charge = 0.5*solar_power_max
            # maximum of power input and to_charge
            to_charge = max(to_charge, max(b.current_power_input for b in self.bess_systems))
            for bess in self.bess_systems:
                bess.rel_optimized_charge_power = to_charge/len(self.bess_systems)
            # for bess in self.bess_systems:
            #     bess.rel_optimized_charge_power = 500
            # #     bess.rel_optimized_charge_power = 0.4*renewable_power/len(self.bess_systems)

        else:
            to_charge= sum(b.rel_optimized_charge_power for b in self.bess_systems)
        # total_active_load += sum(b.rel_optimized_charge_power for b in self.bess_systems)
        total_active_load += to_charge
        remaining_active_load = total_active_load
        
        # Calculate remaining load after BESS optimization
        total_bess_optimized_discharge = sum(b.rel_optimized_discharge_power for b in self.bess_systems)
        remaining_active_load = total_active_load - total_bess_optimized_discharge
        
        # Allocate remaining load
        unallocated_load = self.allocate_rel_active_power(available_sources, remaining_active_load, use_effective=True)
        
        # Apply load sharing
        self.apply_load_sharing_rel(available_sources, redundancy_level=self.redundancy_level, use_effective=True)

        # Validate and balance final allocation
        self.validate_and_balance_rel_allocation(available_sources, remaining_active_load)

        kvar_sources = [s for s in self.sources]
        kvar_sources.extend(self.bess_systems)
        
        self.allocate_rel_reactive_power(kvar_sources, total_reactive_load)
        
        if not self.grid_connected:
            self.handle_off_grid_operation_rel(available_sources)

    def validate_and_balance_rel_allocation(self, sources, target_load):
        """Validate that allocated reliability load matches target and balance if needed"""
        total_allocated = sum(s.optimized_rel_active_load for s in sources)
        difference = target_load - total_allocated
        
        print(f"Target load: {target_load:.2f} kW")
        print(f"Total allocated: {total_allocated:.2f} kW")
        print(f"Difference: {difference:.2f} kW")
        
        if abs(difference) > 0.01:  # Tolerance of 0.01 kW
            if difference > 0:  # Need to allocate more
                print(f"Allocating additional {difference:.2f} kW")
                self.allocate_additional_load_rel(sources, difference)
            else:  # Need to reduce allocation
                print(f"Reducing allocation by {abs(difference):.2f} kW")
                self.reduce_allocation_rel(sources, abs(difference))

    def allocate_additional_load_rel(self, sources, additional_load):
        """Allocate additional load to available sources for reliability optimization"""
        remaining = additional_load
        
        # In enhanced reliability mode, avoid using dispatchable sources unless absolutely necessary
        if self.enhanced_reliability_mode:
            # Try non-dispatchable sources first
            non_dispatchable_sources = [s for s in sources if s.source_type != 'grid' and 'bess' not in s.name.lower()]
            for source in non_dispatchable_sources:
                if remaining <= 0:
                    break
                available_capacity = source.effective_max - source.optimized_rel_active_load
                if available_capacity > 0:
                    allocation = min(remaining, available_capacity)
                    source.optimized_rel_active_load += allocation
                    remaining -= allocation
                    print(f"Added {allocation:.2f} kW to {source.name}")
            
            # Only if absolutely necessary, use minimal dispatchable sources
            if remaining > 0:
                dispatchable_sources = [s for s in sources if s.source_type == 'grid' or 'bess' in s.name.lower()]
                for source in dispatchable_sources:
                    if remaining <= 0:
                        break
                    available_capacity = min(10, source.effective_max - source.optimized_rel_active_load)  # Limit to minimal
                    if available_capacity > 0:
                        allocation = min(remaining, available_capacity)
                        source.optimized_rel_active_load += allocation
                        remaining -= allocation
                        print(f"Minimal emergency allocation to {source.name}: {allocation:.2f} kW")
        else:
            # Normal mode - try to allocate to sources with available capacity
            for source in sources:
                if remaining <= 0:
                    break
                available_capacity = source.effective_max - source.optimized_rel_active_load
                if available_capacity > 0:
                    allocation = min(remaining, available_capacity)
                    source.optimized_rel_active_load += allocation
                    remaining -= allocation
                    print(f"Added {allocation:.2f} kW to {source.name}")
        
        # If still remaining, try to use BESS charging (only if not in enhanced mode)
        if remaining > 0 and not self.enhanced_reliability_mode:
            for bess in self.bess_systems:
                if remaining <= 0:
                    break
                bess.check_availability()
                if bess.available and bess.can_charge(remaining):
                    assignable = min(remaining, bess.power_rating_kw - bess.rel_optimized_charge_power)
                    bess.rel_optimized_charge_power += assignable
                    bess.mode = 'charging' if bess.rel_optimized_charge_power > 0 else bess.mode
                    remaining -= assignable
                    print(f"Assigned {assignable:.2f} kW to BESS {bess.name} for charging")
        
        # If still remaining, try grid feed (only if not in enhanced mode)
        if remaining > 0 and self.allow_grid_feed and not self.enhanced_reliability_mode:
            grid_feed_capacity = self.grid_feed_limit - self.grid_feed
            if grid_feed_capacity > 0:
                assignable = min(remaining, grid_feed_capacity)
                self.grid_feed += assignable
                remaining -= assignable
                print(f"Fed {assignable:.2f} kW to grid")
                
                # Update grid source if exists
                for source in self.sources:
                    if source.name.lower() == 'grid':
                        source.grid_feed_power = self.grid_feed
                        break
        
        if remaining > 0:
            print(f"WARNING: Could not allocate {remaining:.2f} kW - insufficient capacity")
            if self.enhanced_reliability_mode:
                print("Enhanced reliability mode: Keeping dispatchable sources as standby took priority")

    def reduce_allocation_rel(self, sources, excess_load):
        """Reduce allocation from sources while respecting minimum capacity"""
        remaining = excess_load
        
        # In enhanced reliability mode, reduce from non-critical sources first
        if self.enhanced_reliability_mode:
            # Sort by priority - reduce from conventional sources first, keep renewables
            sources_by_priority = []
            conventional_sources = [s for s in sources if s.source_type == 'conventional' and s.name.lower() != 'grid']
            renewable_sources = [s for s in sources if s.source_type == 'renewable']
            grid_sources = [s for s in sources if s.name.lower() == 'grid']
            
            sources_by_priority.extend(conventional_sources)
            sources_by_priority.extend(renewable_sources)
            sources_by_priority.extend(grid_sources)
            
            for source in sources_by_priority:
                if remaining <= 0:
                    break
                if source.name.lower() == 'grid':
                    # Keep minimal grid standby
                    reducible = max(0, source.optimized_rel_active_load - 5)
                else:
                    reducible = source.optimized_rel_active_load - source.min_capacity
                
                if reducible > 0:
                    reduction = min(remaining, reducible)
                    source.optimized_rel_active_load -= reduction
                    remaining -= reduction
                    print(f"Reduced {reduction:.2f} kW from {source.name}")
        else:
            # Normal mode - reduce from sources with load above minimum capacity
            sources_sorted = sorted(sources, key=lambda x: x.optimized_rel_active_load - x.min_capacity, reverse=True)
            
            for source in sources_sorted:
                if remaining <= 0:
                    break
                reducible = source.optimized_rel_active_load - source.min_capacity
                if reducible > 0:
                    reduction = min(remaining, reducible)
                    source.optimized_rel_active_load -= reduction
                    remaining -= reduction
                    print(f"Reduced {reduction:.2f} kW from {source.name}")
        
        if remaining > 0:
            print(f"WARNING: Could not reduce {remaining:.2f} kW - minimum capacity constraints")

    def allocate_cost_reactive_power(self, sources, total_reactive_load):
        """Allocate reactive power among sources"""
        remaining_reactive_load = total_reactive_load
        
        for source in sources:
            if remaining_reactive_load <= 0:
                break

            if hasattr(source, 'power_rating_kw'):  # BESS
                max_reactive = source.power_rating_kw * 0.8
                allocation = min(max_reactive, remaining_reactive_load)
                source.optimized_cost_reactive_load = allocation
                remaining_reactive_load -= allocation
            
            elif source.name.lower() != 'solar':
                max_reactive = source.optimized_cost_active_load * 0.6
                allocation = min(max_reactive, remaining_reactive_load)
                source.optimized_cost_reactive_load = allocation
                remaining_reactive_load -= allocation

            else:
                allocation = 0
                source.optimized_cost_reactive_load = allocation

            print(f"Reactive power allocation to {source.name}: {allocation:.2f} kVAR")

        if remaining_reactive_load > 0:
            print(f"Unallocated reactive power: {remaining_reactive_load:.2f} kVAR")
            grid_source = next((s for s in sources if hasattr(s, 'name') and s.name.lower() == 'grid'), None)
            if grid_source:
                grid_source.optimized_cost_reactive_load += remaining_reactive_load
                print(f"Allocated {remaining_reactive_load:.2f} kVAR to grid")

        return remaining_reactive_load

    def allocate_rel_reactive_power(self, sources, total_reactive_load):
        """Allocate reactive power among sources for reliability optimization"""
        remaining_reactive_load = total_reactive_load
        
        for source in sources:
            if remaining_reactive_load <= 0:
                break

            if hasattr(source, 'power_rating_kw'):  # BESS
                if self.enhanced_reliability_mode:
                    # Keep BESS reactive power minimal for standby
                    max_reactive = source.power_rating_kw * 0.2
                else:
                    max_reactive = source.power_rating_kw * 0.8
                allocation = min(max_reactive, remaining_reactive_load)
                source.optimized_rel_reactive_load = allocation
                remaining_reactive_load -= allocation
            
            elif source.name.lower() != 'solar':
                max_reactive = source.optimized_rel_active_load * 0.6
                allocation = min(max_reactive, remaining_reactive_load)
                source.optimized_rel_reactive_load = allocation
                remaining_reactive_load -= allocation

            else:
                allocation = 0
                source.optimized_rel_reactive_load = allocation

            print(f"Reactive power allocation to {source.name}: {allocation:.2f} kVAR")

        if remaining_reactive_load > 0:
            print(f"Unallocated reactive power: {remaining_reactive_load:.2f} kVAR")
            grid_source = next((s for s in sources if hasattr(s, 'name') and s.name.lower() == 'grid'), None)
            if grid_source and not self.enhanced_reliability_mode:
                grid_source.optimized_rel_reactive_load += remaining_reactive_load
                print(f"Allocated {remaining_reactive_load:.2f} kVAR to grid")
            elif grid_source and self.enhanced_reliability_mode:
                # In enhanced mode, only allocate minimal reactive power to grid
                minimal_reactive = min(remaining_reactive_load, 10)
                grid_source.optimized_rel_reactive_load += minimal_reactive
                remaining_reactive_load -= minimal_reactive
                print(f"Allocated minimal {minimal_reactive:.2f} kVAR to grid (enhanced reliability mode)")

        return remaining_reactive_load

    def apply_load_sharing_rel(self, sources, redundancy_level=0, use_effective=False):
        """Apply load sharing among similar sources with optional redundancy"""
        source_groups = {}
        for source in sources:
            source_type = source.name.split('_')[0] if '_' in source.name else source.name
            if source_type not in source_groups:
                source_groups[source_type] = []
            source_groups[source_type].append(source)
            
        for group_name, group_sources in source_groups.items():
            if len(group_sources) > 1:
                total_group_load = sum(s.optimized_rel_active_load for s in group_sources)
                if total_group_load > 0:
                    # Store original total for validation
                    original_total = total_group_load
                    
                    # In enhanced reliability mode, modify load sharing strategy
                    if self.enhanced_reliability_mode and (group_name.lower() == 'grid' or 'bess' in group_name.lower()):
                        # Keep dispatchable sources at minimal levels
                        print(f"Enhanced reliability mode: Minimizing {group_name} load sharing")
                        active_sources = [s for s in group_sources if s.optimized_rel_active_load > 0]
                        if active_sources:
                            # Distribute load evenly but keep minimal
                            avg_load = min(total_group_load / len(active_sources), 10)  # Cap at 10 kW each
                            for s in group_sources:
                                if s in active_sources:
                                    s.optimized_rel_active_load = avg_load
                                else:
                                    s.optimized_rel_active_load = 0
                        continue
                    
                    # Equal load sharing for similar capacity units
                    if all(abs(s.max_capacity - group_sources[0].max_capacity) < 50 for s in group_sources):
                        active_sources = group_sources.copy()
                        
                        while len(active_sources) > 1:
                            avg_load = total_group_load / len(active_sources)
                            min_c = min(s.min_capacity for s in active_sources)
                            
                            if avg_load >= min_c:
                                for s in group_sources:
                                    if s in active_sources:
                                        s.optimized_rel_active_load = avg_load*0.9 + s.min_capacity*0.1
                                    else:
                                        s.optimized_rel_active_load = 0
                                break
                            else:
                                machine_to_shutdown = max(active_sources, key=lambda x: x.min_capacity)
                                active_sources.remove(machine_to_shutdown)
                                machine_to_shutdown.optimized_rel_active_load = 0
                        
                        if len(active_sources) == 1:
                            remaining_machine = active_sources[0]
                            if total_group_load >= remaining_machine.min_capacity:
                                remaining_machine.optimized_rel_active_load = total_group_load*0.9 + remaining_machine.min_capacity*0.1
                            else:
                                remaining_machine.optimized_rel_active_load = remaining_machine.min_capacity
                            
                            for s in group_sources:
                                if s != remaining_machine:
                                    s.optimized_rel_active_load = 0
                                    
                    else:
                        # Proportional load sharing
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
                                        s.optimized_rel_active_load = max(new_load, s.min_capacity)*0.9 + s.min_capacity*0.1
                                    else:
                                        s.optimized_rel_active_load = 0
                            else:
                                best_machine = min(group_sources, key=lambda x: x.min_capacity)
                                best_machine.optimized_rel_active_load = max(total_group_load, best_machine.min_capacity)*0.9 + best_machine.min_capacity*0.1
                                for s in group_sources:
                                    if s != best_machine:
                                        s.optimized_rel_active_load = 0
                        else:
                            for s in group_sources:
                                s.optimized_rel_active_load = temp_loads[s]
                    
                    # Add redundancy if required (but not in enhanced reliability mode for dispatchable sources)
                    if not (self.enhanced_reliability_mode and (group_name.lower() == 'grid' or 'bess' in group_name.lower())):
                        active_sources = [s for s in group_sources if s.optimized_rel_active_load > 0]
                        number_needed = len(active_sources)
                        desired = number_needed + redundancy_level
                        if desired > len(group_sources):
                            desired = len(group_sources)
                        additional = desired - number_needed
                        if additional > 0:
                            shut_off = [s for s in group_sources if s.optimized_rel_active_load == 0]
                            additional = min(additional, len(shut_off))
                            for i in range(additional):
                                extra = shut_off[i]
                                extra.optimized_rel_active_load = extra.min_capacity
                            print(f"Added {additional} redundant units for {group_name} at min capacity")
                    
                    # Validate that group total is maintained after load sharing
                    new_group_total = sum(s.optimized_rel_active_load for s in group_sources)
                    if abs(new_group_total - original_total) > 0.01:
                        print(f"WARNING: Load sharing changed group {group_name} total from {original_total:.2f} to {new_group_total:.2f}")
                        # Adjust proportionally to maintain total (except in enhanced mode for dispatchable sources)
                        if new_group_total > 0 and not (self.enhanced_reliability_mode and (group_name.lower() == 'grid' or 'bess' in group_name.lower())):
                            adjustment_factor = original_total / new_group_total
                            for s in group_sources:
                                if s.optimized_rel_active_load > 0:
                                    s.optimized_rel_active_load *= adjustment_factor
                
                print(f"Load sharing applied to {group_name} sources")

    def apply_load_sharing_cost(self, sources, redundancy_level=0, use_effective=False):
        """Apply load sharing among similar sources with optional redundancy"""
        source_groups = {}
        for source in sources:
            source_type = source.name.split('_')[0] if '_' in source.name else source.name
            if source_type not in source_groups:
                source_groups[source_type] = []
            source_groups[source_type].append(source)
            
        for group_name, group_sources in source_groups.items():
            if len(group_sources) > 1:
                total_group_load = sum(s.optimized_cost_active_load for s in group_sources)
                if total_group_load > 0:
                    # Store original total for validation
                    original_total = total_group_load
                    
                    # Equal load sharing for similar capacity units
                    if all(abs(s.max_capacity - group_sources[0].max_capacity) < 50 for s in group_sources):
                        active_sources = group_sources.copy()
                        
                        while len(active_sources) > 1:
                            avg_load = total_group_load / len(active_sources)
                            min_c = min(s.min_capacity for s in active_sources)
                            
                            if avg_load >= min_c:
                                for s in group_sources:
                                    if s in active_sources:
                                        s.optimized_cost_active_load = avg_load
                                    else:
                                        s.optimized_cost_active_load = 0
                                break
                            else:
                                machine_to_shutdown = max(active_sources, key=lambda x: x.min_capacity)
                                active_sources.remove(machine_to_shutdown)
                                machine_to_shutdown.optimized_cost_active_load = 0
                        
                        if len(active_sources) == 1:
                            remaining_machine = active_sources[0]
                            if total_group_load >= remaining_machine.min_capacity:
                                remaining_machine.optimized_cost_active_load = total_group_load
                            else:
                                remaining_machine.optimized_cost_active_load = remaining_machine.min_capacity
                            
                            for s in group_sources:
                                if s != remaining_machine:
                                    s.optimized_cost_active_load = 0
                                    
                    else:
                        # Proportional load sharing
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
                                        s.optimized_cost_active_load = max(new_load, s.min_capacity)
                                    else:
                                        s.optimized_cost_active_load = 0
                            else:
                                best_machine = min(group_sources, key=lambda x: x.min_capacity)
                                best_machine.optimized_cost_active_load = max(total_group_load, best_machine.min_capacity)
                                for s in group_sources:
                                    if s != best_machine:
                                        s.optimized_cost_active_load = 0
                        else:
                            for s in group_sources:
                                s.optimized_cost_active_load = temp_loads[s]
                    
                    # Add redundancy if required
                    active_sources = [s for s in group_sources if s.optimized_cost_active_load > 0]
                    number_needed = len(active_sources)
                    desired = number_needed + redundancy_level
                    if desired > len(group_sources):
                        desired = len(group_sources)
                    additional = desired - number_needed
                    if additional > 0:
                        shut_off = [s for s in group_sources if s.optimized_cost_active_load == 0]
                        additional = min(additional, len(shut_off))
                        for i in range(additional):
                            extra = shut_off[i]
                            extra.optimized_cost_active_load = extra.min_capacity
                        print(f"Added {additional} redundant units for {group_name} at min capacity")
                    
                    # Validate that group total is maintained after load sharing
                    new_group_total = sum(s.optimized_cost_active_load for s in group_sources)
                    if abs(new_group_total - original_total) > 0.01:
                        print(f"WARNING: Load sharing changed group {group_name} total from {original_total:.2f} to {new_group_total:.2f}")
                        # Adjust proportionally to maintain total
                        if new_group_total > 0:
                            adjustment_factor = original_total / new_group_total
                            for s in group_sources:
                                if s.optimized_cost_active_load > 0:
                                    s.optimized_cost_active_load *= adjustment_factor
                
                print(f"Load sharing applied to {group_name} sources")
    

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
            cost_optimized_kvar = source.optimized_cost_reactive_load
            rel_optimized_kvar = source.optimized_rel_reactive_load

            current_cost_hr = current_load * cost_per_kwh
            
            if source.name.lower() == 'grid':
                cost_effective_optimized = cost_optimized_load - source.grid_feed_power
                rel_effective_optimized = rel_optimized_load - source.grid_feed_power
                cost_optimized_cost_hr = max(0, cost_effective_optimized) * cost_per_kwh
                rel_optimized_cost_hr = max(0, rel_effective_optimized) * cost_per_kwh
                total_grid_feed += source.grid_feed_power
            else:
                cost_effective_optimized = cost_optimized_load
                rel_effective_optimized = rel_optimized_load
                cost_optimized_cost_hr = cost_optimized_load * cost_per_kwh
                rel_optimized_cost_hr = rel_optimized_load * cost_per_kwh

            # Add status indicator for enhanced reliability mode
            status_indicator = 'Active' if source.available else 'Inactive'
            if self.enhanced_reliability_mode and (source.source_type == 'grid' or 'bess' in source.name.lower()):
                if rel_optimized_load <= 10:  # Minimal allocation indicates standby
                    status_indicator += ' (Standby)'

            row = {
                'ENERGY SOURCE': source.name,
                'MAXIMUM CAPACITY (kW)': source.max_capacity,
                'MINIMUM CAPACITY (kW)': source.min_capacity,
                'CURRENT GENERATION (kW)': round(current_load, 2),
                'COST OPTIMIZED GENERATION (kW)': round(cost_effective_optimized, 2),
                'RELIABILITY OPTIMIZED GENERATION (kW)': round(rel_effective_optimized, 2),
                'CURRENT KVAR (kVAR)': round(current_kvar, 2),
                'COST OPTIMIZED KVAR (kVAR)': round(cost_optimized_kvar, 2),
                'RELIABILITY OPTIMIZED KVAR (kVAR)': round(rel_optimized_kvar, 2),
                'TOTAL COST (PKR/kWh)': round(cost_per_kwh, 2),
                'PRODUCTION COST (PKR/kWh)': round(source.production_cost, 2),
                'CARBON COST (PKR/kWh)': round(source.carbon_emission * self.carbon_cost_pkr_kg, 2),
                'CURRENT COST/HR': round(current_cost_hr, 2),
                'COST OPTIMIZED COST/HR': round(cost_optimized_cost_hr, 2),
                'RELIABILITY OPTIMIZED COST/HR': round(rel_optimized_cost_hr, 2),
                'COST OPTIMIZED CHARGE (kW)': 0.0,
                'RELIABILITY OPTIMIZED CHARGE (kW)': 0.0,
                'COST OPTIMIZED DISCHARGE (kW)': 0.0,
                'RELIABILITY OPTIMIZED DISCHARGE (kW)': 0.0,
                'GRID FEED (kW)': round(source.grid_feed_power, 2),
                'RELIABILITY SCORE': round(source.reliability_score, 2),
                'EFFICIENCY SCORE': round(source.efficiency_score, 2),
                'STATUS': status_indicator
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
            current_bess_discharge = abs(bess.current_power_input) if bess.current_power_input < 0 else 0
            
            cost_optimized_bess_load = bess.cost_optimized_discharge_power
            rel_optimized_bess_load = bess.rel_optimized_discharge_power
            
            # optimized_bess_load = bess.optimized_discharge_power

            try:
                current_kvar = bess.current_reactive_load
            except AttributeError:
                current_kvar = 0

            try:
                cost_optimized_kvar = bess.optimized_cost_reactive_load
            except AttributeError:
                cost_optimized_kvar = 0

            try:
                rel_optimized_kvar = bess.optimized_rel_reactive_load
            except AttributeError:
                rel_optimized_kvar = 0

            current_cost_hr = bess.get_current_operating_cost()
            cost_optimized_cost_hr = bess.get_cost_optimized_operating_cost()
            rel_optimized_cost_hr = bess.get_rel_optimized_operating_cost()
            
            try:
                total_cost = bess.total_cost
            except AttributeError:
                total_cost = 0.0

            try:
                production_cost = bess.production_cost
            except AttributeError:
                production_cost = 0.0

            try:
                carbon_emission = bess.carbon_emission
            except AttributeError:
                carbon_emission = 0.0

            try:
                reliability_score = bess.reliability_score
                efficiency_score = self.calculate_efficiency_score(bess)
            except AttributeError:
                reliability_score = 0.0
                efficiency_score = 0.0

            if bess.mode == 'charging':
                status = f'Charging (SOC: {bess.current_soc}%)'
            elif bess.mode == 'discharging':
                status = f'Discharging (SOC: {bess.current_soc}%)'
            else:
                status = f'Standby (SOC: {bess.current_soc}%)'
                
            # Add enhanced reliability mode indicator
            if self.enhanced_reliability_mode and bess.mode == 'standby':
                status += ' - Emergency Ready'

            # bess.optimized_charge_power must be same as input power to bess in charging mode
            print("BESS Mode:", bess.mode, "Current Power Input:", bess.current_power_input, "Cost Optimized Charge Power before:", bess.cost_optimized_charge_power)
            print("BESS Mode:", bess.mode, "Current Power Input:", bess.current_power_input, "Reliability Optimized Charge Power before:", bess.rel_optimized_charge_power)
            # if bess.mode == 'charging':
            #     print('Currently charging Cost optimized, setting optimized charge power to current input')
            #     bess.cost_optimized_charge_power = bess.current_power_input

            # print optimized bess load
            print(f"BESS {bess.name} - Current Discharge: {current_bess_discharge} kW, Cost Optimized Discharge: {cost_optimized_bess_load} kW")
            print(f"BESS {bess.name} - Current Discharge: {current_bess_discharge} kW, Reliability Optimized Discharge: {rel_optimized_bess_load} kW")

            row = {
                'ENERGY SOURCE': bess.name,
                'MAXIMUM CAPACITY (kW)': bess.power_rating_kw,
                'MINIMUM CAPACITY (kW)': 0,
                'CURRENT GENERATION (kW)': round(current_bess_discharge, 2),

                'COST OPTIMIZED GENERATION (kW)': round(cost_optimized_bess_load, 2),
                'RELIABILITY OPTIMIZED GENERATION (kW)': round(rel_optimized_bess_load, 2),


                # 'COST OPTIMIZED GENERATION (kW)': round(optimized_bess_load, 2),
                # # 'COST OPTIMIZED DISCHARGE (kW)': round(optimized_bess_load, 2),
                # 'RELIABILITY OPTIMIZED GENERATION (kW)': round(optimized_bess_load, 2),

                'CURRENT KVAR (kVAR)': round(current_kvar, 2),
                'COST OPTIMIZED KVAR (kVAR)': round(cost_optimized_kvar, 2),
                'RELIABILITY OPTIMIZED KVAR (kVAR)': round(rel_optimized_kvar, 2),
                'TOTAL COST (PKR/kWh)': round(total_cost, 2),
                'PRODUCTION COST (PKR/kWh)': round(production_cost, 2),
                'CARBON COST (PKR/kWh)': round(carbon_emission * self.carbon_cost_pkr_kg, 2),

                'CURRENT COST/HR': round(current_cost_hr, 2),
                'COST OPTIMIZED COST/HR': round(cost_optimized_cost_hr, 2),
                # 'CURRENT COST/HR': 0.0,
                # 'COST OPTIMIZED COST/HR': 0.0,
                # 'RELIABILITY OPTIMIZED COST/HR': 0.0,

                'RELIABILITY OPTIMIZED COST/HR': round(rel_optimized_cost_hr, 2),

                'COST OPTIMIZED CHARGE (kW)': round(bess.cost_optimized_charge_power, 2),
                'RELIABILITY OPTIMIZED CHARGE (kW)': round(bess.rel_optimized_charge_power, 2),

                # 'COST OPTIMIZED CHARGE (kW)': round(bess.optimized_charge_power, 2),
                # 'RELIABILITY OPTIMIZED CHARGE (kW)': round(bess.optimized_charge_power, 2),


                # 'COST OPTIMIZED DISCHARGE (kW)': round(bess.optimized_discharge_power, 2),
                # 'RELIABILITY OPTIMIZED DISCHARGE (kW)': round(bess.optimized_discharge_power, 2),

                'COST OPTIMIZED DISCHARGE (kW)': round(bess.cost_optimized_discharge_power, 2),
                'RELIABILITY OPTIMIZED DISCHARGE (kW)': round(bess.rel_optimized_discharge_power, 2),


                # 'COST OPTIMIZED DISCHARGE (kW)': round(optimized_bess_load, 2),
                # 'RELIABILITY OPTIMIZED DISCHARGE (kW)': round(optimized_bess_load, 2),
                'GRID FEED (kW)': 0.0,
                'RELIABILITY SCORE': round(reliability_score, 2),
                'EFFICIENCY SCORE': round(efficiency_score, 2),
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

        # Add total row
        total_savings_hr = total_current_cost - total_cost_optimized_cost
        
        # Add enhanced reliability mode indicator to status
        status_text = f'Savings: PKR {total_savings_hr:.2f}/hr'
        if self.enhanced_reliability_mode:
            status_text += ' (Enhanced Reliability Mode)'
        
        results.append({
            'ENERGY SOURCE': 'TOTAL',
            'MAXIMUM CAPACITY (kW)': round(sum(s.max_capacity for s in self.sources) + sum(b.power_rating_kw for b in self.bess_systems), 2),
            'MINIMUM CAPACITY (kW)': round(sum(s.min_capacity for s in self.sources), 2),
            'CURRENT GENERATION (kW)': round(total_current_load, 2),
            'COST OPTIMIZED GENERATION (kW)': round(total_cost_optimized_load, 2),
            'RELIABILITY OPTIMIZED GENERATION (kW)': round(total_rel_optimized_load, 2),
            'CURRENT KVAR (kVAR)': round(total_current_kvar, 2),
            'COST OPTIMIZED KVAR (kVAR)': round(total_cost_optimized_kvar, 2),
            'RELIABILITY OPTIMIZED KVAR (kVAR)': round(total_rel_optimized_kvar, 2),
            'TOTAL COST (PKR/kWh)': '',
            'PRODUCTION COST (PKR/kWh)': '',
            'CARBON COST (PKR/kWh)': '',
            'CURRENT COST/HR': round(total_current_cost, 2),
            'COST OPTIMIZED COST/HR': round(total_cost_optimized_cost, 2),
            'RELIABILITY OPTIMIZED COST/HR': round(total_rel_optimized_cost, 2),
            'COST OPTIMIZED CHARGE (kW)': round(sum(b.cost_optimized_charge_power for b in self.bess_systems), 2),
            'RELIABILITY OPTIMIZED CHARGE (kW)': round(sum(b.rel_optimized_charge_power for b in self.bess_systems), 2),
            'COST OPTIMIZED DISCHARGE (kW)': round(sum(b.cost_optimized_discharge_power for b in self.bess_systems), 2),
            'RELIABILITY OPTIMIZED DISCHARGE (kW)': round(sum(b.rel_optimized_discharge_power for b in self.bess_systems), 2),
            'GRID FEED (kW)': round(total_grid_feed, 2),
            'RELIABILITY SCORE': '',
            'EFFICIENCY SCORE': '',
            'STATUS': status_text
        })

        df_r = pd.DataFrame(results)
        df_r['EFFICIENCY SCORE'] = pd.to_numeric(df_r['EFFICIENCY SCORE'], errors='coerce')
        df_r['priority'] = df_r['EFFICIENCY SCORE'].rank(method='min').fillna(0).astype(int)

        return df_r, total_current_cost, total_savings_hr

    def generate_recommendations(self, results_df):
        """Generate optimization recommendations"""
        recommendations = []
        
        # Determine optimization mode (prefer cost if non-zero, else reliability)
        total_cost_optimized_load = results_df[results_df['ENERGY SOURCE'] == 'TOTAL']['COST OPTIMIZED GENERATION (kW)'].iloc[0]
        use_cost_optimization = total_cost_optimized_load > 0
        
        # Add enhanced reliability mode context
        if self.enhanced_reliability_mode:
            recommendations.append(f"""
**ENHANCED RELIABILITY MODE ACTIVATED**:
• Total potential loss exceeds PKR {self.critical_threshold:,.2f}
• Dispatchable sources (Grid & BESS) kept as emergency standby
• Prioritizing renewable and fixed conventional sources
• Minimal allocation to dispatchable sources for emergency readiness
            """)
        
        # Source-specific recommendations
        for _, row in results_df.iterrows():
            source = row['ENERGY SOURCE']
            if source == 'TOTAL':
                continue
            
            current_kw = row['CURRENT GENERATION (kW)']
            cost_optimized_kw = row['COST OPTIMIZED GENERATION (kW)']
            rel_optimized_kw = row['RELIABILITY OPTIMIZED GENERATION (kW)']
            optimized_kw = cost_optimized_kw if use_cost_optimization else rel_optimized_kw

            rec = f"**{source}**:\n"
            
            if 'BESS' in source:
                charge_kw = row.get('COST OPTIMIZED CHARGE (kW)', 0)
                discharge_kw = row.get('COST OPTIMIZED DISCHARGE (kW)', 0)
                
                if self.enhanced_reliability_mode:
                    rec += f"• Keep in standby mode for emergency availability\n"
                    rec += f"• Maintain SOC for rapid response capability\n"
                elif charge_kw > 0:
                    rec += f"• Charge at {charge_kw:.1f} kW\n"
                elif discharge_kw > 0:
                    rec += f"• Discharge at {discharge_kw:.1f} kW to provide {discharge_kw:.1f} kW to load\n"
                else:
                    rec += f"• Maintain standby mode\n"
                
                rec += f"• Current status: {row['STATUS']}\n"
                rec += f"• Reliability score: {row['RELIABILITY SCORE']}\n"
            else:
                if source.lower() == 'grid' and self.enhanced_reliability_mode:
                    rec += f"• Maintain minimal connection ({optimized_kw:.1f} kW) for emergency backup\n"
                    rec += f"• Keep full capacity available for emergency use\n"
                elif optimized_kw > current_kw:
                    rec += f"• Increase load from {current_kw:.1f} kW to {optimized_kw:.1f} kW\n"
                elif optimized_kw < current_kw:
                    rec += f"• Reduce load from {current_kw:.1f} kW to {optimized_kw:.1f} kW\n"
                else:
                    rec += f"• Maintain current load of {current_kw:.1f} kW\n"
                rec += f"• Reliability score: {row['RELIABILITY SCORE']}\n"
            
            recommendations.append(rec)
        
        # Overall summary
        total_row = results_df[results_df['ENERGY SOURCE'] == 'TOTAL'].iloc[0]
        optimized_load_key = 'COST OPTIMIZED GENERATION (kW)' if use_cost_optimization else 'RELIABILITY OPTIMIZED GENERATION (kW)'
        optimized_cost_key = 'COST OPTIMIZED COST/HR' if use_cost_optimization else 'RELIABILITY OPTIMIZED COST/HR'
        optimized_charge_key = 'COST OPTIMIZED CHARGE (kW)' if use_cost_optimization else 'RELIABILITY OPTIMIZED CHARGE (kW)'
        optimized_discharge_key = 'COST OPTIMIZED DISCHARGE (kW)' if use_cost_optimization else 'RELIABILITY OPTIMIZED DISCHARGE (kW)'
        
        optimization_type = 'lowest total cost' if use_cost_optimization else 'reliability'
        if self.enhanced_reliability_mode:
            optimization_type = 'enhanced reliability with emergency preparedness'
        
        summary = f"""
**OPTIMIZATION SUMMARY**:
• Total current load: {total_row['CURRENT GENERATION (kW)']:.2f} kW
• Total optimized load: {total_row[optimized_load_key]:.2f} kW
• Current hourly cost: PKR {total_row['CURRENT COST/HR']:,.2f}
• Optimized hourly cost: PKR {total_row[optimized_cost_key]:,.2f}
• Hourly cost savings: PKR {total_row['CURRENT COST/HR'] - total_row[optimized_cost_key]:,.2f}
• Total BESS charge power: {total_row[optimized_charge_key]:.2f} kW
• Total BESS discharge power: {total_row[optimized_discharge_key]:.2f} kW
• Grid feed: {total_row['GRID FEED (kW)']:.2f} kW
• Optimization prioritizes {optimization_type}
        """
        
        if self.enhanced_reliability_mode:
            summary += f"""
• **CRITICAL LOSS THRESHOLD EXCEEDED**: PKR {self.tripping_cost + self.production_loss_hourly:,.2f}
• Dispatchable sources maintained as emergency backup
• Primary focus on operational continuity over cost savings
            """
        
        recommendations.append(summary)
        return '\n\n'.join(recommendations)