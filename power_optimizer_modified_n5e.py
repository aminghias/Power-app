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

        # Store both cost and reliability optimized values
        self.cost_optimized_charge_power = 0
        self.cost_optimized_discharge_power = 0
        self.rel_optimized_charge_power = 0
        self.rel_optimized_discharge_power = 0
        
        # Reactive power attributes
        self.current_reactive_load = 0
        self.cost_optimized_reactive_load = 0
        self.rel_optimized_reactive_load = 0


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
        # Assumes a 1-hour interval where kWh capacity is numerically equal to max kW power
        max_charge_power = min(self.power_rating_kw,
                               self.get_available_charge_capacity())
        return power_kw <= max_charge_power

    def can_discharge(self, power_kw):
        """Check if BESS can discharge at given power"""
        if self.current_soc <= self.min_soc:
            return False
        # Assumes a 1-hour interval where kWh capacity is numerically equal to max kW power
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
        # Cost is incurred when charging from a source
        if self.current_power_input > 0:  # Charging
            return abs(self.current_power_input) * self.total_cost
        # Discharging avoids cost, but here we model it as neutral or based on degradation (simplified to 0)
        return 0

    def get_optimized_operating_cost(self, mode='cost'):
        """Calculate optimized operating cost for specified mode"""
        charge_power = 0
        if mode == 'cost':
            charge_power = self.cost_optimized_charge_power
        else:  # reliability
            charge_power = self.rel_optimized_charge_power

        if charge_power > 0:
            return charge_power * self.total_cost
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
        
        self.optimized_active_load = 0
        self.optimized_reactive_load = 0

        self.grid_feed_power = 0
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
        self.power_sources = {}
        self.bess_systems = {}
        self.carbon_cost_pkr_kg = 50.0
        self.global_params = {
            'wind_speed': None, 'fuel_pressure': None, 'fuel_level': None,
            'gas_pressure': None, 'ghi': None
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

    def set_global_params(self, wind_speed=None, fuel_pressure=None, fuel_level=None, gas_pressure=None, ghi=None):
        """Set global operational parameters"""
        self.global_params.update({
            'wind_speed': wind_speed, 'fuel_pressure': fuel_pressure,
            'fuel_level': fuel_level, 'gas_pressure': gas_pressure, 'ghi': ghi
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
            return pd.read_sql(query, connection)
        finally:
            if connection.is_connected():
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

        nom_voltage, v_min, v_max = 400, 400 * 0.9, 400 * 1.1
        f_min, f_max = 49.5, 50.5

        voltages = df[['a_u', 'b_u', 'c_u']].values.flatten()
        v_std = np.std(voltages)
        v_in_range = np.mean((voltages >= v_min) & (voltages <= v_max)) * 10

        f_std = df['grid_frequency'].std()
        f_in_range = np.mean((df['grid_frequency'] >= f_min) & (df['grid_frequency'] <= f_max)) * 10

        v_score = max(0, 10 - (v_std / 10)) * 0.6 + v_in_range * 0.4
        f_score = max(0, 10 - (f_std * 10)) * 0.6 + f_in_range * 0.4
        return (v_score + f_score) / 2

    def initialize_sources(self, site_id):
        """Initialize power sources and BESS systems from configuration."""
        sources_config = self.load_sources_config(site_id)
        if not sources_config:
            print("No sources configuration found")
            return

        self.power_sources.clear()
        for config in sources_config:
            source_type = 'conventional'
            name_lower = config['name'].lower()
            if name_lower in ['solar', 'wind']:
                source_type = 'renewable'
            elif name_lower == 'grid':
                source_type = 'grid'
            
            total_cost = config['production_cost'] + (config['carbon_emission'] * self.carbon_cost_pkr_kg)
            
            source = PowerSource(
                name=config['name'],
                production_cost=float(config['production_cost']),
                carbon_emission=float(config['carbon_emission']),
                total_cost=total_cost,
                min_capacity=float(config['min_capacity']),
                max_capacity=float(config['max_capacity']),
                source_type=source_type,
                reliability_score=float(config.get('reliability_score', 10.0)),
                wind_speed=self.global_params['wind_speed'] if name_lower.startswith('wind') else None,
                fuel_pressure=self.global_params['fuel_pressure'] if name_lower.startswith('diesel') else None,
                fuel_level=self.global_params['fuel_level'] if name_lower.startswith('diesel') else None,
                gas_pressure=self.global_params['gas_pressure'] if name_lower.startswith('gas') else None,
                ghi=self.global_params['ghi'] if name_lower.startswith('solar') else None
            )

            if source.name.lower() == 'grid':
                device_id = config.get('device_id', 0)
                source.reliability_score = self.calculate_grid_reliability(device_id)
                print(f"Grid reliability score from DB: {source.reliability_score:.2f}")

            source.check_availability()
            
            csv_file = f"{source.name}_data.csv"
            if os.path.exists(csv_file):
                source_data = pd.read_csv(csv_file)
                source.power_reading = source.current_active_load = source_data['active_power'].iloc[0]
                source.reactive_power_reading = source.current_reactive_load = source_data['reactive_power'].iloc[0]

            self.power_sources[source.name] = source

        bess_config = self.load_bess_config(site_id)
        self.bess_systems.clear()
        for config in bess_config:
            total_cost = config['production_cost'] + (config['carbon_emission'] * self.carbon_cost_pkr_kg)
            bess = BatteryEnergyStorageSystem(
                name=config['name'],
                capacity_kwh=float(config['capacity_kwh']),
                power_rating_kw=float(config['power_rating_kw']),
                current_soc=float(config.get('current_soc', 50)),
                production_cost=float(config['production_cost']),
                carbon_emission=float(config['carbon_emission']),
                total_cost=total_cost,
                reliability_score=float(config.get('reliability_score', 9.0))
            )
            bess.discharge_threshold = float(config.get('discharge_threshold', 50))
            bess.charge_threshold = float(config.get('charge_threshold', 85))
            bess.current_power_input = float(config.get('power_input', 0))
            bess.grid_synced = self.grid_connected
            bess.check_availability()

            if bess.current_power_input > 0:
                bess.current_charge_power = bess.current_power_input
                bess.mode = 'charging'
            elif bess.current_power_input < 0:
                bess.current_discharge_power = abs(bess.current_power_input)
                bess.mode = 'discharging'

            self.bess_systems[bess.name] = bess

    def calculate_efficiency_score(self, source):
        """Calculate efficiency score as sum of total cost and inverted reliability"""
        inverted_reliability = 11 - source.reliability_score
        return source.total_cost + inverted_reliability

    def calculate_current_cost(self):
        """Calculate the current total cost per hour based on active power loads and production costs."""
        total_cost = 0.0
        for source in self.power_sources.values():
            if source.available:
                total_cost += source.current_active_load * source.total_cost
        for bess in self.bess_systems.values():
            if bess.available:
                total_cost += bess.get_current_operating_cost()
        return total_cost

    def calculate_optimized_cost(self, mode='cost'):
        """Calculate the optimized total cost per hour based on optimized loads and costs."""
        total_cost = 0.0
        for source in self.power_sources.values():
            if source.available:
                active_power = source.cost_optimized_active_load if mode == 'cost' else source.rel_optimized_active_load
                total_cost += active_power * source.total_cost
        for bess in self.bess_systems.values():
            if bess.available:
                total_cost += bess.get_optimized_operating_cost(mode=mode)
        return total_cost

    def calculate_total_loss(self):
        """Calculate total loss from tripping cost and production loss."""
        return self.tripping_cost + self.production_loss_hourly

    def optimize_power_allocation_cost(self):
        """Cost-optimized power allocation"""
        available_sources = [s for s in self.power_sources.values() if s.available]
        if not available_sources:
            print("No available power sources")
            return

        total_active_load = sum(s.current_active_load for s in self.power_sources.values())
        total_reactive_load = sum(s.current_reactive_load for s in self.power_sources.values())
        total_bess_discharge = sum(abs(b.current_power_input) for b in self.bess_systems.values() if b.current_power_input < 0)
        total_active_load += total_bess_discharge

        for source in available_sources:
            source.efficiency_score = self.calculate_efficiency_score(source)
        available_sources.sort(key=lambda x: x.efficiency_score)

        for source in available_sources:
            source.cost_optimized_active_load = 0
            source.cost_optimized_reactive_load = 0

        self.optimize_bess_operation(mode='cost')
        total_bess_optimized_discharge = sum(b.cost_optimized_discharge_power for b in self.bess_systems.values())
        remaining_active_load = total_active_load - total_bess_optimized_discharge
        
        self.allocate_active_power(available_sources, remaining_active_load, use_effective=False, mode='cost')
        self.apply_load_sharing(available_sources, redundancy_level=0, use_effective=False, mode='cost')
        
        all_kvar_sources = list(self.power_sources.values()) + list(self.bess_systems.values())
        self.allocate_reactive_power(all_kvar_sources, total_reactive_load, mode='cost')

        if not self.grid_connected:
            self.handle_off_grid_operation(available_sources)

    def optimize_power_allocation_reliability(self):
        """Reliability-optimized power allocation"""
        for source in self.power_sources.values():
            source.rel_optimized_active_load = 0
            source.rel_optimized_reactive_load = 0
            source.effective_max = source.max_capacity
            source.rel_grid_feed_power = 0
        for bess in self.bess_systems.values():
            bess.rel_optimized_charge_power = 0
            bess.rel_optimized_discharge_power = 0
        self.rel_grid_feed = 0

        total_loss = self.tripping_cost + self.production_loss_hourly
        if total_loss > self.very_high_threshold:
            self.redundancy_level, self.prefer_bess = 2, True
        elif total_loss > self.high_threshold:
            self.redundancy_level, self.prefer_bess = 1, False
        else:
            self.redundancy_level, self.prefer_bess = 0, False

        available_sources = [s for s in self.power_sources.values() if s.available]
        if not available_sources:
            print("No available power sources")
            return

        total_active_load = sum(s.current_active_load for s in self.power_sources.values())
        total_reactive_load = sum(s.current_reactive_load for s in self.power_sources.values())
        total_bess_discharge = sum(abs(b.current_power_input) for b in self.bess_systems.values() if b.current_power_input < 0)
        total_active_load += total_bess_discharge

        for source in available_sources:
            source.efficiency_score = self.calculate_efficiency_score(source)
        available_sources.sort(key=lambda x: (-x.reliability_score, x.total_cost))

        solar_running = sum(s.current_active_load for s in self.power_sources.values() if s.name.lower().startswith('solar'))
        wind_running = sum(s.current_active_load for s in self.power_sources.values() if s.name.lower().startswith('wind'))
        sum_engine_max = sum(s.max_capacity for s in self.power_sources.values() if s.name.lower().startswith(('diesel', 'gas')))
        
        sp = (0.5 * solar_running) / sum_engine_max if sum_engine_max > 0 else 0
        wp = (0.7 * wind_running) / sum_engine_max if sum_engine_max > 0 else 0
        reserve_factor = max(0.2, 1 - 0.1 - sp - wp)

        for source in self.power_sources.values():
            if source.name.lower().startswith(('diesel', 'gas')):
                source.effective_max = source.max_capacity * reserve_factor

        if self.prefer_bess:
            for bess in self.bess_systems.values():
                if bess.check_availability():
                    max_discharge = bess.get_available_discharge_capacity()
                    bess.rel_optimized_discharge_power = min(bess.power_rating_kw, max_discharge)
                    bess.mode = 'discharging'
        else:
            self.optimize_bess_operation(mode='reliability')

        total_bess_optimized_discharge = sum(b.rel_optimized_discharge_power for b in self.bess_systems.values())
        remaining_active_load = total_active_load - total_bess_optimized_discharge
        
        self.allocate_active_power(available_sources, remaining_active_load, use_effective=True, mode='reliability')
        self.apply_load_sharing(available_sources, redundancy_level=self.redundancy_level, use_effective=True, mode='reliability')

        all_kvar_sources = list(self.power_sources.values()) + list(self.bess_systems.values())
        self.allocate_reactive_power(all_kvar_sources, total_reactive_load, mode='reliability')

        if not self.grid_connected:
            self.handle_off_grid_operation(available_sources)

    def optimize_bess_operation(self, mode='cost'):
        """Optimize BESS charging/discharging strategy"""
        for bess in self.bess_systems.values():
            bess.check_availability()
            if mode == 'cost':
                bess.cost_optimized_charge_power = 0
                bess.cost_optimized_discharge_power = 0
            else:
                bess.rel_optimized_charge_power = 0
                bess.rel_optimized_discharge_power = 0

            if bess.should_discharge():
                max_discharge = min(bess.power_rating_kw, bess.get_available_discharge_capacity())
                if mode == 'cost':
                    bess.cost_optimized_discharge_power = max_discharge
                else:
                    bess.rel_optimized_discharge_power = max_discharge
                bess.mode = 'discharging'
            elif bess.should_charge():
                max_charge = min(bess.power_rating_kw, bess.get_available_charge_capacity())
                if mode == 'cost':
                    bess.cost_optimized_charge_power = max_charge
                else:
                    bess.rel_optimized_charge_power = max_charge
                bess.mode = 'charging'
            else:
                bess.mode = 'standby'

    def handle_off_grid_operation(self, sources):
        """Adjust generation in off-grid mode to meet demand."""
        print("\nOff-grid operation mode activated")
        non_grid_sources = [s for s in sources if s.source_type != 'grid']
        total_generation = sum(s.cost_optimized_active_load for s in non_grid_sources)
        for bess in self.bess_systems.values():
            if bess.mode == 'discharging':
                total_generation += bess.cost_optimized_discharge_power
        
        if total_generation < self.total_load_demand:
            deficit = self.total_load_demand - total_generation
            print(f"Generation deficit in off-grid mode: {deficit:.2f} kW")
            for source in sorted(non_grid_sources, key=lambda x: x.total_cost):
                if deficit <= 0: break
                if source.cost_optimized_active_load < source.effective_max:
                    additional_capacity = min(deficit, source.effective_max - source.cost_optimized_active_load)
                    source.cost_optimized_active_load += additional_capacity
                    deficit -= additional_capacity
                    print(f"Increased {source.name} output by {additional_capacity:.2f} kW")
            if deficit > 0:
                print(f"Load shedding required: {deficit:.2f} kW")
    
    def allocate_active_power(self, sources, total_load, use_effective=False, mode='cost'):
        """Allocate active power load to available sources based on priority."""
        remaining_load = total_load
        for source in sources:
            if remaining_load <= 0:
                break
            max_cap = source.effective_max if use_effective else source.max_capacity
            allocation = min(max_cap, remaining_load)
            if allocation >= source.min_capacity:
                if mode == 'cost':
                    source.cost_optimized_active_load = allocation
                else:
                    source.rel_optimized_active_load = allocation
                remaining_load -= allocation
        return remaining_load

    def allocate_reactive_power(self, all_sources, total_reactive_load, mode='cost'):
        """Allocate reactive power among all available sources, including BESS."""
        remaining_kvar = total_reactive_load
        
        # Prioritize non-solar and BESS sources for reactive power
        sorted_sources = sorted(all_sources, key=lambda x: 1 if 'solar' in x.name.lower() else 0)

        for source in sorted_sources:
            if remaining_kvar <= 0: break
            
            allocation = 0
            if isinstance(source, BatteryEnergyStorageSystem):
                # BESS can provide reactive power up to a certain limit (e.g., 80% of power rating)
                max_reactive = source.power_rating_kw * 0.8
                allocation = min(max_reactive, remaining_kvar)
            elif 'solar' not in source.name.lower():
                # Other generators can provide reactive power (e.g., up to 60% of their active load)
                active_load = source.cost_optimized_active_load if mode == 'cost' else source.rel_optimized_active_load
                max_reactive = active_load * 0.6
                allocation = min(max_reactive, remaining_kvar)
            
            if mode == 'cost':
                source.cost_optimized_reactive_load = allocation
            else:
                source.rel_optimized_reactive_load = allocation
            remaining_kvar -= allocation
        
        if remaining_kvar > 0:
            print(f"Unallocated reactive power: {remaining_kvar:.2f} kVAR")
            grid = self.power_sources.get('grid')
            if grid:
                if mode == 'cost':
                    grid.cost_optimized_reactive_load += remaining_kvar
                else:
                    grid.rel_optimized_reactive_load += remaining_kvar
                print(f"Allocated remaining {remaining_kvar:.2f} kVAR to grid")
    
    def apply_load_sharing(self, sources, redundancy_level=0, use_effective=False, mode='cost'):
        """Apply load sharing and redundancy among similar source types."""
        source_groups = {}
        for source in sources:
            source_type = source.name.split('_')[0] if '_' in source.name else source.name
            source_groups.setdefault(source_type, []).append(source)

        for group_name, group_sources in source_groups.items():
            if len(group_sources) > 1:
                total_group_load = sum(s.cost_optimized_active_load if mode == 'cost' else s.rel_optimized_active_load for s in group_sources)
                if total_group_load <= 0: continue

                # Filter to only active sources for load sharing calculation
                active_sources = [s for s in group_sources if (s.cost_optimized_active_load if mode == 'cost' else s.rel_optimized_active_load) > 0]
                if not active_sources: continue

                # Proportional load sharing based on capacity
                total_capacity = sum(s.effective_max if use_effective else s.max_capacity for s in active_sources)
                if total_capacity == 0: continue

                for s in active_sources:
                    proportion = (s.effective_max if use_effective else s.max_capacity) / total_capacity
                    new_load = proportion * total_group_load
                    
                    # Ensure load is not below minimum capacity
                    if new_load < s.min_capacity and total_group_load > s.min_capacity:
                        new_load = s.min_capacity
                    
                    if mode == 'cost':
                        s.cost_optimized_active_load = new_load
                    else:
                        s.rel_optimized_active_load = new_load
                
                # Re-distribute any excess/deficit from min capacity adjustments (simplified)
                current_total = sum(s.cost_optimized_active_load if mode == 'cost' else s.rel_optimized_active_load for s in active_sources)
                if abs(current_total - total_group_load) > 1: # Tolerance for float precision
                    # Simple redistribution logic can be added here if needed
                    pass
    
    def generate_results(self):
        """Generate a comprehensive DataFrame comparing current, cost-optimized, and reliability-optimized states."""
        # Run both optimization scenarios to populate results
        self.optimize_power_allocation_cost()
        self.optimize_power_allocation_reliability()

        results_data = []
        all_entities = list(self.power_sources.values()) + list(self.bess_systems.values())

        for entity in all_entities:
            is_bess = isinstance(entity, BatteryEnergyStorageSystem)
            row = {
                'ENERGY SOURCE': entity.name,
                'CURRENT LOAD (kW)': entity.current_active_load if not is_bess else (abs(entity.current_power_input) if entity.current_power_input < 0 else 0),
                'COST OPT LOAD (kW)': entity.cost_optimized_active_load if not is_bess else entity.cost_optimized_discharge_power,
                'REL OPT LOAD (kW)': entity.rel_optimized_active_load if not is_bess else entity.rel_optimized_discharge_power,
                'CURRENT KVAR (kVAR)': entity.current_reactive_load,
                'COST OPT KVAR (kVAR)': entity.cost_optimized_reactive_load,
                'REL OPT KVAR (kVAR)': entity.rel_optimized_reactive_load,
                'TOTAL COST (PKR/kWh)': entity.total_cost,
                'PRODUCTION COST (PKR/kWh)': entity.production_cost,
                'CARBON COST (PKR/kWh)': entity.carbon_emission * self.carbon_cost_pkr_kg,
                'CURRENT COST/HR': (entity.current_active_load * entity.total_cost) if not is_bess else entity.get_current_operating_cost(),
                'COST OPT COST/HR': (entity.cost_optimized_active_load * entity.total_cost) if not is_bess else entity.get_optimized_operating_cost(mode='cost'),
                'REL OPT COST/HR': (entity.rel_optimized_active_load * entity.total_cost) if not is_bess else entity.get_optimized_operating_cost(mode='reliability'),
                'COST OPT CHARGE (kW)': entity.cost_optimized_charge_power if is_bess else 0,
                'COST OPT DISCHARGE (kW)': entity.cost_optimized_discharge_power if is_bess else 0,
                'REL OPT CHARGE (kW)': entity.rel_optimized_charge_power if is_bess else 0,
                'REL OPT DISCHARGE (kW)': entity.rel_optimized_discharge_power if is_bess else 0,
                'RELIABILITY SCORE': entity.reliability_score,
                'EFFICIENCY SCORE': entity.efficiency_score if not is_bess else 0,
                'STATUS': 'Active' if entity.available else 'Inactive'
            }
            results_data.append(row)
        
        df = pd.DataFrame(results_data).round(2)

        # Add Totals Row
        total_row = df.sum(numeric_only=True)
        total_row['ENERGY SOURCE'] = 'TOTAL'
        cost_savings = total_row['CURRENT COST/HR'] - total_row['COST OPT COST/HR']
        rel_savings = total_row['CURRENT COST/HR'] - total_row['REL OPT COST/HR']
        total_row['STATUS'] = f"Cost Savings: {cost_savings:.2f} PKR/hr | Rel. Savings: {rel_savings:.2f} PKR/hr"
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

        # Decide final optimization mode
        cost_diff = total_row['REL OPT COST/HR'] - total_row['COST OPT COST/HR']
        total_loss = self.calculate_total_loss()

        if total_loss > cost_diff:
            self.optimized_mode = 'reliability'
            final_cost = total_row['REL OPT COST/HR']
            final_savings = rel_savings
            for source in self.power_sources.values():
                source.optimized_active_load = source.rel_optimized_active_load
            for bess in self.bess_systems.values():
                bess.optimized_charge_power = bess.rel_optimized_charge_power
                bess.optimized_discharge_power = bess.rel_optimized_discharge_power
        else:
            self.optimized_mode = 'cost'
            final_cost = total_row['COST OPT COST/HR']
            final_savings = cost_savings
            for source in self.power_sources.values():
                source.optimized_active_load = source.cost_optimized_active_load
            for bess in self.bess_systems.values():
                bess.optimized_charge_power = bess.cost_optimized_charge_power
                bess.optimized_discharge_power = bess.cost_optimized_discharge_power
        
        return df, total_row['CURRENT COST/HR'], final_savings

    def generate_recommendations(self, results_df):
        """Generate optimization recommendations based on the chosen mode."""
        recommendations = []
        if not self.optimized_mode:
            return "Optimization has not been run yet."

        recs = [f"**Chosen Optimization Mode: {self.optimized_mode.upper()}**\n"]

        for _, row in results_df.iterrows():
            source_name = row['ENERGY SOURCE']
            if source_name == 'TOTAL': continue

            if 'BESS' in source_name:
                charge = row[f'{self.optimized_mode.upper()} OPT CHARGE (kW)']
                discharge = row[f'{self.optimized_mode.upper()} OPT DISCHARGE (kW)']
                if charge > 0:
                    recs.append(f"- **{source_name}**: Charge at {charge:.1f} kW.")
                elif discharge > 0:
                    recs.append(f"- **{source_name}**: Discharge at {discharge:.1f} kW.")
                else:
                    recs.append(f"- **{source_name}**: Remain in standby mode.")
            else:
                current_load = row['CURRENT LOAD (kW)']
                optimized_load = row[f'{self.optimized_mode.upper()} OPT LOAD (kW)']
                delta = optimized_load - current_load
                if abs(delta) > 1: # Threshold for change
                    action = "Increase" if delta > 0 else "Decrease"
                    recs.append(f"- **{source_name}**: {action} load from {current_load:.1f} kW to {optimized_load:.1f} kW.")
                else:
                    recs.append(f"- **{source_name}**: Maintain current load of {current_load:.1f} kW.")
        
        # Overall Summary
        total_row = results_df[results_df['ENERGY SOURCE'] == 'TOTAL'].iloc[0]
        summary = f"""
**OPTIMIZATION SUMMARY**:
- **Chosen Mode**: {self.optimized_mode.upper()}
- **Current Hourly Cost**: {total_row['CURRENT COST/HR']:,.2f} PKR
- **Optimized Hourly Cost**: {total_row[f'{self.optimized_mode.upper()} OPT COST/HR']:,.2f} PKR
- **Estimated Hourly Savings**: {(total_row['CURRENT COST/HR'] - total_row[f'{self.optimized_mode.upper()} OPT COST/HR']):,.2f} PKR
"""
        recommendations.append("\n".join(recs))
        recommendations.append(summary)
        return "\n".join(recommendations)