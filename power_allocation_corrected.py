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
    self.optimize_bess_operation()
    
    # Calculate remaining load after BESS optimization
    total_bess_optimized_discharge = sum(b.optimized_discharge_power for b in self.bess_systems)
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

def handle_off_grid_operation_cost(self, sources):
    print("\nOff-grid operation mode activated")
    
    sources = [s for s in sources if s.source_type != 'grid']
    
    total_generation = sum(s.optimized_cost_active_load for s in sources)
    total_demand = self.total_load_demand
    
    for bess in self.bess_systems:
        if bess.mode == 'discharging':
            total_generation += bess.optimized_discharge_power
    
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
            total_generation += bess.optimized_discharge_power
    
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
    bess_discharging = any(b.optimized_discharge_power > 0 for b in self.bess_systems)
    
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

    solar_sources = [s for s in sources if s.name.lower().startswith('solar')]
    non_renewable_sources = [s for s in sources if not s.name.lower().startswith('solar') and not s.name.lower().startswith('wind') and not s.name.lower().startswith('bess')]
    bess_discharging = any(b.optimized_discharge_power > 0 for b in self.bess_systems)
    
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
    """Reliability-optimized power allocation"""
    # Reset optimized
    for source in self.sources:
        source.optimized_rel_active_load = 0
        source.optimized_rel_reactive_load = 0
        source.effective_max = source.max_capacity
        source.grid_feed_power = 0
    
    for bess in self.bess_systems:
        bess.optimized_charge_power = 0
        bess.optimized_discharge_power = 0
    
    self.grid_feed = 0
    
    # Determine redundancy level and prefer_bess
    total_loss = self.tripping_cost + self.production_loss_hourly
    if total_loss > self.very_high_threshold:
        self.redundancy_level = 2
        self.prefer_bess = True
    elif total_loss > self.high_threshold:
        self.redundancy_level = 1
        self.prefer_bess = False
    else:
        self.redundancy_level = 0
        self.prefer_bess = False
    
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
    
    # Optimize BESS
    if self.prefer_bess:
        for bess in self.bess_systems:
            bess.check_availability()
            if bess.available:
                max_discharge = bess.get_available_discharge_capacity()
                bess.optimized_discharge_power = min(bess.power_rating_kw, max_discharge)
                bess.mode = 'discharging'
    else:
        self.optimize_bess_operation()
    
    # Calculate remaining load after BESS optimization
    total_bess_optimized_discharge = sum(b.optimized_discharge_power for b in self.bess_systems)
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
    
    # Try to allocate to sources with available capacity
    for source in sources:
        if remaining <= 0:
            break
        available_capacity = source.effective_max - source.optimized_rel_active_load
        if available_capacity > 0:
            allocation = min(remaining, available_capacity)
            source.optimized_rel_active_load += allocation
            remaining -= allocation
            print(f"Added {allocation:.2f} kW to {source.name}")
    
    # If still remaining, try to use BESS charging
    if remaining > 0:
        for bess in self.bess_systems:
            if remaining <= 0:
                break
            bess.check_availability()
            if bess.available and bess.can_charge(remaining):
                assignable = min(remaining, bess.power_rating_kw - bess.optimized_charge_power)
                bess.optimized_charge_power += assignable
                bess.mode = 'charging' if bess.optimized_charge_power > 0 else bess.mode
                remaining -= assignable
                print(f"Assigned {assignable:.2f} kW to BESS {bess.name} for charging")
    
    # If still remaining, try grid feed
    if remaining > 0 and self.allow_grid_feed:
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

def reduce_allocation_rel(self, sources, excess_load):
    """Reduce allocation from sources while respecting minimum capacity"""
    remaining = excess_load
    
    # Reduce from sources with load above minimum capacity
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
    """Allocate reactive power among sources"""
    remaining_reactive_load = total_reactive_load
    
    for source in sources:
        if remaining_reactive_load <= 0:
            break

        if hasattr(source, 'power_rating_kw'):  # BESS
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
        if grid_source:
            grid_source.optimized_rel_reactive_load += remaining_reactive_load
            print(f"Allocated {remaining_reactive_load:.2f} kVAR to grid")

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
                
                # Equal load sharing for similar capacity units
                if all(abs(s.max_capacity - group_sources[0].max_capacity