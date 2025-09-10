import pandas as pd

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
    self.allocate_cost_active_power(available_sources, remaining_active_load, use_effective=False)
    
    self.apply_load_sharing_cost(available_sources, redundancy_level=0, use_effective=False)

    # FIX: Reconcile to ensure allocated == target (remaining_active_load)
    target = remaining_active_load
    allocated = sum(s.optimized_cost_active_load for s in available_sources)
    diff = target - allocated  # >0 means deficit, <0 means surplus

    if abs(diff) > 1e-6:
        if diff > 0:
            # Need to add 'diff' kW across cheapest sources with available headroom
            for src in sorted(available_sources, key=lambda x: x.efficiency_score):
                if diff <= 0:
                    break
                max_cap = getattr(src, 'effective_max', src.max_capacity)
                headroom = max_cap - src.optimized_cost_active_load
                if headroom <= 0:
                    continue
                # If turning on from zero, respect min_capacity
                assignable = min(diff, headroom)
                if src.optimized_cost_active_load == 0 and assignable > 0:
                    assignable = max(assignable, src.min_capacity, 0)
                    # Do not exceed headroom
                    assignable = min(assignable, headroom)
                src.optimized_cost_active_load += assignable
                diff -= assignable
        else:
            # Surplus: remove -diff starting from least efficient while keeping >= min if already on
            surplus = -diff
            for src in sorted(available_sources, key=lambda x: x.efficiency_score, reverse=True):
                if surplus <= 0:
                    break
                current = src.optimized_cost_active_load
                if current <= 0:
                    continue
                # Keep min if the unit remains on; allow going to 0 if current == min_capacity
                reducible_floor = src.min_capacity if current > src.min_capacity else 0
                reducible = max(0.0, current - reducible_floor)
                take = min(surplus, reducible)
                if take > 0:
                    src.optimized_cost_active_load -= take
                    surplus -= take
            # If still surplus remains (all at floors), reduce proportionally even if some dip below min,
            # as a last resort to force exact balance (rare edge). This preserves exact equality.
            if surplus > 1e-6:
                total_now = sum(s.optimized_cost_active_load for s in available_sources if s.optimized_cost_active_load > 0)
                if total_now > 0:
                    for src in available_sources:
                        if surplus <= 0:
                            break
                        if src.optimized_cost_active_load <= 0:
                            continue
                        share = surplus * (src.optimized_cost_active_load / total_now)
                        take = min(share, src.optimized_cost_active_load)
                        src.optimized_cost_active_load -= take

        # Final numerical snap
        allocated = sum(s.optimized_cost_active_load for s in available_sources)
        if abs(allocated - target) > 1e-6:
            # small numerical correction on the cheapest source
            cheapest = min(available_sources, key=lambda x: x.efficiency_score)
            cheapest.optimized_cost_active_load += (target - allocated)

    kvar_sources = [s for s in self.sources]
    kvar_sources.extend(self.bess_systems)
    print('The KVAR sources are:')
    for kvar in kvar_sources:
        print(f" - {kvar.name}")

    self.allocate_cost_reactive_power(kvar_sources, total_reactive_load)
    
    if not self.grid_connected:
        self.handle_off_grid_operation_cost(available_sources)


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
            bess.optimized_discharge_power = max(0.0, max_discharge)  # guard
            bess.mode = 'discharging' if bess.optimized_discharge_power > 0 else 'standby'
            print(f"BESS {bess.name} optimized for discharging: {bess.optimized_discharge_power:.2f} kW")
            
        elif bess.current_soc <= bess.charge_threshold:
            # Should charge when excess power available
            max_charge = min(
                bess.power_rating_kw,
                bess.get_available_charge_capacity()
            )
            bess.optimized_charge_power = max(0.0, max_charge)  # guard
            bess.mode = 'charging' if bess.optimized_charge_power > 0 else 'standby'
            print(f"BESS {bess.name} optimized for charging: {bess.optimized_charge_power:.2f} kW")
            
        else:
            # Maintain current operation or standby
            bess.mode = 'standby'
            print(f"BESS {bess.name} in standby mode")


def handle_off_grid_operation_cost(self, sources):
    print("\nOff-grid operation mode activated")
    
    sources = [s for s in sources if s.source_type != 'grid']
    
    total_generation = sum(s.optimized_cost_active_load for s in sources)
    total_demand = self.total_load_demand  # NOTE: ensure this is consistent with total_active_load
    
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
                if additional_capacity > 0:  # guard
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
                    if reduction > 0:  # guard
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
                if additional_capacity > 0:  # guard
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
    if solar_sources and total_load <= sum((s.effective_max if use_effective else s.max_capacity) for s in solar_sources) and not bess_discharging:
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
    if solar_sources and total_load <= sum((s.effective_max if use_effective else s.max_capacity) for s in solar_sources) and not bess_discharging:
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
    
    # Sort by efficiency_score ascending (your chosen logic)
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
                bess.optimized_discharge_power = min(bess.power_rating_kw, max(0.0, max_discharge))  # guard
                bess.mode = 'discharging' if bess.optimized_discharge_power > 0 else 'standby'
    else:
        self.optimize_bess_operation()
    
    # Remaining load
    total_bess_optimized_discharge = sum(b.optimized_discharge_power for b in self.bess_systems)
    remaining_active_load = total_active_load - total_bess_optimized_discharge
    
    # Allocate
    self.allocate_rel_active_power(available_sources, remaining_active_load, use_effective=True)
    
    self.apply_load_sharing_rel(available_sources, redundancy_level=self.redundancy_level, use_effective=True)

    # sum of allocated reliability optimized loads
    total_allocated_rel_opt_load = sum(s.optimized_rel_active_load for s in available_sources)
    print(f"Total allocated reliability optimized load: {total_allocated_rel_opt_load:.2f} kW")

    # print total running load 
    total_running_load = sum(s.current_active_load for s in available_sources)
    print(f"Total running load: {total_running_load:.2f} kW")

    # find difference
    load_difference = total_running_load - total_allocated_rel_opt_load

    # Kept name 'excess_load' for structure; this represents a deficit to fill
    excess_load = load_difference if load_difference > 0 else 0

    # Check and fix sources exceeding effective_max
    for source in available_sources:
        if source.optimized_rel_active_load > source.effective_max:
            excess = source.optimized_rel_active_load - source.effective_max
            source.optimized_rel_active_load = source.effective_max
            excess_load += excess
            print(f"Source {source.name} exceeded effective_max. Reduced to {source.effective_max:.2f} kW. Excess: {excess:.2f} kW")
    
    # Redistribute excess load (deficit to assign)
    if excess_load > 0:
        print(f"Redistributing excess load: {excess_load:.2f} kW")
        
        # Step 1: Try to assign to machines with zero active power
        zero_load_sources = [s for s in available_sources if s.optimized_rel_active_load == 0 and s.effective_max > 0]
        zero_load_sources.sort(key=lambda x: x.efficiency_score)
        for source in zero_load_sources:
            if excess_load <= 0:
                break
            assignable = min(excess_load, source.effective_max)
            assignable = max(source.min_capacity, assignable)  # respect min if turning on
            assignable = min(assignable, source.effective_max)
            source.optimized_rel_active_load = assignable
            excess_load -= assignable
            print(f"Assigned {assignable:.2f} kW to previously unused source {source.name}")
        
        # Step 2: Try to assign remaining to machines that still have capacity
        if excess_load > 0:
            for source in available_sources:
                if excess_load <= 0:
                    break
                available_capacity = source.effective_max - source.optimized_rel_active_load
                if available_capacity > 0:
                    assignable = min(excess_load, available_capacity)
                    assignable = max(source.min_capacity if source.optimized_rel_active_load == 0 else 0, assignable)
                    assignable = min(assignable, available_capacity)
                    source.optimized_rel_active_load += assignable
                    excess_load -= assignable
                    print(f"Assigned additional {assignable:.2f} kW to source {source.name}")
        
        # Step 3: Try to assign to BESS (charging mode)
        if excess_load > 0:
            for bess in self.bess_systems:
                if excess_load <= 0:
                    break
                bess.check_availability()
                if bess.available and bess.can_charge(excess_load):
                    assignable = min(excess_load, bess.power_rating_kw - bess.optimized_charge_power)
                    if assignable > 0:
                        bess.optimized_charge_power += assignable
                        bess.mode = 'charging' if bess.optimized_charge_power > 0 else bess.mode
                        excess_load -= assignable
                        print(f"Assigned {assignable:.2f} kW to BESS {bess.name} for charging")
        
        # Step 4: Try to assign to grid feed
        if excess_load > 0 and self.allow_grid_feed:
            grid_feed_capacity = self.grid_feed_limit - self.grid_feed
            if grid_feed_capacity > 0:
                assignable = min(excess_load, grid_feed_capacity)
                self.grid_feed += assignable
                excess_load -= assignable
                print(f"Fed {assignable:.2f} kW to grid")
                
                # Update grid source if exists
                for source in self.sources:
                    if source.name.lower() == 'grid':
                        source.grid_feed_power = self.grid_feed
                        break
        
        # Step 5: Report any remaining unassigned load
        if excess_load > 0:
            print(f"WARNING: Unable to assign {excess_load:.2f} kW of load. System capacity insufficient.")
    
    kvar_sources = [s for s in self.sources]
    kvar_sources.extend(self.bess_systems)
    
    self.allocate_rel_reactive_power(kvar_sources, total_reactive_load)
    
    # Handle excess if any (original logic)
    remaining_active_load = total_active_load - sum(b.optimized_discharge_power for b in self.bess_systems)
    total_allocated = sum(s.optimized_rel_active_load for s in self.sources)
    excess = total_allocated - remaining_active_load
    if excess > 0:
        # Try to charge BESS with excess
        for bess in self.bess_systems:
            bess.check_availability()
            if bess.available and bess.can_charge(excess):
                bess.optimized_charge_power += excess
                if bess.optimized_discharge_power > 0:
                    bess.optimized_discharge_power = max(0, bess.optimized_discharge_power - excess)
                bess.mode = 'charging' if bess.optimized_charge_power > 0 else bess.mode
                excess = 0
                break
        if excess > 0:
            for bess in self.bess_systems:
                if bess.available and bess.can_charge(excess):
                    bess.optimized_charge_power += excess
                    bess.mode = 'charging' if bess.optimized_charge_power > 0 else bess.mode
                    excess = 0
                    break
        if excess > 0 and self.allow_grid_feed:
            self.grid_feed = min(excess, self.grid_feed_limit)
            for s in self.sources:
                if s.name.lower() == 'grid':
                    s.grid_feed_power = self.grid_feed
                    break
    
    if not self.grid_connected:
        self.handle_off_grid_operation_rel(available_sources)


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
            max_reactive = getattr(source, 'optimized_cost_active_load', 0) * 0.6  # guard
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
            current = getattr(grid_source, 'optimized_cost_reactive_load', 0)
            setattr(grid_source, 'optimized_cost_reactive_load', current + remaining_reactive_load)
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
            max_reactive = getattr(source, 'optimized_rel_active_load', 0) * 0.6  # guard
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
            current = getattr(grid_source, 'optimized_rel_reactive_load', 0)
            setattr(grid_source, 'optimized_rel_reactive_load', current + remaining_reactive_load)
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
                    total_max = sum((s.effective_max if use_effective else s.max_capacity) for s in group_sources)
                    
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
                            remaining_total_max = sum((s.effective_max if use_effective else s.max_capacity) for s in active_sources)
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
                
                # Add redundancy if required
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
                    total_max = sum((s.effective_max if use_effective else s.max_capacity) for s in group_sources)
                    
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
                            remaining_total_max = sum((s.effective_max if use_effective else s.max_capacity) for s in active_sources)
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
        
        print(f"Load sharing applied to {group_name} sources")


def generate_results(self):
    """Generate comprehensive results including all sources and BESS"""
    results = []
    total_current_load = 0.0
    total_cost_optimized_load = 0.0
    total_current_cost = 0.0
    total_cost_optimized_cost = 0.0
    total_current_kvar = 0.0
    total_cost_optimized_kvar = 0.0
    total_grid_feed = 0.0
    total_rel_optimized_kvar = 0.0
    total_rel_optimized_load = 0.0
    total_rel_optimized_cost = 0.0

    # Process all power sources
    for source in self.sources:
        current_load = float(getattr(source, 'current_active_load', 0.0))
        cost_optimized_load = float(getattr(source, 'optimized_cost_active_load', 0.0))
        rel_optimized_load = float(getattr(source, 'optimized_rel_active_load', 0.0))
        cost_per_kwh = float(getattr(source, 'total_cost', 0.0))

        current_kvar = float(getattr(source, 'current_reactive_load', 0.0))
        cost_optimized_kvar = float(getattr(source, 'optimized_cost_reactive_load', 0.0))
        rel_optimized_kvar = float(getattr(source, 'optimized_rel_reactive_load', 0.0))

        current_cost_hr = current_load * cost_per_kwh

        grid_feed = float(getattr(source, 'grid_feed_power', 0.0))
        if getattr(source, 'name', '').lower() == 'grid':
            # Net import = optimized - feed; display imports ≥ 0; feed shown separately
            net_cost_opt = cost_optimized_load - grid_feed
            net_rel_opt = rel_optimized_load - grid_feed
            cost_effective_optimized = max(0.0, net_cost_opt)
            rel_effective_optimized = max(0.0, net_rel_opt)
            cost_optimized_cost_hr = cost_effective_optimized * cost_per_kwh
            rel_optimized_cost_hr = rel_effective_optimized * cost_per_kwh
            total_grid_feed += grid_feed
        else:
            cost_effective_optimized = cost_optimized_load
            rel_effective_optimized = rel_optimized_load
            cost_optimized_cost_hr = cost_effective_optimized * cost_per_kwh
            rel_optimized_cost_hr = rel_effective_optimized * cost_per_kwh

        row = {
            'ENERGY SOURCE': source.name,
            'CURRENT LOAD (kW)': round(current_load, 2),
            'COST OPTIMIZED LOAD (kW)': round(cost_effective_optimized, 2),
            'RELIABILITY OPTIMIZED LOAD (kW)': round(rel_effective_optimized, 2),
            'CURRENT KVAR (kVAR)': round(current_kvar, 2),
            'COST OPTIMIZED KVAR (kVAR)': round(cost_optimized_kvar, 2),
            'RELIABILITY OPTIMIZED KVAR (kVAR)': round(rel_optimized_kvar, 2),
            'TOTAL COST (PKR/kWh)': round(cost_per_kwh, 2),
            'PRODUCTION COST (PKR/kWh)': round(float(getattr(source, 'production_cost', 0.0)), 2),
            'CARBON COST (PKR/kWh)': round(float(getattr(source, 'carbon_emission', 0.0)) * float(getattr(self, 'carbon_cost_pkr_kg', 0.0)), 2),
            'CURRENT COST/HR': round(current_cost_hr, 2),
            'COST OPTIMIZED COST/HR': round(cost_optimized_cost_hr, 2),
            'RELIABILITY OPTIMIZED COST/HR': round(rel_optimized_cost_hr, 2),
            'COST OPTIMIZED CHARGE (kW)': 0.0,
            'RELIABILITY OPTIMIZED CHARGE (kW)': 0.0,
            'COST OPTIMIZED DISCHARGE (kW)': 0.0,
            'RELIABILITY OPTIMIZED DISCHARGE (kW)': 0.0,
            'GRID FEED (kW)': round(grid_feed, 2),
            'RELIABILITY SCORE': round(float(getattr(source, 'reliability_score', 0.0)), 2),
            'EFFICIENCY SCORE': round(float(getattr(source, 'efficiency_score', 0.0)), 2),
            'STATUS': 'Active' if getattr(source, 'available', False) else 'Inactive'
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
        current_bess_discharge = float(abs(getattr(bess, 'current_power_input', 0.0))) if getattr(bess, 'current_power_input', 0.0) < 0 else 0.0
        optimized_bess_load = float(getattr(bess, 'optimized_discharge_power', 0.0))

        current_kvar = float(getattr(bess, 'current_reactive_load', 0.0))
        cost_optimized_kvar = float(getattr(bess, 'optimized_cost_reactive_load', 0.0))
        # ensure rel_optimized_kvar is defined for BESS (fallback to cost value)
        rel_optimized_kvar = float(getattr(bess, 'optimized_rel_reactive_load', cost_optimized_kvar))

        current_cost_hr = float(getattr(bess, 'get_current_operating_cost', lambda: 0.0)())
        cost_optimized_cost_hr = float(getattr(bess, 'get_optimized_operating_cost', lambda: 0.0)())
        rel_optimized_cost_hr = float(getattr(bess, 'get_optimized_operating_cost', lambda: 0.0)())

        mode = getattr(bess, 'mode', 'standby')
        soc = getattr(bess, 'current_soc', 0)
        if mode == 'charging':
            status = f'Charging (SOC: {soc}%)'
        elif mode == 'discharging':
            status = f'Discharging (SOC: {soc}%)'
        else:
            status = f'Standby (SOC: {soc}%)'
        
        row = {
            'ENERGY SOURCE': bess.name,
            'CURRENT LOAD (kW)': round(current_bess_discharge, 2),
            'COST OPTIMIZED LOAD (kW)': round(optimized_bess_load, 2),
            'RELIABILITY OPTIMIZED LOAD (kW)': round(optimized_bess_load, 2),
            'CURRENT KVAR (kVAR)': round(current_kvar, 2),
            'COST OPTIMIZED KVAR (kVAR)': round(cost_optimized_kvar, 2),
            'RELIABILITY OPTIMIZED KVAR (kVAR)': round(rel_optimized_kvar, 2),
            'TOTAL COST (PKR/kWh)': round(float(getattr(bess, 'total_cost', 0.0)), 2),
            'PRODUCTION COST (PKR/kWh)': round(float(getattr(bess, 'production_cost', 0.0)), 2),
            'CARBON COST (PKR/kWh)': round(float(getattr(bess, 'carbon_emission', 0.0)) * float(getattr(self, 'carbon_cost_pkr_kg', 0.0)), 2),
            'CURRENT COST/HR': round(current_cost_hr, 2),
            'COST OPTIMIZED COST/HR': round(cost_optimized_cost_hr, 2),
            'RELIABILITY OPTIMIZED COST/HR': round(rel_optimized_cost_hr, 2),
            'COST OPTIMIZED CHARGE (kW)': round(float(getattr(bess, 'optimized_charge_power', 0.0)), 2),
            'RELIABILITY OPTIMIZED CHARGE (kW)': round(float(getattr(bess, 'optimized_charge_power', 0.0)), 2),
            'COST OPTIMIZED DISCHARGE (kW)': round(float(getattr(bess, 'optimized_discharge_power', 0.0)), 2),
            'RELIABILITY OPTIMIZED DISCHARGE (kW)': round(float(getattr(bess, 'optimized_discharge_power', 0.0)), 2),
            'GRID FEED (kW)': 0.0,
            'RELIABILITY SCORE': round(float(getattr(bess, 'reliability_score', 0.0)), 2),
            'EFFICIENCY SCORE': round(0.0, 2),  # Placeholder (not used for BESS priority)
            'STATUS': status
        }
        
        results.append(row)
        
        total_current_load += current_bess_discharge
        total_cost_optimized_load += optimized_bess_load
        total_rel_optimized_load += optimized_bess_load
        total_current_cost += current_cost_hr
        total_cost_optimized_cost += cost_optimized_cost_hr
        total_rel_optimized_cost += rel_optimized_cost_hr
        total_current_kvar += current_kvar
        total_cost_optimized_kvar += cost_optimized_kvar
        total_rel_optimized_kvar += rel_optimized_kvar

    # Add total row
    total_savings_hr = total_current_cost - total_cost_optimized_cost
    results.append({
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
        'COST OPTIMIZED CHARGE (kW)': round(sum(float(getattr(b, 'optimized_charge_power', 0.0)) for b in self.bess_systems), 2),
        'RELIABILITY OPTIMIZED CHARGE (kW)': round(sum(float(getattr(b, 'optimized_charge_power', 0.0)) for b in self.bess_systems), 2),
        'COST OPTIMIZED DISCHARGE (kW)': round(sum(float(getattr(b, 'optimized_discharge_power', 0.0)) for b in self.bess_systems), 2),
        'RELIABILITY OPTIMIZED DISCHARGE (kW)': round(sum(float(getattr(b, 'optimized_discharge_power', 0.0)) for b in self.bess_systems), 2),
        'GRID FEED (kW)': round(total_grid_feed, 2),
        'RELIABILITY SCORE': '',
        'EFFICIENCY SCORE': '',
        'STATUS': f'Savings: PKR {total_savings_hr:.2f}/hr'
    })

    df_r = pd.DataFrame(results)
    df_r['EFFICIENCY SCORE'] = pd.to_numeric(df_r['EFFICIENCY SCORE'], errors='coerce')
    df_r['priority'] = df_r['EFFICIENCY SCORE'].rank(method='min').fillna(0).astype(int)

    return df_r, total_current_cost, total_savings_hr


def generate_recommendations(self, results_df):
    """Generate optimization recommendations"""
    recommendations = []
    
    # Source-specific recommendations
    for _, row in results_df.iterrows():
        source = row['ENERGY SOURCE']
        if source == 'TOTAL':
            continue
        
        current_kw = float(row.get('CURRENT LOAD (kW)', 0.0))
        cost_optimized_kw = float(row.get('COST OPTIMIZED LOAD (kW)', 0.0))
        rel_optimized_kw = float(row.get('RELIABILITY OPTIMIZED LOAD (kW)', 0.0))
        optimized_kw = cost_optimized_kw if cost_optimized_kw != 0 else rel_optimized_kw

        rec = f"**{source}**:\n"
        
        if 'BESS' in str(source):
            charge_kw = float(row.get('COST OPTIMIZED CHARGE (kW)', 0.0))
            discharge_kw = float(row.get('COST OPTIMIZED DISCHARGE (kW)', 0.0))
            
            if charge_kw > 0:
                rec += f"• Charge at {charge_kw:.1f} kW\n"
            elif discharge_kw > 0:
                rec += f"• Discharge at {discharge_kw:.1f} kW to provide {discharge_kw:.1f} kW to load\n"
            else:
                rec += f"• Maintain standby mode\n"
            
            rec += f"• Current status: {row.get('STATUS', '')}\n"
            rec += f"• Reliability score: {row.get('RELIABILITY SCORE', 0)}\n"
        else:
            if optimized_kw > current_kw:
                rec += f"• Increase load from {current_kw:.1f} kW to {optimized_kw:.1f} kW\n"
            elif optimized_kw < current_kw:
                rec += f"• Reduce load from {current_kw:.1f} kW to {optimized_kw:.1f} kW\n"
            else:
                rec += f"• Maintain current load of {current_kw:.1f} kW\n"
            rec += f"• Reliability score: {row.get('RELIABILITY SCORE', 0)}\n"
        
        recommendations.append(rec)
    
    # Overall summary block kept commented, matching your original style
    # total_row = results_df[results_df['ENERGY SOURCE'] == 'TOTAL'].iloc[0]
    # summary = f"""
    # **OPTIMIZATION SUMMARY**:
    # • Total current load: {total_row['CURRENT LOAD (kW)']} kW
    # • Total cost-optimized load: {total_row['COST OPTIMIZED LOAD (kW)']} kW
    # • Total reliability-optimized load: {total_row['RELIABILITY OPTIMIZED LOAD (kW)']} kW
    # • Current hourly cost: PKR {total_row['CURRENT COST/HR']:,.2f}
    # • Cost-optimized hourly cost: PKR {total_row['COST OPTIMIZED COST/HR']:,.2f}
    # • Reliability-optimized hourly cost: PKR {total_row['RELIABILITY OPTIMIZED COST/HR']:,.2f}
    # • Hourly cost savings: PKR {total_row['CURRENT COST/HR'] - total_row['COST OPTIMIZED COST/HR']:,.2f}
    # • Total BESS charge power: {total_row['COST OPTIMIZED CHARGE (kW)']} kW
    # • Total BESS discharge power: {total_row['COST OPTIMIZED DISCHARGE (kW)']} kW
    # • Grid feed: {total_row['GRID FEED (kW)']} kW
    # """
    # recommendations.append(summary)

    return '\n\n'.join(recommendations)


def validate_results(self, results_df, tol_kw=1e-3, verbose=True):
    """
    Validate that optimized allocations match the running load (within tolerance).
    Checks, using the TOTAL row:
      - Cost path:   cost_opt_load + cost_charge - grid_feed == current_load
      - Reliability: rel_opt_load  + rel_charge  - grid_feed == current_load
    Returns: (ok: bool, report: dict)
    """
    import math

    if results_df is None or results_df.empty:
        if verbose:
            print("Allocation check FAILED: results DataFrame is empty.")
        return False, {"error": "empty_results"}

    total_row = results_df[results_df['ENERGY SOURCE'] == 'TOTAL']
    if total_row.empty:
        if verbose:
            print("Allocation check FAILED: TOTAL row not found.")
        return False, {"error": "total_row_missing"}

    total = total_row.iloc[0]

    def g(key, default=0.0):
        val = total.get(key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(default)

    current_load = g('CURRENT LOAD (kW)')
    cost_opt_load = g('COST OPTIMIZED LOAD (kW)')
    rel_opt_load  = g('RELIABILITY OPTIMIZED LOAD (kW)')
    cost_charge   = g('COST OPTIMIZED CHARGE (kW)')
    rel_charge    = g('RELIABILITY OPTIMIZED CHARGE (kW)')
    grid_feed     = g('GRID FEED (kW)')

    # Net optimized power that must equal running load
    cost_net = cost_opt_load + cost_charge - grid_feed
    rel_net  = rel_opt_load  + rel_charge  - grid_feed

    cost_ok = math.isclose(cost_net, current_load, abs_tol=tol_kw)
    rel_ok  = math.isclose(rel_net,  current_load, abs_tol=tol_kw)
    ok = cost_ok and rel_ok

    issues = []
    if not cost_ok:
        issues.append(f"Cost path mismatch: net={cost_net:.3f} kW vs current={current_load:.3f} kW (Δ={cost_net - current_load:.3f} kW)")
    if not rel_ok:
        issues.append(f"Reliability path mismatch: net={rel_net:.3f} kW vs current={current_load:.3f} kW (Δ={rel_net - current_load:.3f} kW)")

    if verbose:
        if ok:
            print(f"Allocation check PASSED (tolerance {tol_kw} kW).")
            print(f"  Current total: {current_load:.3f} kW | Cost net: {cost_net:.3f} kW | Rel net: {rel_net:.3f} kW | Grid feed: {grid_feed:.3f} kW")
        else:
            print(f"Allocation check FAILED (tolerance {tol_kw} kW).")
            for msg in issues:
                print(" - " + msg)

    report = {
        "tolerance_kw": tol_kw,
        "current_total_kw": round(current_load, 3),
        "cost_net_kw": round(cost_net, 3),
        "rel_net_kw": round(rel_net, 3),
        "grid_feed_kw": round(grid_feed, 3),
        "cost_charge_kw": round(cost_charge, 3),
        "rel_charge_kw": round(rel_charge, 3),
        "ok": ok,
        "issues": issues,
    }
    return ok, report
