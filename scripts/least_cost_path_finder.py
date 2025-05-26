from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterPoint, QgsProcessingParameterVectorDestination,
                       QgsProcessingMultiStepFeedback, QgsVectorLayer, QgsField, QgsFields, 
                       QgsWkbTypes, QgsFeature, QgsGeometry, QgsFeatureSink,
                       QgsLineString, QgsRasterLayer, QgsDistanceArea, QgsProcessingException,
                       QgsPointXY, QgsProcessingParameterNumber, QgsProcessingParameterEnum,
                       QgsPoint, QgsProcessingUtils)
from qgis.PyQt.QtCore import QDateTime
import os
from math import sqrt, floor, isnan, ceil, exp
import heapq
import collections
from time import time

class PathFinder:
    def __init__(self, matrix, heuristic_weight=1.0, slope_weight=0.5, diagonal_factor=1.0, flat_sensitivity=0.5, feedback=None):
        self.feedback = feedback
        self.calculation_started = time()
        
        # Convert original matrix to our internal format
        self.height = len(matrix)
        self.width = len(matrix[0]) if matrix else 0
        
        # Use dictionaries instead of 2D arrays for sparse data
        # Only store valid cells to reduce memory usage
        self.cost_map = {}
        
        # Transfer valid values
        valid_values = []
        valid_count = 0
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val is not None:
                    # Store in format (row, col): value
                    self.cost_map[(i, j)] = float(val)
                    valid_values.append(float(val))
                    valid_count += 1
        
        # Calculate statistics (only once)
        if valid_values:
            self.min_cost = min(valid_values)
            self.max_cost = max(valid_values)
            self.mean_cost = sum(valid_values) / valid_count
            
            # Standard deviation with one pass
            variance = sum((x - self.mean_cost) ** 2 for x in valid_values) / valid_count
            self.std_cost = sqrt(variance)
            
            # Calculate distribution statistics for flat terrain detection
            sorted_values = sorted(valid_values)
            self.percentile_10 = sorted_values[int(valid_count * 0.1)]
            self.percentile_25 = sorted_values[int(valid_count * 0.25)]
            self.median_cost = sorted_values[int(valid_count * 0.5)]
            self.percentile_75 = sorted_values[int(valid_count * 0.75)]
            self.percentile_90 = sorted_values[int(valid_count * 0.9)]
            
            # Calculate the interquartile range for flat detection sensitivity
            self.iqr = self.percentile_75 - self.percentile_25
            
            # Calculate flat threshold - values below this are considered "flat terrain"
            self.flat_threshold = self.percentile_25 + (self.iqr * 0.3)
        else:
            self.min_cost = 0
            self.max_cost = 1
            self.mean_cost = 0.5
            self.std_cost = 0.1
            self.percentile_10 = 0.1
            self.percentile_25 = 0.25
            self.median_cost = 0.5
            self.percentile_75 = 0.75
            self.percentile_90 = 0.9
            self.iqr = 0.5
            self.flat_threshold = 0.4
        
        # Parameters
        self.heuristic_weight = heuristic_weight
        self.slope_weight = slope_weight
        self.diagonal_factor = diagonal_factor
        self.flat_sensitivity = flat_sensitivity  # New parameter for flat terrain sensitivity
        
        # Directions (8 neighbors) - prioritize orthogonal movements first
        self.directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # Adjacent (orthogonal) - check these first
            (1, 1), (-1, 1), (-1, -1), (1, -1)  # Diagonals
        ]
        
        # Prepare gradients for costs - precalculated but not parallel
        self._calculate_gradients()
        
        # Debug
        if self.feedback:
            elapsed = time() - self.calculation_started
            self.feedback.pushInfo(f"Matrix shape: {self.height}x{self.width}")
            self.feedback.pushInfo(f"Valid cells: {valid_count}")
            self.feedback.pushInfo(f"Cost range: {self.min_cost} to {self.max_cost}")
            self.feedback.pushInfo(f"Median cost: {self.median_cost}, IQR: {self.iqr}")
            self.feedback.pushInfo(f"Flat terrain threshold: {self.flat_threshold}")
            self.feedback.pushInfo(f"Initialization took {elapsed:.2f} seconds")
    
    def _calculate_gradients(self):
        """Calculate gradients and microrelief indicators"""
        # Use a dictionary for gradient storage - only store cells that need it
        self.gradient_map = {}
        
        # Also track micro-relief for flat area enhancement
        self.microrelief_map = {}
        
        # Normalize factor (precomputed)
        norm_factor = max(1, (self.max_cost - self.min_cost))
        flat_norm_factor = max(0.01, self.iqr)  # For flat areas, use IQR to normalize
        
        # Calculate gradients for all valid cells that have valid neighbors
        for pos in self.cost_map:
            i, j = pos
            
            # Track maximum local gradient (for microrelief detection)
            max_local_gradient = 0
            
            # Check all 8 neighbors for gradient calculation
            for di, dj in self.directions:
                ni, nj = i + di, j + dj
                if (ni, nj) in self.cost_map:
                    center_val = self.cost_map[(i, j)]
                    neighbor_val = self.cost_map[(ni, nj)]
                    
                    # Calculate absolute gradient
                    abs_gradient = abs(center_val - neighbor_val)
                    
                    # Choose normalization factor based on terrain type
                    is_flat_terrain = (center_val < self.flat_threshold and 
                                      neighbor_val < self.flat_threshold)
                    
                    if is_flat_terrain:
                        # In flat areas, we amplify small differences
                        # Apply exponential scaling for very small differences
                        # This makes the algorithm more sensitive to micro-variations in flat terrain
                        rel_gradient = abs_gradient / flat_norm_factor
                        amplified_gradient = rel_gradient * (2.0 + self.flat_sensitivity)
                        
                        # Add extra penalty for even tiny variations in flat terrain
                        # with flat_sensitivity as a control parameter (0-1)
                        if abs_gradient > 0:
                            micro_bonus = self.flat_sensitivity * (1.0 - exp(-abs_gradient * 10))
                            amplified_gradient += micro_bonus
                        
                        # Store normalized gradient, amplified for flat areas
                        self.gradient_map[((i, j), (ni, nj))] = amplified_gradient
                    else:
                        # Regular gradient normalization for non-flat terrain
                        norm_gradient = abs_gradient / norm_factor
                        self.gradient_map[((i, j), (ni, nj))] = norm_gradient
                    
                    # Update maximum local gradient
                    max_local_gradient = max(max_local_gradient, abs_gradient)
            
            # Store microrelief information for this cell
            # This helps identify areas with subtle terrain variations
            self.microrelief_map[pos] = max_local_gradient
    
    def is_valid(self, pos):
        """Checks if a position has valid cost data"""
        return pos in self.cost_map
    
    def get_neighbors(self, pos):
        """Gets valid neighbors of a position - optimized version"""
        r, c = pos
        neighbors = []
        
        # Check all 8 neighbors - orthogonal first, then diagonal
        for dr, dc in self.directions:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            
            if self.is_valid(neighbor):
                # Check if it's diagonal
                if dr != 0 and dc != 0:
                    # For diagonals, verify if there's a direct path
                    if self.is_valid((r, c + dc)) or self.is_valid((r + dr, c)):
                        neighbors.append(neighbor)
                else:
                    neighbors.append(neighbor)
        
        return neighbors
    
    def cost_between(self, current, next_pos):
        """Calculates the cost between two adjacent cells - with enhanced flat terrain sensitivity"""
        # Base values
        current_val = self.cost_map[current]
        next_val = self.cost_map[next_pos]
        
        # Detect if we're in flat terrain (both cells below threshold)
        is_flat_terrain = (current_val < self.flat_threshold and 
                          next_val < self.flat_threshold)
        
        # Base cost (average)
        base_cost = (current_val + next_val) / 2.0
        
        # In very flat terrain, ensure minimum cost differentiation
        if is_flat_terrain and self.flat_sensitivity > 0:
            # Get microrelief information for both cells
            current_relief = self.microrelief_map.get(current, 0)
            next_relief = self.microrelief_map.get(next_pos, 0)
            
            # Calculate a local relief factor based on how flat the area is
            # This adds extra cost variations in flat areas
            local_relief_factor = (current_relief + next_relief) * 0.5
            
            # Scale factor based on sensitivity 
            # Higher flat_sensitivity means more emphasis on micro-terrain
            flat_factor = 1.0 + (self.flat_sensitivity * 
                               (1.0 - exp(-local_relief_factor * 20)))
            
            # Apply the flat terrain factor to base cost
            base_cost *= flat_factor
        
        # Check if diagonal (compute only once)
        r1, c1 = current
        r2, c2 = next_pos
        is_diagonal = r1 != r2 and c1 != c2
        
        # Apply movement factor
        if is_diagonal:
            movement_factor = sqrt(2) * self.diagonal_factor
        else:
            movement_factor = 1.0
        
        # Slope factor (only if needed)
        if self.slope_weight > 0:
            # Get gradient between cells (or 0 if not calculated)
            gradient = self.gradient_map.get((current, next_pos), 
                      self.gradient_map.get((next_pos, current), 0))
            
            # Apply slope factor directly
            slope_factor = 1.0 + self.slope_weight * gradient
        else:
            # Skip calculation if slope weight is 0
            slope_factor = 1.0
        
        # Calculate final cost in one step
        return max(0.0001, base_cost * movement_factor * slope_factor)
    
    def heuristic(self, pos, goal):
        """Estimates the distance to the goal"""
        r1, c1 = pos
        r2, c2 = goal
        
        # Calculate both Manhattan and Euclidean distances
        manhattan = abs(r1 - r2) + abs(c1 - c2)
        euclidean = sqrt((r1 - r2)**2 + (c1 - c2)**2)
        
        # Use a weighted combination for better results
        # Higher heuristic_weight emphasizes Euclidean, lower emphasizes Manhattan
        if self.heuristic_weight > 1.5:
            # For speed focus, prioritize Euclidean distance
            h_distance = euclidean
        elif self.heuristic_weight < 0.7:
            # For accuracy focus, use a blend with more Manhattan influence
            h_distance = (euclidean + manhattan) / 2
        else:
            # For balanced, use a weighted blend
            h_distance = (euclidean * 0.7) + (manhattan * 0.3)
        
        # In flat regions, we want to reduce the heuristic influence to allow
        # the algorithm to find more optimal paths over micro-terrain
        if self.cost_map.get(pos, float('inf')) < self.flat_threshold:
            # Reduce heuristic weight in flat areas to favor exploration
            flat_heuristic_factor = max(0.6, 1.0 - (self.flat_sensitivity * 0.4))
            return h_distance * self.mean_cost * self.heuristic_weight * flat_heuristic_factor
        else:
            # Normal heuristic for non-flat areas
            return h_distance * self.mean_cost * self.heuristic_weight
    
    def find_path(self, start, end, max_iterations=1000000):
        """Finds the optimal path between two points - optimized"""
        calculation_started = time()
        
        # Validate start and end
        if not self.is_valid(start) or not self.is_valid(end):
            if self.feedback:
                self.feedback.pushInfo("Invalid start or end position")
            return None
        
        # Direct distance for progress estimation
        direct_distance = sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
        
        # Scale max iterations by distance, complexity, and optimization level
        base_iterations = direct_distance * 2000
        complexity_factor = max(1, (self.max_cost - self.min_cost) / max(0.1, self.mean_cost))
        
        # Adjust max_iterations based on heuristic_weight
        if self.heuristic_weight >= 1.5:  # Speed focus
            # Speed focus needs fewer iterations
            max_iterations = min(max_iterations, int(base_iterations * 1.2))
        elif self.heuristic_weight <= 0.7:  # Accuracy focus
            # Accuracy focus needs more iterations
            max_iterations = min(2500000, int(base_iterations * 2.5 * complexity_factor))
        else:  # Balanced
            # Balanced approach
            max_iterations = min(2000000, int(base_iterations * 1.8 * complexity_factor))
        
        # If in flat terrain, increase iterations to allow more exploration
        if (self.cost_map.get(start, float('inf')) < self.flat_threshold or 
            self.cost_map.get(end, float('inf')) < self.flat_threshold):
            # Increase iterations based on flat_sensitivity
            flat_factor = 1.0 + (self.flat_sensitivity * 1.5)
            max_iterations = int(max_iterations * flat_factor)
        
        # Debug
        if self.feedback:
            self.feedback.pushInfo(f"Direct distance: {direct_distance:.1f} cells")
            self.feedback.pushInfo(f"Max iterations set to: {max_iterations}")
            self.feedback.pushInfo(f"Start cost: {self.cost_map.get(start, 'N/A')}")
            self.feedback.pushInfo(f"End cost: {self.cost_map.get(end, 'N/A')}")
        
        # Initialize A* with optimized data structures
        open_set = []  # Priority queue
        counter = 0  # Counter for tiebreakers
        heapq.heappush(open_set, (0, counter, start))
        
        came_from = {start: None}
        cost_so_far = {start: 0}
        closed_set = set()
        
        iterations = 0
        progress_interval = max(1, int(max_iterations / 20))  # Less frequent updates
        
        # Search for path
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Cancellation
            if self.feedback and self.feedback.isCanceled():
                return None
            
            # Update progress less frequently
            if self.feedback and iterations % progress_interval == 0:
                self.feedback.setProgress(min(99, 100 * iterations / max_iterations))
            
            # Get current node
            _, _, current = heapq.heappop(open_set)
            
            # If already processed, skip
            if current in closed_set:
                continue
                
            # Mark as processed
            closed_set.add(current)
            
            # If destination reached, reconstruct path
            if current == end:
                # Reconstruct path efficiently
                path = []
                costs = []
                node = current
                
                while node is not None:
                    path.append(node)
                    costs.append(cost_so_far[node])
                    node = came_from[node]
                
                path.reverse()
                costs.reverse()
                
                elapsed = time() - calculation_started
                
                if self.feedback:
                    self.feedback.pushInfo(f"Path found after {iterations} iterations")
                    self.feedback.pushInfo(f"Path length: {len(path)} cells")
                    self.feedback.pushInfo(f"Total cells visited: {len(closed_set)}")
                    self.feedback.pushInfo(f"Pathfinding took {elapsed:.2f} seconds")
                
                return path, costs
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                # Get accumulated cost (single calculation)
                new_cost = cost_so_far[current] + self.cost_between(current, neighbor)
                
                # If we found a better path
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    
                    # Priority = current cost + heuristic
                    priority = new_cost + self.heuristic(neighbor, end)
                    
                    counter += 1
                    heapq.heappush(open_set, (priority, counter, neighbor))
                    came_from[neighbor] = current
        
        # No path found
        elapsed = time() - calculation_started
        if self.feedback:
            if iterations >= max_iterations:
                self.feedback.pushInfo(f"Search terminated after {max_iterations} iterations")
            else:
                self.feedback.pushInfo("No path found - all possible routes exhausted")
            self.feedback.pushInfo(f"Cells visited: {len(closed_set)}")
            self.feedback.pushInfo(f"Pathfinding took {elapsed:.2f} seconds")
        
        return None


class FastOptimalPathAlgorithm(QgsProcessingAlgorithm):
    INPUT_COST_RASTER = 'INPUT_COST_RASTER'
    POINT_START = 'POINT_START'
    POINT_INTERMEDIATE = 'POINT_INTERMEDIATE'
    POINT_END = 'POINT_END'
    OUTPUT_PATH = 'OUTPUT_PATH'
    
    # Parameters in the requested order
    PARAM_OPTIMIZATION_LEVEL = 'PARAM_OPTIMIZATION_LEVEL'
    PARAM_HEURISTIC_WEIGHT = 'PARAM_HEURISTIC_WEIGHT'
    PARAM_SLOPE_WEIGHT = 'PARAM_SLOPE_WEIGHT'
    PARAM_DIAGONAL_FACTOR = 'PARAM_DIAGONAL_FACTOR'
    PARAM_FLAT_SENSITIVITY = 'PARAM_FLAT_SENSITIVITY'  # New parameter

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_COST_RASTER,
                self.tr('Cost Raster'),
                optional=False
            )
        )

        self.addParameter(
            QgsProcessingParameterPoint(
                self.POINT_START,
                self.tr('Starting Point'),
                optional=False
            )
        )

        self.addParameter(
            QgsProcessingParameterPoint(
                self.POINT_INTERMEDIATE,
                self.tr('Intermediate Point'),
                optional=True
            )
        )

        self.addParameter(
            QgsProcessingParameterPoint(
                self.POINT_END,
                self.tr('Ending Point'),
                optional=False
            )
        )
        
        # Parameters in the requested order
        self.addParameter(
            QgsProcessingParameterEnum(
                self.PARAM_OPTIMIZATION_LEVEL,
                self.tr('Optimization Level'),
                options=['Balanced', 'Speed Focus', 'Accuracy Focus', 'Flat Terrain Focus'],  # Added new option
                defaultValue=0,
                optional=True
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_HEURISTIC_WEIGHT,
                self.tr('Heuristic Weight (1.0 = balanced)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
                optional=True,
                minValue=0.1,
                maxValue=5.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_SLOPE_WEIGHT,
                self.tr('Slope Influence (0.0 = ignore slopes)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5,
                optional=True,
                minValue=0.0,
                maxValue=1.0
            )
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_DIAGONAL_FACTOR,
                self.tr('Diagonal Movement Cost (1.0 = normal)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
                optional=True,
                minValue=0.5,
                maxValue=2.0
            )
        )
        
        # New parameter for flat terrain sensitivity
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_FLAT_SENSITIVITY,
                self.tr('Flat Terrain Sensitivity (0.0 = ignore, 1.0 = high)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5,
                optional=True,
                minValue=0.0,
                maxValue=1.0
            )
        )

        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_PATH,
                self.tr('Output Optimal Path'),
                type=QgsProcessing.TypeVectorLine
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        total_steps = 3
        multi_feedback = QgsProcessingMultiStepFeedback(total_steps, feedback)
        
        try:
            # Get parameters
            cost_raster = self.parameterAsRasterLayer(parameters, self.INPUT_COST_RASTER, context)
            point_start = self.parameterAsPoint(parameters, self.POINT_START, context, cost_raster.crs())
            point_intermediate = self.parameterAsPoint(parameters, self.POINT_INTERMEDIATE, context, cost_raster.crs())
            point_end = self.parameterAsPoint(parameters, self.POINT_END, context, cost_raster.crs())
            
            heuristic_weight = self.parameterAsDouble(parameters, self.PARAM_HEURISTIC_WEIGHT, context)
            slope_weight = self.parameterAsDouble(parameters, self.PARAM_SLOPE_WEIGHT, context)
            diagonal_factor = self.parameterAsDouble(parameters, self.PARAM_DIAGONAL_FACTOR, context)
            flat_sensitivity = self.parameterAsDouble(parameters, self.PARAM_FLAT_SENSITIVITY, context)
            optimization_level = self.parameterAsEnum(parameters, self.PARAM_OPTIMIZATION_LEVEL, context)
            
            # Adjust parameters based on optimization level
            if optimization_level == 1:  # Speed focus
                heuristic_weight = max(1.8, heuristic_weight)  # More aggressive heuristic
                slope_weight = min(0.3, slope_weight)          # Less concern for slopes
                diagonal_factor = min(0.9, diagonal_factor)    # Encourage diagonals for speed
                flat_sensitivity = min(0.3, flat_sensitivity)  # Lower flat terrain sensitivity
            elif optimization_level == 2:  # Accuracy focus
                heuristic_weight = min(0.65, heuristic_weight) # Less heuristic influence
                slope_weight = max(0.6, slope_weight)          # More attention to slopes
                diagonal_factor = max(1.1, diagonal_factor)    # More careful with diagonals
                flat_sensitivity = max(0.6, flat_sensitivity)  # Higher flat terrain sensitivity
            elif optimization_level == 3:  # Flat Terrain focus - new option
                heuristic_weight = min(0.6, heuristic_weight)  # Low heuristic for exploration
                slope_weight = max(0.7, slope_weight)          # High slope sensitivity
                diagonal_factor = max(1.1, diagonal_factor)    # Careful with diagonals
                flat_sensitivity = max(0.8, flat_sensitivity)  # Maximum flat terrain sensitivity
            else:  # Balanced
                heuristic_weight = 1.0     # Neutral heuristic
                slope_weight = 0.4         # Moderate slope consideration
                diagonal_factor = 0.95     # Slight diagonal preference
                flat_sensitivity = 0.5     # Moderate flat terrain sensitivity
            
            # Validate inputs
            if not cost_raster.isValid():
                raise QgsProcessingException("Invalid Cost Raster")

            # Step 1: Prepare matrix
            multi_feedback.setCurrentStep(0)
            multi_feedback.pushInfo(f"Step 1/{total_steps}: Preparing cost matrix...")
            
            # Load raster data (only once)
            t_start = time()
            block = cost_raster.dataProvider().block(1, cost_raster.extent(), cost_raster.width(), cost_raster.height())
            matrix = [[None if block.isNoData(i, j) else abs(block.value(i, j)) for j in range(block.width())]
                      for i in range(block.height())]
            
            multi_feedback.pushInfo(f"Raster loading took {time() - t_start:.2f} seconds")
            
            # Step 2: Find optimal path
            multi_feedback.setCurrentStep(1)
            multi_feedback.pushInfo(f"Step 2/{total_steps}: Finding optimal path...")
            
            # Convert points to raster coordinates
            start_row, start_col = self.point_to_raster_coordinates(point_start, cost_raster)
            end_row, end_col = self.point_to_raster_coordinates(point_end, cost_raster)
            
            multi_feedback.pushInfo(f"Start point: ({point_start.x()}, {point_start.y()}) -> cell ({start_row}, {start_col})")
            multi_feedback.pushInfo(f"End point: ({point_end.x()}, {point_end.y()}) -> cell ({end_row}, {end_col})")
            
            # Normalize coordinates to be within bounds
            start_row = max(0, min(start_row, block.height() - 1))
            start_col = max(0, min(start_col, block.width() - 1))
            end_row = max(0, min(end_row, block.height() - 1))
            end_col = max(0, min(end_col, block.width() - 1))
            
            # Create path finder with parameters
            pathfinder = PathFinder(
                matrix, 
                heuristic_weight=heuristic_weight,
                slope_weight=slope_weight,
                diagonal_factor=diagonal_factor,
                flat_sensitivity=flat_sensitivity,
                feedback=multi_feedback
            )
            
            # Check for intermediate point
            has_intermediate = point_intermediate is not None and point_intermediate != QgsPointXY()
            
            if has_intermediate:
                # Process with intermediate point
                interm_row, interm_col = self.point_to_raster_coordinates(point_intermediate, cost_raster)
                interm_row = max(0, min(interm_row, block.height() - 1))
                interm_col = max(0, min(interm_col, block.width() - 1))
                
                multi_feedback.pushInfo(f"Intermediate point: ({point_intermediate.x()}, {point_intermediate.y()}) -> cell ({interm_row}, {interm_col})")
                
                # Find first segment
                multi_feedback.pushInfo("Finding path from start to intermediate point...")
                result_start_to_interm = pathfinder.find_path((start_row, start_col), (interm_row, interm_col))
                
                if not result_start_to_interm:
                    # Try with fallback parameters for intermediate segment
                    multi_feedback.pushWarning("Initial path to intermediate point failed. Trying with different parameters...")
                    
                    alt_pathfinder = PathFinder(
                        matrix, 
                        heuristic_weight=0.6,      # More thorough search
                        slope_weight=0.7,          # Higher slope sensitivity
                        diagonal_factor=0.8,       # More diagonal freedom
                        flat_sensitivity=0.8,      # High flat terrain sensitivity
                        feedback=multi_feedback
                    )
                    
                    result_start_to_interm = alt_pathfinder.find_path((start_row, start_col), (interm_row, interm_col))
                    
                    if not result_start_to_interm:
                        raise QgsProcessingException("No path found between start and intermediate points. Try adjusting parameters or positions.")
                
                path_part1, costs_part1 = result_start_to_interm
                
                # Find second segment
                multi_feedback.pushInfo("Finding path from intermediate to end point...")
                result_interm_to_end = pathfinder.find_path((interm_row, interm_col), (end_row, end_col))
                
                if not result_interm_to_end:
                    # Try with fallback parameters for second segment
                    multi_feedback.pushWarning("Initial path from intermediate to end failed. Trying with different parameters...")
                    
                    alt_pathfinder = PathFinder(
                        matrix, 
                        heuristic_weight=0.6,      # More thorough search
                        slope_weight=0.7,          # Higher slope sensitivity
                        diagonal_factor=0.8,       # More diagonal freedom
                        flat_sensitivity=0.8,      # High flat terrain sensitivity
                        feedback=multi_feedback
                    )
                    
                    result_interm_to_end = alt_pathfinder.find_path((interm_row, interm_col), (end_row, end_col))
                    
                    if not result_interm_to_end:
                        raise QgsProcessingException("No path found between intermediate and end points. Try adjusting parameters or positions.")
                
                path_part2, costs_part2 = result_interm_to_end
                
                # Combine paths
                path = path_part1[:-1] + path_part2  # Avoid duplicate intermediate point
                
                # Adjust costs for second segment
                base_cost = costs_part1[-1]
                adjusted_costs = costs_part1[:-1] + [base_cost + c for c in costs_part2]
                costs = adjusted_costs
                
            else:
                # Direct path
                multi_feedback.pushInfo("Finding direct path from start to end...")
                result = pathfinder.find_path((start_row, start_col), (end_row, end_col))
                
                if not result:
                    # Try alternative parameters if first attempt fails
                    multi_feedback.pushWarning("Initial path finding failed. Trying with different parameters...")
                    
                    # Better fallback parameters with enhanced flat terrain sensitivity
                    alt_param_sets = [
                        # Try with lower heuristic and high flat sensitivity
                        {"heuristic_weight": 0.55, "slope_weight": 0.6, "diagonal_factor": 0.9, "flat_sensitivity": 0.9},
                        # Try with balanced approach and high flat sensitivity
                        {"heuristic_weight": 0.8, "slope_weight": 0.5, "diagonal_factor": 0.8, "flat_sensitivity": 0.8},
                        # Finally try with minimal slope consideration but still sensitive to flat terrain
                        {"heuristic_weight": 1.2, "slope_weight": 0.2, "diagonal_factor": 0.7, "flat_sensitivity": 0.7}
                    ]
                    
                    for i, params in enumerate(alt_param_sets):
                        multi_feedback.pushInfo(f"Trying alternate parameter set {i+1}...")
                        
                        alt_pathfinder = PathFinder(
                            matrix,
                            heuristic_weight=params["heuristic_weight"],
                            slope_weight=params["slope_weight"], 
                            diagonal_factor=params["diagonal_factor"],
                            flat_sensitivity=params["flat_sensitivity"],
                            feedback=multi_feedback
                        )
                        
                        result = alt_pathfinder.find_path((start_row, start_col), (end_row, end_col))
                        
                        if result:
                            multi_feedback.pushInfo(f"Alternate parameter set {i+1} found a path!")
                            break
                    
                    if not result:
                        # Try straight line approach as last resort
                        multi_feedback.pushWarning("All alternative path finding failed. Trying simple line approach...")
                        path, costs = self.create_direct_line_path((start_row, start_col), (end_row, end_col), matrix)
                    else:
                        path, costs = result
                else:
                    path, costs = result
            
            # Step 3: Convert path to geometry
            multi_feedback.setCurrentStep(2)
            multi_feedback.pushInfo(f"Step 3/{total_steps}: Creating output geometry...")

            # Create fields for output
            fields = QgsFields()
            fields.append(QgsField('id', QVariant.Int))
            fields.append(QgsField('length_m', QVariant.Double))
            fields.append(QgsField('total_cost', QVariant.Double))
            fields.append(QgsField('segments', QVariant.Int))
            fields.append(QgsField('created', QVariant.String))
            fields.append(QgsField('flat_analyzed', QVariant.Int))  # New field to indicate flat terrain analysis

            # Setup output layer
            (sink, dest_id) = self.parameterAsSink(
                parameters, self.OUTPUT_PATH, context,
                fields, QgsWkbTypes.LineString, cost_raster.crs()
            )

            if sink is None:
                raise QgsProcessingException("Failed to create output layer")

            # Convert raster coordinates to world coordinates
            path_points = self.raster_path_to_world_coordinates(path, cost_raster, point_start, point_end, 
                                                             point_intermediate if has_intermediate else None,
                                                             flat_sensitivity=flat_sensitivity)

            # Critical validation: Ensure at least 2 points in the geometry
            if len(path_points) < 2:
                raise QgsProcessingException("Path must contain at least 2 points to form a line.")

            # Create geometry and validate
            line_geom = QgsGeometry.fromPolylineXY(path_points)
            if not line_geom.isGeosValid():
                line_geom = line_geom.makeValid()
                if line_geom.isEmpty():
                    raise QgsProcessingException("Corrected geometry is empty.")

            # Length calculation with CRS verification
            distance_area = QgsDistanceArea()
            distance_area.setSourceCrs(cost_raster.crs(), context.transformContext())
            distance_area.setEllipsoid(cost_raster.crs().ellipsoidAcronym())
            
            try:
                path_length = distance_area.measureLength(line_geom)
            except Exception as e:
                raise QgsProcessingException(f"Error calculating length: {str(e)}")

            # Safe attribute assignment
            feat = QgsFeature(fields)
            feat.setGeometry(line_geom)
            feat.setAttributes([
                1,
                float(path_length) if path_length else 0.0,
                float(costs[-1]) if costs else 0.0,
                len(path) - 1,
                QDateTime.currentDateTime().toString('yyyy-MM-dd HH:mm:ss'),
                1 if flat_sensitivity > 0 else 0  # Indicate if flat terrain analysis was used
            ])

            sink.addFeature(feat, QgsFeatureSink.FastInsert)

            # Success message
            multi_feedback.pushInfo(f"Path calculation complete!")
            multi_feedback.pushInfo(f"Path length: {path_length:.2f} meters")
            multi_feedback.pushInfo(f"Total cost: {costs[-1]:.2f}")
            multi_feedback.pushInfo(f"Path segments: {len(path) - 1}")
            multi_feedback.pushInfo(f"Flat terrain sensitivity: {flat_sensitivity:.2f}")
            
            return {self.OUTPUT_PATH: dest_id}
            
        except Exception as e:
            import traceback
            feedback.pushWarning(traceback.format_exc())
            raise QgsProcessingException(f"Error in path calculation: {str(e)}")
    
    def point_to_raster_coordinates(self, point, raster_layer):
        """Converts a point from world coordinates to raster coordinates"""
        xres = raster_layer.rasterUnitsPerPixelX()
        yres = raster_layer.rasterUnitsPerPixelY()
        extent = raster_layer.dataProvider().extent()
        
        col = int((point.x() - extent.xMinimum()) / xres)
        row = int((extent.yMaximum() - point.y()) / yres)
        
        return row, col
    
    def raster_coordinates_to_world(self, row, col, raster_layer):
        """Converts raster coordinates to world coordinates"""
        xres = raster_layer.rasterUnitsPerPixelX()
        yres = raster_layer.rasterUnitsPerPixelY()
        extent = raster_layer.dataProvider().extent()
        
        x = (col + 0.5) * xres + extent.xMinimum()
        y = extent.yMaximum() - (row + 0.5) * yres
        
        return QgsPointXY(x, y)
    
    def raster_path_to_world_coordinates(self, path, raster_layer, start_point, end_point, intermediate_point=None, flat_sensitivity=0.5):
        """Converts a path from raster coordinates to world coordinates"""
        result = []
        
        # First point (start)
        result.append(start_point)
        
        # Intermediate points
        if len(path) > 2:
            # If there's a specified intermediate point
            interm_idx = -1
            if intermediate_point:
                interm_row, interm_col = self.point_to_raster_coordinates(intermediate_point, raster_layer)
                
                # Find the index of the point closest to the intermediate in the path
                for i, (row, col) in enumerate(path[1:-1], 1):
                    if abs(row - interm_row) <= 1 and abs(col - interm_col) <= 1:
                        interm_idx = i
                        break
            
            # Convert path points efficiently
            # In flat terrain with high sensitivity, include more points to capture micro-variations
            path_len = len(path)
            
            # Adaptive simplification based on path length
            if path_len > 1000:
                # For very long paths in flat terrain, keep more points
                max_points = 800 if flat_sensitivity > 0.7 else 500
                skip_factor = max(1, path_len // max_points)
                
                for i, (row, col) in enumerate(path[1:-1], 1):
                    if i == interm_idx:
                        # Always include intermediate point
                        result.append(intermediate_point)
                    elif i % skip_factor == 0:
                        # Only include every nth point
                        result.append(self.raster_coordinates_to_world(row, col, raster_layer))
            else:
                # For shorter paths, include all points
                for i, (row, col) in enumerate(path[1:-1], 1):
                    if i == interm_idx:
                        result.append(intermediate_point)
                    else:
                        result.append(self.raster_coordinates_to_world(row, col, raster_layer))
        
        # Last point (end)
        result.append(end_point)
        
        return result
    
    def create_direct_line_path(self, start, end, matrix):
        """Creates a direct line path between two points - optimized"""
        path = [start]
        costs = [0]
        
        # Bresenham algorithm for line
        start_row, start_col = start
        end_row, end_col = end
        
        dx = abs(end_col - start_col)
        dy = abs(end_row - start_row)
        
        sx = 1 if start_col < end_col else -1
        sy = 1 if start_row < end_row else -1
        
        err = dx - dy
        
        current_row, current_col = start_row, start_col
        cost_so_far = 0
        
        while current_row != end_row or current_col != end_col:
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                current_col += sx
            
            if e2 < dx:
                err += dx
                current_row += sy
            
            # Only include if valid
            if 0 <= current_row < len(matrix) and 0 <= current_col < len(matrix[0]):
                if matrix[current_row][current_col] is not None:
                    # Add to path
                    path.append((current_row, current_col))
                    
                    # Calculate approximate accumulated cost
                    if len(path) > 1:
                        prev_row, prev_col = path[-2]
                        
                        # Cost using simple average
                        if matrix[prev_row][prev_col] is not None:
                            segment_cost = (matrix[prev_row][prev_col] + matrix[current_row][current_col]) / 2
                            
                            # Adjust for diagonal movement
                            if prev_row != current_row and prev_col != current_col:
                                segment_cost *= sqrt(2)
                                
                            cost_so_far += segment_cost
                            costs.append(cost_so_far)
        
        # Ensure the last point is the destination
        if path[-1] != end:
            path.append(end)
            
            # Estimate cost of last segment
            if len(path) > 1 and matrix[end_row][end_col] is not None:
                prev_row, prev_col = path[-2]
                if matrix[prev_row][prev_col] is not None:
                    segment_cost = (matrix[prev_row][prev_col] + matrix[end_row][end_col]) / 2
                    
                    # Adjust for diagonal movement
                    if prev_row != end_row and prev_col != end_col:
                        segment_cost *= sqrt(2)
                        
                    cost_so_far += segment_cost
                    costs.append(cost_so_far)
        
        return path, costs

    def name(self):
        return 'fastoptimalpath'

    def displayName(self):
        return self.tr('Fast Optimal Path Finder')
    
    def group(self):
        return self.tr('ArcGeek Calculator')
    
    def groupId(self):
        return 'arcgeekcalculator'
    
    def shortHelpString(self):
        return self.tr("""Calculates the optimal path between points based on a cost raster.
        
        Parameters:
        - Cost Raster: Raster layer representing the cost of movement
        - Starting Point: Origin of the path
        - Intermediate Point (optional): Optional waypoint to pass through
        - Ending Point: Destination of the path
        
        Advanced Options:
        - Optimization Level: Pre-configured parameter sets for different scenarios
          * Balanced: Good for general routing with reasonable balance of speed and accuracy
          * Speed Focus: Faster calculation but may miss optimal routes in complex terrain
          * Accuracy Focus: More thorough search for optimal paths in complex terrain
          * Flat Terrain Focus: Specialized mode for capturing subtle variations in flat areas
        
        - Heuristic Weight (0.1-5.0): Controls path finding behavior
          * Lower values (0.1-0.8): More thorough search, better for complex terrain with barriers
          * Values around 1.0: Balanced approach for most scenarios
          * Higher values (1.2-5.0): Faster computation but may miss optimal routes
        
        - Slope Influence (0.0-1.0): How much elevation changes affect path selection
          * 0.0: Ignores terrain slopes completely
          * 0.1-0.3: Minimal consideration, good for flat areas or where slope isn't critical
          * 0.4-0.6: Moderate consideration, suitable for most scenarios
          * 0.7-1.0: High sensitivity to slopes, ideal for hiking trails or drainage channels
        
        - Diagonal Movement Cost (0.5-2.0): Controls path shape and preference for diagonal movement
          * Below 1.0: Encourages diagonal movement, resulting in more direct routes
          * 1.0: Standard cost (diagonal = sqrt(2) times orthogonal)
          * Above 1.0: Discourages diagonal movement, creates paths with more right angles
          
        - Flat Terrain Sensitivity (0.0-1.0): Controls how the algorithm responds to subtle terrain variations
          * 0.0: Ignores micro-variations in flat terrain (straight lines through flat areas)
          * 0.3-0.5: Moderate sensitivity to micro-terrain features (slight deviations in flat areas)
          * 0.6-1.0: High sensitivity to even tiny variations (follows minor terrain undulations)
          
        Practical Applications:
        - For hiking trails: Use Accuracy Focus with Slope Influence 0.7-0.9
        - For drainage channels: Use Flat Terrain Focus with Slope Influence 0.8-1.0
        - For irrigation channels: Use Flat Terrain Focus with Slope Influence 0.9 and Diagonal 1.2+
        - For electric/utility lines: Use Speed Focus with Slope Influence 0.3-0.5
        - For general access roads: Use Balanced mode with Slope Influence 0.5-0.7
        - For detailed hydrological analysis: Use Flat Terrain Focus with Flat Sensitivity 0.9-1.0
        """)
    
    def tr(self, string):
        return QCoreApplication.translate('Processing', string)
    
    def createInstance(self):
        return FastOptimalPathAlgorithm()


class LeastCostPathFinder(FastOptimalPathAlgorithm):
    def name(self):
        return 'leastcostpathfinder'
    
    def displayName(self):
        return self.tr('Least Cost Path Finder')
        
    def createInstance(self):
        return LeastCostPathFinder()                