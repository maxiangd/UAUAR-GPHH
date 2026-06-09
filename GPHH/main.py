import copy
import operator
import random
from collections import defaultdict, namedtuple
import multiprocessing

import numpy

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Global configuration (all original values preserved)
START_NODE = 1  # Depot/charging station
MAX_BATTERY = 100  # Full battery capacity
# Training and test dataset configuration
TRAIN_SEED_BASE = 1000  # Base seed for training dataset
TEST_SEED_BASE = 42  # Base seed for test dataset (completely separated from training)
DATASET_SIZE = 30  # Number of graphs per dataset
K_PARAM = 20  # Shape parameter of Gamma distribution
graph_structure = {
    1: [(2, 4, 8, False), (4, 3, 3, False), (5, 1, 5, False), (6, 2, 8, False)],
    2: [(1, 4, 8, False), (3, 1, 4, False), (4, 9, 6, False), (5, 5, 1, False), (7, 2, 9, False)],
    3: [(2, 1, 4, False), (7, 6, 8, False)],
    4: [(1, 3, 3, False), (2, 9, 6, False)],
    5: [(1, 1, 5, False), (2, 5, 1, False), (7, 7, 9, False)],
    6: [(1, 2, 8, False), (8, 5, 5, False)],
    7: [(2, 2, 9, False), (5, 7, 9, False), (3, 6, 8, False)],
    8: [(6, 5, 5, False)]
}


# Protected division to avoid zero division error
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def path_contains_edge(path, edge_u, edge_v):
    """Check if the path contains edge (edge_u, edge_v) or its reverse"""
    if len(path) < 2:
        return False
    edge_set = set(zip(path[:-1], path[1:]))
    return (edge_u, edge_v) in edge_set or (edge_v, edge_u) in edge_set


# Precompute edge information for O(1) lookup
def precompute_edge_maps(graph):
    """Cache edge information as dictionaries for fast access"""
    EdgeInfo = namedtuple('EdgeInfo', ['cost', 'demand'])
    edge_map = {}
    for u in graph:
        for v, cost, demand, _ in graph[u]:
            edge = tuple(sorted((u, v)))  # Normalize undirected edge to avoid duplication
            if edge not in edge_map:
                edge_map[edge] = EdgeInfo(cost, demand)
    return edge_map


# Get direct edge cost
def get_direct_edge_cost(edge_map, u, v):
    edge = tuple(sorted((u, v)))
    return edge_map[edge].cost if edge in edge_map else 0


def are_all_edges_true(graph):
    """Check if all edges in the graph have been visited"""
    for node in graph:
        for neighbor, cost, demand, visit in graph[node]:
            if not visit:
                return False
    return True


# Get direct edge demand
def get_direct_edge_demand(edge_map, u, v):
    edge = tuple(sorted((u, v)))
    return edge_map[edge].demand if edge in edge_map else 0


# Floyd-Warshall algorithm for all-pairs shortest paths
def floyd_warshall(graph):
    nodes = sorted(graph.keys())
    INF = float('inf')
    shortest_dist = {node: {n: INF for n in nodes} for node in nodes}
    for node in nodes:
        shortest_dist[node][node] = 0

    path_matrix = {node: {n: None for n in nodes} for node in nodes}

    for start in nodes:
        for neighbor_info in graph[start]:
            end = neighbor_info[0]
            weight = neighbor_info[1]
            shortest_dist[start][end] = weight
            path_matrix[start][end] = end

    for mid in nodes:
        for start in nodes:
            for end in nodes:
                if shortest_dist[start][mid] + shortest_dist[mid][end] < shortest_dist[start][end]:
                    shortest_dist[start][end] = shortest_dist[start][mid] + shortest_dist[mid][end]
                    path_matrix[start][end] = path_matrix[start][mid]

    shortest_path = {start: {} for start in nodes}
    for start in nodes:
        for end in nodes:
            if shortest_dist[start][end] == INF:
                shortest_path[start][end] = []
            else:
                current = start
                path_list = [current]
                while current != end:
                    current = path_matrix[current][end]
                    path_list.append(current)
                shortest_path[start][end] = path_list

    return shortest_dist, shortest_path


# Find the endpoint after completing the task on an edge
def search_mission_endpoint(graph, shortest_dist, shortest_path, current_location, head, tail):
    cost1 = shortest_dist[current_location][head]
    cost2 = shortest_dist[current_location][tail]
    tail_of_task = tail if cost1 <= cost2 else head
    return tail_of_task


# Calculate costs for all unserved edges in batch
def calculate_all_unserve_edge_costs(graph, edge_map, shortest_dist, shortest_path, current_location):
    """Calculate costs for all unserved edges once to reduce redundant computation"""
    unserve_costs = {}
    processed_edges = set()
    for node in graph:
        for neighbor, _, _, visit in graph[node]:
            edge = tuple(sorted((node, neighbor)))
            if not visit and edge not in processed_edges:
                cost = calculate_each_cost(edge_map, shortest_dist, shortest_path, current_location, node, neighbor)
                unserve_costs[edge] = cost
                processed_edges.add(edge)
    return unserve_costs


# Calculate cost for a single unserved edge
def calculate_each_cost(edge_map, shortest_dist, shortest_path, current_location, head, tail):
    cost_to_head = shortest_dist[current_location][head]
    path_to_head = shortest_path[current_location][head]
    cost_to_tail = shortest_dist[current_location][tail]
    path_to_tail = shortest_path[current_location][tail]
    cost_head_tail = get_direct_edge_cost(edge_map, head, tail)
    candidate_costs = []

    if path_contains_edge(path_to_head, head, tail):
        candidate_costs.append(cost_to_head)
    else:
        candidate_costs.append(cost_to_head + cost_head_tail)

    if path_contains_edge(path_to_tail, head, tail):
        candidate_costs.append(cost_to_tail)
    else:
        candidate_costs.append(cost_to_tail + cost_head_tail)

    return min(candidate_costs)


# Predict if remaining battery is sufficient for task and return to depot
def monitoring_whether_return(graph, shortest_dist, shortest_path, edge_map, current_location, head, tail):
    cost1 = shortest_dist[current_location][head]
    cost2 = shortest_dist[current_location][tail]
    cost = min(cost1, cost2)
    tail_of_task = search_mission_endpoint(graph, shortest_dist, shortest_path, current_location, head, tail)

    return cost + get_direct_edge_demand(edge_map, head, tail) + get_direct_edge_cost(edge_map, head, tail) + \
        shortest_dist[tail_of_task][START_NODE]


def generate_random_value(static_value, k, rng):
    theta = static_value / k
    return round(rng.gamma(shape=k, scale=theta))


def update_graph_with_random(graph, k, seed):
    rng = numpy.random.RandomState(seed)
    updated_graph = copy.deepcopy(graph)
    processed_edges = set()

    for node in updated_graph:
        neighbors = updated_graph[node]
        for i in range(len(neighbors)):
            neighbor, cost, demand, visited = neighbors[i]
            edge = (min(node, neighbor), max(node, neighbor))
            if edge not in processed_edges:
                new_cost = generate_random_value(cost, k, rng)
                new_demand = generate_random_value(demand, k, rng)

                updated_graph[node][i] = (neighbor, new_cost, new_demand, visited)
                for j in range(len(updated_graph[neighbor])):
                    n_node, n_cost, n_demand, n_visited = updated_graph[neighbor][j]
                    if n_node == node:
                        updated_graph[neighbor][j] = (node, new_cost, new_demand, n_visited)
                        break
                processed_edges.add(edge)
    return updated_graph


def has_zero_in_edges(graph):
    """Check if any edge has zero cost or zero demand"""
    for edges in graph.values():
        for edge_tuple in edges:
            if edge_tuple[1] == 0 or edge_tuple[2] == 0:
                return True
    return False


def generate_dataset(graph, base_seed, k, size):
    """Generate dataset of graphs with Gamma distributed random edge values"""
    dataset = []
    for i in range(1, size + 1):
        current_seed = base_seed + i
        graph_update = update_graph_with_random(graph, k, current_seed)
        while has_zero_in_edges(graph):
            current_seed += 1
            graph_update = update_graph_with_random(graph, k, current_seed)
        dataset.append(graph_update)
    return dataset


# ========== Feature calculation functions ==========
# Task demand feature
def calculate_demand(edge_map, head, tail):
    demand = get_direct_edge_demand(edge_map, head, tail)
    return demand / MAX_BATTERY


# Remaining battery feature
def calculate_surplus(current_battery):
    return current_battery / MAX_BATTERY


# Edge service cost feature
def calculate_unserve_edg_cost(unserve_costs, edge, max_cost_edg):
    """Get normalized edge cost from precomputed results"""
    return unserve_costs[edge] / max_cost_edg


# Return to depot cost feature (renamed from Recharge)
def calculate_return(graph, shortest_dist, shortest_path, current_location, head, tail):
    mission_endpoint = search_mission_endpoint(graph, shortest_dist, shortest_path, current_location, head, tail)
    cost1 = shortest_dist[mission_endpoint][START_NODE]

    # Find the edge with maximum return to depot cost for normalization
    max_mission_endpoint = None
    max_cost = 1e-6  # Initialize to avoid division by zero
    for node in graph:
        for neighbor, cost, demand, visit in graph[node]:
            if not visit:
                candidate_endpoint = search_mission_endpoint(graph, shortest_dist, shortest_path, current_location,
                                                             node, neighbor)
                candidate_cost = shortest_dist[candidate_endpoint][START_NODE]
                if candidate_cost > max_cost:
                    max_mission_endpoint = candidate_endpoint
                    max_cost = candidate_cost
    return cost1 / max_cost


# Task completion rate feature
def calculate_served(graph):
    double_count_edg = 0
    double_complete_served = 0
    for node in graph:
        for neighbor, cost, demand, visit in graph[node]:
            double_count_edg += 1
            if visit:
                double_complete_served += 1
    count_edg = double_count_edg / 2
    complete_served = double_complete_served / 2
    return complete_served / count_edg


def get_unserved_edges(graph):
    """Get all unserved edges once to avoid duplicate counting"""
    unserve_edges = []
    processed_edges = set()
    for node in graph:
        for neighbor, _, _, visit in graph[node]:
            edge = tuple(sorted((node, neighbor)))
            if not visit and edge not in processed_edges:
                unserve_edges.append((node, neighbor))
                processed_edges.add(edge)
    return unserve_edges


def get_adjacent_unserved_edges(graph, current_location):
    """Get unserved edges adjacent to current location first"""
    adjacent_edges = []
    processed_edges = set()
    for neighbor, _, _, visit in graph[current_location]:
        edge = tuple(sorted((current_location, neighbor)))
        if not visit and edge not in processed_edges:
            adjacent_edges.append((current_location, neighbor))
            processed_edges.add(edge)
    return adjacent_edges


def select(graph, shortest_dist, shortest_path, edge_map, current_location, current_battery):
    """Select candidate edges: adjacent first, then global; filter out unsafe edges"""
    # Prefer adjacent unserved edges; fall back to global if none exist
    unserve_edges = get_adjacent_unserved_edges(graph, current_location)
    if not unserve_edges:
        unserve_edges = get_unserved_edges(graph)

    # Filter edges that would leave insufficient battery to return to depot
    valid_edges = []
    for node, neighbor in unserve_edges:
        if monitoring_whether_return(graph, shortest_dist, shortest_path, edge_map, current_location, node,
                                     neighbor) < current_battery:
            valid_edges.append((node, neighbor))

    return valid_edges


def flyto(graph, shortest_dist, shortest_path, edge_map, current_location, current_battery, current_path_node,
          total_power_consumption, next_traverse_edge):
    """Fly to the selected edge and complete the task"""
    cost1 = shortest_dist[current_location][next_traverse_edge[0]]
    cost2 = shortest_dist[current_location][next_traverse_edge[1]]

    if cost1 > cost2:
        current_location = next_traverse_edge[0]
        current_path_node.extend(shortest_path[current_location][next_traverse_edge[1]][1:])
        current_path_node.append(next_traverse_edge[0])
        edge_cost = get_direct_edge_cost(edge_map, next_traverse_edge[0], next_traverse_edge[1])
        edge_demand = get_direct_edge_demand(edge_map, next_traverse_edge[0], next_traverse_edge[1])
        total_power_consumption += cost2 + edge_cost + edge_demand
        current_battery -= cost2 + edge_cost + edge_demand
    else:
        current_location = next_traverse_edge[1]
        current_path_node.extend(shortest_path[current_location][next_traverse_edge[0]][1:])
        current_path_node.append(next_traverse_edge[1])
        edge_cost = get_direct_edge_cost(edge_map, next_traverse_edge[0], next_traverse_edge[1])
        edge_demand = get_direct_edge_demand(edge_map, next_traverse_edge[0], next_traverse_edge[1])
        total_power_consumption += cost1 + edge_cost + edge_demand
        current_battery -= cost1 + edge_cost + edge_demand

    # Mark edge as visited in both directions
    for i, (neighbor, cost, demand, visit) in enumerate(graph[next_traverse_edge[0]]):
        if neighbor == next_traverse_edge[1]:
            graph[next_traverse_edge[0]][i] = (neighbor, cost, demand, True)
    for j, (n, c, d, v) in enumerate(graph[next_traverse_edge[1]]):
        if n == next_traverse_edge[0]:
            graph[next_traverse_edge[1]][j] = (n, c, d, True)

    return current_location, current_battery, total_power_consumption


def replace(shortest_dist, shortest_path, current_location, current_battery, current_path_node,
            total_power_consumption):
    """Return to depot and recharge battery"""
    if current_location == START_NODE:
        current_battery = MAX_BATTERY
    else:
        cost3 = shortest_dist[current_location][START_NODE]
        path3 = shortest_path[current_location][START_NODE]
        total_power_consumption += cost3
        current_path_node.extend(path3[1:])
        current_location = START_NODE
        current_battery = MAX_BATTERY
    return current_location, current_battery, total_power_consumption


# Define GP primitives
pset = gp.PrimitiveSet("MAIN", 5)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(min, 2)
pset.addPrimitive(max, 2)
pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1))
# Rename arguments: Recharge → Return
pset.renameArguments(ARG0='Demand', ARG1='Surplus', ARG2='Cost', ARG3='Return', ARG4='Served')

# Define fitness and individual types
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


# UAV routing simulation
def uav_simulation(graph, func):
    graph_copy = defaultdict(list)
    for node in graph:
        graph_copy[node] = [(neighbor, cost, demand, visit) for neighbor, cost, demand, visit in graph[node]]

    # Precompute all-pairs shortest paths and edge information
    shortest_dist, shortest_path = floyd_warshall(graph_copy)
    edge_map = precompute_edge_maps(graph_copy)

    current_location = START_NODE
    current_battery = MAX_BATTERY
    current_path_node = [START_NODE]
    total_power_consumption = 0

    while not are_all_edges_true(graph_copy):
        candidates = select(graph_copy, shortest_dist, shortest_path, edge_map, current_location, current_battery)

        if candidates:
            min_func_value = float('inf')
            next_traverse_edge = None

            # Batch calculate costs for all unserved edges
            unserve_costs = calculate_all_unserve_edge_costs(graph_copy, edge_map, shortest_dist, shortest_path,
                                                             current_location)
            max_cost_edg = max(unserve_costs.values()) if unserve_costs else 1e-6

            # Evaluate all candidate edges using GP heuristic
            for node, neighbor in candidates:
                edge = tuple(sorted((node, neighbor)))
                if edge not in unserve_costs:
                    continue
                # Calculate features
                Demand = calculate_demand(edge_map, node, neighbor)
                Surplus = calculate_surplus(current_battery)
                Cost = calculate_unserve_edg_cost(unserve_costs, edge, max_cost_edg)
                Return = calculate_return(graph_copy, shortest_dist, shortest_path, current_location, node, neighbor)
                Served = calculate_served(graph_copy)

                # GP decision making
                gp_output = func(Demand, Surplus, Cost, Return, Served)
                if gp_output < min_func_value:
                    min_func_value = gp_output
                    next_traverse_edge = [node, neighbor]

            if next_traverse_edge is not None:
                current_location, current_battery, total_power_consumption = flyto(
                    graph_copy, shortest_dist, shortest_path, edge_map, current_location,
                    current_battery, current_path_node, total_power_consumption, next_traverse_edge)
            else:
                break
        else:
            if not get_unserved_edges(graph_copy):
                break

            current_location, current_battery, total_power_consumption = replace(
                shortest_dist, shortest_path, current_location, current_battery,
                current_path_node, total_power_consumption)

            if current_location == START_NODE and not select(graph_copy, shortest_dist, shortest_path, edge_map,
                                                             current_location, current_battery):
                break

    # Final return to depot after all tasks
    if current_location != START_NODE:
        cost4 = shortest_dist[current_location][START_NODE]
        path4 = shortest_path[current_location][START_NODE]
        current_path_node.extend(path4[1:])
        total_power_consumption += cost4

    return total_power_consumption, current_path_node


# Evaluation function for GP individuals
def evalUAV(individual, dataset):
    func = toolbox.compile(expr=individual)
    total_consumption = []
    for graph in dataset:
        consumption, current_path = uav_simulation(graph=graph, func=func)
        total_consumption.append(consumption)
    average = sum(total_consumption) / len(total_consumption) if total_consumption else 0
    return (float(average),)


def test_best_individual(best_ind, test_dataset):
    """Test the best evolved individual on the test dataset"""
    print("\n========== Testing Best Individual ==========")
    test_fitness = evalUAV(best_ind, test_dataset)
    print(f"Average power consumption on test set: {test_fitness[0]:.2f}")
    print(f"\nBest GP individual expression:")
    print(str(best_ind))
    return test_fitness[0]


# Main function
def main():
    # Generate training and test datasets
    train_dataset = generate_dataset(graph_structure, TRAIN_SEED_BASE, K_PARAM, DATASET_SIZE)
    test_dataset = generate_dataset(graph_structure, TEST_SEED_BASE, K_PARAM, DATASET_SIZE)

    # Register evolutionary operators
    toolbox.register("evaluate", evalUAV, dataset=train_dataset)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=14))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=14))

    # Initialize population
    pop = toolbox.population(n=1024)
    hof = tools.HallOfFame(1)

    # Statistics configuration
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    # Enable multiprocessing for parallel evaluation
    pool = multiprocessing.Pool(processes=14)
    toolbox.register("map", pool.map)

    # Run genetic programming algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, 0.8, 0.15, 50, stats=mstats,
                                   halloffame=hof, verbose=True)

    # Test the best individual
    if hof:
        test_best_individual(hof[0], test_dataset)

    # Clean up process pool
    pool.close()
    pool.join()

    return pop, log, hof


if __name__ == "__main__":
    main()