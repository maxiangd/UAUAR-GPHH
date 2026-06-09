from collections import defaultdict
import numpy as np
import copy  # Deep copy the graph to avoid modifying the original


def are_all_edges_true(graph_list):
    """Check whether the visited flag of all edges is True"""
    for head, tail, cost, demand, is_visited in graph_list:
        if not is_visited:
            return False
    return True


def shortest_path_routing(graph, start, battery):
    """Traverse all unvisited edges and return to the starting depot (revised version)"""
    # Create a copy of the graph to mark visited edges without altering the original
    graph_copy = defaultdict(list)
    for node in graph:
        graph_copy[node] = [(neighbor, cost, demand, visit) for neighbor, cost, demand, visit in graph[node]]

    max_cost = []  # Store the ratio of demand to service cost for each edge
    max_path = defaultdict(list)  # Key: demand-cost ratio, Value: edge list [[head, tail, cost, demand, visit]]
    total_path = []  # Record the complete travel path
    current_battery = battery  # Remaining battery capacity
    current_location = start  # Current vehicle position
    total_power_consumption = 0

    # Initialization: calculate demand-cost ratio and group edges
    for node in graph:
        for neighbor, cost, demand, visit in graph[node]:
            # Calculate demand to service cost ratio of the edge
            demand_ratio_servecost = demand / cost
            max_cost.append(demand_ratio_servecost)
            # Standardize undirected edge to eliminate duplicates (e.g. (2,3) and (3,2) are identical)
            std_head = min(node, neighbor)
            std_tail = max(node, neighbor)
            # Add edge to corresponding group without duplication
            if [std_head, std_tail, cost, demand, visit] not in max_path[demand_ratio_servecost]:
                max_path[demand_ratio_servecost].append([std_head, std_tail, cost, demand, visit])

    # Sort ratio values in descending order
    max_cost = sorted(set(max_cost), reverse=True)

    # Process edge groups from high ratio to low ratio
    for mcost in max_cost:
        edges = max_path[mcost]
        # Process all unvisited edges in current group
        while not are_all_edges_true(edges):
            current_min_unserved = float('inf')
            current_head = None
            current_tail = None
            current_idx = None

            predicted_power_consumption, task_power_consumption = monitoring_whether_return(graph, current_location, max_cost, max_path, start)
            # Predict power consumption for task execution and return trip to depot
            if predicted_power_consumption < current_battery:
                # Select the optimal unvisited edge
                for i, (head, tail, cost, demand, visit) in enumerate(edges):
                    if not visit:  # Only process unvisited edges
                        # Calculate travel cost from current position to this edge
                        unserved_cost = depot_to_path(graph_copy, current_location, head, tail)
                        if unserved_cost < current_min_unserved:
                            current_min_unserved = unserved_cost
                            current_head = head
                            current_tail = tail
                            current_idx = i

                # Defensive check for empty unvisited edges
                if current_head is None or current_tail is None or current_idx is None:
                    break

                # Compute shortest paths to two endpoints of the edge
                cost1, path1 = floyd_shortest_with_path(graph, current_location, current_head)
                cost2, path2 = floyd_shortest_with_path(graph, current_location, current_tail)

                # Assemble travel path and avoid duplicate nodes
                if cost1 <= cost2:  # Prioritize moving to head node
                    if not total_path:
                        total_path.extend(path1)
                    else:
                        total_path.extend(path1[1:])  # Skip redundant current node
                    total_path.append(current_tail)  # Arrive at the other endpoint
                    current_location = current_tail  # Update current position
                else:  # Prioritize moving to tail node
                    if not total_path:
                        total_path.extend(path2)
                    else:
                        total_path.extend(path2[1:])
                    total_path.append(current_head)
                    current_location = current_head

                # Mark edge as visited after service
                edges[current_idx] = [current_head, current_tail, edges[current_idx][2], edges[current_idx][3], True]
                current_battery -= task_power_consumption
                total_power_consumption += task_power_consumption

            else:
                # Return to depot when battery is insufficient
                if current_location != start:
                    cost3, path3 = floyd_shortest_with_path(graph, current_location, start)
                    total_path.extend(path3[1:])
                    total_power_consumption += cost3
                    current_battery = battery
                    current_location = start
                else:
                    current_battery = battery

    # Return to depot after all tasks are finished
    if current_location != start:
        cost4, path4 = floyd_shortest_with_path(graph, current_location, start)
        total_path.extend(path4[1:])
        total_power_consumption += cost4

    return total_path, total_power_consumption


def monitoring_whether_return(graph, current_location, max_cost, max_path, start):
    """Select one optimal unvisited edge and return corresponding cost values"""
    total_cost1 = None  # Total cost including task and return to depot
    total_cost2 = None  # Cost of executing the task only

    for mcost in max_cost:
        edges = max_path[mcost]
        current_min_unserved = float('inf')
        current_head = None
        current_tail = None
        current_idx = None
        current_demand = None
        total_path = []

        # Search for the optimal unvisited edge
        for i, (head, tail, cost, demand, visit) in enumerate(edges):
            if not visit:
                unserved_cost = depot_to_path(graph, current_location, head, tail)
                if unserved_cost < current_min_unserved:
                    current_min_unserved = unserved_cost
                    current_head = head
                    current_tail = tail
                    current_idx = i
                    current_demand = demand

        # Calculate cost if valid edge is found
        if current_head is not None and current_tail is not None and current_idx is not None:
            cost1, path1 = floyd_shortest_with_path(graph, current_location, current_head)
            cost2, path2 = floyd_shortest_with_path(graph, current_location, current_tail)

            if cost1 <= cost2:
                total_path.extend(path1)
                total_path.append(current_tail)
                current_location = current_tail
                result1 = floyd_shortest_with_path(graph, current_tail, start)
                total_cost1 = min_cost(graph, total_path) + result1[0] + current_demand
                total_cost2 = min_cost(graph, total_path) + current_demand
            else:
                total_path.extend(path2)
                total_path.append(current_head)
                current_location = current_head
                result2 = floyd_shortest_with_path(graph, current_head, start)
                total_cost1 = min_cost(graph, total_path) + result2[0] + current_demand
                total_cost2 = min_cost(graph, total_path) + current_demand

            break  # Stop after selecting one edge

    return total_cost1, total_cost2


def min_cost(graph, path_list):
    """Calculate total travel cost of a given path"""
    if not path_list or len(path_list) < 2:
        return 0
    total = 0
    for i in range(len(path_list) - 1):
        u = path_list[i]
        v = path_list[i + 1]
        # Accumulate travel cost between adjacent nodes
        for neighbor, cost, _, _ in graph.get(u, []):
            if neighbor == v:
                total += cost
                break
    return total


def depot_to_path(graph, depot, path_start, path_end):
    """Calculate minimum cost from depot to the specified edge"""
    cost1, path1 = floyd_shortest_with_path(graph, depot, path_start)
    cost2, path2 = floyd_shortest_with_path(graph, depot, path_end)

    # Check graph connectivity
    if cost1 is None:
        raise ValueError(f"Node {depot} is disconnected from {path_start}")
    if cost2 is None:
        raise ValueError(f"Node {depot} is disconnected from {path_end}")

    return min(cost1, cost2)


# Floyd-Warshall algorithm for shortest path calculation
def floyd_shortest_with_path(graph, start, end):
    """Floyd-Warshall algorithm to compute shortest distance and path (vertex extraction optimized)"""
    if start == end:
        return 0, [start]

    # Collect all vertices in the graph
    vertices = set(graph.keys())
    for u in graph:
        for v, _, _, _ in graph[u]:
            vertices.add(v)
    vertices = list(vertices)

    if start not in vertices or end not in vertices:
        return None, None  # Target node does not exist

    INF = float('inf')
    # Initialize distance and predecessor matrices
    dist = {u: {v: INF for v in vertices} for u in vertices}
    prev = {u: {v: None for v in vertices} for u in vertices}

    # Distance from a node to itself is zero
    for u in vertices:
        dist[u][u] = 0

    # Load direct edge information
    for u in graph:
        for neighbor_info in graph[u]:
            v, cost, _, _ = neighbor_info
            if cost < dist[u][v]:
                dist[u][v] = cost
                prev[u][v] = u
            # Add reverse edge for undirected graph
            if cost < dist[v][u]:
                dist[v][u] = cost
                prev[v][u] = v

    # Core relaxation process of Floyd-Warshall
    for k in vertices:
        for i in vertices:
            for j in vertices:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    prev[i][j] = prev[k][j]

    shortest_dist = dist[start][end]
    if shortest_dist == INF:
        return None, None  # Two nodes are disconnected

    # Reconstruct the complete path
    path = []
    current = end
    while current is not None:
        path.append(current)
        if current == start:
            break
        current = prev[start][current]
    path.reverse()

    return shortest_dist, path


def generate_random_value(static_value, k, rng):
    """
    Generate integer value following Gamma distribution
    :param static_value: Base value
    :param k: Shape parameter of Gamma distribution
    :param rng: Independent random number generator
    :return: Generated integer value
    """
    theta = static_value / k
    return round(rng.gamma(shape=k, scale=theta))


def update_graph_with_random(graph, k, seed):
    """
    Update undirected graph with Gamma distributed random values (deterministic with fixed seed)
    :param graph: Original undirected graph
    :param k: Shape parameter of Gamma distribution
    :param seed: Random seed for reproducibility
    :return: Updated graph
    """
    rng = np.random.RandomState(seed)
    updated_graph = copy.deepcopy(graph)
    processed_edges = set()

    for node in updated_graph:
        neighbors = updated_graph[node]
        for i in range(len(neighbors)):
            neighbor, cost, demand, visited = neighbors[i]
            edge = (min(node, neighbor), max(node, neighbor))
            if edge not in processed_edges:
                # Generate new random cost and demand
                new_cost = generate_random_value(cost, k, rng)
                new_demand = generate_random_value(demand, k, rng)

                updated_graph[node][i] = (neighbor, new_cost, new_demand, visited)
                # Synchronize the reverse edge
                for j in range(len(updated_graph[neighbor])):
                    n_node, n_cost, n_demand, n_visited = updated_graph[neighbor][j]
                    if n_node == node:
                        updated_graph[neighbor][j] = (node, new_cost, new_demand, n_visited)
                        break
                processed_edges.add(edge)
    return updated_graph


def has_zero_in_edges(graph):
    """Check if any edge contains zero cost or zero demand"""
    for edges in graph.values():
        for edge_tuple in edges:
            if edge_tuple[1] == 0 or edge_tuple[2] == 0:
                return True
    return False


if __name__ == "__main__":
    # Undirected graph format: (adjacent node, cost, demand, visited flag)
    graph = {
        1: [(2, 4, 8, False), (4, 3, 3, False), (5, 1, 5, False), (6, 2, 8, False)],
        2: [(1, 4, 8, False), (3, 1, 4, False), (4, 9, 6, False), (5, 5, 1, False), (7, 2, 9, False)],
        3: [(2, 1, 4, False), (7, 6, 8, False)],
        4: [(1, 3, 3, False), (2, 9, 6, False)],
        5: [(1, 1, 5, False), (2, 5, 1, False), (7, 7, 9, False)],
        6: [(1, 2, 8, False), (8, 5, 5, False)],
        7: [(2, 2, 9, False), (5, 7, 9, False), (3, 6, 8, False)],
        8: [(6, 5, 5, False)]
    }

    # Algorithm configuration parameters
    k = 20
    count = 30
    start = 1
    battery = 100
    seed = 42
    all_power = 0
    graph_list = []

    for i in range(1, count + 1):
        current_seed = seed + i
        graph_random = update_graph_with_random(graph, k, current_seed)
        # Regenerate graph if zero value exists
        while has_zero_in_edges(graph_random):
            current_seed += 1
            graph_random = update_graph_with_random(graph, k, current_seed)
        graph_list.append(graph_random)
        all_power += shortest_path_routing(graph_random, start, battery)[1]

    print(all_power / count)
