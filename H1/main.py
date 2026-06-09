from collections import defaultdict
import numpy as np
import copy  # Perform deep copy to avoid modifying the original graph


def are_all_edges_true(graph_list):
    """Check if the 'visited' flag of all edges is True"""
    for head, tail, cost, demand, is_visited in graph_list:
        if not is_visited:
            return False
    return True


def shortest_path_routing(graph, start, battery):
    """Traverse all unvisited edges and return to the starting depot (revised version)"""
    # Create a graph copy to mark visited edges, keep original graph unchanged
    graph_copy = defaultdict(list)
    for node in graph:
        graph_copy[node] = [(neighbor, cost, demand, visit) for neighbor, cost, demand, visit in graph[node]]

    max_cost = []  # Store the minimum cost from depot to each edge
    max_path = defaultdict(list)  # Key: cost value, Value: edge list [[head, tail, cost, demand, visit]]
    total_path = []  # Record the complete travel path
    current_battery = battery  # Remaining battery power
    current_location = start  # Current position of the vehicle
    total_power_consumption = 0

    # Initialization: calculate minimum cost from depot to all edges and build edge set
    for node in graph:
        for neighbor, cost, demand, visit in graph[node]:
            # Calculate minimum cost from depot to current edge (node, neighbor)
            edge_cost = depot_to_path(graph_copy, start, node, neighbor)
            max_cost.append(edge_cost)
            # Standardize edge to eliminate duplicate (e.g. (2,3) and (3,2) are the same edge)
            std_head = min(node, neighbor)
            std_tail = max(node, neighbor)
            # Add edge without duplication
            if [std_head, std_tail, cost, demand, visit] not in max_path[edge_cost]:
                max_path[edge_cost].append([std_head, std_tail, cost, demand, visit])

    max_cost = sorted(set(max_cost), reverse=True)
    # Process edge groups in descending order of cost
    for mcost in max_cost:
        edges = max_path[mcost]
        # Process all unvisited edges under current cost

        while not are_all_edges_true(edges):
            current_min_unserved = float('inf')
            current_head = None
            current_tail = None
            current_idx = None
            # Calculate power consumption for task execution and return trip
            predicted_power_consumption, task_power_consumption = monitoring_whether_return(graph, current_location, max_cost, max_path, start)

            # Execute task if remaining battery is sufficient
            if predicted_power_consumption < current_battery:
                # Step 1: Select the optimal unvisited edge under current cost
                for i, (head, tail, cost, demand, visit) in enumerate(edges):
                    if not visit:  # Only process unvisited edges
                        # Calculate cost from current position to this edge
                        unserved_cost = depot_to_path(graph_copy, current_location, head, tail)
                        if unserved_cost < current_min_unserved:
                            current_min_unserved = unserved_cost
                            current_head = head
                            current_tail = tail
                            current_idx = i

                # Exit if no valid unvisited edge is found
                if current_head is None or current_tail is None or current_idx is None:
                    break

                # Step 2: Compute shortest paths to two ends of the edge and select the better one
                cost1, path1 = floyd_shortest_with_path(graph, current_location, current_head)
                cost2, path2 = floyd_shortest_with_path(graph, current_location, current_tail)

                # Update travel path and current position
                if cost1 <= cost2:  # Move to head node first
                    if not total_path:
                        total_path.extend(path1)
                    else:
                        total_path.extend(path1[1:])  # Skip duplicate current node
                    total_path.append(current_tail)  # Arrive at the other end of the edge
                    current_location = current_tail
                else:  # Move to tail node first
                    if not total_path:
                        total_path.extend(path2)
                    else:
                        total_path.extend(path2[1:])
                    total_path.append(current_head)
                    current_location = current_head

                # Mark current edge as visited
                edges[current_idx] = [current_head, current_tail, edges[current_idx][2], edges[current_idx][3], True]
                current_battery = current_battery - task_power_consumption
                total_power_consumption = total_power_consumption + task_power_consumption

            else:
                # Return to depot for charging when battery is insufficient
                if current_location != start:
                    cost3, path3 = floyd_shortest_with_path(graph, current_location, start)
                    total_path.extend(path3[1:])
                    total_power_consumption = total_power_consumption + cost3
                    current_battery = battery
                    current_location = start
                else:
                    current_battery = battery
        # All edges under current cost have been processed

    # Return to depot finally
    if current_location != start:
        cost4, path4 = floyd_shortest_with_path(graph, current_location, start)
        total_path.extend(path4[1:])
        total_power_consumption = total_power_consumption + cost4
    return total_path, total_power_consumption


def monitoring_whether_return(graph, current_location, max_cost, max_path, start):
    """Select one optimal unvisited edge and return total power consumption"""
    total_cost1 = None  # Total power for completing task and returning to depot
    total_cost2 = None  # Power consumption for completing the current task
    for mcost in max_cost:
        edges = max_path[mcost]
        current_min_unserved = float('inf')
        current_head = None
        current_tail = None
        current_idx = None
        current_demand = None
        total_path = []

        # Step 1: Find the optimal unvisited edge (only select one)
        for i, (head, tail, cost, demand, visit) in enumerate(edges):
            if not visit:  # Only process unvisited edges
                unserved_cost = depot_to_path(graph, current_location, head, tail)
                if unserved_cost < current_min_unserved:
                    current_min_unserved = unserved_cost
                    current_head = head
                    current_tail = tail
                    current_idx = i
                    current_demand = demand

        # Process the selected edge if exists
        if current_head is not None and current_tail is not None and current_idx is not None:
            # Step 2: Calculate travel cost
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

            # Stop searching after one edge is selected
            break
    return (total_cost1, total_cost2)


def min_cost(graph, path_list):
    """Calculate total travel cost of a given path"""
    if not path_list or len(path_list) < 2:
        return 0
    total = 0
    for i in range(len(path_list) - 1):
        u = path_list[i]
        v = path_list[i + 1]
        # Search cost from node u to node v
        for neighbor, cost, _, _ in graph.get(u, []):
            if neighbor == v:
                total += cost
                break
    return total


def depot_to_path(graph, depot, path_start, path_end):
    """Calculate minimum cost from depot to a specified edge (path_start-path_end)"""
    cost1, path1 = floyd_shortest_with_path(graph, depot, path_start)
    cost2, path2 = floyd_shortest_with_path(graph, depot, path_end)

    # Check graph connectivity
    if cost1 is None:
        raise ValueError(f"Node {depot} is disconnected with {path_start}")
    if cost2 is None:
        raise ValueError(f"Node {depot} is disconnected with {path_end}")

    return min(cost1, cost2)


# Floyd-Warshall algorithm for shortest path searching
def floyd_shortest_with_path(graph, start, end):
    """Floyd-Warshall algorithm to compute shortest distance and path between two nodes (vertex extraction revised)"""
    if start == end:
        return (0, [start])
    # Extract all vertices including neighbor nodes
    vertices = set(graph.keys())
    for u in graph:
        for v, _, _, _ in graph[u]:
            vertices.add(v)
    vertices = list(vertices)
    if start not in vertices or end not in vertices:
        return (None, None)  # Start or end node does not exist

    INF = float('inf')
    # Initialize distance matrix
    dist = {u: {v: INF for v in vertices} for u in vertices}
    # Initialize predecessor matrix for path reconstruction
    prev = {u: {v: None for v in vertices} for u in vertices}

    # Distance from a node to itself is zero
    for u in vertices:
        dist[u][u] = 0

    # Fill direct edge cost and predecessor information
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

    # Core iteration of Floyd-Warshall algorithm
    for k in vertices:
        for i in vertices:
            for j in vertices:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    prev[i][j] = prev[k][j]

    # Check connectivity
    shortest_dist = dist[start][end]
    if shortest_dist == INF:
        return (None, None)  # Two nodes are disconnected

    # Reconstruct the complete path
    path = []
    current = end
    while current is not None:
        path.append(current)
        if current == start:
            break
        current = prev[start][current]
    path.reverse()  # Reverse to get path from start to end

    return (shortest_dist, path)


def generate_random_value(static_value, k, rng):
    """
    Generate integer random value following Gamma distribution
    :param static_value: Base static value
    :param k: Shape parameter of Gamma distribution
    :param rng: Independent random number generator
    :return: Generated integer random value
    """
    theta = static_value / k
    return round(rng.gamma(shape=k, scale=theta))


def update_graph_with_random(graph, k, seed):
    """
    Update undirected graph with Gamma distributed random values (fixed result with given seed)
    :param graph: Original undirected graph
    :param k: Shape parameter of Gamma distribution
    :param seed: Unique random seed for current graph instance
    :return: Updated graph
    """
    # Create independent random generator to avoid global interference
    rng = np.random.RandomState(seed)

    updated_graph = copy.deepcopy(graph)
    processed_edges = set()

    for node in updated_graph:
        neighbors = updated_graph[node]
        for i in range(len(neighbors)):
            neighbor, cost, demand, visited = neighbors[i]
            edge = (min(node, neighbor), max(node, neighbor))
            if edge not in processed_edges:
                # Generate new value via independent random generator
                new_cost = generate_random_value(cost, k, rng)
                new_demand = generate_random_value(demand, k, rng)

                updated_graph[node][i] = (neighbor, new_cost, new_demand, visited)
                # Update the reverse edge
                for j in range(len(updated_graph[neighbor])):
                    n_node, n_cost, n_demand, n_visited = updated_graph[neighbor][j]
                    if n_node == node:
                        updated_graph[neighbor][j] = (node, new_cost, new_demand, n_visited)
                        break
                processed_edges.add(edge)
    return updated_graph


def has_zero_in_edges(graph):
    """Check if any edge has cost or demand equal to zero"""
    # Traverse all edges in the graph
    for edges in graph.values():
        for edge_tuple in edges:
            # Check cost and demand value
            if edge_tuple[1] == 0 or edge_tuple[2] == 0:
                return True
    return False


if __name__ == "__main__":
    # Example undirected graph format: (neighbor, cost, demand, is_visited)
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
    # Algorithm parameters
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
        # Regenerate graph if zero value exists in edges
        while has_zero_in_edges(graph_random):
            current_seed = current_seed + 1
            graph_random = update_graph_with_random(graph, k, current_seed)
        graph_list.append(graph_random)
        all_power = all_power + shortest_path_routing(graph_random, start, battery)[1]

    print(all_power / count)
