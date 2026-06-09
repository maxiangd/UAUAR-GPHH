from collections import defaultdict
import numpy as np
import copy  # Deep copy graph to prevent modifying the original structure


def are_all_edges_true(graph_list):
    """Check if the visited flag of all edges is True"""
    for head, tail, cost, demand, is_visited in graph_list:
        if not is_visited:
            return False
    return True


def shortest_path_routing(graph, start, battery):
    """Traverse all unvisited edges and return to the starting depot (revised version)"""
    # Create graph copy for marking visited edges
    graph_copy = defaultdict(list)
    for node in graph:
        graph_copy[node] = [(neighbor, cost, demand, visit)
                            for neighbor, cost, demand, visit in graph[node]]

    ratio_list = []  # Store demand-cost ratio of each edge
    edge_groups = defaultdict(list)  # Key: ratio, Value: edge list

    total_path = []
    current_battery = battery
    current_location = start
    total_power_consumption = 0

    # Calculate demand-cost ratio and group edges
    for node in graph:
        for neighbor, cost, demand, visit in graph[node]:
            demand_ratio_servecost = demand / cost
            ratio_list.append(demand_ratio_servecost)

            # Standardize undirected edge to avoid duplicate records
            std_head = min(node, neighbor)
            std_tail = max(node, neighbor)
            edge_item = [std_head, std_tail, cost, demand, visit]

            if edge_item not in edge_groups[demand_ratio_servecost]:
                edge_groups[demand_ratio_servecost].append(edge_item)

    # Sort ratio values in ascending order
    sorted_ratios = sorted(set(ratio_list), reverse=False)

    # Process edge groups by ratio from low to high
    for ratio in sorted_ratios:
        edges = edge_groups[ratio]
        while not are_all_edges_true(edges):
            current_min_cost = float('inf')
            sel_head = None
            sel_tail = None
            sel_idx = None

            pred_total_cost, task_cost = monitoring_whether_return(
                graph, current_location, sorted_ratios, edge_groups, start
            )

            # Execute task if remaining battery is sufficient
            if pred_total_cost < current_battery:
                # Select the nearest unvisited edge
                for idx, (head, tail, cost, demand, visit) in enumerate(edges):
                    if not visit:
                        dist = depot_to_path(graph_copy, current_location, head, tail)
                        if dist < current_min_cost:
                            current_min_cost = dist
                            sel_head = head
                            sel_tail = tail
                            sel_idx = idx

                # Exit if no available unvisited edge
                if sel_head is None or sel_tail is None or sel_idx is None:
                    break

                # Compute shortest paths to two endpoints of the edge
                dist_h, path_h = floyd_shortest_with_path(graph, current_location, sel_head)
                dist_t, path_t = floyd_shortest_with_path(graph, current_location, sel_tail)

                # Assemble travel path
                if dist_h <= dist_t:
                    if not total_path:
                        total_path.extend(path_h)
                    else:
                        total_path.extend(path_h[1:])
                    total_path.append(sel_tail)
                    current_location = sel_tail
                else:
                    if not total_path:
                        total_path.extend(path_t)
                    else:
                        total_path.extend(path_t[1:])
                    total_path.append(sel_head)
                    current_location = sel_head

                # Mark edge as visited
                edges[sel_idx] = [sel_head, sel_tail, edges[sel_idx][2], edges[sel_idx][3], True]
                current_battery -= task_cost
                total_power_consumption += task_cost

            # Return to depot for recharging when battery is insufficient
            else:
                if current_location != start:
                    dist_back, path_back = floyd_shortest_with_path(graph, current_location, start)
                    total_path.extend(path_back[1:])
                    total_power_consumption += dist_back
                    current_battery = battery
                    current_location = start
                else:
                    current_battery = battery

    # Return to depot after all tasks are completed
    if current_location != start:
        dist_final, path_final = floyd_shortest_with_path(graph, current_location, start)
        total_path.extend(path_final[1:])
        total_power_consumption += dist_final

    return total_path, total_power_consumption


def monitoring_whether_return(graph, current_location, ratio_list, edge_groups, start):
    """Select one optimal unvisited edge and return predicted consumption cost"""
    cost_with_return = None
    cost_task_only = None

    for ratio in ratio_list:
        edges = edge_groups[ratio]
        current_min_cost = float('inf')
        sel_head = None
        sel_tail = None
        sel_idx = None
        sel_demand = None
        temp_path = []

        # Search for the nearest unvisited edge
        for idx, (head, tail, cost, demand, visit) in enumerate(edges):
            if not visit:
                dist = depot_to_path(graph, current_location, head, tail)
                if dist < current_min_cost:
                    current_min_cost = dist
                    sel_head = head
                    sel_tail = tail
                    sel_idx = idx
                    sel_demand = demand

        if sel_head and sel_tail and sel_idx is not None:
            dist_h, path_h = floyd_shortest_with_path(graph, current_location, sel_head)
            dist_t, path_t = floyd_shortest_with_path(graph, current_location, sel_tail)

            if dist_h <= dist_t:
                temp_path.extend(path_h)
                temp_path.append(sel_tail)
                loc_temp = sel_tail
            else:
                temp_path.extend(path_t)
                temp_path.append(sel_head)
                loc_temp = sel_head

            dist_back, _ = floyd_shortest_with_path(graph, loc_temp, start)
            path_cost = min_cost(graph, temp_path)
            cost_with_return = path_cost + dist_back + sel_demand
            cost_task_only = path_cost + sel_demand
            break

    return cost_with_return, cost_task_only


def min_cost(graph, path_list):
    """Calculate total travel cost of a given path"""
    if not path_list or len(path_list) < 2:
        return 0
    total = 0
    for i in range(len(path_list) - 1):
        u = path_list[i]
        v = path_list[i + 1]
        for neighbor, cost, _, _ in graph.get(u, []):
            if neighbor == v:
                total += cost
                break
    return total


def depot_to_path(graph, depot, path_start, path_end):
    """Calculate minimum cost from depot to a specified edge"""
    cost_s, _ = floyd_shortest_with_path(graph, depot, path_start)
    cost_e, _ = floyd_shortest_with_path(graph, depot, path_end)

    if cost_s is None:
        raise ValueError(f"Node {depot} is disconnected from {path_start}")
    if cost_e is None:
        raise ValueError(f"Node {depot} is disconnected from {path_end}")

    return min(cost_s, cost_e)


def floyd_shortest_with_path(graph, start, end):
    """Floyd-Warshall algorithm to compute shortest distance and path"""
    if start == end:
        return (0, [start])

    # Collect all vertices in graph
    vertices = set(graph.keys())
    for u in graph:
        for v, _, _, _ in graph[u]:
            vertices.add(v)
    vertices = list(vertices)

    if start not in vertices or end not in vertices:
        return (None, None)

    INF = float('inf')
    # Initialize distance and predecessor matrix
    dist = {u: {v: INF for v in vertices} for u in vertices}
    prev = {u: {v: None for v in vertices} for u in vertices}

    for u in vertices:
        dist[u][u] = 0

    # Load direct edge weights
    for u in graph:
        for v, cost, _, _ in graph[u]:
            if cost < dist[u][v]:
                dist[u][v] = cost
                prev[u][v] = u
            if cost < dist[v][u]:
                dist[v][u] = cost
                prev[v][u] = v

    # Floyd-Warshall core iteration
    for k in vertices:
        for i in vertices:
            for j in vertices:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    prev[i][j] = prev[k][j]

    shortest_dist = dist[start][end]
    if shortest_dist == INF:
        return (None, None)

    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        if current == start:
            break
        current = prev[start][current]
    path.reverse()

    return (shortest_dist, path)


def generate_random_value(static_value, k, rng):
    """
    Generate integer value following Gamma distribution
    :param static_value: Base value
    :param k: Shape parameter of Gamma distribution
    :param rng: Independent random generator
    :return: Generated integer
    """
    theta = static_value / k
    return round(rng.gamma(shape=k, scale=theta))


def update_graph_with_random(graph, k, seed):
    """
    Update undirected graph with Gamma distributed random values (reproducible with fixed seed)
    :param graph: Original graph
    :param k: Shape parameter
    :param seed: Random seed
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
                new_cost = generate_random_value(cost, k, rng)
                new_demand = generate_random_value(demand, k, rng)

                updated_graph[node][i] = (neighbor, new_cost, new_demand, visited)
                # Synchronize reverse edge
                for j in range(len(updated_graph[neighbor])):
                    n_node, n_c, n_d, n_v = updated_graph[neighbor][j]
                    if n_node == node:
                        updated_graph[neighbor][j] = (node, new_cost, new_demand, n_v)
                        break
                processed_edges.add(edge)
    return updated_graph


def has_zero_in_edges(graph):
    """Check whether any edge has zero cost or zero demand"""
    for edges in graph.values():
        for edge_tuple in edges:
            if edge_tuple[1] == 0 or edge_tuple[2] == 0:
                return True
    return False


if __name__ == "__main__":
    # Undirected graph: (adjacent node, cost, demand, visited flag)
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
        # Regenerate graph if zero value exists
        while has_zero_in_edges(graph_random):
            current_seed += 1
            graph_random = update_graph_with_random(graph, k, current_seed)
        graph_list.append(graph_random)
        all_power += shortest_path_routing(graph_random, start, battery)[1]

    print(all_power / count)
