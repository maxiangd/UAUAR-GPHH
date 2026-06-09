from collections import defaultdict
import numpy as np
import copy  # Deep copy graph to avoid modifying the original structure


def are_all_edges_true(graph_list):
    """Check if the visited flag of all edges in the list is True"""
    for head, tail, cost, demand, is_visited in graph_list:
        if not is_visited:
            return False
    return True


def are_all_true(max_path):
    """Check if the visited flag of all edges in all groups is True"""
    for cost_group in max_path.values():
        if not are_all_edges_true(cost_group):
            return False
    return True


def shortest_path_routing(graph, start, battery):
    """Traverse all unvisited edges (optimized: pre-judge return feasibility)"""
    graph_copy = defaultdict(list)
    for node in graph:
        graph_copy[node] = [(neighbor, cost, demand, visit)
                            for neighbor, cost, demand, visit in graph[node]]

    max_cost = []
    minimal_cost = []
    max_path = defaultdict(list)
    total_path = []
    current_battery = battery
    current_location = start
    total_power_consumption = 0

    # Initialize edge groups by cost from depot
    for node in graph:
        for neighbor, cost, demand, visit in graph[node]:
            std_head = min(node, neighbor)
            std_tail = max(node, neighbor)
            edge_cost = depot_to_path(graph_copy, start, std_head, std_tail)
            max_cost.append(edge_cost)
            edge_item = [std_head, std_tail, cost, demand, visit]
            if edge_item not in max_path[edge_cost]:
                max_path[edge_cost].append(edge_item)

    max_cost = sorted(set(max_cost), reverse=True)
    minimal_cost = sorted(set(max_cost))

    # Main loop until all edges are visited
    while not are_all_true(max_path):
        # Dynamically select strategy H1 / H2
        if current_battery / battery >= 0.5:
            target_costs = max_cost
        else:
            target_costs = minimal_cost

        processed = False
        for mcost in target_costs:
            edges = max_path[mcost]
            if are_all_edges_true(edges):
                continue

            # Get predicted total power and task-only power consumption
            predicted_total, task_power = monitoring_whether_return(
                graph, current_location, [mcost], max_path, start
            )
            return_cost = predicted_total - task_power if predicted_total != float('inf') else 0

            # Pre-judge: ensure enough power to finish task and return to depot
            if predicted_total <= current_battery:
                # Select the nearest unvisited edge
                current_min_unserved = float('inf')
                best_head = best_tail = best_idx = None
                for i, (h, t, c, d, v) in enumerate(edges):
                    if not v:
                        unserved_cost = depot_to_path(graph_copy, current_location, h, t)
                        if unserved_cost < current_min_unserved:
                            current_min_unserved = unserved_cost
                            best_head, best_tail, best_idx = h, t, i

                if best_head is None:
                    continue

                # Choose shorter path to one endpoint of the edge
                cost1, path1 = floyd_shortest_with_path(graph, current_location, best_head)
                cost2, path2 = floyd_shortest_with_path(graph, current_location, best_tail)

                if cost1 <= cost2:
                    if total_path:
                        total_path.extend(path1[1:])
                    else:
                        total_path.extend(path1)
                    total_path.append(best_tail)
                    current_location = best_tail
                else:
                    if total_path:
                        total_path.extend(path2[1:])
                    else:
                        total_path.extend(path2)
                    total_path.append(best_head)
                    current_location = best_head

                # Mark edge as visited and update power
                edges[best_idx][4] = True
                current_battery -= task_power
                total_power_consumption += task_power
               
                processed = True
                break

            # Return to depot for recharging if power is insufficient
            else:

                if current_location != start:
                    cost_back, path_back = floyd_shortest_with_path(graph, current_location, start)
                    if cost_back is not None:
                        total_path.extend(path_back[1:])
                        total_power_consumption += cost_back
                        current_battery = battery
                        current_location = start

                else:
                    current_battery = battery
                processed = True
                break

        if not processed:
            break

    # Final return to depot after all tasks
    if current_location != start:
        cost_final, path_final = floyd_shortest_with_path(graph, current_location, start)
        if cost_final is not None:
            total_path.extend(path_final[1:])
            total_power_consumption += cost_final


    return total_path, total_power_consumption


def monitoring_whether_return(graph, current_location, max_cost, max_path, start):
    """Calculate predicted power: (task + return) and (task only)"""
    total_cost1 = float('inf')  # Total power: task + return to depot
    total_cost2 = 0             # Power for task only

    for mcost in max_cost:
        edges = max_path[mcost]
        current_min_unserved = float('inf')
        current_head = None
        current_tail = None
        current_idx = None
        current_demand = None
        temp_path = []

        for i, (head, tail, cost, demand, visit) in enumerate(edges):
            if not visit:
                unserved_cost = depot_to_path(graph, current_location, head, tail)
                if unserved_cost < current_min_unserved:
                    current_min_unserved = unserved_cost
                    current_head = head
                    current_tail = tail
                    current_idx = i
                    current_demand = demand

        if current_head and current_tail and current_idx is not None:
            cost1, path1 = floyd_shortest_with_path(graph, current_location, current_head)
            cost2, path2 = floyd_shortest_with_path(graph, current_location, current_tail)

            if cost1 <= cost2:
                temp_path.extend(path1)
                temp_path.append(current_tail)
                loc_temp = current_tail
                res_cost, _ = floyd_shortest_with_path(graph, loc_temp, start)
            else:
                temp_path.extend(path2)
                temp_path.append(current_head)
                loc_temp = current_head
                res_cost, _ = floyd_shortest_with_path(graph, loc_temp, start)

            path_cost = min_cost(graph, temp_path)
            total_cost1 = path_cost + res_cost + current_demand
            total_cost2 = path_cost + current_demand
            break

    return total_cost1, total_cost2


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
    """Calculate minimum cost from depot to the target edge"""
    cost1, _ = floyd_shortest_with_path(graph, depot, path_start)
    cost2, _ = floyd_shortest_with_path(graph, depot, path_end)

    if cost1 is None:
        raise ValueError(f"Node {depot} is disconnected from {path_start}")
    if cost2 is None:
        raise ValueError(f"Node {depot} is disconnected from {path_end}")

    return min(cost1, cost2)


def floyd_shortest_with_path(graph, start, end):
    """Floyd-Warshall algorithm to compute shortest distance and path"""
    if start == end:
        return (0, [start])

    vertices = set(graph.keys())
    for u in graph:
        for v, _, _, _ in graph[u]:
            vertices.add(v)
    vertices = list(vertices)

    if start not in vertices or end not in vertices:
        return (None, None)

    INF = float('inf')
    dist = {u: {v: INF for v in vertices} for u in vertices}
    prev = {u: {v: None for v in vertices} for u in vertices}

    for u in vertices:
        dist[u][u] = 0

    # Load edge weights
    for u in graph:
        for neighbor_info in graph[u]:
            v, cost, _, _ = neighbor_info
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

    if dist[start][end] == INF:
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
    return (dist[start][end], path)


def generate_random_value(static_value, k, rng):
    """
    Generate integer value following Gamma distribution
    :param static_value: Base value
    :param k: Shape parameter of Gamma distribution
    :param rng: Independent random number generator
    :return: Random integer
    """
    theta = static_value / k
    return round(rng.gamma(shape=k, scale=theta))


def update_graph_with_random(graph, k, seed):
    """
    Update undirected graph with Gamma distributed random values (reproducible with fixed seed)
    :param graph: Original undirected graph
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
