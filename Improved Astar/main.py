from collections import defaultdict
import numpy as np
import copy  # For deep copy of graph to avoid modifying original graph
import heapq  # For priority queue


def are_all_edges_true(graph_list):
    """Check if the 'visit' status of all edges in the graph is True"""
    for head, tail, cost, demand, is_visited in graph_list:
        if not is_visited:
            return False
    return True


def a_star_shortest_path(graph, start, end):
    """A* algorithm to calculate the shortest distance and path between two points
       Heuristic value is set to 0, degraded to Dijkstra algorithm
    """
    if start == end:
        return (0, [start])

    # Extract all vertices
    vertices = set(graph.keys())
    for u in graph:
        for v, _, _, _ in graph[u]:
            vertices.add(v)
    vertices = list(vertices)

    if start not in vertices or end not in vertices:
        return (None, None)  # Start or end node not in graph

    # Heuristic function
    def heuristic(node):
        return 0  # Use 0 for Dijkstra, can use Euclidean distance with coordinates

    # Priority queue: (f_score, node)
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {node: float('inf') for node in vertices}
    g_score[start] = 0

    f_score = {node: float('inf') for node in vertices}
    f_score[start] = heuristic(start)

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == end:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return (g_score[end], path)

        for neighbor, cost, _, _ in graph.get(current, []):
            tentative_g = g_score[current] + cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return (None, None)  # Disconnected graph


def shortest_path_routing(graph, start, battery):
    """Select the unvisited edge with minimum cost step by step, and return to start point"""
    # Create graph copy to mark visited edges, keep original graph unchanged
    graph_copy = defaultdict(list)
    for node in graph:
        graph_copy[node] = [(neighbor, cost, demand, visit) for neighbor, cost, demand, visit in graph[node]]

    max_cost = []  # Store minimum cost from start to each edge
    max_path = defaultdict(list)  # Key: cost, Value: edge list [[head, tail, cost, demand, visit]]
    total_path = []  # Record complete travel path
    current_battery = battery  # Remaining battery power
    current_location = start  # Current position
    total_power_consumption = 0

    # Initialize: calculate minimum cost from start to all edges and build max_path
    for node in graph:
        for neighbor, cost, demand, visit in graph[node]:
            # Calculate minimum cost from depot to current edge (node, neighbor)
            edge_cost = depot_to_path(graph_copy, start, node, neighbor)
            max_cost.append(edge_cost)
            # Standardize edge to avoid duplicate (e.g. (2,3) and (3,2) are the same edge)
            std_head = min(node, neighbor)
            std_tail = max(node, neighbor)
            # Add edge without duplication
            if [std_head, std_tail, cost, demand, visit] not in max_path[edge_cost]:
                max_path[edge_cost].append([std_head, std_tail, cost, demand, visit])
    max_cost = sorted(set(max_cost), reverse=False)

    def has_unvisited_edges(edge_map):
        """Check if there exists any unvisited edge"""
        for edge_list in edge_map.values():
            for _, _, _, _, visited in edge_list:
                if not visited:
                    return True
        return False

    while has_unvisited_edges(max_path):
        previous_location = total_path[-2] if len(total_path) >= 2 else None

        predicted_power_consumption, task_power_consumption, selected_edge = monitoring_whether_return(
            graph,
            current_location,
            max_cost,
            max_path,
            start,
            previous_location,
        )

        if selected_edge is None:
            break

        selected_mcost, selected_idx, current_head, current_tail = selected_edge

        # Check if remaining battery supports completing task and returning to start
        if predicted_power_consumption <= current_battery:
            cost1, path1 = floyd_shortest_with_path(graph, current_location, current_head)
            cost2, path2 = floyd_shortest_with_path(graph, current_location, current_tail)

            if cost1 is None or cost2 is None:
                # Mark unreachable edge as visited to avoid infinite loop
                edge_row = max_path[selected_mcost][selected_idx]
                max_path[selected_mcost][selected_idx] = [
                    edge_row[0],
                    edge_row[1],
                    edge_row[2],
                    edge_row[3],
                    True,
                ]
                continue

            # Append path and update current position
            if cost1 <= cost2:  # Go to head node first
                if not total_path:
                    total_path.extend(path1)
                else:
                    total_path.extend(path1[1:])
                total_path.append(current_tail)
                current_location = current_tail
            else:  # Go to tail node first
                if not total_path:
                    total_path.extend(path2)
                else:
                    total_path.extend(path2[1:])
                total_path.append(current_head)
                current_location = current_head

            # Mark current edge as visited
            edge_row = max_path[selected_mcost][selected_idx]
            max_path[selected_mcost][selected_idx] = [
                edge_row[0],
                edge_row[1],
                edge_row[2],
                edge_row[3],
                True,
            ]
            current_battery = current_battery - task_power_consumption
            total_power_consumption = total_power_consumption + task_power_consumption
            print(
                f"Process edge {current_head}-{current_tail}: Task power {task_power_consumption}, Remaining battery {current_battery} (Enough to return)")

        else:
            print(
                f"Predicted total power {predicted_power_consumption} exceeds current battery {current_battery}, return to charge")
            if current_location != start:
                cost3, path3 = floyd_shortest_with_path(graph, current_location, start)
                total_path.extend(path3[1:])
                total_power_consumption = total_power_consumption + cost3
                current_battery = battery
                current_location = start
                total_path.append("Charging")  # Mark charging status
                print(f"Charge completed: Return power {cost3}, Total power {total_power_consumption}")
            else:
                # Terminate if fully charged at start with no executable edges
                if current_battery == battery:
                    print("Fully charged at start, no available edges. Terminate early.")
                    break
                current_battery = battery
    # Return to start point finally
    if current_location != start:
        cost4, path4 = floyd_shortest_with_path(graph, current_location, start)
        total_path.extend(path4[1:])
        total_power_consumption = total_power_consumption + cost4
        print(f"Return to start finally: Power {cost4}, Total power {total_power_consumption}")
    return total_path, total_power_consumption


def monitoring_whether_return(graph, current_location, max_cost, max_path, start, previous_location=None):
    """Select the minimum cost unvisited edge: prefer adjacent edges first, then global edges"""
    local_candidates = []
    global_candidates = []

    for mcost in max_cost:
        for idx, edge in enumerate(max_path[mcost]):
            head, tail, cost, demand, visit = edge
            if visit:
                continue

            g_cost = depot_to_path(graph, current_location, head, tail)
            h_n = depot_to_path(graph, start, head, tail)

            h_j = 0
            if previous_location is not None:
                h_j, _ = floyd_shortest_with_path(graph, start, previous_location)
                if h_j is None:
                    h_j = 0

            f_cost = g_cost + h_n + h_j + demand
            candidate = (f_cost, mcost, idx, head, tail, cost, demand)
            global_candidates.append(candidate)

            if head == current_location or tail == current_location:
                local_candidates.append(candidate)

    candidates = local_candidates if local_candidates else global_candidates
    if not candidates:
        return (float('inf'), 0, None)

    best_f, best_mcost, best_idx, head, tail, cost, demand = min(candidates, key=lambda x: x[0])

    cost1, _ = floyd_shortest_with_path(graph, current_location, head)
    cost2, _ = floyd_shortest_with_path(graph, current_location, tail)
    if cost1 is None or cost2 is None:
        return (float('inf'), 0, (best_mcost, best_idx, head, tail))

    if cost1 <= cost2:
        goto_edge_cost = cost1
        arrival_node = tail
    else:
        goto_edge_cost = cost2
        arrival_node = head

    task_individual_cost = goto_edge_cost + cost + demand
    return_to_start_dist, _ = floyd_shortest_with_path(graph, arrival_node, start)
    if return_to_start_dist is None:
        return (float('inf'), task_individual_cost, (best_mcost, best_idx, head, tail))

    total_cost_to_return = task_individual_cost + return_to_start_dist
    return (total_cost_to_return, task_individual_cost, (best_mcost, best_idx, head, tail))


def min_cost(graph, path_list):
    """Calculate total cost of a given path"""
    if not path_list or len(path_list) < 2:
        return 0
    total = 0
    for i in range(len(path_list) - 1):
        u = path_list[i]
        v = path_list[i + 1]
        # Find cost from u to v
        for neighbor, cost, _, _ in graph.get(u, []):
            if neighbor == v:
                total += cost
                break
    return total


def depot_to_path(graph, depot, path_start, path_end):
    """Calculate minimum cost from depot to a specified edge (path_start-path_end)"""
    cost1, path1 = floyd_shortest_with_path(graph, depot, path_start)
    cost2, path2 = floyd_shortest_with_path(graph, depot, path_end)

    # Check connectivity
    if cost1 is None:
        raise ValueError(f"Node {depot} is disconnected with {path_start}")
    if cost2 is None:
        raise ValueError(f"Node {depot} is disconnected with {path_end}")

    return min(cost1, cost2)


# Floyd-Warshall algorithm for shortest path
def floyd_shortest_with_path(graph, start, end):
    """Floyd-Warshall algorithm to compute shortest distance and path between two nodes"""
    if start == end:
        return (0, [start])
    # Extract all vertices
    vertices = set(graph.keys())
    for u in graph:
        for v, _, _, _ in graph[u]:
            vertices.add(v)
    vertices = list(vertices)
    if start not in vertices or end not in vertices:
        return (None, None)  # Node not exists

    INF = float('inf')
    # Initialize distance matrix
    dist = {u: {v: INF for v in vertices} for u in vertices}
    # Initialize predecessor matrix for path reconstruction
    prev = {u: {v: None for v in vertices} for u in vertices}

    # Distance from node to itself is 0
    for u in vertices:
        dist[u][u] = 0

    # Fill direct edge distance and predecessor
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

    # Core of Floyd-Warshall algorithm
    for k in vertices:
        for i in vertices:
            for j in vertices:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    prev[i][j] = prev[k][j]

    shortest_dist = dist[start][end]
    if shortest_dist == INF:
        return (None, None)  # Disconnected

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
    Generate integer random value based on Gamma distribution
    :param static_value: Base static value
    :param k: Shape parameter of Gamma distribution
    :param rng: Independent random generator instance
    :return: Random integer value
    """
    theta = static_value / k
    return round(rng.gamma(shape=k, scale=theta))


def update_graph_with_random(graph, k, seed):
    """
    Update undirected graph with Gamma distributed random values (fixed result with seed)
    :param graph: Original undirected graph
    :param k: Shape parameter of Gamma distribution
    :param seed: Unique random seed for current graph
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
                new_cost = generate_random_value(cost, k, rng)
                new_demand = generate_random_value(demand, k, rng)

                updated_graph[node][i] = (neighbor, new_cost, new_demand, visited)
                # Update reverse edge
                for j in range(len(updated_graph[neighbor])):
                    n_node, n_cost, n_demand, n_visited = updated_graph[neighbor][j]
                    if n_node == node:
                        updated_graph[neighbor][j] = (node, new_cost, new_demand, n_visited)
                        break
                processed_edges.add(edge)
    return updated_graph


def has_zero_in_edges(graph):
    """Check if any edge has cost or demand equal to zero"""
    for edges in graph.values():
        for edge_tuple in edges:
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
    all_paths = []  # Store all generated paths
    graph_list = []

    for i in range(1, count + 1):
        current_seed = seed + i
        graph_random = update_graph_with_random(graph, k, current_seed)
        # Regenerate if zero value exists in edges
        while has_zero_in_edges(graph_random):
            current_seed = current_seed + 1
            graph_random = update_graph_with_random(graph, k, current_seed)
        graph_list.append(graph_random)
        total_path, total_power = shortest_path_routing(graph_random, start, battery)
        all_paths.append(total_path)
        all_power += total_power
        print(f"\nTotal power consumption: {total_power}")

    print(f"Average total power consumption: {all_power / count}")
    print("All travel paths:")
    for i, path in enumerate(all_paths):
        print(f"  Path {i + 1}: {path}")
