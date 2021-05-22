import networkx as nx
import matplotlib.pyplot as plt
import copy
from itertools import combinations
import collections
from ro import Ro

max_weight = 100000


# todo: после написание алгоритма '# comment', раскомментить
def initialize_graph(G):
    G.add_node(1, pos=(20, 50))
    G.add_node(2, pos=(45, 50))
    G.add_node(3, pos=(65, 60))
    # G.add_node(4, pos=(90, 50))  # comment
    G.add_node(5, pos=(30, 30))
    G.add_node(6, pos=(60, 30))
    # G.add_node(7, pos=(80, 35))  # comment
    G.add_node(8, pos=(10, 10))
    G.add_node(9, pos=(30, 10))
    G.add_node(10, pos=(50, 10))
    G.add_node(11, pos=(70, 10))  # comment
    # G.add_node(12, pos=(95, 20))  # comment
    e = [(1, 2, 1), (1, 5, 4), (2, 5, 2),
         (2, 3, 1),
         # (3, 4, 4), (4, 7, 1),  # comment
         (2, 6, 3),
         # (6, 7, 4), (7, 12, 4),  # comment
         (5, 8, 1), (8, 9, 3), (5, 9, 2),
         (9, 10, 1), (5, 6, 4), (6, 10, 4),
         (10, 11, 1), (6, 11, 2),
         # (11, 12, 3)  # comment
         ]
    G.add_weighted_edges_from(e)


def print_graph(G, result_path):
    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edges(G, pos, edge_color="#dedede", )
    nx.draw_networkx_edges(G, pos, edgelist=result_path, edge_color="#f70909", )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='#015e98')
    plt.show()


# дейкстра для двух точек
def dijkstra2(G, a, b):
    print("min path:", nx.dijkstra_path(G, a, b))
    print("min path length:", nx.dijkstra_path_length(G, a, b))


# дейкстра для 3 точек
def dijkstra3(G, init_nodes):
    min = 10000
    array_min_i = []
    for i in G.nodes:
        local_min = 0

        for init_node in init_nodes:
            local_min += nx.dijkstra_path_length(G, i, init_node)

        if local_min <= min:
            if local_min < min:
                array_min_i = []
            min = local_min
            array_min_i.append(i)

    print("min path length:", min)
    print("Steiner's points:", array_min_i)
    print("Steiner's point:", array_min_i[0], "\n")

    print("More detailed:")
    for init_node in init_nodes:
        dijkstra2(G, array_min_i[0], init_node)


# получение всевозможных разбиений множества вершин на два непустых непересекающихся подмножества
def get_disjoint_subsets(arr):
    result = get_disjoint_subsets_function_body(arr, 2, [[]], 0)
    result.sort(key=lambda val: len(val[0]))
    return result


# основное тело функции получения непустых непересекающихся подмножеств
def get_disjoint_subsets_function_body(arr, k, accum, index):
    if index == len(arr):
        return copy.deepcopy(accum) if k == 0 else []

    element = arr[index]
    result = []

    for set_i in range(len(accum)):
        if k > 0:
            clone_new = copy.deepcopy(accum)
            clone_new[set_i].append([element])
            result.extend(get_disjoint_subsets_function_body(arr, k - 1, clone_new, index + 1))

        for elem_i in range(len(accum[set_i])):
            clone_new = copy.deepcopy(accum)
            clone_new[set_i][elem_i].append(element)
            result.extend(get_disjoint_subsets_function_body(arr, k, clone_new, index + 1))

    return result


# input: [1, 2, 5, 9, 10], output: [(1, 2), (2, 5), (5, 9), (9, 10)]
def get_path_by_nodes(nodes_in_path):
    true_path = []
    for i in range(1, len(nodes_in_path)):
        true_path.append(tuple([nodes_in_path[i - 1], nodes_in_path[i]]))

    true_path = [tuple(sorted(path)) for path in true_path]  # [(2, 1), (5, 2), (5, 9)] -> [(1, 2), (2, 5), (5, 9)]
    return true_path


def sort_dict_by_keys(main_dict, keys):
    result = {}
    for key in keys:
        result[key] = main_dict.get(key)
    return result


# дейкстра для более, чем 3 точек
def dijkstra4(G, init_nodes):
    global check_print_debug
    if check_print_debug: print("init_nodes:", init_nodes)
    global max_weight

    all_coordinates = {}  # здесь будут храниться все координаты
    all_paths = {}  # здесь будут храниться все пути
    init_nodes_with_out_last = init_nodes[:len(init_nodes) - 1]
    if check_print_debug: print("init_nodes_with_out_last", init_nodes_with_out_last)

    # 1 этап - 1-координаты
    for node in G.nodes:
        path = {}
        first_coordinate = {}

        for init_node in init_nodes_with_out_last:
            first_coordinate[(init_node,)] = nx.dijkstra_path_length(G, node, init_node)  # храним координаты в tuple
            path[(init_node,)] = get_path_by_nodes(nx.dijkstra_path(G, node, init_node))

        all_coordinates[node] = first_coordinate
        all_paths[node] = path

    # ключ - вершина, значение - длина пути
    if check_print_debug:
        for key in all_coordinates:
            print(key, ": ", all_coordinates[key], sep="")
        print()

    # остальные этапы
    for count_nodes_in_omegas in range(2, len(init_nodes)):
        lots_of_omegas = list(combinations(init_nodes_with_out_last, count_nodes_in_omegas))
        if check_print_debug: print("\nlots_of_omegas:", lots_of_omegas)
        # первая стадия
        for lot_omega in lots_of_omegas:
            nodes_to_subsets_omega = [lot_omega[0]]
            quasi_coordinates = {}
            quasi_paths = {}

            omegas = []
            for i in range(1, len(lot_omega)):
                nodes_to_subsets_omega.append(lot_omega[i])
                omegas = get_disjoint_subsets(nodes_to_subsets_omega)

            if check_print_debug: print("\n-----------------\nOmegas:", omegas)
            nodes_in_omega = []
            for omega in omegas:
                for sublist in omega:
                    nodes_in_omega += [elem for elem in sublist if elem not in nodes_in_omega]

            for node in G.nodes:
                if nodes_in_omega.count(node) == 0:  # перебираем все вершины кроме тех, что попали в омегу
                    quasi_coordinates[node] = max_weight
                    quasi_paths[node] = []

            for omega in omegas:
                if check_print_debug: print("\nOmega", omega)

                for node in quasi_coordinates:
                    current_coordinates = all_coordinates[node]
                    current_paths = all_paths[node]
                    sum_current_coordinates = current_coordinates[tuple(sorted(omega[0]))] + current_coordinates[
                        tuple(sorted(omega[1]))]

                    if quasi_coordinates[node] > sum_current_coordinates:
                        quasi_coordinates[node] = sum_current_coordinates
                        quasi_paths[node] = list(
                            set(current_paths[tuple(sorted(omega[0]))] + current_paths[tuple(sorted(omega[1]))]))

            # упорядочиваем квазикоординаты в порядке возрастания
            quasi_coordinates = dict(
                collections.OrderedDict(sorted(quasi_coordinates.items(), key=lambda kv: kv[1])))
            # упорядочиваем пути по квазикоординатам
            quasi_paths = sort_dict_by_keys(quasi_paths, quasi_coordinates.keys())

            if check_print_debug:
                print("Quasi coordinates:\t", quasi_coordinates)
                for key in quasi_coordinates:
                    print(key, ": ", quasi_coordinates[key], sep="")
                print("Quasi paths:", )
                for key in quasi_paths:
                    print(key, ": ", quasi_paths[key], sep="")

            # вторая стадия
            fi = []
            lot_of_pi = {}
            number_ro = 1
            lot_of_ro = {}

            if check_print_debug: print(nodes_in_omega)

            # 1 шаг
            coordinate = sorted([node for node in nodes_in_omega if node != nodes_in_omega[0]])
            first_weight = all_coordinates[nodes_in_omega[0]][tuple(coordinate)]
            fi.extend(
                nodes_in_omega + [quasi for quasi in quasi_coordinates if quasi_coordinates[quasi] == first_weight])
            first_edges = all_paths[nodes_in_omega[0]][tuple(coordinate)]
            for vertex in fi:
                all_paths[vertex][tuple(sorted(nodes_in_omega))] = first_edges

            for key, edge in quasi_paths.items():
                if key in fi:
                    for e in edge:
                        if e not in first_edges:
                            first_edges.append(e)
            pometka_lambda = {node: max_weight for node in G.adj if node not in fi}
            pometka_lambda_path = {node: [] for node in G.adj if node not in fi}

            lot_of_ro[number_ro] = Ro(fi.copy(), first_edges, first_weight)
            lot_of_pi[number_ro] = fi.copy()

            quasi_coordinates = {key: value for key, value in quasi_coordinates.items() if value != first_weight}
            quasi_paths = {key: value for key, value in quasi_paths.items() if key in quasi_coordinates}

            # следующие шаги
            all_nodes = [node for node in G.nodes if node not in fi]

            while len(all_nodes) > 0:
                new_min_weight = max_weight
                next_nodes = [key for key in G.nodes if
                              key not in fi]  # вершины, для которых сейчас будем заново строить лямбды
                for node in next_nodes:
                    for ro in lot_of_ro.values():
                        for vertex in ro.vertex:
                            if G.adj[node].get(vertex) is not None:
                                new_min_weight = min(new_min_weight,
                                                     ro.weight + G.adj[node].get(vertex).get("weight"))
                                if pometka_lambda[node] > ro.weight + G.adj[node].get(vertex).get("weight"):
                                    pometka_lambda[node] = ro.weight + G.adj[node].get(vertex).get("weight")
                                    pometka_lambda_path[node] = ro.edges + get_path_by_nodes(
                                        # todo скорее всего неправильно
                                        nx.dijkstra_path(G, node, vertex))
                new_min_weight = min(new_min_weight, quasi_coordinates[list(quasi_coordinates.keys())[0]])
                new_vertex = []
                all_new_edges = []
                for coordinate in quasi_coordinates:
                    if quasi_coordinates[coordinate] == new_min_weight:
                        new_vertex.append(coordinate)
                        all_new_edges.append(quasi_paths[coordinate])
                        all_paths[coordinate][tuple(sorted(nodes_in_omega))] = quasi_paths[coordinate]

                for node in pometka_lambda:
                    if pometka_lambda[node] == new_min_weight:
                        new_vertex.append(node)
                        all_new_edges.append(pometka_lambda_path[node])
                        all_paths[node][tuple(sorted(nodes_in_omega))] = pometka_lambda_path[node]
                new_vertex = list(set(new_vertex))
                new_edges = set()
                for path in all_new_edges:
                    new_edges.update(path)
                new_edges = list(new_edges)

                quasi_coordinates = {key: quasi_coordinates[key] for key in quasi_coordinates if
                                     key not in new_vertex}
                quasi_paths = {key: quasi_paths[key] for key in quasi_paths if key not in new_vertex}
                for vertex in new_vertex:
                    pometka_lambda.pop(vertex)
                    pometka_lambda_path.pop(vertex)
                new_ro = Ro(new_vertex, new_edges, new_min_weight)
                number_ro += 1
                lot_of_pi[number_ro] = new_vertex
                fi += [vertex for vertex in new_vertex if vertex not in fi]
                lot_of_ro[number_ro] = new_ro
                all_nodes = [node for node in all_nodes if node not in fi]

            if check_print_debug:
                print("lot_of_ro", end=' { ')
                for key, value in lot_of_ro.items():
                    print(f"{key}: {value}", end=" |||| ")
                print("}")
                # print("lot_of_pi", lot_of_pi)
                # print("fi", fi, "\n")
            for ro in lot_of_ro.values():
                for vertex in ro.vertex:
                    all_coordinates[vertex][tuple(sorted(nodes_in_omega))] = ro.weight

            # stein_nodes = lot_of_ro[1].vertex + [init_nodes[len(init_nodes) - 1]]
            # print("stein nodes:", stein_nodes)

    if check_print_debug:
        print("\nall_coordinates")
        for key in all_coordinates:
            print(key, ": ", all_coordinates[key], sep="")
        print()
        print("all_paths")
        for key in all_paths:
            print(key, ": ", all_paths[key], sep="")
        print()

    result_length = all_coordinates[init_nodes[len(init_nodes) - 1]][tuple(sorted(init_nodes[:len(init_nodes) - 1]))]
    result_path = all_paths[init_nodes[len(init_nodes) - 1]][tuple(sorted(init_nodes[:len(init_nodes) - 1]))]

    print(
        f"length: {result_length}")
    print(
        f"path: {result_path}")

    return result_path


# main:
G = nx.Graph()
initialize_graph(G)

# дейкстра для 4 точек
# init_nodes4 = [1, 3, 8, 9]  # норм
# init_nodes4 = [1, 6, 8, 10]
# dijkstra4(G, init_nodes4)
print()
check_print_debug = True
result_path = dijkstra4(G, [1, 8, 6, 10])
# dijkstra4(G, [1, 10, 6, 8])
# dijkstra4(G, [1, 8, 6, 10, 3])
print_graph(G, result_path)
