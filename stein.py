import networkx as nx
import matplotlib.pyplot as plt
import copy
from itertools import combinations
import collections
import time
import random
import sys
from ro import Ro

max_weight = 10000000


def initialize_random_graph(n, m, min_random_weight, max_random_weight):
    G = nx.dense_gnm_random_graph(n, m)  # второй параметр - количество ребер
    attempt = 10  # 10 попыток на то, чтобы построить связный граф
    while attempt > 0 and not nx.is_connected(G):  # проверка на связность графа
        G = nx.dense_gnm_random_graph(n, m)
        attempt -= 1
    if not nx.is_connected(G):
        return None
    weights = [(edge[0], edge[1], random.randint(min_random_weight, max_random_weight)) for edge in G.edges]
    G.add_weighted_edges_from(weights)
    return G


def print_random_graph(G, result_path, pos):
    if pos is None:
        pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=200, font_size=9)
    nx.draw_networkx_edges(G, pos, edge_color="#dedede")
    nx.draw_networkx_edges(G, pos, edgelist=result_path, edge_color="#f70909")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='#015e98')
    plt.show()
    return pos


def initialize_graph():
    G = nx.Graph()
    G.add_node(1, pos=(20, 50))
    G.add_node(2, pos=(45, 50))
    G.add_node(3, pos=(65, 60))
    G.add_node(4, pos=(90, 50))
    G.add_node(5, pos=(30, 30))
    G.add_node(6, pos=(60, 30))
    G.add_node(7, pos=(80, 35))
    G.add_node(8, pos=(10, 10))
    G.add_node(9, pos=(30, 10))
    G.add_node(10, pos=(50, 10))
    G.add_node(11, pos=(70, 10))
    G.add_node(12, pos=(95, 20))
    e = [
        (1, 2, 1), (1, 5, 4), (2, 5, 2),
        (2, 3, 1), (3, 4, 4), (4, 7, 1),
        (2, 6, 3), (6, 7, 4), (7, 12, 4),
        (5, 8, 1), (8, 9, 3), (5, 9, 2),
        (9, 10, 1), (5, 6, 4), (6, 10, 4),
        (10, 11, 1), (6, 11, 2), (11, 12, 3)
    ]
    G.add_weighted_edges_from(e)
    return G


def print_graph(G, result_path):
    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=200, font_size=9)
    nx.draw_networkx_edges(G, pos, edge_color="#dedede")
    nx.draw_networkx_edges(G, pos, edgelist=result_path, edge_color="#f70909")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='#015e98')
    plt.show()


# дейкстра для двух точек
def dijkstra2(G, a, b):
    print("min path:", nx.dijkstra_path(G, a, b))
    print("min path length:", nx.dijkstra_path_length(G, a, b))


# дейкстра для 3 точек
def dijkstra3(G, init_nodes):
    min = max_weight
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


def sort_dict_by_keys(my_dict, keys):
    result = {}
    for key in keys:
        result[key] = my_dict.get(key)
    return result


# алгоритм Левина для более, чем 4 точек
def levin_algorithm(G, init_nodes, check_print, check_print_result):
    global max_weight

    all_coordinates = {}  # здесь будут храниться все координаты
    all_paths = {}  # здесь будут храниться все пути

    # по алгоритму надо находить пути до всех точек из init_nodes, кроме последней
    init_nodes_with_out_last = init_nodes[:len(init_nodes) - 1]
    if check_print: print("init_nodes_with_out_last", init_nodes_with_out_last)

    # 1 этап - 1-координаты
    for node in G.nodes:
        first_path = {}
        first_coordinate = {}

        for init_node in init_nodes_with_out_last:
            # храним пути и координаты в tuple
            first_coordinate[(init_node,)] = nx.dijkstra_path_length(G, node, init_node)
            first_path[(init_node,)] = get_path_by_nodes(nx.dijkstra_path(G, node, init_node))

        all_coordinates[node] = first_coordinate  # ключ - вершина, значение - длина пути
        all_paths[node] = first_path  # ключ - вершина, значение - путь

    if check_print:
        for key in all_coordinates:
            print(key, ": ", all_coordinates[key], sep="")
        print()

    # остальные этапы
    for count_nodes_in_omegas in range(2, len(init_nodes)):
        # всевозможные множества вершин омега (на k-ом этапе в омеге будет k вершин)
        lots_of_omegas = list(combinations(init_nodes_with_out_last, count_nodes_in_omegas))  # всевозможные
        if check_print: print("\nlots_of_omegas:", lots_of_omegas)
        # первая стадия. находим квазикоординаты и квазипути
        for lot_omega in lots_of_omegas:
            nodes_to_subsets_omega = [lot_omega[0]]
            quasi_coordinates = {}
            quasi_paths = {}

            # omegas - всевозможные разбиения множества вершин омега
            omegas = []
            for i in range(1, len(lot_omega)):
                nodes_to_subsets_omega.append(lot_omega[i])
                omegas = get_disjoint_subsets(nodes_to_subsets_omega)

            if check_print: print("\n-----------------\nOmegas:", omegas)
            nodes_in_omega = []
            for omega in omegas:
                for sublist in omega:
                    nodes_in_omega += [elem for elem in sublist if elem not in nodes_in_omega]

            # инициализируем квазикоординаты и квазипути
            for node in G.nodes:
                if nodes_in_omega.count(node) == 0:  # перебираем все вершины кроме тех, что попали в омегу
                    quasi_coordinates[node] = max_weight
                    quasi_paths[node] = []

            # рассматриваем всевозможные разбиения множества вершин омега
            for omega in omegas:
                if check_print: print("\nOmega", omega)

                # находим квазикоординаты и квазипути
                for node in quasi_coordinates:
                    current_coordinates = all_coordinates[node]
                    current_paths = all_paths[node]
                    sum_current_coordinates = current_coordinates[tuple(sorted(omega[0]))] \
                                              + current_coordinates[tuple(sorted(omega[1]))]

                    if quasi_coordinates[node] > sum_current_coordinates:
                        quasi_coordinates[node] = sum_current_coordinates
                        quasi_paths[node] = list(
                            set(current_paths[tuple(sorted(omega[0]))] + current_paths[tuple(sorted(omega[1]))]))

            # упорядочиваем квазикоординаты в порядке возрастания
            quasi_coordinates = dict(
                collections.OrderedDict(sorted(quasi_coordinates.items(), key=lambda kv: kv[1])))
            # упорядочиваем квазипути по квазикоординатам
            quasi_paths = sort_dict_by_keys(quasi_paths, quasi_coordinates.keys())

            if check_print:
                print("Quasi coordinates:\t", quasi_coordinates)
                for key in quasi_coordinates:
                    print(key, ": ", quasi_coordinates[key], sep="")
                print("Quasi paths:", )
                for key in quasi_paths:
                    print(key, ": ", quasi_paths[key], sep="")

            # вторая стадия
            fi = []  # множество вершин, снабженных координатами на текущем шаге
            lot_of_ro = {}  # множество для хранения координат ro. Используется отдельный класс Ro
            number_ro = 1  # счетчик для записи в lot_of_ro

            if check_print: print(nodes_in_omega)

            # 1 шаг:
            # наименьшей координатой снабжаются вершины nodes_in_omega и вершины, чьи кавзикоординаты равны first_weight
            coordinate = sorted([node for node in nodes_in_omega if node != nodes_in_omega[0]])
            first_weight = all_coordinates[nodes_in_omega[0]][tuple(coordinate)]
            fi.extend(
                nodes_in_omega + [quasi for quasi in quasi_coordinates if quasi_coordinates[quasi] == first_weight])
            first_edges = {}
            for node_in_omega in nodes_in_omega:
                first_edges[node_in_omega] = all_paths[nodes_in_omega[0]][tuple(coordinate)]

            for key, edge in quasi_paths.items():
                if key in fi:
                    first_edges[key] = edge

            for vertex in fi:
                all_paths[vertex][tuple(sorted(nodes_in_omega))] = first_edges[vertex]

            lot_of_ro[number_ro] = Ro(fi.copy(), first_edges, first_weight)

            quasi_coordinates = {key: value for key, value in quasi_coordinates.items() if value != first_weight}
            quasi_paths = {key: value for key, value in quasi_paths.items() if key in quasi_coordinates}

            # следующие шаги
            pometka_lambda = {node: max_weight for node in G.adj if node not in fi}
            pometka_lambda_path = {node: [] for node in G.adj if node not in fi}
            nodes_outside_fi = [node for node in G.nodes if node not in fi]

            # идем по циклу до тех пор, пока все вершины не будут снабжены координатами
            while len(nodes_outside_fi) > 0:
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
                                    pometka_lambda_path[node] = ro.edges[vertex] + get_path_by_nodes(
                                        nx.dijkstra_path(G, node, vertex))
                new_min_weight = min(new_min_weight, quasi_coordinates[list(quasi_coordinates.keys())[0]])
                new_vertex = []  # вершины, которые будут помещены в следующее ro
                new_edges = {}  # ребра, которые будут помещены в следующее ro
                for coordinate in quasi_coordinates:
                    if quasi_coordinates[coordinate] == new_min_weight:
                        new_vertex.append(coordinate)
                        new_edges[coordinate] = quasi_paths[coordinate]
                        all_paths[coordinate][tuple(sorted(nodes_in_omega))] = quasi_paths[coordinate]

                for node in pometka_lambda:
                    if pometka_lambda[node] == new_min_weight:
                        new_vertex.append(node)
                        new_edges[node] = pometka_lambda_path[node]
                        all_paths[node][tuple(sorted(nodes_in_omega))] = pometka_lambda_path[node]
                new_vertex = list(set(new_vertex))

                # пересчитываем квазикоординаты, квазипути, пометки для лямбд
                quasi_coordinates = {key: quasi_coordinates[key] for key in quasi_coordinates if
                                     key not in new_vertex}
                quasi_paths = {key: quasi_paths[key] for key in quasi_paths if key not in new_vertex}
                for vertex in new_vertex:
                    pometka_lambda.pop(vertex)
                    pometka_lambda_path.pop(vertex)

                new_ro = Ro(new_vertex, new_edges, new_min_weight)
                fi += [vertex for vertex in new_vertex if vertex not in fi]
                number_ro += 1
                lot_of_ro[number_ro] = new_ro
                nodes_outside_fi = [node for node in nodes_outside_fi if node not in fi]

            if check_print:
                print("lot_of_ro", end=' { ')
                for key, value in lot_of_ro.items():
                    print(f"{key}: {value}", end=" |||| ")
                print("}")
            for ro in lot_of_ro.values():
                for vertex in ro.vertex:
                    all_coordinates[vertex][tuple(sorted(nodes_in_omega))] = ro.weight

    if check_print:
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
    steiner_points = {vertex for edge in result_path for vertex in edge if vertex not in init_nodes}  # точки штейнера
    if len(steiner_points) == 0:
        steiner_points = "{}"

    if check_print_result:
        print(f"number of the nodes: {len(G.nodes)}")
        print(f"number of the edges: {len(G.edges)}")
        print(f"init_nodes: {init_nodes}")
        print(f"result length: {result_length}")
        print(f"result path: {result_path}")
        print(f"Steiner points: {steiner_points}")

    return result_path


if __name__ == "__main__":
    # G = initialize_graph()
    n, m, min_random_weight, max_random_weight = 13, 30, 2, 9
    G = initialize_random_graph(n, m, min_random_weight, max_random_weight)
    if G is None:
        sys.exit("You can't build a connected graph with such input data")

    pos = None
    pos = print_random_graph(G, [], pos)

    # init_nodes - инициализирующие вершины, на которых будет строиться дерево Штейнера
    number_init_nodes = 9
    init_nodes = random.sample(range(0, n), number_init_nodes)
    result_path = levin_algorithm(G, init_nodes, False, True)
    print_random_graph(G, result_path, pos)

    start_time = time.time()
    for i in range(20):
        levin_algorithm(G, init_nodes, False, False)
    end_time = time.time()
    result_time = (end_time - start_time) / 20
    print(f"\nresult time: {result_time}")

    # print_graph(G, result_path)
