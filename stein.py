import networkx as nx
import matplotlib.pyplot as plt
import copy
from itertools import combinations
import collections
from ro import Ro

count_operation = 0
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
    # G.add_node(11, pos=(70, 10))  # comment
    # G.add_node(12, pos=(95, 20))  # comment
    e = [(1, 2, 1), (1, 5, 4), (2, 5, 2),
         (2, 3, 1),
         # (3, 4, 4), (4, 7, 1),  # comment
         (2, 6, 3),
         # (6, 7, 4), (7, 12, 4),  # comment
         (5, 8, 1), (8, 9, 3), (5, 9, 2),
         (9, 10, 1), (5, 6, 4), (6, 10, 4),
         # (10, 11, 2), (6, 11, 2), (11, 12, 3)  # comment
         ]
    G.add_weighted_edges_from(e)


def print_graph(G):
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
    plt.show()


# дейкстра для двух точек
def dijkstra2(G, a, b):
    global count_operation
    count_operation += 1
    print("min path:", nx.dijkstra_path(G, a, b))
    print("min path length:", nx.dijkstra_path_length(G, a, b))


# дейкстра для 3 точек
def dijkstra3(G, init_nodes):
    global count_operation
    min = 10000
    array_min_i = []
    for i in G.nodes:
        local_min = 0

        for init_node in init_nodes:
            count_operation += 1
            local_min += nx.dijkstra_path_length(G, i, init_node)

        if local_min <= min:
            if local_min < min:
                array_min_i = []
            count_operation += 1
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


# дейкстра для более, чем 3 точек
def dijkstra4(G, init_nodes):
    print("init_nodes:", init_nodes)
    global count_operation
    global max_weight
    count_operation = 0
    # 1 этап
    first_path_length = {}
    coordinates = {}  # todo здесь буду хранить все координаты

    for node in G.nodes:
        path_length = {}

        for init_node in init_nodes:
            count_operation += 1
            # if node != init_node:
            path_length[init_node] = nx.dijkstra_path_length(G, node, init_node)

        first_path_length[node] = path_length

    # ключ - вершина, значение - длина пути
    for key in first_path_length:
        print(key, ": ", first_path_length[key], sep="")
    print()

    init_nodes_with_out_last = init_nodes[:len(init_nodes) - 1]  # [1, 3, 8]
    print("init_nodes_with_out_last", init_nodes_with_out_last)

    for count_nodes_in_omegas in range(2, len(init_nodes)):
        lots_of_omegas = list(combinations(init_nodes_with_out_last, count_nodes_in_omegas))
        print("\nlots_of_omegas:", lots_of_omegas)
        # первая стадия
        for lot_omega in lots_of_omegas:
            nodes_to_subsets_omega = [lot_omega[0]]
            quasi_coordinates = {}

            omegas = []
            for i in range(1, len(lot_omega)):
                nodes_to_subsets_omega.append(lot_omega[i])
                omegas = get_disjoint_subsets(nodes_to_subsets_omega)

            print("\n-----------------\nOmegas:")
            for omega in omegas:
                print("\nOmega", omega)
                nodes_in_omega = [item for sublist in omega for item in sublist]

                for node in G.nodes:
                    if nodes_in_omega.count(node) == 0:  # перебираем все вершины кроме тех, что попали в омегу
                        tempmap = first_path_length[node]
                        quasi_coordinates[node] = tempmap[omega[0][0]] + tempmap[omega[1][0]]

                # упорядочиваем квазикоординаты в порядке возрастания
                quasi_coordinates = dict(
                    collections.OrderedDict(sorted(quasi_coordinates.items(), key=lambda kv: kv[1])))

                print("Quasi coordinates:\t", quasi_coordinates)
                for key in quasi_coordinates:
                    print(key, ": ", quasi_coordinates[key], sep="")

                # вторая стадия
                fi = []
                lot_of_pi = {}
                number_ro = 1
                lot_of_ro = {}
                pometka_lambda = {node: max_weight for node in G.adj}

                print("\n1 шаг")
                print(nodes_in_omega)

                # 1 шаг
                weight = first_path_length[nodes_in_omega[0]][nodes_in_omega[1]]
                fi.extend(nodes_in_omega + [quasi for quasi in quasi_coordinates if quasi_coordinates[quasi] == weight])

                lot_of_ro[number_ro] = Ro(fi.copy(), weight)
                lot_of_pi[number_ro] = fi.copy()

                quasi_coordinates = {key: value for key, value in quasi_coordinates.items() if value != weight}

                # следующие шаги
                print("\nследующие шаги")
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
                                    pometka_lambda[node] = min(pometka_lambda[node],
                                                               ro.weight + G.adj[node].get(vertex).get("weight"))
                                    # print(node, "and", vertex, "=",
                                    #       ro.weight + G.adj[node].get(vertex).get("weight"))
                    new_min_weight = min(new_min_weight, quasi_coordinates[list(quasi_coordinates.keys())[0]])
                    # print("new_min_weight", new_min_weight)
                    new_vertex = list(
                        set([quasi for quasi in quasi_coordinates if quasi_coordinates[quasi] == new_min_weight] \
                            + [node for node in pometka_lambda if pometka_lambda[node] == new_min_weight]))
                    quasi_coordinates = {key: quasi_coordinates[key] for key in quasi_coordinates if
                                         key not in new_vertex}
                    new_ro = Ro(new_vertex, new_min_weight)
                    number_ro += 1
                    lot_of_pi[number_ro] = new_vertex
                    fi += [vertex for vertex in new_vertex if vertex not in fi]
                    lot_of_ro[number_ro] = new_ro
                    all_nodes = [node for node in all_nodes if node not in fi]
                    print(f"lot_of_ro[{number_ro}]: {lot_of_ro[number_ro]}")
                    print("fi", fi)

                # if any([(i in nodes_in_omega) for i in G.adj[node]]):
                #     print(node)
                print("\n\npometka_lambda", pometka_lambda)
                print("\nlot_of_ro")
                for key_dictionary in lot_of_ro:
                    print(f"{key_dictionary}: {lot_of_ro[key_dictionary]}")

                print("\nlot_of_pi")

                for key_dictionary in lot_of_pi:
                    print(f"{key_dictionary}: {lot_of_pi[key_dictionary]}")

                print("\nfi", fi)

                # print("\nffff", G.adj)
                # print("ffff", G.adj[1])
                # print("ffff", G.adj[1][5])
                # print("ffff", G.adj[1].get(6))
                # print("ffff", G.adj[1][5]['weight'])

                # break  # todo удалить все брейки
            # break
        # break


# main:
G = nx.Graph()
initialize_graph(G)

# дейкстра для 2 и 3 точек
# init_nodes = [1, 3, 9]
# print("init nodes: ", init_nodes, "\n")
#
# if len(init_nodes) == 2:
#     dijkstra2(G, init_nodes[0], init_nodes[1])
# elif len(init_nodes) == 3:
#     dijkstra3(G, init_nodes)
#
# print("\ncount operation: %d" % count_operation)

# print("----------------------")
# print("min path:", nx.single_source_dijkstra_path(G, 1))
# print("min path length:", nx.single_source_dijkstra_path_length(G, 1))
# print("----------------------")


# дейкстра для 4 точек
init_nodes4 = [1, 3, 8, 9]
dijkstra4(G, init_nodes4)
print_graph(G)
