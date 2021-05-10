import networkx as nx
import matplotlib.pyplot as plt
import copy

count_operation = 0


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
def get_disjoint_sets(arr):
    result = get_disjoint_sets_function_body(arr, 2, [[]], 0)
    result.sort(key=lambda val: len(val[0]))
    return result


# основное тело функции получения непустых непересекающихся подмножеств
def get_disjoint_sets_function_body(arr, k, accum, index):
    if index == len(arr):
        if k == 0:
            return copy.deepcopy(accum)
        else:
            return []

    element = arr[index]
    result = []

    for set_i in range(len(accum)):
        if k > 0:
            clone_new = copy.deepcopy(accum)
            clone_new[set_i].append([element])
            result.extend(get_disjoint_sets_function_body(arr, k - 1, clone_new, index + 1))

        for elem_i in range(len(accum[set_i])):
            clone_new = copy.deepcopy(accum)
            clone_new[set_i][elem_i].append(element)
            result.extend(get_disjoint_sets_function_body(arr, k, clone_new, index + 1))

    return result


# дейкстра для более, чем 3 точек
def dijkstra4(G, init_nodes):
    print("init_nodes:", init_nodes)
    global count_operation
    count_operation = 0
    # 1 этап
    first_path_length = {}
    for i in G.nodes:
        path_length = {}

        for init_node in init_nodes:
            count_operation += 1
            if i != init_node:
                path_length[str(init_node)] = nx.dijkstra_path_length(G, i, init_node)

        first_path_length[i] = path_length

    # ключ - вершина, значение - длина пути
    for key in first_path_length:
        print(key, ": ", first_path_length[key], sep="")
    print()

    new_arr = [init_nodes[0]]
    for i in range(1, len(init_nodes)):
        new_arr.append(init_nodes[i])
        omegas = get_disjoint_sets(new_arr)
        print("\n\nOmegas:")
        for r in omegas:
            print(r)


# main:
G = nx.Graph()
initialize_graph(G)

# дейкстра для 2 и 3 точек
init_nodes = [1, 3, 9]
print("init nodes: ", init_nodes, "\n")

if len(init_nodes) == 2:
    dijkstra2(G, init_nodes[0], init_nodes[1])
elif len(init_nodes) == 3:
    dijkstra3(G, init_nodes)

print("\ncount operation: %d" % count_operation)
print_graph(G)

print("----------------------")
print("min path:", nx.single_source_dijkstra_path(G, 1))
print("min path length:", nx.single_source_dijkstra_path_length(G, 1))
print("----------------------")

# дейкстра для 4 точек
init_nodes4 = [1, 3, 8, 9]
dijkstra4(G, init_nodes4)
