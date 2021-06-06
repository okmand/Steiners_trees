import xlsxwriter as xls
import time
import random
from stein import initialize_random_graph, levin_algorithm
import webbrowser
import pickle

# тесты по алгоритму.
# number_iterations_random - количество раз, сколько сгенерируется граф с данными n и m
# number_iterations_algorithm - количество раз, сколько пройдет алгоритм
# number_iterations_random * number_iterations_algorithm = сколько раз пройдет алгоритм на текущих n и m
def get_algorithm_time(n, m, min_random_weight, max_random_weight, number_init_nodes, number_iterations_random,
                       number_iterations_algorithm):
    result_time_algorithm = []
    for i in range(number_iterations_random):
        G = initialize_random_graph(n, m, min_random_weight, max_random_weight)
        if G is None:
            return None
        init_nodes = random.sample(range(0, n), number_init_nodes)
        start_time = time.time()
        for j in range(number_iterations_algorithm):
            levin_algorithm(G, init_nodes, False, False)
        end_time = time.time()
        result_time_algorithm.append((end_time - start_time) / number_iterations_algorithm)
    result_time = 0
    for time_alg in result_time_algorithm:
        result_time += time_alg
    return result_time / number_iterations_random


if __name__ == "__main__":
    start_time_main = time.time()
    workbook = xls.Workbook('stein_excel_20.xlsx')
    worksheet = workbook.add_worksheet("first")
    bold = workbook.add_format({'bold': True})

    current_column = 0
    min_n, max_n = 20, 20  # будет генерироваться эксель с разным количеством вершин графа
    step_m = 50
    intermediate_data = {}

    for current_n in range(min_n, max_n + 1):
        worksheet.write(0, current_column, current_n)
        worksheet.write(1, current_column, 'm', bold)
        worksheet.write(0, current_column + 1, 'init_nodes', bold)
        max_current_m = int(current_n * (current_n - 1) / 2)

        current_row = 2
        # запись ребер в экселе
        for i in range(current_n - 1, max_current_m + 1, step_m):
            worksheet.write(current_row, current_column, i)
            current_row += 1

        for init_nodes in range(2, current_n + 1):
            current_row = 2
            current_column += 1
            worksheet.write(1, current_column, init_nodes)
            for current_m in range(current_n - 1, max_current_m + 1, step_m):
                current_data = get_algorithm_time(current_n, current_m, 10, 50, init_nodes, 1, 2)
                print(f"current_n: {current_n}, current_m: {current_m}, init_nodes: {init_nodes}, time: {time.time() - start_time_main},\tcurrent_data: {current_data}")
                intermediate_data[(current_n, current_m, init_nodes)] = current_data  # сохраняем промежуточные данные
                with open('data.txt', 'wb') as f:
                    pickle.dump(intermediate_data, f)  # сохраняем промежуточные данные в файл data.txt
                worksheet.write(current_row, current_column, current_data)
                current_row += 1

        current_column += 2

    workbook.close()
    webbrowser.open('stein_excel_20.xlsx')

    end_time_main = time.time()
    result_time_main = (end_time_main - start_time_main)
    print(f"time spent for excel: {result_time_main}")
