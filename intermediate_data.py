import pickle

with open('data.txt', 'rb') as f:
    intermediate = pickle.load(f)

print(intermediate)

# for current_n, current_m, init_nodes in intermediate:
#      print(current_n, current_m)
