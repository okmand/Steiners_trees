class Ro:

    def __init__(self, vertex, edges, weight):
        self.vertex = vertex
        self.edges = edges
        self.weight = weight

    def __repr__(self):
        return f"vertex -> {str(self.vertex)}, edges -> {str(self.edges)}, weight -> {str(self.weight)}"

