class Ro:

    def __init__(self, vertex, weight):
        self.vertex = vertex
        self.weight = weight

    def __repr__(self):
        return f"vertex -> {str(self.vertex)}; weight -> {str(self.weight)}"

