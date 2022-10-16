#  File: Graph.py

#  Description:

#  Student Name: Tomas Matteson

#  Student UT EID: tlm3347

#  Partner Name:

#  Partner UT EID:

#  Course Name: CS 313E

#  Unique Number:

#  Date Created: 8/10/18

#  Date Last Modified: 8/13/18


class Stack(object):
    def __init__(self):
        self.stack = []

    # add an item to the top of the stack
    def push(self, item):
        self.stack.append(item)

    # remove an item from the top of the stack
    def pop(self):
        return self.stack.pop()

    # check what item is on top of the stack without removing it
    def peek(self):
        return self.stack[len(self.stack) - 1]

    # check if a stack is empty
    def isEmpty(self):
        return (len(self.stack) == 0)

    # return the number of elements in the stack
    def size(self):
        return (len(self.stack))


class Queue(object):
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        return (self.queue.pop(0))

    def isEmpty(self):
        return (len(self.queue) == 0)

    def size(self):
        return len(self.queue)


class Vertex(object):
    def __init__(self, label):
        self.label = label
        self.visited = False

    # determine if a vertex was visited
    def was_visited(self):
        return self.visited

    # determine the label of the vertex
    def get_label(self):
        return self.label

    # string representation of the vertex
    def __str__(self):
        return str(self.label)


# initializing the Graph class
class Graph(object):
    # The graph is visualized by a list of vertices and by an adjacency matrix (list of list of weights)
    def __init__(self):
        self.Vertices = []
        self.adjMat = []

    # check if a vertex is already in the graph
    def has_vertex(self, label):
        # n number of vertices
        nVert = len(self.Vertices)
        # for each vertex
        for i in range(nVert):
            # if the label we're looking for is that vertex's label, return True
            if (label == (self.Vertices[i]).label):
                return True
        # if we can't find it, say False
        return False

    # given a label get the index of a vertex
    def get_index(self, label):
        # n number of vertices
        nVert = len(self.Vertices)
        # for each vertex
        for i in range(nVert):
            # if the label we're looking for is that vertex's label, return the vertex
            if ((self.Vertices[i]).label == label):
                return i
        # otherwise return -1
        return -1

    # add a Vertex with a given label to the graph
    def add_vertex(self, label):
        # if the vertex in question cannot be found, add a new vertex with the label data to the vertex list
        if not self.has_vertex(label):
            self.Vertices.append(Vertex(label))

            # then add a new column in the adjacency matrix for the new Vertex
            nVert = len(self.Vertices)
            for i in range(nVert - 1):
                (self.adjMat[i]).append(0)

            # then add a new row for the new Vertex in the adjacency matrix
            new_row = []
            for i in range(nVert):
                new_row.append(0)
            self.adjMat.append(new_row)

    # add weighted directed edge to graph, needs start finish and weight
    def add_directed_edge(self, start, finish, weight=1):
        self.adjMat[start][finish] = weight

    # add weighted undirected edge to graph, needs start finish and weight
    def add_undirected_edge(self, start, finish, weight=1):
        self.adjMat[start][finish] = weight
        self.adjMat[finish][start] = weight

    # return an unvisited vertex adjacent to vertex v
    def get_adj_unvisited_vertex(self, v):
        nVert = len(self.Vertices)
        # for each vertex in the list of vertices if an edge exists between it and the specified vertex,
        # but that vertex has not been visited, return these unvisited adjacent vertices
        for i in range(nVert):
            if (self.adjMat[v][i] > 0) and (not (self.Vertices[i]).was_visited()):
                return i
        # if no vertices are adjacent, return -1
        return -1

    # do the depth first search in a graph, starting at a vertex v
    def dfs(self, v):
        # create the Stack
        theStack = Stack()

        # mark vertex v as visited and push it on the stack
        (self.Vertices[v]).visited = True
        print(self.Vertices[v])
        theStack.push(v)

        # visit other vertices according to depth
        while (not theStack.isEmpty()):
            # get an adjacent unvisited vertex
            u = self.get_adj_unvisited_vertex(theStack.peek())
            if (u == -1):
                u = theStack.pop()
            # if the there really is an adjacent unvisited vertex, visit the one on the top of the stack
            # push it on the stack
            else:
                (self.Vertices[u]).visited = True
                print(self.Vertices[u])
                theStack.push(u)

        # the stack is empty let us reset the flags
        nVert = len(self.Vertices)
        for i in range(nVert):
            (self.Vertices[i]).visited = False

    # return the the index of a vertex with a given vertex_label
    # if there are more than one vertices return any of the indices
    # label is a string, and index is an integer
    # return -1 if label is not found in the graph
    def get_index (self, vertex_label):
        nVert = len(self.Vertices)
        for i in range(nVert):
            if (vertex_label == (self.Vertices[i]).label):
                return i
        return -1
        # it works!

    # return edge weight of edge starting at from_vertex_label
    # and ending at to_vertex_label
    # from_vertex_label and to_vertex_label are strings
    # returns an integer
    # return -1 if edge does not exist
    def get_edge_weight (self, from_vertex_label, to_vertex_label):
        nVert = len(self.Vertices)
        from_index = -1
        to_index = -1
        # for each vertex
        for i in range(nVert):
            # if the label we're looking for is that vertex's label, return the vertex
            if ((self.Vertices[i]).label == from_vertex_label):
                from_index = i
        nVert = len(self.Vertices)
        # for each vertex
        for i in range(nVert):
            # if the label we're looking for is that vertex's label, return the vertex
            if ((self.Vertices[i]).label == to_vertex_label):
                to_index = i
        if self.adjMat[from_index][to_index] != 0:
            return self.adjMat[from_index][to_index]
        return -1

        # it works for both pos and neg cases

    # return a list of indices of immediate neighbors that
    # you can reach from the vertex with the given label
    # vertex_label is a string and the function returns a list of integers
    # return empty list if there are none or if the given label is not
    # in the graph
    def get_neighbors (self, vertex_label):
        nVert = len(self.Vertices)
        neighbor_list = []
        v = None
        if not self.has_vertex(vertex_label):
            return neighbor_list
        for i in range(nVert):
            if (vertex_label == (self.Vertices[i]).label):
                v = i

        # for each vertex in the list of vertices if an edge exists between it and the specified vertex,
        # , return these unvisited adjacent vertices
        for i in range(nVert):
            if (self.adjMat[v][i] > 0):
                neighbor_list.append(i)
        # if no vertices are adjacent, return -1
        return neighbor_list
        # it works for both pos and neg cases

    # return a list of the vertices in the graph
    # returns a list of Vertex objects
    def get_vertices (self):
        vert_list = []
        for i in self.Vertices:
            vert_list.append(str(i))
        return vert_list
    # it works!

    # delete an edge from the graph
    # from_vertex_label and to_vertex_label are strings
    # if there is no edge, does nothing
    # does not return anything
    # make sure to modify the adjacency matrix appropriately
    def delete_edge (self, from_vertex_label, to_vertex_label):

        if self.has_vertex(from_vertex_label) and self.has_vertex(to_vertex_label):
            nVert = len(self.Vertices)
            from_index = -1
            to_index = -1
            # for each vertex
            for i in range(nVert):
                # if the label we're looking for is that vertex's label, return the vertex
                if ((self.Vertices[i]).label == from_vertex_label):
                    from_index = i
            nVert = len(self.Vertices)
            # for each vertex
            for i in range(nVert):
                # if the label we're looking for is that vertex's label, return the vertex
                if ((self.Vertices[i]).label == to_vertex_label):
                    to_index = i
            if self.adjMat[from_index][to_index] != 0:
                self.adjMat[from_index][to_index] = 0

        # I think this works for both pos and neg cases

    # delete a vertex from the graph
    # vertex_label is a string
    # if there is no such vertex, does nothing
    # does not return anything
    # make sure to remove vertex from vertex list AND
    # remove the appropriate row/column of the adjacency matrix
    def delete_vertex (self, vertex_label):
        # does remove work by object or by index?
        nVert = len(self.Vertices)
        vert_copy = self.Vertices[:]
        for i in range(nVert):
            if (vertex_label == (vert_copy[i]).label):
                self.Vertices.remove(self.Vertices[i])
                del self.adjMat[i]
        # it works for both pos and neg cases



def main():
    # create a Graph object
    cities = Graph()

    # open file for reading
    in_file = open("./graph.txt", "r")

    # read the Vertices
    num_vertices = int((in_file.readline()).strip())
    print(num_vertices)

    for i in range(num_vertices):
        city = (in_file.readline()).strip()
        print(city)
        cities.add_vertex(city)

    # read the edges
    num_edges = int((in_file.readline()).strip())
    print(num_edges)

    for i in range(num_edges):
        edge = (in_file.readline()).strip()
        print(edge)
        edge = edge.split()
        start = int(edge[0])
        finish = int(edge[1])
        weight = int(edge[2])

        cities.add_directed_edge(start, finish, weight)

    # print the adjacency matrix
    print("\nAdjacency Matrix")
    for i in range(num_vertices):
        for j in range(num_vertices):
            print(cities.adjMat[i][j], end=' ')
        print()
    print()

    # read the starting vertex for dfs and bfs
    start_vertex = (in_file.readline()).strip()
    print(start_vertex)

    # get the index of the starting vertex
    start_index = cities.get_index(start_vertex)
    print(start_index)

    # do depth first search
    print("\nDepth First Search from " + start_vertex)
    cities.dfs(start_index)
    print()

    # Divide between Mitra's output code and my own
    print("----------------------------")
    # test get_index on known has
    print(cities.get_index("Seattle"))
    # test get_index on known has not (should be -1)
    print(cities.get_index("Jo"))
    # test delete_edge on known has (it works)
    # cities.delete_edge("San Francisco", "Denver")
    # test delete_edge on known has not (should be -1)
    cities.delete_edge("San Francisco", "jo")
    # test get_edge_weight on known has
    print(cities.get_edge_weight("San Francisco", "Seattle"))
    # test get_edge_weight on known has not
    print(cities.get_edge_weight("San Francisco", "Jo"))
    # test get_vertices on filled list
    print(cities.get_vertices())
    # test get_vertices on empty list (it gives an empty list, like it should)
    # cities.Vertices = []
    # print(cities.get_vertices())

    # test get_neighbors of known has (it works)
    print(cities.get_neighbors("San Francisco"))
    # test get_neighbors of known has not (it works)
    print(cities.get_neighbors("San Jo"))

    # test delete_vertex of known has (it works)
    # print(len(cities.adjMat))
    # cities.delete_vertex("San Francisco")
    # print(len(cities.adjMat))
    # print(cities.has_vertex("San Francisco"))

    # test delete_vertex of known has not
    print(len(cities.adjMat))
    print(cities.has_vertex("Jo"))
    cities.delete_vertex("Jo")
    print(len(cities.adjMat))
    print(cities.has_vertex("Jo"))
    # close the file
    in_file.close()


main()
