#  File: TopoSort.py

#  Description:

#  Student Name: Tomas Matteson

#  Student UT EID: tlm3347

#  Partner Name: Hamza Sait

#  Partner UT EID: hs26386

#  Course Name: CS 313E

#  Unique Number:

#  Date Created: 8/10/18

#  Date Last Modified: 8/18/18


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


class Graph(object):
    def __init__(self):
        self.Vertices = []
        self.adjMat = []

    # check if a vertex is already in the graph
    def has_vertex(self, label):
        nVert = len(self.Vertices)
        for i in range(nVert):
            if (label == (self.Vertices[i]).label):
                return True
        return False

    def get_vertices(self):
        return self.Vertices

    def num_vertices(self):
        return len(self.Vertices)

    # given a label get the index of a vertex
    def get_index(self, label):
        nVert = len(self.Vertices)
        for i in range(nVert):
            if ((self.Vertices[i]).label == label):
                return i
        return -1

    # add a Vertex with a given label to the graph
    def add_vertex(self, label):
        if not self.has_vertex(label):
            self.Vertices.append(Vertex(label))

            # add a new column in the adjacency matrix for the new Vertex
            nVert = len(self.Vertices)
            for i in range(nVert - 1):
                (self.adjMat[i]).append(0)

            # add a new row for the new Vertex in the adjacency matrix
            new_row = []
            for i in range(nVert):
                new_row.append(0)
            self.adjMat.append(new_row)

    # add weighted directed edge to graph
    def add_directed_edge(self, start, finish, weight=1):
        s = self.get_index(start)
        f = self.get_index(finish)
        self.adjMat[s][f] = weight

    # add weighted undirected edge to graph
    def add_undirected_edge(self, start, finish, weight=1):
        self.adjMat[start][finish] = weight
        self.adjMat[finish][start] = weight

    # return an unvisited vertex adjacent to vertex v
    def get_adj_unvisited_vertex(self, v):
        nVert = len(self.Vertices)
        for i in range(nVert):
            if (self.adjMat[v][i] > 0) and (not (self.Vertices[i]).was_visited()):
                return i
        return -1

    def refresh(self):
        nVert = len(self.Vertices)
        for i in range(nVert):
            (self.Vertices[i]).visited = False

    def bfs(self, v):

        theQueue = Queue()
        print(self.Vertices[v])
        self.Vertices[v].visited = True

        while (True):
            while (True):
                u = self.get_adj_unvisited_vertex(v)
                if (u == -1):
                    break
                self.Vertices[u].visited = True
                theQueue.enqueue(self.Vertices[u])
                print(self.Vertices[u].label)
            if (theQueue.isEmpty() == True):
                break
            else:
                v = self.get_index(theQueue.dequeue().label)

        for x in range(len(self.Vertices)):
            (self.Vertices[x]).visited = False

    def get_edge_weight(self, from_vertex_label, to_vertex_label):

        a = self.get_index(from_vertex_label)
        b = self.get_index(to_vertex_label)

        return self.adjMat[a][b]

    def print_matrix(self, ):
        num_vertices = self.num_vertices()
        print("\nAdjacency Matrix")
        for i in range(num_vertices):
            print(self.Vertices[i])
            for j in range(num_vertices):
                print(self.adjMat[i][j], end=' ')
            print()
        print()

    def get_neighbors(self, v):

        output = []
        num = v

        num = self.adjMat[num]

        for x in range(len(num)):
            if (num[x] > 0):
                output.append(self.get_index(self.Vertices[x].label))
        return output

    def delete_edge(self, from_vertex_label, to_vertex_label):

        a = self.get_index(from_vertex_label)
        b = self.get_index(to_vertex_label)

        self.adjMat[a][b] = 0
        self.adjMat[b][a] = 0

    def delete_vertex(self, vertex_label):

        n = self.get_index(vertex_label)

        del self.Vertices[n]
        del self.adjMat[n]

        for x in range(len(self.adjMat)):
            del self.adjMat[x][n]

    def check_vertex(self, v):

        neighbors = (self.get_neighbors(v))
        for x in neighbors:
            if (self.Vertices[x].visited == True):
                return False
        return True

    def check_vertex2(self, v):

        neighbors = (self.get_neighbors(v))
        for x in neighbors:
            if (self.Vertices[x].visited != True):
                return False
        return True

    def has_cycle(self):
        for v in range(len(self.Vertices)):
            theStack = Stack()

            (self.Vertices[v]).visited = True
            theStack.push(v)

            while (not theStack.isEmpty()):
                u = self.get_adj_unvisited_vertex(theStack.peek())
                if (u == -1):
                    u = theStack.pop()
                else:
                    (self.Vertices[u]).visited = True
                    theStack.push(u)

                checks = self.get_neighbors(u)
                if (v in checks):
                    return True
            self.refresh()
        return False


    def fixer(self, listy):
        listy = sorted(listy)
        compare = []
        for x in range(len(self.adjMat)):
            compare.append(x)

        for y in range(len(listy)):
            if (listy[y] != compare[y]):
                return y
        else:
            return len(listy)

    def topo_sort(self):
        otherStack = Stack()

        for x in range(len(self.adjMat) - 1, -1, -1):
            output = self.topohelper(x, otherStack)

        outer = []
        if (len(self.adjMat) > len(output.stack)):
            output.push(self.fixer(output.stack))

        for x in range(len(output.stack)):
            outer.append(self.Vertices[output.pop()])

        return(outer)

    def topohelper(self, v, otherStack):
        theStack = Stack()
        theStack.push(v)
        u = v

        while (not theStack.isEmpty()):
            if (self.check_vertex2(u)):
                if (u not in otherStack.stack):
                    otherStack.push(u)
            u = self.get_adj_unvisited_vertex(theStack.peek())
            if (u == -1):
                u = theStack.pop()
            else:
                (self.Vertices[u]).visited = True
                theStack.push(u)
        self.refresh()
        for x in range(len(self.adjMat)):
            if (x in otherStack.stack):
                self.Vertices[x].visited = True

        return otherStack


def read_file(graph):
    vertex_number = None
    edge_number = None
    file = open("topo.txt", "r")
    file = file.read().splitlines()
    for i in range(len(file)):
        if i == 0:
            vertex_number = file[i]
        elif 1 <= i <= int(vertex_number):
            graph.add_vertex(file[i])
        elif i == (int(vertex_number) + 1):
            edge_number = file[i]
        else:
            to_from = file[i].split(" ")
            graph.add_directed_edge(to_from[0], to_from[1])



def main():
    graph = Graph()
    read_file(graph)
    if graph.has_cycle():
        print("Cycle detected, remove cycle to use topological sort")
    else:
        output = graph.topo_sort()
        for x in range(len(output)):
            print(output[x].label, end= " ")


if __name__ == "__main__":
    main()
