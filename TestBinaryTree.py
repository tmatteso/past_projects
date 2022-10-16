#  File: TestBinaryTree.py

#  Description:

#  Student Name: Tomas Matteson

#  Student UT EID: tlm3347

#  Partner Name: Uche Okonma

#  Partner UT EID: uo389

#  Course Name: CS 313E

#  Unique Number:

#  Date Created: 8/7/18

#  Date Last Modified: 8/9/18

class Node(object):
    def __init__(self, data):
        self.data = data
        self.lChild = None
        self.rChild = None

    def __str__(self):
        return str(self.data)


class Tree(object):
    def __init__(self):
        self.root = None

    # Search for a node with the key
    def search(self, key):
        current = self.root
        while ((current != None) and (current.data != key)):
            if (key < current.data):
                current = current.lChild
            else:
                current = current.rChild
        return current

    # Insert a node in the tree
    def insert(self, val):
        newNode = Node(val)

        if (self.root == None):
            self.root = newNode
        else:
            current = self.root
            parent = self.root
            while (current != None):
                parent = current
                if (val < current.data):
                    current = current.lChild
                else:
                    current = current.rChild

            if (val < parent.data):
                parent.lChild = newNode
            else:
                parent.rChild = newNode

    # Find the node with the smallest value
    def minimum(self):
        current = self.root
        parent = current
        while (current != None):
            parent = current
            current = current.lChild
        return parent

    # Find the node with the largest value
    def maximum(self):
        current = self.root
        parent = current
        while (current != None):
            parent = current
            current = current.rChild
        return parent

    # Delete a node with a given key
    def delete(self, key):
        deleteNode = self.root
        parent = self.root
        isLeft = False

        # If empty tree
        if (deleteNode == None):
            return False

        # Find the delete node
        while ((deleteNode != None) and (deleteNode.data != key)):
            parent = deleteNode
            if (key < deleteNode.data):
                deleteNode = deleteNode.lChild
                isLeft = True
            else:
                deleteNode = deleteNode.rChild
                isLeft = False

        # If node not found
        if (deleteNode == None):
            return False

        # Delete node is a leaf node
        if ((deleteNode.lChild == None) and (deleteNode.rChild == None)):
            if (deleteNode == self.root):
                self.root = None
            elif (isLeft):
                parent.lChild = None
            else:
                parent.rChild = None

        # Delete node is a node with only left child
        elif (deleteNode.rChild == None):
            if (deleteNode == self.root):
                self.root = deleteNode.lChild
            elif (isLeft):
                parent.lChild = deleteNode.lChild
            else:
                parent.rChild = deleteNode.lChild

        # Delete node is a node with only right child
        elif (deleteNode.lChild == None):
            if (deleteNode == self.root):
                self.root = deleteNode.rChild
            elif (isLeft):
                parent.lChild = deleteNode.rChild
            else:
                parent.rChild = deleteNode.rChild

        # Delete node is a node with both left and right child
        else:
            # Find delete node's successor and successor's parent nodes
            successor = deleteNode.rChild
            successorParent = deleteNode

            while (successor.lChild != None):
                successorParent = successor
                successor = successor.lChild

            # Successor node right child of delete node
            if (deleteNode == self.root):
                self.root = successor
            elif (isLeft):
                parent.lChild = successor
            else:
                parent.rChild = successor

            # Connect delete node's left child to be successor's left child
            successor.lChild = deleteNode.lChild

            # Successor node left descendant of delete node
            if (successor != deleteNode.rChild):
                successorParent.lChild = successor.rChild
                successor.rChild = deleteNode.rChild

        return True

    # Returns true if two binary trees are similar (returns true if the nodes have the same key values
    # and are arranged in the same order and false otherwise)

    def is_similar(self, pNode):
        return self.sim_helper(self.root, pNode.root)

    def sim_helper(self, node, pNode):
        # 1. Both empty
        if node is None and pNode is None:
            return True

        # 2. Both non-empty -> Compare them
        if node is not None and pNode is not None:
            return ((node.data == pNode.data) and
                    self.sim_helper(node.lChild, pNode.lChild) and
                    self.sim_helper(node.rChild, pNode.rChild))

        # 3. one empty, one not -- false
        return False

    # Prints out all nodes at the given level ( takes as input the level and prints out all the nodes at that level.
    # If that level does not exist for that binary search tree it prints nothing. Use the convention that
    # the root is at level 1)
    def print_level(self, level):
        self.printGivenLevel(self.root, level)

    def printGivenLevel(self, node, level):
        if node == None:
            return
        if level == 1:
            print(node.data)
        elif level > 1:
                self.printGivenLevel(node.lChild, level - 1)
                self.printGivenLevel(node.rChild, level - 1)

    # Returns the height of the tree ( returns the height of a binary search tree. Recall that the height of a tree is
    # the longest path length from the root to a leaf)

    def get_height(self):
        return self.actual_height(self.root)

    def actual_height(self, bst_node):
        if bst_node is None:
            return 0
        else:
            return 1 + max(self.actual_height(bst_node.lChild), self.actual_height(bst_node.rChild))


    # Returns the number of nodes in the left subtree and
    # the number of nodes in the right subtree and the root (returns the number of nodes in the left subtree from the
    # root and the number of nodes in the right subtree from the root and the root itself. This function will be useful
    # to determine if the tree is balanced)
    def num_nodes(self):
        return self.actual_num(self.root)


    def actual_num (self, node):
        if node is None:
            return 0
        if (node.lChild is None and node.rChild is None):
            return 1
        else:
            return 1 + self.actual_num(node.lChild) + self.actual_num(node.rChild)



# In the class TestBinaryTree you will create several trees and show convincingly that your methods are working.
# For example you can create a tree by inserting the following integers
# in this order: 50, 30, 70, 10, 40, 60, 80, 7, 25, 38, 47, 58, 65, 77, 96.
def main():
    # Create three trees - two are the same and the third is different
    z = Tree()
    a = Tree()
    b = Tree()
    z.insert(12)
    z.insert(33)
    z.insert(45)
    z.insert(6)
    z.insert(28)
    a.insert(12)
    a.insert(33)
    a.insert(45)
    a.insert(6)
    a.insert(28)
    b.insert(28)
    b.insert(45)
    b.insert(12)
    # Test your method is_similar()
    print("Is Tree Z similar to Tree A?")
    print(z.is_similar(a))
    print("Is Tree Z similar to Tree B?")
    print(z.is_similar(b))
    # Print the various levels of two of the trees that are different
    print("Tree Z level 1: ")
    z.print_level(1)
    print("Tree Z level 2: ")
    z.print_level(2)
    print("Tree Z level 3: ")
    z.print_level(3)
    print("Tree B level 1: ")
    b.print_level(1)
    print("Tree B level 2: ")
    b.print_level(2)
    # Get the height of the two trees that are different
    print("The height of Tree Z")
    print(z.get_height())
    print("The height of Tree B")
    print(b.get_height())
    # Get the total number of nodes in a binary search tree
    print("The number of nodes in Tree Z")
    print(z.num_nodes())
    print("The number of nodes in Tree A")
    print(a.num_nodes())
    print("The number of nodes in Tree B")
    print(b.num_nodes())

main()



