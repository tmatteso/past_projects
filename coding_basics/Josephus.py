#  File: Josephus.py

#  Description:

#  Student Name: Tomas Matteson

#  Student UT EID: tlm3347

#  Partner Name:

#  Partner UT EID: N/A

#  Course Name: CS 313E

#  Unique Number:

#  Date Created: 7/31/18

#  Date Last Modified:8/2/18

# the Link class constructs a Link object with two attributes, data and next
class Link (object):
    def __init__ (self, data, next = None):
        self.data = data
        self.next = next

    def __lt__(self, other):
        if self.data < other.data:
            return True
        else:
            return False

    def __str__(self):
        return str(self.data)

    def __int__(self):
        return int(self.data)

# the Linked List class constructs a singly Linked List compose of single direction links
class LinkedList(object):
    def __init__(self):
        # creates an empty linked list
        self.first = None
    # this is the part of the code that deals with inserting a the first link to the list
    def addFirst(self, data):
        # it creates a new link
        newLink = Link(data)
        # makes it's next address point to self.first (i.e. the old Empty Head)
        self.first = newLink
        newLink.next = self.first
        # makes the new link the list head


    # inserting a link at the end of the list
    def insert(self, data):
        # make a new link
        newLink = Link(data)
        # current is initialized as the list's head
        current = self.first
        if current.next == current:
            current.next = newLink
            newLink.next = self.first

        else:
            while current.next != self.first:
                current = current.next
            # the last element's next pointer goes to the newly created link
            current.next = newLink
            newLink.next = self.first


    # deleting a link in the linked list
    def delete(self, data):
        current = self.first
        previous = self.first
        # if there is no head, return none
        if (current == None):
            return None
        while previous.next != self.first:
                previous = previous.next

        while (current.data != data):
            previous = current
            current = current.next
        if self.first != self.first.next:
            self.first = current.next
        else:
            self.first = None
        # if the current is not the head, then previous next pointer points to the link directly after the deleted current
        previous.next = current.next

    def __str__(self):
        current = self.first
        linked_list = []
        while current.next != self.first:
            linked_list.append(str(current))
            current = current.next
            if current.next == current:
                break

        linked_list.append(str(current))
        str_list =  ', '.join(linked_list)
        return str_list

    # Search in an unordered list, return None if not found
    def find(self, data):
        # initialize current as the lists head
        current = self.first
        # if there is no head, return None
        if (current == None):
            return None
        while current.next != self.first:
            if int(current) == data:
                return current
            current = current.next
        # if you've found the link, return current
        return None

    def deleteAfter(self, start, n):
        current = self.find(start)

        for i in range(1, n):
            current = current.next

        print(str(current.data), end=' ')

        self.delete(current.data)

        return current.next

    def josephus(self, k):
        current = self.first
        Link2 = self.first
        while current.next != current:

            count = 1
            while count != k:
                Link2 = current
                current = current.next
                count += 1

            print(current)
            Link2.next = current.next
            current = Link2.next


        print(current)
        self.first = None
        self.addFirst(current)
def main():
    file = open('josephus.txt', 'r')
    lines = file.read().splitlines()
    n = int(lines[0])
    start = int(lines[1])
    k = int(lines[2])
    ll = LinkedList()
    for i in range(start, n + 1):
        newLink = Link(i)
        if i == 1:
            ll.addFirst(newLink)
        else:
            ll.insert(newLink)
    ll.josephus(k)
main()
