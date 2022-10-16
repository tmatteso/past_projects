#  File: TestLinkedList.py

#  Description:

#  Student Name: Tomas Matteson

#  Student UT EID: tlm3347

#  Partner Name: N/A

#  Partner UT EID: N/A

#  Course Name: CS 313E

#  Unique Number:

#  Date Created: 7/29/18

#  Date Last Modified:7/31/18

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
        newLink.next = self.first
        # makes the new link the list head
        self.first = newLink

    # inserting a link at the end of the list
    def addLast(self, data):
        # make a new link
        newLink = Link(data)
        # current is initialized as the list's head
        current = self.first
        # if the list's head is empty, insert the new element here
        if (current == None):
            self.first = newLink
            return
        # if there's an element after current, then current is equal to this next element
        while (current.next != None):
            current = current.next
        # the last element's next pointer goes to the newly created link
        current.next = newLink


    def delete(self, data):
        # initialize current as the list's head, initialize previous as the list's head
        current = self.first
        previous = self.first
        # if there is no head, return none
        if (current == None):
            return None
        # while the data of current is the data you are looking for
        while (int(current) != data):
            # if it reaches the end of the list and the data is not found, return none
            if (current.next == None):
                return None
            # otherwise reassign previous as the last current
            else:
                previous = current
                current = current.next
        # if current is the head
        if (current == self.first):
            # then the head becomes the second element
            self.first = self.first.next
        # if the current is not the head, then previous next pointer points to the link directly after the deleted current
        else:
            previous.next = current.next
        return current

    def getNumLinks(self):
        current = self.first
        if current == None:
            return 0
        count = 1
        while (current.next != None):
            current = current.next
            count += 1
        return count

    def sortList(self):
        current = self.first
        if current == None:
            return
        sortedlist = []
        while current != None:
            sortedlist.append(current)
            current = current.next

        sortedlist = sorted(sortedlist, key=lambda Link: Link.data, reverse=False)
        newll = LinkedList()
        for link in sortedlist:
            newll.addLast(link)
        return newll

    def addInOrder(self, data):
        current = self.first
        newLink = Link(data)
        if current is None:
            self.first = newLink
            return
        if current.data > data:
            newLink.next = current
            self.first = newLink
            return

        while current.next is not None:
            if current.next.data > data:
                break
            current = current.next
        newLink.next = current.next
        current.next = newLink
        return

    def __str__(self):
        if self.first is None:
            return
        current = self.first
        linked_list = []
        while current is not None:
            linked_list.append(str(current))
            current = current.next
        str_list =  ', '.join(linked_list)
        return str_list

    # Search in an unordered list, return None if not found
    def findUnordered(self, data):
        # initialize current as the lists head
        current = self.first
        # if there is no head, return None
        if (current == None):
            return None
        while current != None:
            if int(current) == data:
                return current
            current = current.next
        # if you've found the link, return current
        return None

    # Search in an ordered list, return None if not found
    def findOrdered(self, data):
        sortedlist = self.sortList()
        current = sortedlist.first
        # if there is no head, return None
        if (current == None):
            return None
        # cycle through the linked list, if the link does not contain the data you are looking ffor, current becomes the next link
        while (int(current) != data):
            # if the end of the list is reached and the data is not found, return None
            if (current.next == None):
                return None
            else:
                current = current.next
        # if you've found the link, return current
        return current


    def copyList(self):
        current = self.first
        if current == None:
            return
        copylist = []
        while current != None:
            copylist.append(current)
            current = current.next

        newll = LinkedList()
        for link in copylist:
            newll.addLast(link)
        return newll

    def reverseList(self):
        current = self.first
        if current == None:
            return
        copylist = []
        while current != None:
            copylist.append(current)
            current = current.next
        reversedlist = list(reversed(copylist))

        newll = LinkedList()
        for link in reversedlist:
            newll.addLast(link)
        return newll

    # Return True if a list is sorted in ascending order or False otherwise
    def isSorted(self):
        current = self.first
        if current == None:
            return
        unsortedlist = []
        while current != None:
            unsortedlist.append(current)
            current = current.next

        sortedlist = self.sortList()
        current = sortedlist.first
        if current == None:
            return
        sortedlist = []
        while current != None:
            sortedlist.append(current)
            current = current.next
        count = 0
        for i in range(len(sortedlist)):
            if int(unsortedlist[i]) == int(sortedlist[i]):
                count += 1
        if count == self.getNumLinks():
            return True
        return False

    # Return True if a list is empty or False otherwise
    def isEmpty(self):
        if self.getNumLinks() == 0:
            return True
        return False

    # Test if two lists are equal, item by item and return True
    def isEqual(self, b):
        selflist = []
        current = self.first
        if current == None:
            return
        while current != None:
            selflist.append(current)
            current = current.next

        current = b.first
        blist = []
        if current == None:
            return
        while current != None:
            blist.append(current)
            current = current.next

        count = 0
        for i in range(len(blist)):
            if int(selflist[i]) == int(blist[i]):
                count += 1
        if count == self.getNumLinks():
            return True
        return False

    # Merge two sorted lists and return new list in ascending order
    def mergeList(self, b):
        sortedlist = self.sortList()
        selflist = []
        current = sortedlist.first
        if current == None:
            return
        while current != None:
            selflist.append(current)
            current = current.next

        bsortedlist = b.sortList()
        current = bsortedlist.first
        blist = []
        if current == None:
            return
        while current != None:
            blist.append(current)
            current = current.next
        for i in range(len(blist)):
            selflist.append(blist[i])
        mergell = LinkedList()
        for link in selflist:
            mergell.addLast(link)
        sortedmerge = mergell.sortList()
        return sortedmerge

    # Return a new list, keeping only the first occurence of an element
    # and removing all duplicates. Do not change the order of the elements.
    def removeDuplicates(self):
        selflist = []
        current = self.first
        if current == None:
            return
        while current != None:
            selflist.append(int(current))
            current = current.next
        from collections import OrderedDict
        copylist = list(OrderedDict.fromkeys(selflist))
        no_dup_ll = LinkedList()
        for link in copylist:
            no_dup_ll.addLast(link)
        return no_dup_ll


def main():
    ll = LinkedList()
    a = Link(1)
    b = Link(3)
    c = Link(2)
    d = Link(2)
    e = Link(3)
    f = Link(5)
    g = Link(4)
    h = Link(5)
    i = Link(5)
    j = Link(0)
    ll.addFirst(a)
    ll.addLast(b)
    ll.addLast(c)
    ll.addLast(d)
    ll.addLast(e)
    ll.addLast(f)
    ll.addLast(g)
    ll.addLast(h)
    ll.addLast(i)
    ll.addInOrder(j)
    rr = LinkedList()
    b = Link(5)
    c = Link(3)
    d = Link(6)
    rr.addFirst(c)
    rr.addLast(d)
    rr.addInOrder(b)
    ok = LinkedList()
    b = Link(5)
    c = Link(3)
    d = Link(6)
    ok.addFirst(c)
    ok.addLast(d)
    ok.addInOrder(b)
    print(ll)
    print(rr)
    print(ll.getNumLinks())
    print(ll.findUnordered(0))
    print(ll.findUnordered(7))
    print(ll.findOrdered(0))
    print(ll.findOrdered(7))
    print(ll.delete(0))
    print(ll.delete(7))
    print(ll.copyList())
    print(ll.reverseList())
    print(ll.sortList())
    print(ll.isSorted())
    print(rr.isSorted())
    print(ll.isEmpty())
    print(ll.mergeList(rr))
    print(ll.isEqual(rr))
    print(rr.isEqual(ok))
    print(ll.removeDuplicates())


if __name__ == "__main__":
    main()

