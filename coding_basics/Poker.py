'''
  File: Poker.py

  Description: This program simulates a Poker game

  Student's Name: Tomas Matteson

  Student's UT EID: tlm3347

  Partner's Name: Hamza Sait

  Partner's UT EID: hs26386

  Course Name: CS 313E

  Unique Number: 86325

  Date Created: June 22 2018

  Date Last Modified: June 25 2018
'''

import random


# this class represents a standard playing card
class Card(object):
    RANKS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

    SUITS = ('C', 'D', 'H', 'S')

    # initialize a card object with given rank (int) and suit (string)
    def __init__(self, rank=12, suit='S'):
        if (rank in Card.RANKS):
            self.rank = rank
        else:
            self.rank = 12

        if (suit in Card.SUITS):
            self.suit = suit
        else:
            self.suit = 'S'

    # string representation takes no arguments and returns a string
    def __str__(self):
        if (self.rank == 14):
            rank = 'A'
        elif (self.rank == 13):
            rank = 'K'
        elif (self.rank == 12):
            rank = 'Q'
        elif (self.rank == 11):
            rank = 'J'
        else:
            rank = str(self.rank)
        return rank + self.suit

    # following functions take a card object as argument and return a boolean
    def __eq__(self, other):
        return (self.rank == other.rank)

    def __ne__(self, other):
        return (self.rank != other.rank)

    def __lt__(self, other):
        return (self.rank < other.rank)

    def __le__(self, other):
        return (self.rank <= other.rank)

    def __gt__(self, other):
        return (self.rank > other.rank)

    def __ge__(self, other):
        return (self.rank >= other.rank)


# class representing a deck of cards
class Deck(object):

    # initializes a deck object
    # self.deck is a list of cards in standard sorted order
    def __init__(self):
        self.deck = []
        for suit in Card.SUITS:
            for rank in Card.RANKS:
                card = Card(rank, suit)
                self.deck.append(card)

    # takes no arguments and does not return any value
    # randomly shuffles the cards in the deck
    def shuffle(self):
        random.shuffle(self.deck)

    # takes no arguments
    # returns a card object at the front of self.deck
    # self.deck has one less card after the deal function is called
    def deal(self):
        if (len(self.deck) == 0):
            return None
        else:
            return self.deck.pop(0)


# class representing a game of poker
class Poker(object):

    # takes two arguments num_players (number of players) and
    # num_cards (number of cards in a hand)
    # each player is represented by a list of num_cards card objects
    # self.all_hands is a list of lists of num_cards card objects
    def __init__(self, num_players = 5, num_cards = 5):
        self.deck = Deck()
        self.deck.shuffle()
        self.all_hands = []
        self.numCards_in_Hand = num_cards

        for i in range(num_players):
            hand = []
            for j in range(self.numCards_in_Hand):
                hand.append(self.deck.deal())
            self.all_hands.append(hand)

    # simulates the play of the game
    # sorts each hand and assigns points to each hand
    # returns a list of players in decreasing order of points
    # example: if there are three players and the points are 20, 16, 18
    # return [1, 3, 2]
    def play(self):
        # sort the hands of each player and print
        for i in range(len(self.all_hands)):
            sorted_hand = sorted(self.all_hands[i],reverse=True)
            self.all_hands[i] = sorted_hand
            hand = ''
            for card in sorted_hand:
                hand = hand + str(card) + ' '
            print('Player ' + str(i + 1) + " : " + hand)
            self.all_hands[i] = sorted(self.all_hands[i])

        #placeholder Poker variable to fit the requirement to have the hand checks within the Poker class
        placeHolder = Poker()

        #this is keeps track of all the players points
        points_hand = []
        name_hand = []

        #this for loop goes through each hand determines what the hand is and adds points
        for x in range (len(self.all_hands)):

            #checks if the hand is royal and adds points
            if (placeHolder.is_royal(self.all_hands[x])):
                points_hand.append(placeHolder.points(10,self.all_hands[x][4],self.all_hands[x][3],self.all_hands[x][2],self.all_hands[x][1],self.all_hands[x][0]))
                name_hand.append("Royal Flush")

            #checks if the hand is a straight flush and adds points
            elif (placeHolder.is_straight_flush(self.all_hands[x])):
                points_hand.append(placeHolder.points(9,self.all_hands[x][4],self.all_hands[x][3],self.all_hands[x][2],self.all_hands[x][1],self.all_hands[x][0]))
                name_hand.append("Straight Flush")

            #checks if the hand is four of a kind and adds points
            elif (placeHolder.is_four_kind(self.all_hands[x])):
                if(self.all_hands[x][0] == self.all_hands[x][3]):
                    points_hand.append(placeHolder.points(8,self.all_hands[x][0],self.all_hands[x][1],self.all_hands[x][2],self.all_hands[x][3],self.all_hands[x][4]))
                else:
                    points_hand.append(placeHolder.points(8,self.all_hands[x][1],self.all_hands[x][2],self.all_hands[x][3],self.all_hands[x][4],self.all_hands[x][0]))
                name_hand.append("Four of a Kind")

            #checks if the hand is a full house and adds points
            elif (placeHolder.is_full_house(self.all_hands[x])):
                if(self.all_hands[x][0] == self.all_hands[x][2]):
                    points_hand.append(placeHolder.points(7,self.all_hands[x][0],self.all_hands[x][1],self.all_hands[x][2],self.all_hands[x][3],self.all_hands[x][4]))
                else:
                    points_hand.append(placeHolder.points(7, self.all_hands[x][2], self.all_hands[x][3], self.all_hands[x][4],self.all_hands[x][1], self.all_hands[x][0]))
                name_hand.append("Full House")

            #checks if the hand is a flush and adds points
            elif (placeHolder.is_flush(self.all_hands[x])):
                    points_hand.append(placeHolder.points(6,self.all_hands[x][4],self.all_hands[x][3],self.all_hands[x][2],self.all_hands[x][1],self.all_hands[x][0]))
                    name_hand.append("Flush")

            #checks if the hand is a straight and adds points
            elif (placeHolder.is_straight(self.all_hands[x])):
                    points_hand.append(placeHolder.points(5, self.all_hands[x][4], self.all_hands[x][3], self.all_hands[x][2],self.all_hands[x][1], self.all_hands[x][0]))
                    name_hand.append("Straight")

            #checks if the hand is three of a kind and adds points
            elif (placeHolder.is_three_kind(self.all_hands[x])):
                if(self.all_hands[x][0] == self.all_hands[x][2]):
                    points_hand.append(placeHolder.points(4,self.all_hands[x][0],self.all_hands[x][1],self.all_hands[x][2],self.all_hands[x][4],self.all_hands[x][3]))
                elif(self.all_hands[x][1] == self.all_hands[x][3]):
                    points_hand.append(placeHolder.points(4, self.all_hands[x][1], self.all_hands[x][2], self.all_hands[x][3],self.all_hands[x][4], self.all_hands[x][0]))
                else:
                    points_hand.append(placeHolder.points(4, self.all_hands[x][2], self.all_hands[x][3], self.all_hands[x][4],self.all_hands[x][1], self.all_hands[x][0]))
                name_hand.append("Three of a Kind")

            #checks if the hand is a two pair and adds points
            elif (placeHolder.is_two_pair(self.all_hands[x])):
                if((self.all_hands[x][0] == self.all_hands[x][1]) and (self.all_hands[x][2] == self.all_hands[x][3])):
                    points_hand.append(placeHolder.points(3,self.all_hands[x][2],self.all_hands[x][3],self.all_hands[x][0],self.all_hands[x][1],self.all_hands[x][4]))
                elif((self.all_hands[x][1] == self.all_hands[x][2]) and (self.all_hands[x][3] == self.all_hands[x][4])):
                    points_hand.append(placeHolder.points(3, self.all_hands[x][3], self.all_hands[x][4], self.all_hands[x][1],self.all_hands[x][2], self.all_hands[x][0]))
                else:
                    points_hand.append(placeHolder.points(3, self.all_hands[x][3], self.all_hands[x][4], self.all_hands[x][0],self.all_hands[x][1], self.all_hands[x][2]))
                name_hand.append("Two Pair")

            #checks if the hand is a one pair and adds points
            elif (placeHolder.is_one_pair(self.all_hands[x])):
                if((self.all_hands[x][0] == self.all_hands[x][1])):
                    points_hand.append(placeHolder.points(2,self.all_hands[x][0],self.all_hands[x][1],self.all_hands[x][4],self.all_hands[x][3],self.all_hands[x][2]))
                elif((self.all_hands[x][1] == self.all_hands[x][2])):
                    points_hand.append(placeHolder.points(2, self.all_hands[x][1], self.all_hands[x][2], self.all_hands[x][4],self.all_hands[x][3], self.all_hands[x][0]))
                elif((self.all_hands[x][2] == self.all_hands[x][3])):
                    points_hand.append(placeHolder.points(2, self.all_hands[x][2], self.all_hands[x][3], self.all_hands[x][4],self.all_hands[x][1], self.all_hands[x][0]))
                else:
                    points_hand.append(placeHolder.points(2, self.all_hands[x][4], self.all_hands[x][3], self.all_hands[x][2],self.all_hands[x][1], self.all_hands[x][0]))
                name_hand.append("One Pair")

            else:
                points_hand.append(placeHolder.points(1, self.all_hands[x][4], self.all_hands[x][3], self.all_hands[x][2],self.all_hands[x][1], self.all_hands[x][0]))
                name_hand.append("High Card")


        #this section prints out the players hands
        print("")
        for x in range (len(name_hand)):
            print ("Player " + str((x+1)) + ": "+ name_hand[x])

        #this piece of code combines the players number, hand value, and point value into a single list
        #the code then determines a winner or determines which players have tied
        print("")
        x = (list(zip(points_hand,name_hand)))
        for y in range (len(x)):
            x[y] = list(x[y])
            x[y].append(y+1)
        x = ((sorted(x, reverse = True)))

        if (x[0][1] != x[1][1]):
            print ("Player " + str(x[0][2]) + " wins.")
        else:
            print("Player " + str(x[0][2]) + " ties.")
            for z in range (len(x)-1):
                if ((x[z][1]) == (x[z+1][1])):
                    print("Player " + str(x[z+1][2]) + " ties.")
                else:
                    break


    # determines if a hand is a royal flush
    # takes as argument a list of 5 card objects and returns a boolean
    def is_royal(self, hand):

        good_suit = 0
        good_rank = 0
        for x in range (len(hand)-1):
            if (hand[x].suit == hand[x+1].suit):
                good_suit += 1
            if (hand[0].rank == 10):
                if (hand[x].rank == hand[x+1].rank+1):
                    good_rank +=1

        return ((good_rank == 4) and (good_suit == 4))

    #determines if a hand is a straight flush
    #take as argument a list of 5 cards objects and returns boolean
    def is_straight_flush (self, hand):
        straightFlush = 0
        for x in range(len(hand)-1):
            if (hand[x].suit == hand[x+1].suit):
                straightFlush += 1
            if (hand[x].rank == hand[x+1].rank+1):
                straightFlush += 1
        return (straightFlush == 8)

    # determines if a hand is four of a kind
    # take as argument a list of 5 cards objects and returns boolean
    def is_four_kind (self, hand):

        forKind = 0
        if (hand[2].rank == hand[0].rank):
            for x in range (len(hand)):
               if (hand[0] == hand[x]):
                   forKind += 1
        elif(hand[2].rank == hand[4].rank):
            for x in range (len(hand)):
               if (hand[4] == hand[x]):
                   forKind += 1
        return (forKind == 4)

    # determines if a hand is a full house
    # take as argument a list of 5 cards objects and returns boolean
    def is_full_house(self, hand):

        if (hand[0] == hand[1]) and ((hand[2] == hand[3]) and (hand[3] == hand[4])):
            return True
        elif (hand[3] == hand[4]) and ((hand[0] == hand[1]) and (hand[1] == hand[3])):
            return True
        else:
            return False

    # determines if a hand is a flush
    # take as argument a list of 5 cards objects and returns boolean
    def is_flush(self, hand):
        isFlush = 0
        for x in range(len(hand)-1):
            if (hand[x].suit == hand[x+1].suit):
                isFlush += 1

        return (isFlush == 4)

    # determines if a hand is a straight
    # take as argument a list of 5 cards objects and returns boolean
    def is_straight(self, hand):
        isStraight = 0
        for x in range(len(hand) - 1):
            if (hand[x] == hand[x + 1]):
                isStraight += 1

        return (isStraight == 4)

    # determines if a hand is three of a kind
    # take as argument a list of 5 cards objects and returns boolean
    def is_three_kind (self, hand):
        if ((hand[0] == hand[1]) and (hand[1] == hand[2])):
            return True
        elif ((hand[4] == hand[3]) and (hand[3] == hand[2])):
            return True
        elif ((hand[1]) == hand[2]) and (hand[2] == hand[3]):
            return True
        else:
            return False

    # determines if a hand is a two pair
    # take as argument a list of 5 cards objects and returns boolean
    def is_two_pair (self, hand):
        if ((hand[0] == hand [1]) and (hand[2] == hand[3])):
            return True
        elif((hand[0] == hand[1]) and (hand[3] == hand[4])):
            return True
        elif ((hand[1] == hand[2]) and (hand[3] == hand[4])):
            return True
        else:
            return False

    # determine if a hand is one pair
    # takes as argument a list of 5 card objects and returns a boolean
    def is_one_pair(self, hand):
        for i in range(len(hand) - 1):
            if (hand[i].rank == hand[i + 1].rank):
                return True
        return False


    #automatically returns True in the case where there no other option exists
    def is_high_card (self, hand):
        return True

    #this section calculates the points for each hand
    def points (self,h,c1,c2,c3,c4,c5):

        c1 = c1.rank
        c2 = c2.rank
        c3 = c3.rank
        c4 = c4.rank
        c5 = c5.rank
        total_points = h * 14 ** 5 + c1 * 14 ** 4 + c2 * 14 ** 3 + c3 * 14 ** 2 + c4 * 14 + c5
        return total_points

def main():
    # prompt user to enter the number of players
    num_players = int(input('Enter number of players: '))
    while ((num_players < 2) or (num_players > 6)):
        num_players = int(input('Enter number of players: '))

    # create the Poker object
    game = Poker(num_players)


    # play the game (poker)
    game.play()

if __name__ == '__main__':
    main()