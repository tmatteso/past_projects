""""
  File: BasicGeometry.py

  Description:

  Student's Name: Tomas Matteson

  Student's UT EID: tlm3347

  Partner's Name: N/A

  Partner's UT EID: N/A

  Course Name: CS 313E

  Unique Number: 86325

  Date Created: 6/19/18

  Date Last Modified: 6/21/18"""

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
    def __init__(self, num_players, num_cards):
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
            sorted_hand = sorted(self.all_hands[i], reverse=True)
            self.all_hands[i] = sorted_hand
            hand = ''
            for card in sorted_hand:
                hand = hand + str(card) + ' '
            print('Player ' + str(i + 1) + ": " + hand)

        # determine the each type of hand and print
        points_hand = []  # create list to store points for each hand

        for i in range(len(self.all_hands)):
            if self.is_royal(self.all_hands[i]):
                h = 10
                points_hand.append(self.score_function(self.all_hands[i], h))
                print("Player %i: Royal Flush" % (i + 1))
            elif self.is_straight_flush(self.all_hands[i]):
                h = 9
                points_hand.append(self.score_function(self.all_hands[i], h))
                print("Player %i: Straight Flush" % (i + 1))
            elif self.is_four_kind(self.all_hands[i]):
                h = 8
                hand = self.all_hands[i]
                if hand[1] == hand[2] == hand[3] == hand[4]:
                    hand1 = []
                    hand1.append(hand[1:5])
                    hand1.append(hand[0])
                    points_hand.append(self.score_function(hand1, h))
                else:
                    points_hand.append(self.score_function(self.all_hands[i], h))
                print("Player %i: Four of a Kind" % (i + 1))
            elif self.is_full_house(self.all_hands[i]):
                h = 7
                hand = self.all_hands[i]

                points_hand.append(self.score_function(self.all_hands[i], h))
                print("Player %i: Full House" % (i + 1))
            elif self.is_flush(self.all_hands[i]):
                h = 6
                points_hand.append(self.score_function(self.all_hands[i], h))
                print("Player %i: Flush" % (i + 1))
            elif self.is_straight(self.all_hands[i]):
                h = 5
                points_hand.append(self.score_function(self.all_hands[i], h))
                print("Player %i: Straight" % (i + 1))
            elif self.is_three_kind(self.all_hands[i]):
                h = 4
                hand = self.all_hands[i]
                hand1 = []
                if hand[1] == hand[2] == hand[3]:

                    hand1.append(hand[1:4])
                    hand1.append(hand[0])
                    hand1.append(hand[4])
                    points_hand.append(self.score_function(hand1, h))
                elif hand[2] == hand[3] == hand[4]:

                    hand1.append(hand[2:5])
                    hand1.append(hand[0:2])
                    points_hand.append(self.score_function(hand1, h))

                else:
                    points_hand.append(self.score_function(self.all_hands[i], h))
                print("Player %i: Three of a Kind" % (i + 1))
            elif self.is_two_pair(self.all_hands[i]):
                h = 3
                hand = self.all_hands[i]
                hand1 = []
                if hand[0] == hand[1] and hand[2] == hand[3] and hand[0] != hand[3]:
                    if hand[0] > hand[2]:

                        hand1.append(hand[0:2])
                        hand1.append(hand[2:4])
                        hand1.append(hand[4])
                        points_hand.append(self.score_function(hand1, h))

                    if hand[0] < hand[2]:
                        hand1.append(hand[2])
                        hand1.append(hand[3])
                        hand1.append(hand[0:2])
                        hand1.append(hand[4])
                        points_hand.append(self.score_function(hand1, h))

                elif hand[0] == hand[1] and hand[3] == hand[4] and hand[0] != hand[3]:
                    if hand[0] > hand[3]:
                        hand1.append(hand[0:2])
                        hand1.append(hand[3:5])
                        hand1.append(hand[2])
                        points_hand.append(self.score_function(hand1, h))

                    elif hand[0] < hand[3]:
                        hand1.append(hand[3:5])
                        hand1.append(hand[0:2])
                        hand1.append(hand[2])
                        points_hand.append(self.score_function(hand1, h))


                elif hand[1] == hand[2] and hand[3] == hand[4] and hand[1] != hand[3]:
                    if hand[1] > hand[3]:
                        hand1.append(hand[1:3])
                        hand1.append(hand[3:5])
                        hand1.append(hand[0])
                        points_hand.append(self.score_function(hand1, h))

                    elif hand[1] < hand[3]:
                        hand1.append(hand[3:5])
                        hand1.append(hand[1:3])
                        hand1.append(hand[0])
                        points_hand.append(self.score_function(hand1, h))

                else:
                    points_hand.append(self.score_function(self.all_hands[i], h))
                print("Player %i: Two Pair" % (i + 1))
            elif self.is_one_pair(self.all_hands[i]):
                h = 2
                hand = (self.all_hands[i])
                hand1 = []
                if hand[1] == hand[2]:
                    hand1.append(hand[1:3])
                    hand1.append(hand[3:5])
                    hand1.append(hand[0])
                    hand1.append(hand[3:5])
                    hand1.append(hand[3:5])
                    points_hand.append(self.score_function(hand1, h))

                elif hand[2] == hand[3]:
                    hand1.append(hand[2])
                    hand1.append(hand[3])
                    hand1.append(hand[0])
                    hand1.append(hand[1])
                    hand1.append(hand[4])
                    points_hand.append(self.score_function(hand1, h))

                elif hand[3] == hand[4]:
                    hand1.append(hand[3])
                    hand1.append(hand[4])
                    hand1.append(hand[0])
                    hand1.append(hand[1])
                    hand1.append(hand[2])
                    points_hand.append(self.score_function(hand1, h))

                else:
                    points_hand.append(self.score_function(self.all_hands[i], h))
                print("Player %i: One Pair" % (i + 1))
            elif self.is_high_card(self.all_hands[i]):
                h = 1
                points_hand.append(self.score_function(self.all_hands[i], h))
                print("Player %i: High Card" % (i + 1))
        # max in points_hand
        winner_score = max(points_hand)
        #sorted_scores = sorted(points_hand, reverse=True)

        """if sorted_scores[0] == sorted_scores[1] or sorted_scores[0] == sorted_scores[1] == sorted_scores[2] or sorted_scores[0] == sorted_scores[1] == sorted_scores[2] == sorted_scores[3] or sorted_scores[0] == sorted_scores[1] == sorted_scores[2] == sorted_scores[3] == sorted_scores[4]:

            print("Player %i ties" % )
            print("Player %i ties" % (i + 1))"""

        winner = points_hand.index(winner_score) + 1
        print("Player %i wins" % winner)



        # index +1 of the max is the player number
        # print the players number and wins

        # determine winner and print

    def score_function(self, hand, h):
        #pass
        #print((hand[0]).rank)
        total_points = (h * 14 ** 5) + (hand[0]).rank * 14 ** 4 + (hand[1]).rank * 14 ** 3 + (hand[2]).rank * 14 ** 2 + (hand[3]).rank * 14 + (hand[4]).rank

        return total_points

    # determine if a hand is a royal flush
    # takes as argument a list of 5 card objects and returns a boolean
    def is_royal(self, hand):
        same_suit = True
        for i in range(len(hand) - 1):
            same_suit = same_suit and (hand[i].suit == hand[i + 1].suit)

        if (not same_suit):
            return False

        rank_order = True
        for i in range(len(hand)):
            rank_order = rank_order and (hand[i].rank == 14 - i)

        if (not rank_order):
            return False
        return (same_suit and rank_order)

    def is_straight_flush(self, hand):
        same_suit = True
        for i in range(len(hand) - 1):
            same_suit = same_suit and (hand[i].suit == hand[i + 1].suit)

        if (not same_suit):
            return False

        rank_order = True
        for i in range(len(hand)):
            rank_order = rank_order

    def is_four_kind(self, hand):

        if hand[0] == hand[1] == hand[2] == hand[3]:

            return True
        elif hand[1] == hand[2] == hand[3] == hand[4]:
            #hand1 = hand
            hand.append(hand[1:5])
            hand.append(hand[0])

            return True
        else:
            return False

    def is_full_house(self, hand):
        if (hand[0] == hand[1] == hand[2]) and (hand[0] != hand[3]) and (hand[3] == hand[4]):
            return True
        elif (hand[2] == hand[3] == hand[4]) and (hand[0] != hand[3]) and (hand[0] == hand[1]):
            #hand1 = hand
            #hand.remove(hand[0:5])
            hand.append(hand[2:5])
            hand.append(hand[0:2])
            #hand.remove(hand[0:5])
            return True
        else:
            return False

    def is_flush(self, hand):
        same_suit = True
        for i in range(len(hand) - 1):
            same_suit = same_suit and (hand[i].suit == hand[i + 1].suit)
        if (not same_suit):
            return False

    def is_straight(self, hand):
        rank_order = True
        for i in range(len(hand)):
            rank_order = rank_order

        if (not rank_order):
            return False

    def is_three_kind(self, hand):

        if hand[0] == hand[1] == hand[2]:

            return True
        elif hand[1] == hand[2] == hand[3]:
            #hand1 = hand
            hand.append(hand[1:4])
            hand.append(hand[0])
            hand.append(hand[4])

            return True
        elif hand[2] == hand[3] == hand[4]:
            #hand1 = hand
            hand.append(hand[2:5])
            hand.append(hand[0:2])
            #hand.remove(hand[0:5])
            return True
        else:
            return False

    def is_two_pair(self, hand):

        if hand[0] == hand[1] and hand[2] == hand[3] and hand[0] != hand[3]:
            if hand[0] > hand[2]:
                #hand1 = hand
                hand.append(hand[0:2])
                hand.append(hand[2:4])
                hand.append(hand[4])
                #hand.remove(hand[0:5])
                return True
            if hand[0] < hand[2]:
                hand.append(hand[2])
                hand.append(hand[3])
                hand.append(hand[0:2])
                hand.append(hand[4])
                #hand.remove(hand[0:5])
                return True
        elif hand[0] == hand[1] and hand[3] == hand[4] and hand[0] != hand[3]:
            if hand[0] > hand[3]:
                hand.append(hand[0:2])
                hand.append(hand[3:5])
                hand.append(hand[2])
                #hand.remove(hand[0:5])
                return True
            elif hand[0] < hand[3]:
                hand.append(hand[3:5])
                hand.append(hand[0:2])
                hand.append(hand[2])
                #hand.remove(hand[0:5])
                return True

        elif hand[1] == hand[2] and hand[3] == hand[4] and hand[1] != hand[3]:
            if hand[1] > hand[3]:
                hand.append(hand[1:3])
                hand.append(hand[3:5])
                hand.append(hand[0])
                #hand.remove(hand[0:5])
                return True
            elif hand[1] < hand[3]:
                hand.append(hand[3:5])
                hand.append(hand[1:3])
                hand.append(hand[0])
                #hand.remove(hand[0:5])
                return True

        else:
            return False

    # determine if a hand is one pair
    # takes as argument a list of 5 card objects and returns a boolean
    def is_one_pair(self, hand):
        if hand[0] == hand[1]:
            pass
        elif hand[1] == hand[2]:
            hand.append(hand[1:3])
            hand.append(hand[0])
            hand.append(hand[3:5])
            #hand.remove(hand[0:5])
        elif hand[2] == hand[3]:
            hand.append(hand[2:4])
            hand.append(hand[0:2])
            hand.append(hand[4])
            #hand.remove(hand[0:5])
        elif hand[3] == hand[4]:
            hand.append(hand[3:5])
            hand.append(hand[0:3])
            #hand.remove(hand[0:5])
        for i in range(len(hand) - 1):
            if (hand[i].rank == hand[i + 1].rank):
                return True
        return False

    def is_high_card(self, hand):
        high_card = hand[0]

        return high_card


def main():
    # prompt user to enter the number of players
    num_players = int(input('Enter number of players: '))
    while ((num_players < 2) or (num_players > 6)):
        num_players = int(input('Enter number of players: '))

    # create the Poker object
    game = Poker(num_players, 5)

    # play the game (poker)
    game.play()


# do not remove this line above main()
if __name__ == '__main__':
    main()