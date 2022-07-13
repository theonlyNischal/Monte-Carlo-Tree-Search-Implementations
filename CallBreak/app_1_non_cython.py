#!python
#cython: language_level=3
# cython: binding=True


from sanic import Sanic
from sanic.response import json
from sanic.request import Request
from sanic_cors import CORS


app = Sanic(__name__)
CORS(app)


# ------------------
from random import choice, shuffle
from math import sqrt, log, ceil
from copy import deepcopy



bhoos_to_ismcts = {
    "1": "14",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "T": "10",
    "J": "11",
    "Q": "12",
    "K": "13",
}

ismcts_to_bhoos = {
    "14": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "10": "T",
    "11": "J",
    "12": "Q",
    "13": "K",
}





class Card:
    """ A playing card, with rank and suit.
        rank must be an integer between 2 and 14 inclusive (Jack=11, Queen=12, King=13, Ace=14)
        suit must be a string of length 1, one of 'C' (Clubs), 'D' (Diamonds), 'H' (Hearts) or 'S' (Spades)
    """
    def __init__(self, r, s):
        # cdef int lower_rank = 2
        # cdef int upper_rank = 15
        # if r not in list(range(lower_rank, upper_rank)):
        #    raise Exception("Invalid rank")
        # if s not in ["C", "D", "H", "S"]:
        #    raise Exception("Invalid suit")
        self.rank = r
        self.suit = s

    def __repr__(self):
        return "??23456789TJQKA"[self.rank] + self.suit

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __ne__(self, other):
        return self.rank != other.rank or self.suit != other.suit

probability_table = {
    # Number of cards of a suit : (probability of winning for A, K, Q)
    0: (0, 0, 0),
    1: (0.992, 0, 0),
    2: (0.986, 0.862, 0),
    3: (0.974, 0.784, 0.275),
    4: (0.955, 0.672, 0.110),
    5: (0.924, 0.523, 0),
    6: (0.872, 0.338, 0),
    7: (0.790, 0.145, 0),
    8: (0.664, 0, 0),
    9: (0.480, 0, 0),
    10: (0.240, 0, 0)
}

def side_suit_high_cards_probability(cards):
    """
    For A, K, Q of non-spade suits, calculate the probability of winning a hand using the probability_table.
    """
    bid_probability = 0.0
    card_count_per_suit = {}
    for card in cards:
        if card.suit not in card_count_per_suit:
            card_count_per_suit[card.suit] = 1
        else:
            card_count_per_suit[card.suit] += 1
    for card in cards:
        if card.suit == "S":
            continue
        if card.rank == 14:
            bid_probability += probability_table[card_count_per_suit[card.suit]][0]
        elif card.rank == 13:
            bid_probability += probability_table[card_count_per_suit[card.suit]][1]
        elif card.rank == 12:
            bid_probability += probability_table[card_count_per_suit[card.suit]][2]
    return bid_probability


def spade_high_cards_probability(cards):
    bid_probability = 0.0
    for card in cards:
        if card.suit == "S":
            if card.rank == 14:
                bid_probability += 1
            elif card.rank == 13:
                # Check if there is more spades than other un-owned higher spades
                if sum([1 for card in cards if card.suit == "S"]) > 1:
                    bid_probability += 1
            elif card.rank == 12:
                # Check if there is 2 more spades than other un-owned spades
                if sum([1 for card in cards if card.suit == "S"]) > 2:
                    bid_probability += 1
            #elif card.rank == 11:
                # Check if there is 3 more spades than other un-owned spades
                #if sum([1 for card in cards if card.suit == "S"]) > 3:
                    #bid_probability += 1
    return bid_probability



def spade_long_suit_probability(cards):
    bid_probability = 0.0
    spade_count = 0
    for card in cards:
        if card.suit == "S":
            spade_count += 1
    if spade_count > 4:
        bid_probability = spade_count - 4
    return bid_probability


short_side_probab_table = {
    0: (0.996, 0.949, 0.729), # first trick
    1: (0, 0.915, 0.605),     # second trick
    2: (0, 0, 0.450)          # third trick
}






def short_side_suit_with_uncounted_spades(cards):
    bid_probability = 0.0
    unassigned_lower_rank_spades = [card for card in cards if card.suit == "S" and card.rank < 12]
    if len(unassigned_lower_rank_spades) == 0:
        return 0
    card_count_per_suit = {
        "C": 0,
        "D": 0,
        "H": 0,
        "S": 0
    }
    for card in cards:
        card_count_per_suit[card.suit] += 1
    for suit in ["C", "D", "H"]:
        count = card_count_per_suit[suit]
        if count == 0:
            for idx, card in enumerate(unassigned_lower_rank_spades):
                # Run the for loop while idx is less than 2
                if idx < 2:
                    bid_probability += short_side_probab_table[0][idx+1]
                    unassigned_lower_rank_spades.pop(idx)
        elif count == 1:
            for idx, card in enumerate(unassigned_lower_rank_spades):
                if idx < 2:
                    bid_probability += short_side_probab_table[1][idx+1]
                    unassigned_lower_rank_spades.pop(idx)
        elif count == 2:
            for idx, card in enumerate(unassigned_lower_rank_spades):
                if idx < 2:
                    bid_probability += short_side_probab_table[2][idx+1]
                    unassigned_lower_rank_spades.pop(idx)
    return bid_probability






def get_bid(cardsStrArr):
    bid_probability = 0.0
    cards = []
    for card in cardsStrArr:
        rank, suit = card[0], card[1]
        if rank in bhoos_to_ismcts:
            rank = bhoos_to_ismcts[rank]
        card = Card(int(rank), suit)
        cards.append(card)

    side_suit_high = side_suit_high_cards_probability(cards)
    spade_high = spade_high_cards_probability(cards)
    spade_long = spade_long_suit_probability(cards)
    short_sides = short_side_suit_with_uncounted_spades(cards)
    #count = round(side_suit_high + spade_high + spade_long)
    count = round(side_suit_high + spade_high + max(spade_long, short_sides))


    # count will be ceiled to the nearest integer
    #count = round(side_suit_high + spade_high) + max(round(spade_long), round(short_sides))
    # print(f"Side Suit High: {side_suit_high}")
    # print(f"Spade High: {spade_high}")
    # print(f"Spade Long: {spade_long}")
    # print(f"Short Sides: {short_sides}")
    # print(f"Count: {count}")
    #print("----")
    

    
    

    count = count if count <= 8 else 8
    return max(1, count)



class CallBreakState:

    def __init__(self, body):
        
        self.body = body
        self.players = body["playerIds"]
        self.numberOfPlayers = 4
        self.playerToMove = body["playerId"] # ToCheck
        self.tricksInRound = len(body["cards"]) # ToCheck
        # ToCheck: Store the cards in each player's hand
        self.playerHands = {p: [] for p in self.players}
        self.playerHands[self.playerToMove] = body["cards"]
        played = body["played"]
        history = body["history"]
        old_history = body["old_history"]
        self.discards = (
            played + history
            # Store the cards that have been played in the current round.
        )
        # Store the cards played by each player in the current trick.
        # ToCheck: Format (player idx, card)
        self.currentTrick = []
        if history == []:
            # This is the first hand of the round.
            # print(f"First hand")
            playerIdx = self.players.index(self.playerToMove) # gives player Index in respect to playerIds
            bidderIdx = (playerIdx - len(played) + 4) % 4 # gives you the index of first player in first hand
            for card in played:
                self.currentTrick.append((self.players[bidderIdx], card))
                bidderIdx = (bidderIdx + 1) % self.numberOfPlayers

        else:
            # This is not the first hand of the round.
            # print(f"Not first hand")
            previous_hand_winner = old_history[-1][2] - 1 
            for i in range(len(body["played"])):
                self.currentTrick.append((self.players[previous_hand_winner], body["played"][i]))
                previous_hand_winner = (previous_hand_winner + 1) % self.numberOfPlayers
        # print(self.currentTrick)
        # Number of tricks won by each player in the current round.
        self.tricksTaken = {
            p: 0 for p in self.players
        }
        self.trumpSuit = "S"

        # self.deal()
    
    def clone(self):
        """Create a deep clone of this game state.
        """
        st = CallBreakState(self.body)
        # st.players = self.players
        # st.numberOfPlayers = self.numberOfPlayers
        # st.playerToMove = self.playerToMove
        # st.tricksInRound = self.tricksInRound
        st.playerHands = deepcopy(self.playerHands)
        st.discards = deepcopy(self.discards)
        st.currentTrick = deepcopy(self.currentTrick)
        # st.trumpSuit = self.trumpSuit
        st.tricksTaken = deepcopy(self.tricksTaken)
        return st
  
    def clone_and_randomize(self, observer):
        """First clone the state, then randomize information not visible to observer.
        # ToCheck: Is this the determinization?
        """
        st = self.clone()
        # Store the cards in observer's hand
        seenCards = (
            st.playerHands[observer] 
            + st.discards
            # + [card for (player, card) in st.currentTrick]
        )
        # Store the unseen cards in the other players' hands
        unSeenCards = [card for card in self.get_card_deck() if card not in seenCards]
        # if we see a player fail to follow suit, we can conclude that they have no cards in that suit.
        # ToDo 

        # Distribute the unseen cards to the players except the observer
        shuffle(unSeenCards)
        current_player = observer
        for card in unSeenCards:
            if current_player == observer:
                current_player = st.players[(st.players.index(current_player) + 1) % len(st.players)]
            st.playerHands[current_player].append(card)
            current_player = st.players[(st.players.index(current_player) + 1) % len(st.players)]
            
        return st

    def get_card_deck(self):
        """Construct a standard deck of 52 cards.
        """
        return [
            Card(rank, suit)
            for rank in range(2, 15)
            for suit in "CDHS"
        ]

    def deal(self):
        """Reset the game state for the beginning of a new round.
        Deal the cards."""
        self.discards = []
        self.currentTrick = []
        self.tricksTaken = {p: 0 for p in self.players}
        # self.tricksTaken = {p: 0 for p in range(1, self.numberOfPlayers + 1)}


        deck = self.get_card_deck()
        shuffle(deck)
        for player in self.players:
            self.playerHands[player] = deck[:13]
            deck = deck[13:]

        # for player in range(1, self.numberOfPlayers + 1):
        #     self.playerHands[player] = deck[13 * (player - 1):13 * player]

    def get_next_player(self, player):
        """Return the id of the next player in the round (clockwise).
        player = P0 or P1 or P2 or P3
        """
        return self.players[(self.players.index(player) + 1) % len(self.players)]

    def do_move(self, move):
        """Apply the move to the state.
        Specify whose turn is next."""
        # Store the played card in the current trick.
        self.currentTrick.append((self.playerToMove, move))
        # Remove the card from the player's hand.
        self.playerHands[self.playerToMove].remove(move)
        # Find the next player
        self.playerToMove = self.get_next_player(self.playerToMove)
        # If the trick is complete, add the points to the player who won it.
        if len(self.currentTrick) == self.numberOfPlayers:
            # Find the player who won the trick.
            # Sort the cards in the trick in ascending order by rank.
            # Sort non-trump cards first, then trump cards.
            # The last card in the trick is the winner.
            (leader, leadCard) = self.currentTrick[0]
            suitedPlays = [
                (player, card.rank)
                for (player, card) in self.currentTrick
                if card.suit == leadCard.suit and card.suit != self.trumpSuit
            ]
            trumpPlays = [
                (player, card.rank)
                for (player, card) in self.currentTrick
                if card.suit == self.trumpSuit
            ]
            sortedPlays = sorted(
                suitedPlays, key=lambda player_rank: player_rank[1]
            ) + sorted(trumpPlays, key=lambda player_rank1: player_rank1[1])
            # The winning play is the last element in sortedPlays
            trickWinner = sortedPlays[-1][0]

            # Update the game state
            self.tricksTaken[trickWinner] += 1
            self.discards += [card for (player, card) in self.currentTrick]
            self.currentTrick = []
            self.playerToMove = trickWinner


    def get_moves(self):
        """Get all possible moves from this state.
        """
        # all the cards in the player's hand
        hand = self.playerHands[self.playerToMove]
        highestTrumpCard = None
        # Check if it's the player turns or the player is following a suit
        if self.currentTrick == []:
            # The player is leading the trick
            # The player can play any card in his hand
            return hand
        else:
            # The player is following the suit
            # The player must play a higher card of the same suit than the lead card if he has one
            # else he can play any card in his hand that matches the suit of the lead card
            # If the player has no cards of the suit, he can play any card
            # in his hand
            highestTrumpCard = None
            highestCardInCurrentTrick = self.currentTrick[0][1]
            for i in range(1, len(self.currentTrick)):
                card = self.currentTrick[i][1]
                if card.suit == highestCardInCurrentTrick.suit:
                    if card.rank > highestCardInCurrentTrick.rank:
                        highestCardInCurrentTrick = card
                elif card.suit == self.trumpSuit:
                    highestTrumpCard = card
            highestCard = highestCardInCurrentTrick
            
            

            
            leadCard = self.currentTrick[0][1]
            cardsInSuit = [
                card
                for card in hand
                if card.suit == leadCard.suit
            ]
            higherCardsInSuit = [
                card
                for card in cardsInSuit
                if card.rank > highestCard.rank
            ]
            trumpCards = [
                card
                for card in hand
                if card.suit == self.trumpSuit
            ]
            otherCards = [
                card
                for card in hand
                if card.suit != leadCard.suit and card.suit != self.trumpSuit
            ]
                
            if higherCardsInSuit == []:
                if cardsInSuit == []:
                    if trumpCards == []:
                        return hand
                    else:
                        # If we do not have higher trump cards, we can play other card.
                        if highestTrumpCard is None:
                            return trumpCards
                        else:
                            # If we have higher trump cards, we can play them.
                            higherTrumpCards = []
                            for card in trumpCards:
                                if card.rank > highestTrumpCard.rank:
                                    higherTrumpCards.append(card)
                            if higherTrumpCards == []:
                                # Return non-trump cards
                                return otherCards if otherCards != [] else trumpCards
                            else:
                                return higherTrumpCards
                else:
                    return cardsInSuit
            else:
                return higherCardsInSuit
            



            # # Previous version
            # leadCard = self.currentTrick[0][1]
            # cardsInSuit = [
            #     card
            #     for card in hand
            #     if card.suit == leadCard.suit
            # ]
            # return cardsInSuit if cardsInSuit != [] else hand

    def get_result(self, player):
        """Get the game result from the perspective of the current player.
        Return 1 if the player wins, 0 if the player loses.
        Player wins if he has most tricks in the round compared to other palyers.
        """
        # ToDo: Check if a player wins if it win 13
        # ToCheck
        # Check if the tricks won by player is maximum
        # ToDo: check if defaulting opponent can be used here like in spades.
        # or w_a - w_b - w_c - w_d can be used here.
        # if self.tricksTaken[player] >= 13:
        #     return 1
        # else:
        #     return 0


        # return (self.tricksTaken[player] - self.bid)

        # if self.tricksTaken[player] == self.bid:
        #     return 1
        # else:
        #     return 0

        # If the player have max tricks, he wins
        # if self.tricksTaken[player] >= max(self.tricksTaken.values()):
        #     return 1.0
        # else:
        #     return 0.0

        # Return 1 if the player wins current trick else return 0
        # tricks won by player
        # tricksWon = self.tricksTaken[player]
        # # tricks won by other players
        # tricksWonByOthers = 13 - tricksWon
        # return 1 if tricksWon >= tricksWonByOthers/3 else 0



        # # Return the difference in tricks won by the players
        # return (self.tricksTaken[player] - self.tricksTaken[self.get_next_player(player)] - self.tricksTaken[self.get_next_player(self.get_next_player(player))] - self.tricksTaken[self.get_next_player(self.get_next_player(self.get_next_player(player)))])

        # Return current player's tricks won
        return self.tricksTaken[player]
    
    def get_random_move(self):
        """Get a random move from the available ones.
        """
        moves = self.get_moves()
        return choice(moves)

    def __repr__(self):
        """Return a representation of the game state
        """
        return "Player to move: {}\nPlayer hands: {}\nDiscards: {}\nCurrent trick: {}\nTricks taken: {}".format(
            self.playerToMove,
            self.playerHands,
            self.discards,
            self.currentTrick,
            self.tricksTaken,
        )

class Node:
    """Represents a single node in the game tree."""
    def __init__(self, move=None, parent=None, playerJustMoved=None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parent = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.avails = 1
        self.playerJustMoved = (
           playerJustMoved
        )

    def get_untried_moves(self, legalMoves):
        """Return the set of moves not yet tried from this node.
        """
        triedMoves = [
            child.move for child in self.childNodes
        ]
        unTriedMoves = [
            move for move in legalMoves if move not in triedMoves
        ]
        return unTriedMoves
    
    def ucb_select_child(self, legalMoves, exploration=0.7):
        """Select the child node with the highest UCB score filtered by the legal moves.
        """
        # legal child nodes
        legalChildren = [
            child for child in self.childNodes if child.move in legalMoves
        ]
        # get the child with highest ucb score
        s = max(
            legalChildren,
            key=lambda c: (c.wins / c.visits)
            + (exploration * sqrt(log(c.avails) / c.visits)),
        )

        # update the child's availablity
        for child in legalChildren:
            child.avails += 1

        return s

    def add_child(self, move, playerJustMoved):
        """Add a new child node to the tree.
        """
        newNode = Node(move=move, parent=self, playerJustMoved=playerJustMoved)
        self.childNodes.append(newNode)
        return newNode
    
    def update(self, terminalState):
        """Update this node - one additional visit and result additional wins.
        """
        self.visits += 1
        if self.playerJustMoved is not None:
            self.wins += terminalState.get_result(self.playerJustMoved)
    
    def __repr__(self):
        """Return a string representation of the tree.
        """
        # return f"{self.move}"
        return "UCB Node\n\tMove: {}\n\tWins: {}\n\tVisits: {}\n\tAvails: {}\n\tChild nodes: {}".format(
            self.move,
            self.wins,
            self.visits,
            self.avails,
            self.childNodes,
        )
    
    def __str__(self):
        """Return a string representation of the tree.
        """
        return self.__repr__()
    
    def is_leaf(self):
        """Check if the node is a leaf node.
        """
        return self.childNodes == []
    
    def is_root(self):
        """Check if the node is a root node.
        """
        return self.parent is None
    
    def get_move(self):
        """Return the move associated with the node.
        """
        return self.move
    
    def get_parent(self):
        """Return the parent node of the node.
        """
        return self.parent
    
    def get_children(self):
        """Return the children of the node.
        """
        return self.childNodes
    
    def get_wins(self):
        """Return the wins of the node.
        """
        return self.wins
    
    def get_visits(self):
        """Return the visits of the node.
        """
        return self.visits
    
    def get_avails(self):
        """Return the availablity of the node.
        """
        return self.avails
    
def ISMCTS(rootstate, itermax, verbose=False):
    """ Conduct an ISMCTS search for itermax iterations starting from rootstate.
    Return the best move from the rootstate.
    """
    # Optimization 1: If rootstate has only one move, return it.
    if len(rootstate.get_moves()) == 1:
        return rootstate.get_moves()[0]
    # Optimiization 2: In first few hands, if turn is ours and we have 2+ trump and we have only 1 card of any
    # other suit, return that 1 one first. So that we can use short side with the trump.
    # calculate the cards of each suit
    if len(rootstate.body["played"]) == 0 and len(rootstate.body["history"]) < 15:
        card_count_per_suit = {
        "C": 0,
        "D": 0,
        "H": 0,
        "S": 0
        }
        cards_per_suit = {
            "C": [],
            "D": [],
            "H": [],
            "S": []

        }
        for card in rootstate.playerHands[rootstate.playerToMove]:
            card_count_per_suit[card.suit] += 1
            cards_per_suit[card.suit].append(card)
        # if we have 2+ trump and we have only 1 card of any other suit, return that one first
        if card_count_per_suit["S"] > 2:
            if card_count_per_suit["C"] == 1:
                return cards_per_suit["C"][0]
            if card_count_per_suit["D"] == 1:
                return cards_per_suit["D"][0]
            if card_count_per_suit["H"] == 1:
                return cards_per_suit["H"][0]

    rootNode = Node()
    for i in range(itermax):
        node = rootNode

        # Determinize
        state = rootstate.clone_and_randomize(rootstate.playerToMove)

        # Select
        # While node is fully expanded and non-terminal
        legalMoves = state.get_moves()
        while (legalMoves != [] and node.get_untried_moves(legalMoves) == []):
        # while (state.get_moves() != [] and node.get_untried_moves(state.get_moves()) != []):
            node = node.ucb_select_child(legalMoves)
            state.do_move(node.get_move())
        
        # Expand
        untriedMoves = node.get_untried_moves(state.get_moves())
        if untriedMoves != []: # If we can expand (i.e. state/node is non-terminal)
            move = choice(untriedMoves)
            # state.do_move(move) ## #################################################################################################
            node = node.add_child(move, state.playerToMove) # Add child and descend tree

        # Rollout
        # ToDo: this can often be made orders of magnitude quicker using a state.get_random_move() function
        while state.get_moves() != []: # while state is non-terminal
            random_move = choice(state.get_moves())
            state.do_move(random_move)

        
        # Backpropagate
        while node is not None: # Rollout reached the root node
            node.update(state) # Update node with result from POV of node.playerJustMoved
            node = node.get_parent()
        
    # Return the move that was most visited
    #for child in rootNode.childNodes:
    #    print(f"Move: {child.get_move()}, Visits: {child.visits}, Wins: {child.wins}")
    #print(f"-------------------------------")
    return max(rootNode.childNodes, key=lambda c: c.visits).get_move()
    # return max(rootNode.childNodes, key=lambda c: c.wins).get_move()




@app.route("/hi", methods=["GET"])
def hi(request: Request):
    return json({"value": "hello"})


@app.route("/bid", methods=["POST"])
def bid(request: Request):
    body = request.json
    bid = get_bid(body["cards"])
    # return should have a single field value which should be an int reprsenting the bid value
    return json({"value": bid})


def change_to_ismcts_format(body, parameter):
    old = body[parameter]
    body[parameter] = []
    for card in old:
        rank, suit = card[0], card[1]
        if rank in bhoos_to_ismcts:
            rank = bhoos_to_ismcts[rank]
            card = Card(int(rank), suit)
            body[parameter].append(card)
        else:
            pass
    return body


@app.route("/play", methods=["POST"])
def play(request: Request):
    body = request.json
    # Change body["cards"] to Card objects
    body = change_to_ismcts_format(body, "cards")
    # Change body["played"] to Card objects
    body = change_to_ismcts_format(body, "played")
    # Change body["history"] to Card objects
    old_history = body["history"]
    body["old_history"] = old_history
    # print(body["history"])
    body["history"] = []


    for hand in old_history:
        for card in hand[1]:
            rank = card[0]
            suit = card[1]

            if rank in bhoos_to_ismcts:
                rank = bhoos_to_ismcts[rank]
                card = Card(int(rank), suit)
                body["history"].append(card)
            else:
                print(f"{rank} not found")

    state = CallBreakState(body)
    #itermax = 12 * (len(body["cards"]) - len(body["played"]) + 2)
    #cdef int card_length = len(body["cards"])
    #cdef int itermax = card_length * 12 + card_length - 26

    #print(f"Iterating for {itermax} iterations")
    itermax = 1000
    play_card = ISMCTS(state, itermax)
    rank = play_card.rank
    suit = play_card.suit
    if str(rank) in ismcts_to_bhoos:
        rank = ismcts_to_bhoos[str(rank)]

    # if len(body["cards"]) == 2:
    #     print(body)
    # if round is 5, then close the sanic server
    #if body["context"]["round"] == 5:
    #    print("Closing server")
    #    app.stop()
    return json({"value": f"{rank}{suit}"})


if __name__ == "__main__":
    # Docker image should always listen in port 7000
    app.run(host="0.0.0.0", port=7000, debug=False, access_log=False)
    # app_cython.app.run(host="0.0.0.0", port=7000, debug=False, access_log=False)