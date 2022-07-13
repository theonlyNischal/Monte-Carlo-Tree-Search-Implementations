from tictactoe import Board
class TreeNode:
    def __init__(self, board, parent):
        self.board = board
        self.parent = parent
        self.is_terminal = True if self.board.is_win() or self.board.is_draw() else False
        self.is_fully_expanded = self.is_terminal
        self.children = {}
        self.wins = 0
        self.visits = 0
        self.avails = 1

class MCTS:
    def search(self, initial_state):
        self.root = TreeNode(initial_state, None)
        for i in range(1000):
            node = self.select(self.root)
            score = self.rollout(node.board)
            self.backpropagate(node, score)
        
        try:
            return get_best_move(self.root, 0)
        except:
            pass
    
    def select(self, node):
        while node.is_terminal() == False:
            if node.is_fully_expanded():
                return self.get_best_move(node, 2)
            else:
                return self.expand(node)
        return node
    
    def expand(node):
        states = node.board.generate_states()
        for state in states:
            if str(state.position) not in node.children:
                new_node = TreeNode(state, node)
                node.children[str(state.position)] = new_node

                if len(states) == len(node.children):
                    node.is_fully_expanded = True
                
                return new_node

    def rollout(self, board):
        pass

    def get_best_move(self, node, exploration_constant):
        pass






if __name__ == '__main__':
    board = Board()
    root = TreeNode(board, None)
    print(root.__dict__)


