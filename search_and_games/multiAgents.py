# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def getClosestFood(pPos, grid):
    minDistance = None
    minPos = None
    for row in range(grid.width):
        for col in range(grid.height):
            if grid[row][col] and (minPos == None or
                manhattanDistance(pPos, (row, col)) < minDistance):
                minDistance = manhattanDistance(pPos, (row, col))
                minPos = (row, col)
    return minPos, minDistance

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        currPos = currentGameState.getPacmanPosition()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        closestFood, closeDist = getClosestFood(currPos, newFood)
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostPositions = successorGameState.getGhostPositions()

        score = successorGameState.getScore()

        x, y = currPos
        cX, cY = newPos
        if newFood[cX][cY]: score += 40
        for pos in ghostPositions:
            dist = manhattanDistance(newPos, pos)
            if dist == 0: score = 0
            # if dist == 1: score -= 100
       
        if closeDist is not None and (
           manhattanDistance(newPos, closestFood) < closeDist):
            score += 20

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 7)
    """
    def getValue(self, gameState, agentIndex, numAgents, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        if agentIndex == 0:
            bestScore = -99999 
            for action in gameState.getLegalActions(agentIndex):
                newState = gameState.generateSuccessor(agentIndex, action)
                
                score = self.getValue(newState, (agentIndex + 1) % numAgents, numAgents, depth)
                if score > bestScore:
                    bestScore = score
                # alpha = max(alpha, score)
                # if beta <= alpha:
                #     break
            return bestScore
        else:
            bestScore = 99999
            depth = depth - 1 if (agentIndex + 1 == numAgents) else depth
            for action in gameState.getLegalActions(agentIndex):
                newState = gameState.generateSuccessor(agentIndex, action)
                
                score = self.getValue(newState, (agentIndex + 1) % numAgents, numAgents, depth)
                if score < bestScore:
                    bestScore = score
                # beta = min(beta, score)
                # if beta <= alpha:
                #     break
            return bestScore

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game
        """
        bestScore, bestAction = None, None
        numAgents = gameState.getNumAgents()
        for action in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0, action)
            cScore = self.getValue(newState, 1, numAgents, self.depth)
            if bestScore == None or bestScore < cScore:
                bestAction = action
                bestScore = cScore
        return bestAction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 8)
    """
    def expectiValue(self, gameState, agentIndex, numAgents, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
            
        if agentIndex == 0:
            bestScore = -99999 
            for action in gameState.getLegalActions(agentIndex):
                newState = gameState.generateSuccessor(agentIndex, action)
                
                score = self.expectiValue(newState, (agentIndex + 1) % numAgents, numAgents, depth)
                if score > bestScore:
                    bestScore = score
                # alpha = max(alpha, score)
                # if beta <= alpha:
                #     break
            return bestScore
        else:
            totalScore = 0
            depth = depth - 1 if (agentIndex + 1 == numAgents) else depth
            for action in gameState.getLegalActions(agentIndex):
                newState = gameState.generateSuccessor(agentIndex, action)
                score = self.expectiValue(newState, (agentIndex + 1) % numAgents, numAgents, depth)
                totalScore += score
            return totalScore/len(gameState.getLegalActions(agentIndex))

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestScore, bestAction = None, None
        numAgents = gameState.getNumAgents()
        for action in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0, action)
            cScore = self.expectiValue(newState, 1, numAgents, self.depth)
            if bestScore == None or bestScore < cScore:
                bestAction = action
                bestScore = cScore
        return bestAction
    

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 9).

    DESCRIPTION: Just returns 0 :(
    """
    return 0
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

