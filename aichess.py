#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Carlos y Danyal

"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy
import math

import chess
import board
import numpy as np
import sys
import queue
from typing import List

RawStateType = List[List[List[int]]]

from itertools import permutations


class Aichess():
    """
    A class to represent the game of chess.

    ...

    Attributes:
    -----------
    chess : Chess
        represents the chess game

    Methods:
    --------
    startGame(pos:stup) -> None
        Promotes a pawn that has reached the other side to another, or the same, piece

    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.listVisitedSituations = []
        self.pathToTarget = []
        self.depthMax = 8;
        # Dictionary to reconstruct the visited path
        self.dictPath = {}
        # Prepare a dictionary to control the visited state and at which
        # depth they were found for DepthFirstSearchOptimized
        self.dictVisitedStates = {}
        self.currentStateW = self.chess.boardSim.currentStateW
        self.currentStateB = self.chess.boardSim.currentStateB
        self.checkMate = False
        self.currentStateW = self.chess.boardSim.currentStateW
        self.currentStateB = self.chess.boardSim.currentStateB
        self.checkMate = False

    def copyState(self, state):

        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState

    def isVisitedSituation(self, color, mystate):

        if (len(self.listVisitedSituations) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedSituations)):
                    if self.isSameState(list(perm_state[j]), self.listVisitedSituations.__getitem__(k)[1]) and color == \
                            self.listVisitedSituations.__getitem__(k)[0]:
                        isVisited = True

            return isVisited
        else:
            return False

    def getCurrentStateW(self):

        return self.myCurrentStateW

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getListNextStatesB(self, myState):
        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):

            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):

            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState

    def isVisited(self, mystate):

        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedStates)):

                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True

            return isVisited
        else:
            return False

    def isWatchedBk(self, currentState):

        self.newBoardSim(currentState)

        bkPosition = self.getPieceState(currentState, 12)[0:2]
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)

        # Si les negres maten el rei blanc, no és una configuració correcta
        if wkState == None:
            return False
        # Mirem les possibles posicions del rei blanc i mirem si en alguna pot "matar" al rei negre
        for wkPosition in self.getNextPositions(wkState):
            if bkPosition == wkPosition:
                # Tindríem un checkMate
                return True
        if wrState != None:
            # Mirem les possibles posicions de la torre blanca i mirem si en alguna pot "matar" al rei negre
            for wrPosition in self.getNextPositions(wrState):
                if bkPosition == wrPosition:
                    return True

        return False

    def isWatchedWk(self, currentState):
        self.newBoardSim(currentState)

        wkPosition = self.getPieceState(currentState, 6)[0:2]

        bkState = self.getPieceState(currentState, 12)
        brState = self.getPieceState(currentState, 8)

        # If whites kill the black king , it is not a correct configuration
        if bkState == None:
            return False
        # We check all possible positions for the black king, and chck if in any of them it may kill the white king
        for bkPosition in self.getNextPositions(bkState):
            if wkPosition == bkPosition:
                # That would be checkMate
                return True
        if brState != None:
            # We check the possible positions of the black tower, and we chck if in any of them it may kill the white king
            for brPosition in self.getNextPositions(brState):
                if wkPosition == brPosition:
                    return True

        return False

    def newBoardSim(self, listStates):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        for state in listStates:
            TA[state[0]][state[1]] = state[2]

        self.chess.newBoardSim(TA)

    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i
                break
        return pieceState

    def getCurrentState(self):
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
            listStates.append(j)
        return listStates

    def getNextPositions(self, state):
        # Given a state, we check the next possible states
        # From these, we return a list with position, i.e., [row, column]
        if state == None:
            return None
        if state[2] > 6:
            nextStates = self.getListNextStatesB([state])
        else:
            nextStates = self.getListNextStatesW([state])
        nextPositions = []
        for i in nextStates:
            nextPositions.append(i[0][0:2])
        return nextPositions

    def getWhiteState(self, currentState):
        whiteState = []
        wkState = self.getPieceState(currentState, 6)
        whiteState.append(wkState)
        wrState = self.getPieceState(currentState, 2)
        if wrState != None:
            whiteState.append(wrState)
        return whiteState

    def getBlackState(self, currentState):
        blackState = []
        bkState = self.getPieceState(currentState, 12)
        blackState.append(bkState)
        brState = self.getPieceState(currentState, 8)
        if brState != None:
            blackState.append(brState)
        return blackState

    def getMovement(self, state, nextState):
        # Given a state and a successor state, return the postiion of the piece that has been moved in both states
        pieceState = None
        pieceNextState = None
        for piece in state:
            if piece not in nextState:
                movedPiece = piece[2]
                pieceNext = self.getPieceState(nextState, movedPiece)
                if pieceNext != None:
                    pieceState = piece
                    pieceNextState = pieceNext
                    break

        return [pieceState, pieceNextState]

    def heuristica(self, currentState, color):
        # In this method, we calculate the heuristics for both the whites and black ones
        # The value calculated here is for the whites,
        # but finally from verything, as a function of the color parameter, we multiply the result by -1
        value = 0

        bkState = self.getPieceState(currentState, 12)
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)
        brState = self.getPieceState(currentState, 8)

        # If the black king has been captured, this is not a valid configuration
        if bkState is None:
            return False

        # Check all possible moves for the black king and see if it can capture the white king
        for bkPosition in self.getNextPositions(bkState):
            if wkPosition == bkPosition:
                # White king would be in check
                return True

        if brState is not None:
            # Check all possible moves for the black rook and see if it can capture the white king
            for brPosition in self.getNextPositions(brState):
                if wkPosition == brPosition:
                    return True

        return False

    def allWkMovementsWatched(self, currentState):

        self.newBoardSim(currentState)
        # In this method, we check if the white king is threatened by black pieces
        # Get the current state of the white king
        wkState = self.getPieceState(currentState, 6)
        allWatched = False

        # If the white king is on the edge of the board, it may be more vulnerable
        if wkState[0] == 0 or wkState[0] == 7 or wkState[1] == 0 or wkState[1] == 7:
            # Get the state of the black pieces
            brState = self.getPieceState(currentState, 8)
            blackState = self.getBlackState(currentState)
            allWatched = True

            # Get the possible future states for the white pieces
            nextWStates = self.getListNextStatesW(self.getWhiteState(currentState))
            for state in nextWStates:
                newBlackState = blackState.copy()
                # Check if the black rook has been captured. If so, remove it from the state
                if brState is not None and brState[0:2] == state[0][0:2]:
                    newBlackState.remove(brState)
                state = state + newBlackState
                # Move the white pieces to their new state
                self.newBoardSim(state)
                # Check if the white king is not threatened in this position,
                # which implies that not all of its possible moves are under threat
                if not self.isWatchedWk(state):
                    allWatched = False
                    break

        # Restore the original board state
        self.newBoardSim(currentState)
        return allWatched


    def isWhiteInCheckMate(self, currentState):
        if self.isWatchedWk(currentState) and self.allWkMovementsWatched(currentState):
            return True
        return False
    
    # Helper method that checks if a black rook can be eliminated
    def eliminarBlack(self, blackState, brState, successor):
        self.newBoardSim(blackState)
        newBlackState = blackState.copy()
        if brState != None:
            if len(successor) >= 2:
                if brState[0:2] == successor[0][0:2] or brState[0:2] == successor[1][0:2]:
                    newBlackState.remove(brState)
            else:
                if brState[0:2] == successor[0][0:2]:
                    newBlackState.remove(brState)
        return newBlackState
    
    # Helper method that checks if a white rook can be eliminated
    def eliminarWhite(self, whiteState, wrState, successor):
        self.newBoardSim(whiteState)
        newWhiteState = whiteState.copy()
        if wrState != None:
            if len(successor) >= 2:
                if wrState[0:2] == successor[0][0:2] or wrState[0:2] == successor[1][0:2]:
                    newWhiteState.remove(wrState)
            else:
                if wrState[0:2] == successor[0][0:2]:
                    newWhiteState.remove(wrState)
        return newWhiteState

    # Method to check checkMate cases to stop the algorithm
    def isCheckMate(self, state):
        self.newBoardSim(state)
        brState = self.getPieceState(state, 8)
        wrState = self.getPieceState(state, 2)
        whiteState = self.getWhiteState(state)
        blackState = self.getBlackState(state)

        for successor in self.getListNextStatesW(whiteState):
            successor += self.eliminarBlack(blackState, brState, successor)
            if not self.isWatchedWk(successor):
                self.newBoardSim(state)
                return False

        for successor in self.getListNextStatesB(blackState):
            successor += self.eliminarWhite(whiteState, wrState, successor)
            if not self.isWatchedBk(successor):
                self.newBoardSim(state)
                return False

        return True

    def heuristica(self, currentState, color):
        # This method calculates the heuristic value for the current state.
        # The value is initially computed from White's perspective.
        # If the 'color' parameter indicates Black, the final value is multiplied by -1.

        value = 0

        bkState = self.getPieceState(currentState, 12)  # Black King
        wkState = self.getPieceState(currentState, 6)   # White King
        wrState = self.getPieceState(currentState, 2)   # White Rook
        brState = self.getPieceState(currentState, 8)   # Black Rook

        filaBk, columnaBk = bkState[0], bkState[1]
        filaWk, columnaWk = wkState[0], wkState[1]

        if wrState is not None:
            filaWr, columnaWr = wrState[0], wrState[1]
        if brState is not None:
            filaBr, columnaBr = brState[0], brState[1]

        # We check if they killed the black tower
        if brState == None:
            value += 50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)
            if distReis >= 3 and wrState != None:
                filaR = abs(filaBk - filaWr)
                columnaR = abs(columnaWr - columnaBk)
                value += (min(filaR, columnaR) + abs(filaR - columnaR)) / 10
            # If we are white, the closer our king from the oponent, the better
            # we substract 7 to the distance between the two kings, since the max distance they can be at in a board is 7 moves
            value += (7 - distReis)
            # If they black king is against a wall, we prioritize him to be at a corner, precisely to corner him
            if bkState[0] == 0 or bkState[0] == 7 or bkState[1] == 0 or bkState[1] == 7:
                value += (abs(filaBk - 3.5) + abs(columnaBk - 3.5)) * 10
            # If not, we will only prioritize that he approahces the wall, to be able to approach the check mate
            else:
                value += (max(abs(filaBk - 3.5), abs(columnaBk - 3.5))) * 10

        # They killed the black tower.
        # Within this method we consider the same conditions than in the previous section, but now with reversed values.
        if wrState == None:
            value += -50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)

            if distReis >= 3 and brState != None:
                filaR = abs(filaWk - filaBr)
                columnaR = abs(columnaBr - columnaWk)
                value -= (min(filaR, columnaR) + abs(filaR - columnaR)) / 10
            # If we are white, the close we have our king from the oponent, the better
            # If we substract 7 to the distance between both kings, as this is the max distance they can be at in a chess board
            value += (-7 + distReis)

            if wkState[0] == 0 or wkState[0] == 7 or wkState[1] == 0 or wkState[1] == 7:
                value -= (abs(filaWk - 3.5) + abs(columnaWk - 3.5)) * 10
            else:
                value -= (max(abs(filaWk - 3.5), abs(columnaWk - 3.5))) * 10

        # We are checking blacks
        if self.isWatchedBk(currentState):
            value += 20

        # We are checking whites
        if self.isWatchedWk(currentState):
            value += -20

            # If black, values are negative, otherwise positive
        if not color:
            value = (-1) * value

        return value

    # Helper method that checks if a black rook can be eliminated
    def eliminarBlack(self, blackState, brState, successor):
        self.newBoardSim(blackState)
        newBlackState = blackState.copy()
        if brState != None:
            if len(successor) >= 2:
                if brState[0:2] == successor[0][0:2] or brState[0:2] == successor[1][0:2]:
                    newBlackState.remove(brState)
            else:
                if brState[0:2] == successor[0][0:2]:
                    newBlackState.remove(brState)
        return newBlackState

    # Helper method that checks if a white rook can be eliminated
    def eliminarWhite(self, whiteState, wrState, successor):
        self.newBoardSim(whiteState)
        newWhiteState = whiteState.copy()
        if wrState != None:
            if len(successor) >= 2:
                if wrState[0:2] == successor[0][0:2] or wrState[0:2] == successor[1][0:2]:
                    newWhiteState.remove(wrState)
            else:
                if wrState[0:2] == successor[0][0:2]:
                    newWhiteState.remove(wrState)
        return newWhiteState

    # Method to check checkMate cases to stop the algorithm
    def isCheckMate(self, state):
        self.newBoardSim(state)
        brState = self.getPieceState(state, 8)
        wrState = self.getPieceState(state, 2)
        whiteState = self.getWhiteState(state)
        blackState = self.getBlackState(state)

        for successor in self.getListNextStatesW(whiteState):
            successor += self.eliminarBlack(blackState, brState, successor)
            if not self.isWatchedWk(successor):
                self.newBoardSim(state)
                return False

        for successor in self.getListNextStatesB(blackState):
            successor += self.eliminarWhite(whiteState, wrState, successor)
            if not self.isWatchedBk(successor):
                self.newBoardSim(state)
                return False

        return True

# ---------------------- MINIMAX START  --------------------------- #

    def minimaxGame(self, depthWhite, depthBlack, playerTurn):
        currentState = self.getCurrentState()
        print("Initial state of all pieces: ", currentState)

        while not self.isCheckMate(currentState):
            currentState = self.getCurrentState()
            #self.newBoardSim(currentState)

            # check player turn
            if playerTurn:
                movimiento = self.minimax(currentState, depthWhite, depthWhite, playerTurn)
            else:
                movimiento = self.minimax(currentState, depthBlack, depthBlack, playerTurn)

            if (movimiento is None):
                if(playerTurn == False):
                    color = "BLANCAS"
                else:
                    color = "NEGRAS"
                return print("JAQUE MATE, GANAN LAS ", color)

            #in case the pieces are repeting movements, stop 
            if (self.isVisitedSituation(playerTurn, self.copyState(movimiento))):
                return print("JUEGO EN TABLAS")

            self.listVisitedSituations.append((playerTurn, self.copyState(movimiento)))

            # make best movement and print on board
            piece_moved = self.getMovement(currentState, self.copyState(movimiento))
            self.chess.move((piece_moved[0][0], piece_moved[0][1]), (piece_moved[1][0], piece_moved[1][1]))
            self.chess.board.print_board()
            playerTurn = not playerTurn


    def minimax(self, state, depth, depthColor, playerTurn):

        # check if it is ternimal node or checkmate scenario to return static heuristic value
        if depth == 0 or self.isCheckMate(state):
            return self.heuristica(state, playerTurn)
        
        # variable that will contain the best movement to make
        maxState = None

        # Maximizing player
        if playerTurn:
            currBestValue = float('-inf')

            blackState = self.getBlackState(state)
            whiteState = self.getWhiteState(state)
            brState = self.getPieceState(state, 8)

            # We see the successors only for the states in White
            for successor in self.getListNextStatesW(whiteState):
                successor += self.eliminarBlack(blackState, brState, successor)

                if not self.isWatchedWk(successor):
                    #self.newBoardSim(state)
                    bestValue = self.minimax(successor, depth - 1, depthColor, False)
                    # check for best value and best movement if any
                    if bestValue > currBestValue:
                        currBestValue = bestValue
                        maxState = successor

        # Minimizing player
        else:
            # initialize minimizer
            currBestValue = float('inf')
            whiteState = self.getWhiteState(state)
            blackState = self.getBlackState(state)
            wrState = self.getPieceState(state, 2)

            # We see the successors only for the states in Black
            for successor in self.getListNextStatesB(blackState):
                successor += self.eliminarWhite(whiteState, wrState, successor)

                if not self.isWatchedBk(successor):
                    #self.newBoardSim(state)
                    # Recursively call minimax with the successor state
                    bestValue = self.minimax(successor, depth - 1, depthColor, True)

                    # Update the best value and maxState if a better successor is found
                    if bestValue < currBestValue:
                        currBestValue = bestValue
                        maxState = successor

        # if back to top level, return the best movement
        if depth == depthColor:
            return maxState

        return currBestValue

    def alphaBetaPoda(self, depthWhite,depthBlack):
        
        currentState = self.getCurrentState()
        # Your code here  
        
    def expectimax(self, depthWhite, depthBlack):
        
        currentState = self.getCurrentState()
        # Your code here       
        

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     sys.exit(usage())

    # Initialize an empty 8x8 chess board
    TA = np.zeros((8, 8))


    # Load initial positions of the pieces
    TA = np.zeros((8, 8))
    TA[7][0] = 2   # White Rook
    TA[7][5] = 6   # White King
    TA[0][7] = 8   # Black Rook
    TA[0][5] = 12  # Black King  

    # Initialise board and print
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    print("printing board")
    aichess.chess.boardSim.print_board()
    
    # Run exercise 1
    aichess.minimaxGame(3,3,True)
    # Add code to save results and continue with other exercises
