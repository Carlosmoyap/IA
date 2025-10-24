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
import matplotlib.pyplot as plt

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

    # Métode que comprova els casos de checkMate per aturar l'algorisme
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

#### EX1 - MINIMAXGAME() ####

    # Contiene toda la lógica para decidir y ejecutar un solo movimiento
    def _perform_turn(self, current_state, depth, is_white_turn):
        """
        Calcula y ejecuta el movimiento para el jugador actual.
        Devuelve el nuevo estado del tablero o None si el juego termina.
        """
        # Paso 1: Obtener el mejor movimiento usando Minimax
        next_state = self.minimax(current_state, depth, depth, is_white_turn)

        # Paso 2: Comprobar si hay jaque mate (no hay movimientos posibles)
        if next_state is None:
            winner = "BLANCAS" if not is_white_turn else "NEGRAS"
            print(f"JAQUE MATE, GANAN LAS {winner}")
            # Devuelve None para señalar el fin del juego
            return None

        # Paso 3: Comprobar si hay tablas por repetición
        if self.isVisitedSituation(is_white_turn, self.copyState(next_state)):
            print("JUEGO EN TABLAS")
            # Devuelve None para señalar el fin del juego
            return None
        
        # Registramos la nueva situación para detectar repeticiones futuras
        self.listVisitedSituations.append((is_white_turn, self.copyState(next_state)))

        # Paso 4: Ejecutamos el movimiento en el tablero
        moved_piece, new_piece_pos = self.getMovement(current_state, self.copyState(next_state))
        self.chess.move((moved_piece[0], moved_piece[1]), (new_piece_pos[0], new_piece_pos[1]))
        
        print(f"Movimiento de las {'Blancas' if is_white_turn else 'Negras'}:")
        self.chess.board.print_board()

        # Paso 5: Devolvemos el nuevo estado del tablero
        return self.getCurrentState()


    # Actua de manager de la partida
    def minimaxGame(self, depthWhite, depthBlack, playerTurn):
        """
        Inicia y gestiona la partida de ajedrez usando el algoritmo Minimax.
        """

        # Paso 1: Preparar el escenario
        current_state = self.getCurrentState()
        print("Estado inicial del tablero:")
        self.chess.board.print_board()

        # Paso 2: Iniciar el bucle del juego
        while True:
            # Paso 3: Comprobar si la partida ya ha terminado (estado CheckMate)
            if self.isCheckMate(current_state):
                # Determinar el ganador basado en quién no puede moverse
                winner = "NEGRAS" if playerTurn else "BLANCAS"
                print(f"JAQUE MATE, GANAN LAS {winner}")
                break # Rompemos el bucle y termina el juego

            # Paso 4: Decidimos que jugador mueve y con qué profundidad
            depth = depthWhite if playerTurn else depthBlack
            
            # Paso 5: Realizar el turno del jugador actual
            new_state = self._perform_turn(current_state, depth, playerTurn)

            # PAso 6: Revisamos si el turno resultó en el fin de juego
            if new_state is None:
                # El juego terminó (por jaque mate o tablas), salir del bucle
                break

            # Paso 7: Actualizamos el estado actual del tablero
            current_state = new_state
            # Paso 8: Intercambiamos turno al siguiente jugador
            playerTurn = not playerTurn

    # Calcula la mejor jugada posible
    def minimax(self, state, depth, depthColor, playerTurn):
        
        # Caso base: si hemos llegado a la profundidad máxima o a un estado de jaque mate
        if depth == 0 or self.isCheckMate(state):
            return self.heuristica(state, playerTurn)
        
        # Variable para almacenar el mejor estado encontrado
        maxState = None

        # Lógica para el turno del jugador maximizador (Blancas)
        if playerTurn:
            currBestValue = float('-inf')

            # Obtenemos los estados de las piezas para generar movimientos
            blackState = self.getBlackState(state)
            whiteState = self.getWhiteState(state)
            brState = self.getPieceState(state, 8)

            # Vemos los sucesores (cada posible movimiento) solo para los estados en Blanco
            for successor in self.getListNextStatesW(whiteState):
                successor += self.eliminarBlack(blackState, brState, successor)

                # Consideramos solo los estados que no ponen al rey blanco en jaque
                if not self.isWatchedWk(successor):
                    # Realizamos una llamada recursiva para minimax con el estado sucesor
                    bestValue = self.minimax(successor, depth - 1, depthColor, False)
                    # Comprobamos si encontramos un mejor valor y un mejor movimiento
                    if bestValue > currBestValue:
                        currBestValue = bestValue
                        maxState = successor

        # Lógica del turno de las negras
        else:
            # Obtenemos los estados de las piezas
            currBestValue = float('inf')
            whiteState = self.getWhiteState(state)
            blackState = self.getBlackState(state)
            wrState = self.getPieceState(state, 2)

            # Exploramos los sucesores y cada posible movimiento de las negras
            for successor in self.getListNextStatesB(blackState):
                successor += self.eliminarWhite(whiteState, wrState, successor)

                # Consideramos los movimientos legales
                if not self.isWatchedBk(successor):
                    # Realizamos una llamada recursiva para minimax con el estado sucesor
                    bestValue = self.minimax(successor, depth - 1, depthColor, True)

                    # Actualizamos el mejor valor y maxState si encontramos un mejor sucesor
                    if bestValue < currBestValue:
                        currBestValue = bestValue
                        maxState = successor

        # Si volvemos al nivel superior, devolvemos el mejor movimiento
        if depth == depthColor:
            return maxState

        return currBestValue
    
    #### EX1 - MINIMAXGAME() ####

    ### EX3 - ALPHABETAPODA ####

    def alphaBetaPoda(self, depthWhite,depthBlack):
        """
        Gestiona una partida donde las Blancas usan Minimax y las Negras usan Alfa-Beta.
        """
        current_state = self.getCurrentState()
        playerTurn = True  # Empiezan las blancas
        print("Iniciando partida: Minimax (Blancas) vs Alfa-Beta (Negras)")
        self.chess.board.print_board()

        while not self.isCheckMate(current_state):
            depth = depthWhite if playerTurn else depthBlack
            
            # Decidir qué algoritmo usar
            if playerTurn: # Turno de las Blancas -> Minimax (sin poda)
                print("Turno de las Blancas (Minimax)...")
                # Llamamos a la función recursiva sin activar la poda para este turno
                next_state = self._minimax_recursive(current_state, depth, depth, True, False, -float('inf'), float('inf'))
            else: # Turno de las Negras -> Alfa-Beta (con poda)
                print("Turno de las Negras (Alfa-Beta)...")
                # Llamamos a la función recursiva activando la poda para este turno
                next_state = self._minimax_recursive(current_state, depth, depth, False, True, -float('inf'), float('inf'))

            # --- Lógica de fin de partida y actualización del tablero ---
            if next_state is None:
                winner = "NEGRAS" if playerTurn else "BLANCAS"
                print(f"JAQUE MATE, GANAN LAS {winner}")
                return

            if self.isVisitedSituation(playerTurn, self.copyState(next_state)):
                print("JUEGO EN TABLAS")
                return
            
            self.listVisitedSituations.append((playerTurn, self.copyState(next_state)))

            moved_piece, new_piece_pos = self.getMovement(current_state, self.copyState(next_state))
            self.chess.move((moved_piece[0], moved_piece[1]), (new_piece_pos[0], new_piece_pos[1]))
            self.chess.board.print_board()

            current_state = self.getCurrentState()
            playerTurn = not playerTurn
        
        # Si el bucle termina, es porque el jugador actual está en jaque mate
        winner = "NEGRAS" if playerTurn else "BLANCAS"
        print(f"JAQUE MATE, GANAN LAS {winner}")
        
    ### EX3 - ALPHABETAPODA ####

    ### EX5 - expectimax ####
    def expectimax(self, depthWhite, depthBlack, playerTurn):
        """
        Ejecuta una partida usando el algoritmo Expectimax para ambos jugadores.
        """
        current_state = self.getCurrentState()
        print("Estado inicial de las piezas:", current_state)

        while not self.isCheckMate(self.copyState(current_state)):
            # Elegimos la profundidad según el turno
            depth = depthWhite if playerTurn else depthBlack

            # Calculamos el siguiente movimiento usando expectimax
            next_state = self.expectimax_recursive(current_state, depth, depth, playerTurn)

            print("Movimiento seleccionado:", next_state)
            if next_state is None:
                ganador = "BLANCAS" if not playerTurn else "NEGRAS"
                print(f"JAQUE MATE, GANAN LAS {ganador}")
                return

            if self.isVisitedSituation(playerTurn, next_state):
                print("JUEGO EN TABLAS")
                return

            self.listVisitedSituations.append((playerTurn, self.copyState(next_state)))
            pieza, nueva_pos = self.getMovement(current_state, self.copyState(next_state))
            self.chess.move((pieza[0], pieza[1]), (nueva_pos[0], nueva_pos[1]))
            self.chess.board.print_board()
            current_state = self.getCurrentState()
            playerTurn = not playerTurn

    def expectimax_recursive(self, state, depth, depthColor, is_white_turn):
        """
        Algoritmo expectimax recursivo.
        """
        if depth == 0 or self.isCheckMate(state):
            return self.heuristica(state, is_white_turn)

        mejor_estado = None

        if is_white_turn:
            mejor_valor = float('-inf')
            blackState = self.getBlackState(state)
            whiteState = self.getWhiteState(state)
            brState = self.getPieceState(state, 8)

            for sucesor in self.getListNextStatesW(whiteState):
                sucesor += self.eliminarBlack(blackState, brState, sucesor)
                if not self.isWatchedWk(sucesor):
                    valor = self.expectimax_recursive(sucesor, depth - 1, depthColor, False)
                    if valor > mejor_valor:
                        mejor_valor = valor
                        mejor_estado = sucesor

            if depth == depthColor:
                return mejor_estado
            return mejor_valor

        else:
            valores = []
            whiteState = self.getWhiteState(state)
            blackState = self.getBlackState(state)
            wrState = self.getPieceState(state, 2)

            for sucesor in self.getListNextStatesB(blackState):
                sucesor += self.eliminarWhite(whiteState, wrState, sucesor)
                if not self.isWatchedBk(sucesor):
                    valor = self.expectimax_recursive(sucesor, depth - 1, depthColor, True)
                    if valor is not None:
                        valores.append(valor)

            if valores:
                esperado = sum(valores) / len(valores)
            else:
                esperado = float('-inf')

            if depth == depthColor:
                # Elegimos el sucesor con valor esperado más alto
                idx = valores.index(max(valores)) if valores else None
                return self.getListNextStatesB(blackState)[idx] + self.eliminarWhite(whiteState, wrState, self.getListNextStatesB(blackState)[idx]) if idx is not None else None

            return esperado

    def expectimax_vs_alphabeta(self, depthWhite, depthBlack, playerTurn):
        """
        Ejecuta una partida donde las blancas usan Expectimax y las negras Alfa-Beta.
        """
        current_state = self.getCurrentState()
        print("Estado inicial del tablero:")
        self.chess.board.print_board()

        while True:
            # Comprobar si la partida terminó
            if self.isCheckMate(current_state):
                winner = "NEGRAS" if playerTurn else "BLANCAS"
                print(f"JAQUE MATE, GANAN LAS {winner}")
                return winner

            depth = depthWhite if playerTurn else depthBlack

            if playerTurn:  # Blancas: Expectimax
                next_state = self.expectimax_recursive(current_state, depth, depth, True)
            else:           # Negras: Alfa-Beta
                next_state = self._minimax_recursive(current_state, depth, depth, False, True, -float('inf'), float('inf'))

            if next_state is None:
                winner = "NEGRAS" if playerTurn else "BLANCAS"
                print(f"JAQUE MATE, GANAN LAS {winner}")
                return winner

            if self.isVisitedSituation(playerTurn, self.copyState(next_state)):
                print("JUEGO EN TABLAS")
                return "TABLAS"

            self.listVisitedSituations.append((playerTurn, self.copyState(next_state)))
            moved_piece, new_piece_pos = self.getMovement(current_state, self.copyState(next_state))
            self.chess.move((moved_piece[0], moved_piece[1]), (new_piece_pos[0], new_piece_pos[1]))
            self.chess.board.print_board()

            current_state = self.getCurrentState()
            playerTurn = not playerTurn
    ### EX5 - expectimax ####
        
    #### EX2 - CÁLCULO DE JUGADAS ####

    # En esta función se simula una partida completa sin imprimir nada por pantalla
    def simulate_game(self, depthWhite, depthBlack, playerTurn, max_moves=50):
        """
        Simula una partida de forma silenciosa y devuelve el ganador.
        'white', 'black', o 'draw'.
        """
        self.listVisitedSituations = []
        current_state = self.getCurrentState()
        move_count = 0

        while move_count < max_moves:
            if self.isCheckMate(current_state):
                return 'black' if playerTurn else 'white'

            depth = depthWhite if playerTurn else depthBlack
            next_state = self.minimax(current_state, depth, depth, playerTurn)

            if next_state is None:
                return 'black' if playerTurn else 'white'

            if self.isVisitedSituation(playerTurn, self.copyState(next_state)):
                return 'draw'
            
            self.listVisitedSituations.append((playerTurn, self.copyState(next_state)))

            # Actualizar el estado sin imprimir
            moved_piece, new_piece_pos = self.getMovement(current_state, self.copyState(next_state))
            self.chess.move((moved_piece[0], moved_piece[1]), (new_piece_pos[0], new_piece_pos[1]))
            
            current_state = self.getCurrentState()
            playerTurn = not playerTurn
            move_count += 1
        
        return 'draw' # Empate si se superan los movimientos máximos
    
    def _minimax_recursive(self, state, depth, depthColor, playerTurn, use_alphabeta, alpha, beta):
        """
        Minimax recursivo con o sin poda alfa-beta.
        Si use_alphabeta es True, activa la poda.
        """
        if depth == 0 or self.isCheckMate(state):
            return self.heuristica(state, playerTurn)

        best_state = None

        if playerTurn:
            max_eval = float('-inf')
            blackState = self.getBlackState(state)
            whiteState = self.getWhiteState(state)
            brState = self.getPieceState(state, 8)

            for successor in self.getListNextStatesW(whiteState):
                successor += self.eliminarBlack(blackState, brState, successor)
                if not self.isWatchedWk(successor):
                    eval = self._minimax_recursive(successor, depth - 1, depthColor, False, use_alphabeta, alpha, beta)
                    if isinstance(eval, list):  # Si eval es un estado, calcula su heurística
                        eval = self.heuristica(eval, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_state = successor
                    if use_alphabeta:
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
            if depth == depthColor:
                return best_state
            return max_eval
        else:
            min_eval = float('inf')
            whiteState = self.getWhiteState(state)
            blackState = self.getBlackState(state)
            wrState = self.getPieceState(state, 2)

            for successor in self.getListNextStatesB(blackState):
                successor += self.eliminarWhite(whiteState, wrState, successor)
                if not self.isWatchedBk(successor):
                    eval = self._minimax_recursive(successor, depth - 1, depthColor, True, use_alphabeta, alpha, beta)
                    if isinstance(eval, list):  # Si eval es un estado, calcula su heurística
                        eval = self.heuristica(eval, True)
                    if eval < min_eval:
                        min_eval = eval
                        best_state = successor
                    if use_alphabeta:
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
            if depth == depthColor:
                return best_state
            return min_eval
    
def run_and_plot_experiments(runs_per_combo=3):
    """
    Ejecuta simulaciones para varias profundidades, calcula los porcentajes de victoria
    de las blancas y muestra un gráfico con los resultados.
    """

    depths_to_test = [3, 4]
    results = {}
    
    print(f"Iniciando experimentos ({runs_per_combo} partidas por combinación)...")

    for dw in depths_to_test:
        for db in depths_to_test:
            config_key = f"W:{dw} vs B:{db}"
            white_wins = 0
            
            for i in range(runs_per_combo):
                # Crear un tablero y un juego nuevos para cada partida para asegurar un estado limpio
                TA = np.zeros((8, 8))
                TA[7][0] = 2; TA[7][5] = 6  # Piezas blancas
                TA[0][7] = 8; TA[0][5] = 12 # Piezas negras
                game = Aichess(TA, True)
                winner = game.simulate_game(dw, db, True)
            if winner == 'white':
                white_wins += 1
            
        # Calcular y guardar el porcentaje de victorias
        win_percentage = (white_wins / runs_per_combo) * 100
        results[config_key] = win_percentage
        print(f"  - {config_key}: {win_percentage:.1f}% de victorias para las blancas.")

    # --- Graficar los resultados ---
    labels = list(results.keys())
    values = list(results.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=['lightblue', 'lightgreen', 'blue', 'green'])
    
    ax.set_ylabel('Porcentaje de Victorias Blancas (%)')
    ax.set_title('Rendimiento de Minimax con Diferentes Profundidades')
    ax.set_ylim(0, 100)

    # Añadir etiquetas de porcentaje encima de cada barra
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 puntos de offset vertical
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.show()

# --- Simulación y gráfico ---
def run_expectimax_vs_alphabeta_experiments(runs_per_combo=3):
    """
    Simula partidas Expectimax (Blancas) vs Alfa-Beta (Negras) y grafica los resultados.
    """
    depths_to_test = [3, 4]
    results_white = {}
    results_black = {}
    results_draw = {}

    print(f"Simulando {runs_per_combo} partidas por combinación...")

    for dw in depths_to_test:
        for db in depths_to_test:
            config_key = f"W:{dw} vs B:{db}"
            white_wins = 0
            black_wins = 0
            draws = 0

            for i in range(runs_per_combo):
                TA = np.zeros((8, 8))
                TA[7][0] = 2; TA[7][5] = 6  # Blancas
                TA[0][7] = 8; TA[0][5] = 12 # Negras
                game = Aichess(TA, True)
                winner = game.expectimax_vs_alphabeta(dw, db, True)
                if winner == "BLANCAS":
                    white_wins += 1
                elif winner == "NEGRAS":
                    black_wins += 1
                else:
                    draws += 1

            total_games = white_wins + black_wins + draws
            if total_games == 0:
                total_games = 1  # evitar división por cero

            results_white[config_key] = (white_wins / total_games) * 100
            results_black[config_key] = (black_wins / total_games) * 100
            results_draw[config_key] = (draws / total_games) * 100
            print(f"  - {config_key}: Blancas {results_white[config_key]:.1f}%, Negras {results_black[config_key]:.1f}%, Tablas {results_draw[config_key]:.1f}%")

    # Graficar
    labels = list(results_white.keys())
    white_values = list(results_white.values())
    black_values = list(results_black.values())
    draw_values = list(results_draw.values())

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_white = ax.bar(x - width, white_values, width, label='Blancas', color='lightblue')
    bars_black = ax.bar(x, black_values, width, label='Negras', color='lightcoral')
    bars_draw = ax.bar(x + width, draw_values, width, label='Tablas', color='lightgray')

    ax.set_ylabel('Porcentaje de Resultados (%)')
    ax.set_title('Expectimax (Blancas) vs Alfa-Beta (Negras)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()

    for bars in [bars_white, bars_black, bars_draw]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.show()


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
    #aichess.minimaxGame(3,3,True)
    # Run exercise 2
    #run_and_plot_experiments(runs_per_combo=3)

    # Run exercise 5
    run_expectimax_vs_alphabeta_experiments(runs_per_combo=3)
    
