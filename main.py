#----------------------------------------------------------------------------------------------
# Monte Carlo Tree Search Algorithm Applied to Industrial Scheduling
# Application developed by Bruno Schrappe as a term project for
# SCS_3547_006 Intelligent Agents and Reinforcement Learning - Prof. Larry Simon - August 2020
#----------------------------------------------------------------------------------------------

import json
import copy
import time
import random
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename  
from tkinter import messagebox
import math

from mctsmodules import helpers
from mctsmodules.scheduling import MCTSnode, MCTSnodeUCT, MCTStree

# Variables with global scope
tasks = []
orders = []
bestSoFar = []
currentOrder = 0
iterations = 0
finishKPI = 0
simulationCount = 0
headerHeight = 50
nodeIndex = 0
mctsNodeFocus = 0
mctsNodeExpand = False


#-------------------------------
#  Exhaustive Search Scheduler
#-------------------------------

# Examines ALL possible nodes or scheduling states
def treescheduler():
    global tasks
    global iterations
    global bestSoFar
    global finishKPI
    bestSoFar = []
    iterations = 0
    finishKPI = 0

    # As an NP-hard problem, it quickly goes to an intractable number of possible states
    if len(tasks) > 13:
        messagebox.showwarning('Warning', 'Too many states for exhaustive search!')
    else:    
        iterations = 0
        # Copies the current list of tasks to be planned
        currentPlan = copy.deepcopy(tasks)
    
        fullTreeSearchAgent(currentPlan)
                
        # And then plots them all
        es.set("Simulations: " + str(iterations).zfill(6))
        #plotGantt(optimizedSchedule)
        plotGantt(bestSoFar)     
        populateMetrics(bestSoFar)
        master.update()    
    return

# Recursive exhaustive tree search
def fullTreeSearchAgent(currentPlanFull):
    global bestSoFar
    global finishKPI
    global iterations
    
    currentSchedulePlanCopy = copy.deepcopy(currentPlanFull)
    for task in possibleScheduleList(currentSchedulePlanCopy):
        newSchedule = scheduleThisTask(task, currentSchedulePlanCopy)
        
        if len(possibleScheduleList(newSchedule)) == 0:
            #lastTaskFinish = helpers.lastFinishMetric(newSchedule)
            iterations += 1
            lastTaskFinish = max(newSchedule, key=lambda x:x['finish'])['finish']
            if finishKPI == 0:
                finishKPI = lastTaskFinish
                bestSoFar = copy.deepcopy(newSchedule)
            elif lastTaskFinish < finishKPI:
                finishKPI = lastTaskFinish
                bestSoFar = copy.deepcopy(newSchedule)
        else:
            fullTreeSearchAgent(newSchedule)
        
    return

#----------------------------------------
#  Basic Operations Research Scheduler
#----------------------------------------

# Basic scheduler picks up the first available task from a list of ready-to-schedule tasks and schedules it (basic operations research technique)
def hwscheduler():
    global tasks
    # Schedules all tasks
    currentPlan = copy.deepcopy(tasks)
    hwSchedule = scheduleAll(currentPlan)
    plotGantt(hwSchedule)
    populateMetrics(hwSchedule)
    return

# Recursive scheduler that picks the earliest possible task from a list, schedules it an reloops until no more tasks are available
def scheduleAll(currentSchedulePlan):
    global finalPlan
    finalPlan = currentSchedulePlan
    readyTasks = possibleScheduleList(currentSchedulePlan)
    if len(readyTasks) > 0:
        # Choose the first possible task to schedule
        taskToSchedule = min(readyTasks, key=lambda x:x['earliest_start'])
        currentSchedulePlan = scheduleThisTask(taskToSchedule, currentSchedulePlan)
        currentSchedulePlan = scheduleAll(currentSchedulePlan)    
    return finalPlan


# Given the current schedule plan, return tasks that are ready for scheduling (no dependencies), with earliest_start dates
def possibleScheduleList(currentSchedulePlan):
    
    # Pruning parameter - pruning is performed by selecting orders able to start within the interval from the earliest possible start task to maxHorizon
    maxHorizon = horizonSlider.get()
    
    latestTaskTime = 0
    readyTasks = []
    for i in range(0, len(currentSchedulePlan)):
        if currentSchedulePlan[i]['earliest_start'] is None:
            # Look at predecessor finish date
            predecessor = currentSchedulePlan[i]['predecessor']
            for predTask in currentSchedulePlan:
                if predTask['index'] == predecessor:
                    # Found predecessor, see if it is finished. If so, task is good to go
                    if predTask['finish'] is None:
                        break
                    else:
                        # There is a predecessor, but it is finished, so the task can start soon
                        currentSchedulePlan[i]['earliest_start'] = predTask['finish']
                        readyTasks.append(currentSchedulePlan[i])
                    
        elif currentSchedulePlan[i]['finish'] is None:
            # There is a time constraint but the task is available for scheduling. Spit it out.
            readyTasks.append(currentSchedulePlan[i])
    
    # Alpha-beta pruning: order by date they can be scheduled and select only tasks with earliest start dates within the horizon
    if len(readyTasks) > 0:
        timeSortedTasks = sorted(readyTasks, key=lambda k: k['earliest_start'])
        earliestTaskTime = timeSortedTasks[0]['earliest_start']
        latestTaskTime = earliestTaskTime + maxHorizon
        prunedTasks = list(filter(lambda f: f['earliest_start'] <= latestTaskTime, timeSortedTasks))
    else:
        prunedTasks = readyTasks
    return(prunedTasks)


# Given a specific task and a copy of the current plan, schedule the task
def scheduleThisTask(task, currentSchedulePlan):
    
    # Need to deepcopy the array of lists, othewise only a reference is passed
    currentSchedulePlanCopy = copy.deepcopy(currentSchedulePlan)
    
    # Get the first task
    taskEarliestStart = task['earliest_start']
    taskIndex = task['index']
    taskDuration = task['duration']
    taskResource = task['resource']
    
    # Verifies the last allocated time for the resource this task is using by looping over all tasks (again!)
    lastAlloc = 0
    lastSequence = 0
    
    # Finds the last sequence number on tasks (needed to establish state)
    allScheduled = list(filter(lambda f: not f['sequence'] is None, currentSchedulePlanCopy))
    
    lastSequence = 0
    if allScheduled:
        lastSequence = max(allScheduled, key=lambda x:x['sequence'])['sequence']
    
    # Filter and sort operations scheduled for that task resource
    resourceTasks = sorted(list(filter(lambda f: f['resource'] == taskResource and not f['finish'] is None, currentSchedulePlanCopy)), key=lambda k: k['start'])
    
    # Compute open windows (if any) for the required resource within which the task could be scheduled
    openWindows = []
    startWindow = 0
    
    # Given sorted operations on the resource, compute its availability windows
    for aTask in resourceTasks:
        resourceInit = aTask['start']
        # There may be a window if the task starts only after the last time the resource was busy
        if resourceInit > startWindow:
            openWindows.append([startWindow, resourceInit])
        startWindow = aTask['finish']
        # Stores the end of the last task on this resource, just in case no window will fit
        lastAlloc = max(lastAlloc, startWindow)
    
    # Now fit the task within the first possible window
    foundWindow = False
    for scheduleWindow in openWindows:
        # Task could start before the open window closes
        if taskEarliestStart <= scheduleWindow[1]:
            # Now let's see if it fits there
            potentialStart = max(taskEarliestStart, scheduleWindow[0])
            if potentialStart + taskDuration <= scheduleWindow[1]:
                # Task fits into the window 
                minEarliestStart = potentialStart
                foundWindow = True
                break
    
    # If no window was found, schedule it after the end of the last task on the resource
    if not foundWindow:
        minEarliestStart = max(taskEarliestStart, lastAlloc)
            
    # Search the schedule plan, find the task and schedule it
    for i in range(0, len(currentSchedulePlanCopy)):
        if currentSchedulePlanCopy[i]['index'] == taskIndex:
            currentSchedulePlanCopy[i]['sequence'] = lastSequence + 1
            currentSchedulePlanCopy[i]['start'] = minEarliestStart
            currentSchedulePlanCopy[i]['finish'] = minEarliestStart + taskDuration
            break
    return currentSchedulePlanCopy


#---------------------------
#  Random Search Scheduler
#---------------------------

# Easy random scheduler that picks up any available task and schedules it
def randomscheduler():
    global tasks
    # Schedules all tasks
    currentPlan = copy.deepcopy(tasks)
    randomSchedulePlan = scheduleFullRandomPlan(currentPlan)
    plotGantt(randomSchedulePlan)
    populateMetrics(randomSchedulePlan)
    return


# Recursive function that randomly schedules a full plan based on task policy, also used for MCTS rollouts
def scheduleFullRandomPlan(currentSchedulePlan):
    global finalPlan
    finalPlan = currentSchedulePlan
    readyTasks = possibleScheduleList(currentSchedulePlan)
    if len(readyTasks) > 0:
        taskToSchedule = random.choice(readyTasks)
        currentSchedulePlan = scheduleThisTask(taskToSchedule, currentSchedulePlan)
        currentSchedulePlan = scheduleFullRandomPlan(currentSchedulePlan)
    return finalPlan



#----------------------------------------------------------------------
#  Monte Carlo Tree Search Scheduler Router (UCT or e-greedy variant)
#----------------------------------------------------------------------

# Launches MCTS scheduler with UCT or epsilon-greedy algorithms, depending on user selection
def montecarloscheduler():
    global finishKPI
    global simulationCount
    finishKPI = 0
    simulationCount = 0
    
    # Verifies the checkbox
    useEpsilonGreedy = useEgreedy.get()
    if useEpsilonGreedy:
        # Go with epsilon-greedy variant    
        montecarloschedulerEpsilonGreedy()
    else:
        # Launch MCTS with UCT controls
        montecarloschedulerUCT()


#----------------------------------------------------------------------------------------------
#  Monte Carlo Tree Search Scheduler With Pruning and UCT (Upper Confidence Bounds for Trees)
#----------------------------------------------------------------------------------------------

# Schedules the best possible task from a current plan state using MCTS with UCT
def montecarloschedulerUCT(currentPlan = tasks):
    global simulationCount
    global bestFinishOfAll
    global nodeIndex

    bestFinishOfAll = 0
    nodeIndex = 1

    # Retrieves the maximum number of rollouts per decision
    simulationDepth = maxRolloutSlider.get()

    # Copies the current state
    currentPlanCopy = copy.deepcopy(currentPlan)
 
    # Check if there is any task to be scheduled...
    possibleTasks = possibleScheduleList(currentPlanCopy)

    if len(possibleTasks) == 0:
        # Review this
        populateMetrics(currentPlanCopy)
        return
    else:
        # There is at least one task still to be scheduled
        # Creates the Monte Carlo tree object from the current state of the scheduling plan
        mctsTree = MCTStree()

        # Creates the first node with unscheduled tasks, parent is 0 (root level)
        rootNode = MCTSnodeUCT(currentPlanCopy)
        rootNode.parent = 0
        rootNode.index = nodeIndex
        mctsTree.nodes.append(rootNode)
    
        # Children can be added, so the parent will not be a leaf mode anymore
        rootNode.children = True
        # Schedule each task and add to the MCTS list, expanding it for the first time         
        mctsTree = expandMCTSnode(mctsTree, rootNode)

        # Iterates through the tree simulationDepth times. Not all of them will result in rollouts, because in some cases a cycle will be used to expand a node, leaving
        # the simulation/rollout step for the next cycle. Something to take care of later, but not required for our early experiments
        for _ in range(simulationDepth):
            mctsTree = MCTSagentUCTcycle(mctsTree)
    
        # Now that a full tree search has been completed, select the node below the initial root node (parent = 1 or first node)
        minAverage = float("inf")
        for node in mctsTree.nodes:
            # One of the decision nodes
            if node.parent == 1:
                average = node.accumulatedFinish / node.numberVisits
                if average < minAverage:
                    minAverage = average
                    bestNode = node

        optimizedPlan = bestNode.plan
        # And then plots it
        plotGantt(optimizedPlan)
        master.update()

        # Repeat until a terminal state (no more tasks to schedule) is reached
        montecarloschedulerUCT(optimizedPlan)


# Agent that executes a single MCTS tree search cycle from seletion, expansion if needed, simulation and backpropagation
def MCTSagentUCTcycle(mctsTree):
    global simulationCount
    global mctsNodeFocus
    global mctsNodeExpand

    #---------------
    # MCTS Selection
    # Traverses the tree starting from the root node (node index=1), with a policy based on UCT evaluation
    parentNodeIndex = 1

    # Start traversing the tree from the parent node of the current simulation
    # I am placing returns (mctsNodeFocus and mctsNodeExpand) on global variables for ease of implementation only.
    # Not proud of it and will do this right later if time permits. If you are reading this, I had no time... but it works just fine.
    mctsSelection(mctsTree, parentNodeIndex)

    #---------------
    # MCTS Expansion
    # Expansion is needed if the leaf node to process has been already sampled
    if mctsNodeExpand:
        mctsTree = expandMCTSnode(mctsTree, mctsNodeFocus)
        return mctsTree
    else:
        #---------------
        # MCTS Simulation
        # With a node to expand selected, do a rollout and update the tree
        # With the chosen node, do a rollout and compute metrics
        nodePlan = copy.deepcopy(mctsNodeFocus.plan)
        randomSimulatedPlan = scheduleFullRandomPlan(nodePlan)
        lastTaskFinish = max(randomSimulatedPlan, key=lambda x:x['finish'])['finish']
        simulationCount += 1
    
        # Updates the metric on screen
        v.set(simulationCount)
        cr.set(lastTaskFinish) 

        #---------------------
        # MCTS Backpropagation
        MCTSbackpropagate(mctsTree, mctsNodeFocus, lastTaskFinish)

        return mctsTree


# Returns a node to either rollout a full simulation for or expand (node + expansionFlag)
def mctsSelection(mctsTree, parentNodeIndex):
    global mctsNodeFocus
    global mctsNodeExpand

    # Initializes UCT
    minUCT = - float("inf")
    hasChildren = False

    # Scans all children nodes
    for node in mctsTree.nodes:
        if node.parent == parentNodeIndex:
            hasChildren = True
            # This is a node to evaluate
            if node.numberVisits == 0:
                # Unvisited child node, return it immediately for a rollout (simulation)
                mctsNodeFocus = node
                mctsNodeExpand = False
                return
            else:
                # Compute UCT metrics for the node
                nodeUCT = computeUCT(node, mctsTree.nodes)
                if nodeUCT > minUCT:
                    minUCT = nodeUCT
                    bestNode = node

    # If the parent node has children, it is not a leaf mode and the first unvisited child has been returned for a rollout or the child with best UCT was selected for expansion
    if hasChildren:
        nextParentIndex = bestNode.index
        # Drill down further
        mctsSelection(mctsTree, nextParentIndex)
    else:
        # Parent node has no children, so it is a leaf node that needs expansion
        # Selects the parent node and spits it back for expansion
        for node in mctsTree.nodes:
            if node.index == parentNodeIndex:
                parentNode = node
                mctsNodeFocus = parentNode
                mctsNodeExpand = True
                break
        return
        

# Expands a leaf node
def expandMCTSnode(mctsTree, mctsNode):
    global nodeIndex

     # Before calling the first MCTS iteration, since this is a root node from which an Action needs to be chosen, we need to populate the tree with possible actions
    possibleTasks = possibleScheduleList(mctsNode.plan)
    
    # Schedule each task and add to the MCTS list
    if len(possibleTasks) > 0 :            
        
        # Children will be added, so the node being expanded is obviously not a leaf node anymore
        mctsNode.children = True

        # Check every task on the current plan
        for task in possibleTasks:
            # Schedule the task and get a new plan
            taskScenario = scheduleThisTask(task, mctsNode.plan)
           
            # Upon scheduling the task, the plan becomes a new node and can be added to the list of MCTS sims
            newMctsNode = MCTSnodeUCT(taskScenario)
            newMctsNode.parent = mctsNode.index
            nodeIndex += 1
            newMctsNode.index = nodeIndex
            mctsTree.nodes.append(newMctsNode)
   
    return mctsTree


# Backpropagates results from a given Node to its parents on a Tree
def MCTSbackpropagate(mctsTree, mctsNode, result):
    mctsNode.numberVisits += 1
    mctsNode.accumulatedFinish += result
    parent = mctsNode.parent
    # Root node, no need to continue backpropagation
    if parent == 0:
        return
    else:
        for node in mctsTree.nodes:
            if node.index == parent:
                break
        # Found the parent node, backpropagate results to it as well
        MCTSbackpropagate(mctsTree, node, result)


# Computes MCTS UCT metrics for a given node in an MCTS tree
def computeUCT(node, nodes):
    # Pass this value later
    c = uctCSlider.get()

    # Find the parent first
    index = node.parent
    for parentNode in nodes:
        if parentNode.index == index:
            parentN = parentNode.numberVisits
            break
    
    # We are minimizing values my maximizing UCT, so Vi is negative
    UCT = - node.accumulatedFinish / node.numberVisits + c * math.sqrt(math.log(parentN)/node.numberVisits)

    return UCT


#-----------------------------------------------------------------------------
#  Monte Carlo Tree Search Scheduler With Pruning and epsilon-greedy Control
#-----------------------------------------------------------------------------

def montecarloschedulerEpsilonGreedy():
    global simulationCount
    global bestFinishOfAll
    
    simulationCount = 0
    bestFinishOfAll = 0

    simulationDepth = maxRolloutSlider.get()
    
    optimizedPlan = MCTS(tasks,simulationDepth)

    # And then plots them all
    plotGantt(optimizedPlan)
    populateMetrics(optimizedPlan)


# Agent that executes MCTS tree search
def MCTS(currentSchedulePlan, nSims):
    # Executes a single move on the current plan by searching on Monte Carlo trees
    global finishKPI
    global master
    global simulationCount
    global bestNodeOverall
    global bestFinishOfAll
    global bestFullSimulation

    # Get epsilon and patience parameters from corresponding UI controls
    epsilon = epsilonSlider.get()
    patience = patienceSlider.get()
    
    # Initializes the tree from the current plan
    mctsList = []

    noImprovement = 0
    
    currentSchedulePlanCopy = copy.deepcopy(currentSchedulePlan)
    
    # Gets the sequence of task to be scheduled
    allScheduled = list(filter(lambda f: not f['sequence'] is None, currentSchedulePlanCopy))
    if allScheduled:
        sequenceToSchedule = max(allScheduled, key=lambda x:x['sequence'])['sequence'] + 1
    else:
        sequenceToSchedule = 1

    # Computes list of potential tasks to schedule    
    possibleTasks = possibleScheduleList(currentSchedulePlanCopy)
    
    if len(possibleTasks) > 0 :            
        # Check every task on the current plan
        for task in possibleTasks:
            # run multiple scenarios
            taskScenario = scheduleThisTask(task, currentSchedulePlanCopy)
            # Upon scheduling the task, the plan becomes a new node and can be added to the list of MCTS sims
            newMCTSnode = MCTSnode(taskScenario)
            # Add the node to MCTS list
            mctsList.append(newMCTSnode)
        
        # Assign bestNode as the first for the time being
        bestNode = mctsList[0]
        
        # With MCTS list properly populated with nodes, run exploration/exploitation rollouts on them
        # until the maximum number of simulations is reached or 
        # patience is exhausted (number or rollouts with no improvement), then select the best node
        
        for _ in range(nSims):
            notVisited = []
            bestAverage = 0
            # Scans and selects a proper node from mctsList
            # Checks for nodes that have not been visited yet
            for node in mctsList:
                if node.rollouts == 0:
                    notVisited.append(node)
                if bestAverage == 0 and node.averageFinish > 0:
                    bestAverage = node.averageFinish
                elif node.averageFinish < bestAverage:
                    bestAverage = node.averageFinish
                    # Most promising node to rollout, unless we have unseen nodes
                    bestNode = node
            
            if len(notVisited) > 0:
                # Choose the first node not visited yet
                selectedNode = random.choice(notVisited)
            else:
                # Choose to explore or exploit based on epsilon
                if np.random.uniform(0, 1) < epsilon:
                    # We will explore possibilities - it is a whole new world out there
                    selectedNode = random.choice(mctsList)
                else:
                    # Select the most promising node to further explore
                    selectedNode = bestNode
            
            # With the chosen node, do a rollout
            nodePlan = copy.deepcopy(selectedNode.plan)
            randomSimulatedPlan = scheduleFullRandomPlan(nodePlan)
            lastTaskFinish = max(randomSimulatedPlan, key=lambda x:x['finish'])['finish']
                        
            # selectedNode is referencing the mcts list object, so it can be changed directly
            # Update average task finish time, from its last average and number of rollouts
            averageTaskFinish = (selectedNode.averageFinish * selectedNode.rollouts + lastTaskFinish) / (selectedNode.rollouts + 1)
            selectedNode.rollouts += 1
            selectedNode.averageFinish = averageTaskFinish
                        
            # Check the metrics for the entire simulation
            if finishKPI == 0:
                finishKPI = lastTaskFinish
                bestNodeOverall = copy.deepcopy(nodePlan)
                bestFullSimulation = copy.deepcopy(randomSimulatedPlan)
            elif lastTaskFinish < finishKPI:
                finishKPI = lastTaskFinish
                bestNodeOverall = copy.deepcopy(nodePlan)
                bestFullSimulation = copy.deepcopy(randomSimulatedPlan)
                noImprovement = 0
            else:
                noImprovement += 1
            
            # Updates the best rollout so far
            if bestFinishOfAll ==0 or finishKPI < bestFinishOfAll:
                bestFinishOfAll = finishKPI
                #br.set(str(bestFinishOfAll))
                
            simulationCount += 1
        
            # Break the loop if we have no improvemnets to the plan
            if noImprovement > patience:
                break
        
        # Extract the task for this sequence that resulted on the best full simulation so far...
        bestTask = list(filter(lambda f: f['sequence'] == sequenceToSchedule, bestFullSimulation))[0]
        
        optimizedPlan = scheduleThisTask(bestTask, currentSchedulePlanCopy)

        v.set(simulationCount)
        cr.set(finishKPI)       
        plotGantt(optimizedPlan)
        master.update() 
        
        mctsList.clear()
        
        # Here we go again...
        MCTS(optimizedPlan, nSims)
    
    # No additional tasks to schedule, returns final plan
    return bestFullSimulation


#----------------------------------
#  Graphical / Plotting Functions
#----------------------------------

# Plots all tasks passed on schedulePlan on a Gantt Chart
def plotGantt(schedulePlan):
    global headerHeight
    xStart = 100
    # Delete all elements on gantt canvas
    w.delete("all")
    w.create_text(5, headerHeight - 25 , font=("TkDefaultFont", 16), text="Gantt Chart ", anchor="w")
    
    # First, collects all machines
    resourceList = []
    for task in schedulePlan:
        resource = task['resource']
        if not(resource in resourceList):
            resourceList.append(resource)

    # Output all resources
    resourceList.sort()

    # Plot resources
    line = 0
    w.create_line(0, headerHeight , canvas_width, headerHeight, fill="#476042")
    
    for resource in resourceList:
        w.create_text(5, headerHeight + 25 + line * 50, text="Resource " + str(resource), anchor="w")
        w.create_line(0, headerHeight + (line + 1) * 50, canvas_width, headerHeight + (line + 1) * 50, fill="#476042")
        line += 1
        
    # Plot all tasks
    for task in schedulePlan:
        resource = task['resource']
        color = task['color']
        start = task['start']
        finish = task['finish']
        if not(start is None or finish is None):
            w.create_rectangle(xStart + start, headerHeight + 10 + (resource - 1) *50, xStart + finish, headerHeight + 40 + (resource - 1)*50, fill=color)
    return


# Plots single orders
def plotOrder():
    
    global currentOrder
    global orders
    
    # Delete all elements on order canvas
    oc.delete("all")

    canvasRow = 0
    
    totalOrders = len(orders)
    
    if currentOrder < 0:
        currentOrder = 0
    if currentOrder > totalOrders - 1:
        currentOrder = totalOrders - 1
    
    order = orders[currentOrder] 
    name = order['name']
    color = order['color']
    earliest_start = order['earliest_start']
    orderName.set(name + " of " + str(totalOrders))
    canvasRow += 1
    oc.create_text(5, 25 + (canvasRow - 1) * 50, text=name, font=('fremono', 12, 'bold'), anchor="w")
    oc.create_line(0, (canvasRow - 1) * 50, canvas_width, (canvasRow - 1) * 50, fill="#476042")        
    xStart = 100
        
    oc.create_line(xStart , 25 + (canvasRow)*50, xStart + earliest_start, 25 + (canvasRow)*50, fill="gray")
    
    for step in order['steps']:
        canvasRow += 1
        resourceName = "Resource " + str(step['resource'])
        duration = step['duration']
        predecessor = step['predecessor']
        
        # Earliest start time
        if predecessor is None:
            startTime = earliest_start
        else:
            for predCheck in order['steps']:
                if predCheck['step'] == predecessor:
                    startTime = predCheck['endTime']
                    break
        
        endTime = startTime + duration
        step['startTime'] = startTime
        step['endTime'] = endTime
            
        # Create line for steps
        oc.create_text(5, 25 + (canvasRow - 1) * 50, text=resourceName, anchor="w")
        oc.create_rectangle(xStart + startTime, 10 + (canvasRow-1)*50, xStart + endTime, 40 + (canvasRow-1)*50, fill=color)
              
    return


# Order display control: advance one
def increaseOrderAndPlot():
    global currentOrder
    currentOrder += 1
    plotOrder()


# Order display control: back to previous one
def decreaseOrderAndPlot():
    global currentOrder
    currentOrder -= 1
    plotOrder()


# Populates schedule plan metrics on the UI
def populateMetrics(currentPlan):
    lastTask, constraint, occupation = helpers.lastFinishMetric(currentPlan)
    lc.set(lastTask)
    taskCount = len(currentPlan)
    nt.set(taskCount)
    cres.set("Resource " + str(constraint))
    cocc.set("{:.0%}".format(occupation))
    master.update()
    return


def aboutMe():
     messagebox.showinfo('About', 'SCS_3547_006 Intelligent Agents and Reinforcement Learning\nFinal Project\n\nBruno Schrappe - July 2020')


#----------------------------
#  File Management - Orders
#----------------------------

# Functions for tkinter application menu
def OpenFile():
    global currentOrder
    
    fileName = askopenfilename()
    loadOrders(fileName)
    currentOrder = 0
    plotOrder()
    tabControl.select(tab1)
    

# Opens JSON file passed as argument with order data and populates global variables orders and tasks
def loadOrders(file):
    global orders
    global tasks
   
    # Just in case we are reloading tasks
    tasks.clear()
    orders.clear()
    f = open(file) 

    # returns JSON object as  a dictionary 
    data = json.load(f)
    f.close() 
    orders = data['orders']
    
    # General order of index
    stepIndex = 0
    
    for order in orders: 
        # Initial index of steps within order
        orderIndex = stepIndex
        name = order['name']
        color = order['color']
        earliestStart = order['earliest_start']
    
        for step in order['steps']:
            stepIndex += 1
            #orderStep = step['step']
            resource = step['resource']
            duration = step['duration']
            predecessor = step['predecessor']
        
            if not(predecessor is None):
                absPredecessor = predecessor + orderIndex
            
            task = {}
            # Sequence is the scheduling order, the series of which defines a State or Node.
            task['sequence'] = None
            task['index'] = stepIndex
            task['order'] = name
            task['color'] = color
            task['resource'] = resource
        
            if predecessor is None:
                task['predecessor'] = None
                task['earliest_start'] = earliestStart
            else:
                task['predecessor'] = absPredecessor
                task['earliest_start'] = None
        
            task['duration'] = duration
            task['start'] = None
            task['finish'] = None
           
            tasks.append(task)
    return


#---------------------------------------
#  Tkinter Setup - User Interface Only
#---------------------------------------

# Tk tabs and canvas setup
master = Tk()
master.title("MCTS Scheduler - Reinforcement Learning")


v = StringVar()
nt = StringVar()
lc = StringVar()
cr = StringVar()
es = StringVar()
cres = StringVar()
cocc = StringVar()
simtime = StringVar()
orderName = StringVar()
useEgreedy = IntVar()

# Menu configuration
menu = Menu(master)
master.config(menu=menu)
filemenu = Menu(menu)
menu.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="Load Orders", command=OpenFile)
filemenu.add_separator()
filemenu.add_command(label="About", command=aboutMe)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=master.quit)

# Definition of tabs                
tabControl = ttk.Notebook(master)

tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)
tabControl.add(tab1, text ='Orders') 
tabControl.add(tab2, text ='Scheduler') 
tabControl.pack(expand = 1, fill ="both") 


tk.Label(tab2, text="Scheduling Algorithms - Benchmarking:", bg="gray93").grid(row=1,column=4, sticky=W)
tk.Button(tab2, text="Random Scheduler!", fg="red", bg="gray93", command=randomscheduler).grid(row=2,column=4, sticky=W)
tk.Button(tab2, text="Basic Factory Scheduler!", fg="green", bg="gray93", command=hwscheduler).grid(row=3,column=4, sticky=W)
tk.Button(tab2, text="Exhaustive Search Scheduler!", bg="gray93", fg="blue", command=treescheduler).grid(row=4,column=4, sticky=W)
exhaustLabel = tk.Label(tab2, textvariable=es, bg="gray93")
exhaustLabel.grid(row=5,column=4, sticky=W)

tk.Button(tab2, text="Schedule with MCTS!", fg="black", bg="gray93", command=montecarloscheduler).grid(row=0,column=0, sticky=W)
epsilonCheck = tk.Checkbutton(tab2, text="Use ϵ-greedy Variant", variable = useEgreedy, bg="gray93")
epsilonCheck.grid(row=0, column=1, sticky=W)
useEgreedy.set(1)

tk.Label(tab2, text="MCTS Max Sims per Action:", bg="gray93").grid(row=1,column=0, sticky=W)
maxRolloutSlider = Scale(tab2, from_=2, to=100, resolution=10, orient=HORIZONTAL, bg="gray93")
maxRolloutSlider.grid(row=1,column=1, sticky=W)
maxRolloutSlider.set(50)

tk.Label(tab2, text="MCTS Pruning Max Horizon:", bg="gray93").grid(row=2,column=0, sticky=W)
horizonSlider = Scale(tab2, from_=0, to=500, resolution=10, orient=HORIZONTAL, bg="gray93")
horizonSlider.grid(row=2,column=1, sticky=W)
horizonSlider.set(300)

tk.Label(tab2, text="MCTS UCT C Parameter:", bg="gray93").grid(row=3,column=0, sticky=W)
uctCSlider = Scale(tab2, from_=10, to=2000, resolution=10, orient=HORIZONTAL, bg="gray93")
uctCSlider.grid(row=3,column=1, sticky=W)
uctCSlider.set(200)

tk.Label(tab2, text="MCTS ϵ-greedy Patience:", bg="gray93").grid(row=4,column=0, sticky=W)
patienceSlider = Scale(tab2, from_=2, to=50, orient=HORIZONTAL, bg="gray93")
patienceSlider.grid(row=4,column=1, sticky=W)
patienceSlider.set(15)

tk.Label(tab2, text="MCTS ϵ-greedy Epsilon:", bg="gray93").grid(row=5,column=0, sticky=W)
epsilonSlider = Scale(tab2, from_=0, to=1,  resolution=0.05, orient=HORIZONTAL, bg="gray93")
epsilonSlider.grid(row=5,column=1, sticky=W)
epsilonSlider.set(0.1)


# grid function returns none object, so if the labels needs to be referenced later, we need to break its definition as shown below
tk.Label(tab2, text="Total Simulations:", bg="gray93").grid(row=6,column=0, sticky=W)
simLabel = tk.Label(tab2, textvariable=v, bg="gray93")
simLabel.grid(row=6,column=1, sticky=W)

# grid function returns none object, so if the labels needs to be referenced later, we need to break its definition as shown below
tk.Label(tab2, text="Current Simulation KPI:", bg="gray93").grid(row=7,column=0, sticky=W)
simRollLabel = tk.Label(tab2, textvariable=cr, bg="gray93")
simRollLabel.grid(row=7,column=1, sticky=W)

tk.Label(tab2, text="Schedule KPIs", bg="gray93").grid(row=1,column=2, sticky=W)

tk.Label(tab2, text="Number of Tasks:", bg="gray93").grid(row=2,column=2, sticky=W)
metrics1Label = tk.Label(tab2, textvariable=nt, bg="gray93")
metrics1Label.grid(row=2,column=3, sticky=W)

tk.Label(tab2, text="Last Completed Task:", bg="gray93").grid(row=3,column=2, sticky=W)
metrics2Label = tk.Label(tab2, textvariable=lc, bg="gray93")
metrics2Label.grid(row=3,column=3, sticky=W)

tk.Label(tab2, text="Constraint Resource:", bg="gray93").grid(row=4,column=2, sticky=W)
metrics3Label = tk.Label(tab2, textvariable=cres, bg="gray93")
metrics3Label.grid(row=4,column=3, sticky=W)

tk.Label(tab2, text="Constraint Occupation:", bg="gray93").grid(row=5,column=2, sticky=W)
metrics4Label = tk.Label(tab2, textvariable=cocc, bg="gray93")
metrics4Label.grid(row=5,column=3, sticky=W)


# Canvas to draw Gantt chart, columnspan set to 5 so it will not push columns in the previous row
canvas_width = 2000
canvas_height = 800
w = Canvas(tab2, width=canvas_width, height=canvas_height)
w.grid(row=8, column=0, columnspan=5)

# Frame to display orders
orderFrame = tk.Frame(tab1, bg="gray93")
orderFrame.grid(row=0,column=0, sticky=W)
button5 = tk.Button(orderFrame, text="< Order", bg="gray93", command=decreaseOrderAndPlot)
button5.grid(row=0,column=0, sticky=W)
orderLabel = tk.Label(orderFrame, textvariable=orderName, bg="gray93")
orderLabel.grid(row=0,column=1)
button6 = tk.Button(orderFrame, text="Order >", bg="gray93", command=increaseOrderAndPlot)
button6.grid(row=0,column=2, sticky=E)

# Draw canvas on the orders tab
oc = Canvas(tab1, width=canvas_width, height=canvas_height)
oc.grid(row=1, column=0)

# Load default set of orders (can be changed by menu action)
loadOrders("orders/orders-default.json")
#Plot initial order
currentOrder=0
plotOrder()

# Launches Application
mainloop()
