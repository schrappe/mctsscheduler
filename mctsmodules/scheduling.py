# Monte Carlo Tree Search object (each holding the a scheduling node as well as rollout metrics)
import copy

class MCTSnode(object):
    def __init__(self, plan):
        # Deep copies the plan so it is not simply referenced in memory
        self.plan = copy.deepcopy(plan)
        
        # Metrics
        self.bestFinish = 0
        self.averageFinish = 0
        self.rollouts = 0

class MCTSnodeUCT(object):
    def __init__(self, plan):
        # Deep copies the plan so it is not simply referenced in memory
        self.plan = copy.deepcopy(plan)
        self.parent = 0
        self.index = 0
        self.children = False

        # Metrics
        self.accumulatedFinish = 0
        self.numberVisits = 0

class MCTStree(object):
    def __init__(self):
        self.nodes = []
        self.size = len(self.nodes)