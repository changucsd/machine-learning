#
# authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#

class Node:

    def __init__(self, question, children, last_answer, isLeaf = False):
        self.question = question
        self.last_answer = last_answer
        self.children = children
        self.isLeaf = isLeaf

    def __str__(self):
        return str(self.question)
