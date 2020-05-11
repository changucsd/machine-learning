from Node import Node
#
# authors:
# Ziqiao Gao 2157371827
# He Chang 5670527576
# Fanlin Qin 5317973858
#
import math

class DecisionTree:
    data = []
    root = None

    def __init__(self, data):
        self.data = data

    # build tree
    def build(self, dataset, last_answer):
        # calculate the entropy of the data set first
        label_list = []
        for data in dataset:
            label_list.append(data["Enjoy"])
        entropy_dataset = self.calEntropy(label_list)
        # termination: entropy == 0, has unanimous answer
        if entropy_dataset == 0:
            print("termination 1")
            print(last_answer)
            print("--------")
            print()
            return Node(question=label_list[0], last_answer=last_answer, isLeaf=True, children=[])
        # termination: run out of attrs
        if len(dataset[0]) == 2:
            # get all possible choice of attrs
            result_ = {}
            attr_name = ""
            for attr in dataset[0]:
                if attr != "Enjoy":
                    attr_name = attr
            for data in dataset:
                result_[data[attr_name]] = data["Enjoy"]
            children_ = []
            for r in result_:
                childNode = Node(question=result_[r], last_answer=r, isLeaf=True, children=[])
                children_.append({r: childNode})
            node = Node(question=attr_name, last_answer=last_answer, isLeaf=False, children=children_)
            print("termination 2")
            print(last_answer)
            print("--------")
            print()
            return node
        # calculate each attr entropy in the data set
        attrs_entropy = []
        for key in dataset[0]:
            # key is the attr name
            # get all possible choice of this atrr
            # then calculate the entropy of each choice
            # then we will have the avg entropy as the entropy of this attr
            if key == "Enjoy":
                continue
            possible_choices = {} #set choice: [entropy, count of choices]
            for data in dataset:
                possible_choices[data[key]] = 0
            for choice in possible_choices:
                label_list = []
                for data in dataset:
                    if data[key] == choice:
                        label_list.append(data["Enjoy"])
                possible_choices[choice] = [self.calEntropy(label_list),len(label_list)]
            #cal avg entropy
            avg_entropy = 0
            for choice in possible_choices:
                avg_entropy += possible_choices[choice][0] * possible_choices[choice][1]
            avg_entropy /= len(dataset)
            attrs_entropy.append({
                "question": key,
                "entropy": avg_entropy
            })
        # calculate max info gain
        max_gain = -100
        max_attr = ""
        for attr_entropy in attrs_entropy:
            info_gain = entropy_dataset - attr_entropy["entropy"]
            print(last_answer)
            print(attr_entropy["question"])
            print(info_gain)
            print()
            if max_gain < info_gain:
                max_attr = attr_entropy["question"]
                max_gain = info_gain
        # the max attr would be the node we pick from this data set
        node_question = max_attr
        # produce new dataset and recursively pick node until termination
        # new dataset will not contain the node attr
        possible_choices = {}
        childrenNode = []
        for data in dataset:
            possible_choices[data[node_question]] = 0
        for choice in possible_choices:
            new_dataset = []
            for data in dataset:
                if data[node_question] == choice:
                    new_dataset_row = {}
                    for d in data:
                        if d != node_question:
                            new_dataset_row[d] = data[d]
                    new_dataset.append(new_dataset_row)
            childNode = self.build(new_dataset, choice)
            childrenNode.append({choice:childNode})
        node = Node(node_question, childrenNode, last_answer, False)
        return node

    def printTree(self):
        self.print_tree(self.root, childattr='children', nameattr='question')
        return

    def print_tree(self, current_node, childattr='children', nameattr='name', indent='', last='updown'):

        name = lambda node: node.question

        children = lambda node: node.children

        """ Creation of balanced lists for "up" branch and "down" branch. """
        up = sorted(children(current_node), key=lambda node: self.nb_children(node))

        """ Printing of "up" branch. """
        for child in up:
            next_last = 'up' if up.index(child) is 0 else ''
            next_indent = '{0}{1}{2}'.format(indent, ' ' if 'up' in last else '│', ' ' * len(name(current_node)))
            child_ = list(child.values())
            self.print_tree(child_[0], childattr, nameattr, next_indent, next_last)

        """ Printing of current node. """
        if last == 'up':
            start_shape = '┌'
        elif last == 'down':
            start_shape = '└'
        elif last == 'updown':
            start_shape = ' '
        else:
            start_shape = '├'

        if up:
            end_shape = '┤'
        else:
            end_shape = ''

        print('{0}{1}{4}-{2}{3}'.format(indent, start_shape, name(current_node), end_shape, current_node.last_answer))

    def nb_children(self, node):
        if node is None:
            return 0
        if not hasattr(node, "children"):
            return 1
        if node.children == []:
            return 1
        sum = 0
        for child_dict in node.children:
            val_list = list(child_dict.values())
            self.nb_children(val_list[0])
            sum += 1
        return sum

    # use current Decision Tree to predict Test Result
    def predict(self, testdata):
        return self.predict_helper(self.root, testdata)

    def predict_helper(self, root, data):
        if root.isLeaf:
            return root
        else:
            next = ""
            for child in root.children:
                for n in child:
                    if n == data[root.question]:
                        next = child[n]
            return self.predict_helper(next, data)

    # calculate entropy of a certain data set
    # @label_list: list of the result label(Enjoy) in certain data set
    def calEntropy(self, label_list):
        n_count = 0
        y_count = 0
        for label in label_list:
            if label == "No":
                n_count += 1
            elif label == "Yes":
                y_count += 1
        n_p = n_count / len(label_list)
        y_p = 1 - n_p
        if n_p == 0 or y_p == 0:
            return 0
        entropy = - n_p * math.log2(n_p) - y_p * math.log2(y_p)
        return entropy

