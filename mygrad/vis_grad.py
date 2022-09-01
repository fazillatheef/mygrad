from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes: # start building from the top. i.e. the final node
            nodes.add(v) # each node is added to the set
            for child in v.prev: # if there is no more children no edge needed
                edges.add((child, v)) # here v is the parent
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='png', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) 
    
    for n in nodes:
        # node for the actual gtype
        dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad %.4f }" % (n.label,n.data, n.grad), shape='record')
        # extra node is created for the op and edge is connected with the current node
        # id of the node plus the op is used for the node id
        if n.op:
            dot.node(name=str(id(n)) + n.op, label=n.op)
            dot.edge(str(id(n)) + n.op, str(id(n)))
    
    # all edges between all the children and op is created
    for n1, n2 in edges: # here n2 is the parent
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)
    
    return dot