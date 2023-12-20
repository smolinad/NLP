
from __future__ import annotations
from typing import Any
from graphviz import Digraph
import numpy as np

"""
Defines an autograd node.
    Attributes: 
        - label: the name (optional) of the node.
        - value: the actual numerical  value.
        - children: The descendant nodes in the backpropagation of a computatioanl node.
        - backward: The back propagation function that calls chain rule of the parent operation.
        - operation: The operation which resulted in the value of the node.
        - grad: The value of the gradient.
"""
class AutoNode:

    def __init__(self, value, children:tuple=(), operation:str="", label=""):
        self.label = label
        self.value = value
        self.children = set(children)
        self.backward = lambda: None
        self.operation = operation
        self.grad = 0.

    def __neg__(self) -> AutoNode:
        return self * -1
    
    def __add__(self, other) -> AutoNode:
        other = other if isinstance(other, AutoNode) else AutoNode(other)
        res = AutoNode(self.value + other.value, children=(self, other), operation="+")

        def backward():
            self.grad += res.grad
            other.grad += res.grad

        res.backward = backward

        return res
    
    def __radd__(self, other): 
        return self + other
    
    def __sub__(self, other) -> AutoNode:
        return self + (-other)
    
    def __rsub__(self, other) -> AutoNode: 
        return other + (-self)
    
    def __mul__(self, other) -> AutoNode:
        other = other if isinstance(other, AutoNode) else AutoNode(other)
        res = AutoNode(self.value * other.value, children=(self, other), operation="*")

        def backward():
            self.grad += other.value * res.grad
            other.grad += self.value * res.grad

        res.backward = backward

        return res
    
    def __rmul__(self, other) -> AutoNode:
        return self * other
    
    def __pow__(self, other) -> AutoNode:
        assert isinstance(other, (int, float))
        res = AutoNode(self.value**other, children=(self, ), operation="pow")

        def backward():
            self.grad += other * ((self.value)**(other-1)) * res.grad

        res.backward = backward

        return res
    
    def __truediv__(self, other) -> AutoNode:
        return self * (other**-1)
    
    def __rtruediv__(self, other): 
        return other * self**-1
    
    def tanh(self) -> AutoNode:
        t = np.tanh(self.value)
        res = AutoNode(t, children=(self, ), operation="tanh")

        def backward():
            self.grad += (1 - t**2) * res.grad

        res.backward = backward

        return res
    
    def exp(self) -> AutoNode:
        e = np.exp(self.value)
        res = AutoNode(e, children=(self, ), operation="e")

        def backward():
            self.grad += res.value * res.grad

        res.backward = backward

        return res
    
    """Orders the computational tree with topological sort and calls backward propagation."""
    def propagate(self) -> None:
        self.grad = 1.
        ordering = []
        visited = set()
        
        def order(node:AutoNode):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    order(child)
                ordering.append(node)
        
        order(self)

        for n in reversed(ordering):
            n.backward()


    def __repr__(self) -> str:
        return f"AutoNode(\n\tvalue={self.value},\n\tgrad={self.grad}\n)"
    
    """Draws computational graph."""
    def draw_graph(self) -> Digraph:
        graph = Digraph(format="svg", graph_attr={'rankdir': "LR"})
        nodes, edges = set(), set()

        """Lists all nodes and edges"""
        def build(node:AutoNode):
            if node not in nodes:
                nodes.add(node)
                for child in node.children:
                    edges.add((child, node))
                    build(child)
        
        build(self)

        for n in nodes:
            """Creates AutoNode node in the graph visualization."""
            uid = str(id(n))
            graph.node(name=uid, label=f"{n.label}\l|val {round(n.value, 2)}\l|grad {round(n.grad, 3)}", shape="record")
            """Connects parent node with dummy subsequent operation node."""
            if n.operation:
                graph.node(name=uid+n.operation, label=n.operation)
                graph.edge(uid+n.operation, uid)

        """Connects the child to the operation of the parent node"""
        for n1, n2 in edges:
            graph.edge(str(id(n1)), str(id(n2)) + n2.operation)

        return graph

"""Defines a neuron"""
class Neuron:
    def __init__(self, num_inputs:int):
        self.w = [AutoNode(np.random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.bias = AutoNode(np.random.uniform(-1, 1))

    def __call__(self, x) -> Any:
        activation = sum((xi*wi for xi, wi in zip(x, self.w)), self.bias) 
        return activation.tanh()
    
    def params(self):
        return self.w + [self.bias]
    
    def __repr__(self) -> str:
        return f"Neuron(\n\tweights={self.w},\n\tbias={self.bias}\n)"
    
    
"""Defines a layer of neurons"""
class Layer:
    def __init__(self, num_inputs:int, num_neurons:int):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    
    def params(self):
        return [p for n in self.neurons for p in n.params()]

"""Defines a MLP."""
class MultiLayerPerceptron:
    def __init__(self, num_inputs:int, dims:list):
        size = [num_inputs] + dims
        self.layers = [Layer(num_inputs=size[i], num_neurons=size[i+1]) for i in range(len(dims))]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x 
        
    def params(self):
        return [p for l in self.layers for p in l.params()]
    
    def train(self, xs, ys):
        loss = np.inf
        while loss==np.inf or loss.value > 0.001:
            ypreds = [self(x) for x in xs]
            loss = sum((y - yout)**2 for y, yout in zip(ys, ypreds))
            print(f"Loss: {loss.value:.5f}")

            for p in self.params():
                p.grad = 0.

            loss.propagate()

            for p in self.params():
                descent = -0.01 * p.grad if loss.value < 1. else -0.1 * p.grad 
                p.value += descent 

    def predict(self, xs):
        return [self(x).value for x in xs]
    
        

