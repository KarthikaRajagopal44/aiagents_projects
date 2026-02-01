import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

class GraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_from_triples(self, triples):
        self.graph.clear()
        for triple in triples:
            if len(triple) == 3:
                s, p, o = triple
                self.graph.add_edge(str(s).title(), str(o).title(), label=str(p).lower())
        return self.graph

    def get_relevant_subgraph(self, query):
        context = []
        query_words = set(query.lower().split())
        for node in self.graph.nodes():
            if any(word in node.lower() for word in query_words) or node.lower() in query.lower():
                # Get Outgoing facts
                for neighbor in self.graph.neighbors(node):
                    context.append(f"{node} {self.graph[node][neighbor]['label']} {neighbor}")
                # Get Incoming facts
                for pred in self.graph.predecessors(node):
                    context.append(f"{pred} {self.graph[pred][node]['label']} {node}")
        return ". ".join(list(set(context)))

    def visualize(self):
        if not self.graph.nodes(): return None
        fig, ax = plt.subplots(figsize=(10, 7))
        pos = nx.spring_layout(self.graph, k=1.5)
        
        nx.draw(self.graph, pos, with_labels=True, node_size=2500, 
                node_color="#ADD8E6", font_size=9, font_weight="bold",
                edge_color="#888888", width=2, arrowsize=20, ax=ax)
        
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        return fig
