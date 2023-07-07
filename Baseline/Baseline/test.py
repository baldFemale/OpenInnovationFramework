# import networkx as nx
# import matplotlib.pyplot as plt
#
# # Create an empty graph
# G = nx.Graph()
#
# # Add nodes to the graph
# num_nodes = 10
# # G.add_nodes_from(range(num_nodes))
# G.add_nodes_from(["0000", "0001", "0010", "0100"])
# # Define the layer information for each node
# # layer_info = {
# #     0: 0,
# #     1: 1,
# #     2: 1,
# #     3: 2,
# #     4: 2,
# #     5: 3,
# #     6: 3,
# #     7: 3,
# #     8: 3,
# #     9: 3
# # }
#
# layer_info = {
#     "0000": 0,
#     "0001": 1,
#     "0010": 1,
#     "0100": 2
# }
#
# # Define the connections between nodes
# # connections = {
# #     0: [1, 2],
# #     1: [3],
# #     2: [4],
# #     3: [5],
# #     4: [6],
# #     5: [7],
# #     6: [8],
# #     7: [9]
# # }
# connections = {
#     "0000": ["0001", "0010", "0100"]
# }
#
# # Add edges to the graph based on the connections
# for node, neighbors in connections.items():
#     for neighbor in neighbors:
#         G.add_edge(node, neighbor)
#
# # Create a dictionary to store the node positions for each layer
# layer_positions = {layer: [] for layer in range(max(layer_info.values()) + 1)}
#
# # Assign node positions within each layer
# for node, layer in layer_info.items():
#     layer_positions[layer].append(node)
#
# # Calculate the x-coordinate for each node within its layer
# layer_width = 1.0 / (len(layer_positions) + 1)  # Width of each layer column
# node_positions = {}
# for layer, nodes in layer_positions.items():
#     x = (layer + 1) * layer_width  # x-coordinate for the layer
#     y_step = 1.0 / (len(nodes) + 1)  # Vertical spacing between nodes in the layer
#     for i, node in enumerate(nodes):
#         y = (i + 1) * y_step  # y-coordinate for the node
#         node_positions[node] = (x, y)
#
# # Set the node positions as node attributes
# nx.set_node_attributes(G, node_positions, 'pos')
#
# # Visualize the graph
# pos = nx.get_node_attributes(G, 'pos')
# nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
# plt.title("Network Visualization with Layers")
# plt.axis('off')  # Turn off the axis
# plt.show()


import networkx as nx
import matplotlib.pyplot as plt

# Create an empty graph
G = nx.Graph()

# Add nodes to the graph
num_nodes = 10
G.add_nodes_from(range(num_nodes))

# Define the layer information for each node
layer_info = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 3,
    7: 3,
    8: 3,
    9: 3
}

# Define the connections between nodes
connections = {
    0: [1, 2],
    1: [3],
    2: [4],
    3: [5],
    4: [6],
    5: [7],
    6: [8],
    7: [9]
}

# Add edges to the graph based on the connections
for node, neighbors in connections.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# Create a dictionary to store the node positions for each layer
layer_positions = {layer: [] for layer in range(max(layer_info.values()) + 1)}

# Assign node positions within each layer
for node, layer in layer_info.items():
    layer_positions[layer].append(node)

# Calculate the x-coordinate for each node within its layer
layer_width = 1.0 / (len(layer_positions) + 1)  # Width of each layer column
node_positions = {}
for layer, nodes in layer_positions.items():
    x = (layer + 1) * layer_width  # x-coordinate for the layer
    y_step = 1.0 / (len(nodes) + 1)  # Vertical spacing between nodes in the layer
    for i, node in enumerate(nodes):
        y = (i + 1) * y_step  # y-coordinate for the node
        node_positions[node] = (x, y)

# Set the node positions as node attributes
nx.set_node_attributes(G, node_positions, 'pos')

# Identify nodes without outward links
nodes_without_links = [node for node in G.nodes if node not in connections.keys()]

# Visualize the graph
pos = nx.get_node_attributes(G, 'pos')

# Set the node color for nodes without outward links to red
node_color = ['red' if node in nodes_without_links else 'lightblue' for node in G.nodes]
print(node_color)
nx.draw(G, pos, with_labels=True, node_color=node_color, node_size=500, edge_color='gray')
plt.title("Network Visualization with Layers")
plt.axis('off')  # Turn off the axis
plt.show()



