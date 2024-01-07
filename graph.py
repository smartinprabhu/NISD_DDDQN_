import pygraphviz as pgv

# Create a directed graph for the workflow diagram
G = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', fontsize=12)

# Define node styles and attributes
node_attributes = {
    'shape': 'box',
    'style': 'rounded,filled',
    'fillcolor': 'lightyellow',
    'fontname': 'Helvetica',
    'fontsize': 8,
    'fontcolor': 'black',
}

# Add nodes (states) with custom attributes
G.add_node("Input State (s)", **node_attributes)
G.add_node("Convolutional Neural Network", **node_attributes)
G.add_node("Feature Vector (f)", **node_attributes)
G.add_node("Dueling Network", **node_attributes)
G.add_node("Value Stream (V)", **node_attributes)
G.add_node("Advantage Stream (A)", **node_attributes)
G.add_node("Q-Values (Q)", **node_attributes)
G.add_node("Select Action with Maximum Q-Value (a)", **node_attributes)
G.add_node("Perform Action (a)", **node_attributes)
G.add_node("Observe Reward (r) and Next State (s')", **node_attributes)
G.add_node("Store Experience (s, a, r, s') in Replay Buffer", **node_attributes)
G.add_node("Sample Random Mini-Batch from Replay Buffer", **node_attributes)
G.add_node("Target Q-Network", **node_attributes)
G.add_node("Target Q-Values (Q')", **node_attributes)
G.add_node("Calculate Target Q-Values using Bellman Equation", **node_attributes)
G.add_node("Calculate Loss using Mean Squared Error", **node_attributes)
G.add_node("Backpropagate the Loss to Update Q-Network Weights", **node_attributes)
G.add_node("Update Target Q-Network Weights with Q-Network Weights", **node_attributes)

# Add edges (relations)
G.add_edge("Input State (s)", "Convolutional Neural Network")
G.add_edge("Convolutional Neural Network", "Feature Vector (f)")
G.add_edge("Feature Vector (f)", "Dueling Network")
G.add_edge("Dueling Network", "Value Stream (V)")
G.add_edge("Dueling Network", "Advantage Stream (A)")
G.add_edge("Value Stream (V)", "Q-Values (Q)")
G.add_edge("Advantage Stream (A)", "Q-Values (Q)")
G.add_edge("Q-Values (Q)", "Select Action with Maximum Q-Value (a)")
G.add_edge("Select Action with Maximum Q-Value (a)", "Perform Action (a)")
G.add_edge("Perform Action (a)", "Observe Reward (r) and Next State (s')")
G.add_edge("Observe Reward (r) and Next State (s')", "Store Experience (s, a, r, s') in Replay Buffer")
G.add_edge("Store Experience (s, a, r, s') in Replay Buffer", "Sample Random Mini-Batch from Replay Buffer")
G.add_edge("Sample Random Mini-Batch from Replay Buffer", "Target Q-Network")
G.add_edge("Target Q-Network", "Target Q-Values (Q')")
G.add_edge("Target Q-Values (Q')", "Calculate Target Q-Values using Bellman Equation")
G.add_edge("Q-Values (Q)", "Calculate Loss using Mean Squared Error")
G.add_edge("Calculate Target Q-Values using Bellman Equation", "Calculate Loss using Mean Squared Error")
G.add_edge("Calculate Loss using Mean Squared Error", "Backpropagate the Loss to Update Q-Network Weights")
G.add_edge("Backpropagate the Loss to Update Q-Network Weights", "Update Target Q-Network Weights with Q-Network Weights")

# Set graph attributes
G.graph_attr['rankdir'] = 'TB'  # Top to bottom layout

# Save the diagram to a file and render it
file_path = 'reinforcement_learning_workflow.png'
G.draw(file_path, format='png', prog='dot')

print("Workflow diagram saved to:", file_path)
