from pyvis.network import Network
import networkx as nx
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class GraphVisualizer:

    def __init__(self):
        self.output_path = str(OUTPUT_DIR / "knowledge_graph.html")

    def render(self, graph: nx.DiGraph, height: str = "600px") -> str:
        net = Network(
            height=height,
            width="100%",
            bgcolor="#0f172a",
            font_color="#e2e8f0",
            directed=True
        )

        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 150,
                    "springConstant": 0.08
                },
                "solver": "forceAtlas2Based",
                "stabilization": { "iterations": 150 }
            },
            "edges": {
                "color": { "color": "#475569" },
                "smooth": { "type": "continuous" },
                "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } },
                "font": { "size": 10, "color": "#94a3b8" }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100
            }
        }
        """)

        # Add nodes
        for node, attrs in graph.nodes(data=True):
            color     = attrs.get("color", "#ffffff")
            size      = attrs.get("size",  20)
            node_type = attrs.get("node_type", "unknown")

            net.add_node(
                node,
                label=node,
                color=color,
                size=size,
                title=f"<b>{node}</b><br>Type: {node_type}",
                font={"size": 12, "color": "#e2e8f0"}
            )

        # Add edges
        for src, dst, attrs in graph.edges(data=True):
            label = attrs.get("label", "")
            net.add_edge(
                src, dst,
                title=label,
                label=label,
                color="#475569"
            )

        net.save_graph(self.output_path)
        logger.info(f"Graph saved to {self.output_path}")
        return self.output_path

    def render_html_string(self, graph: nx.DiGraph) -> str:
        path = self.render(graph)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()