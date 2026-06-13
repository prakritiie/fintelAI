import networkx as nx
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Sample financial knowledge data ─────────────────────────
COMPANY_DATA = {
    "Apple": {
        "ceo":         "Tim Cook",
        "industry":    "Technology",
        "sector":      "Consumer Electronics",
        "competitors": ["Microsoft", "Samsung", "Google"],
        "risks":       ["Supply Chain Risk", "Regulatory Risk", "Competition Risk"],
        "products":    ["iPhone", "Mac", "Services", "Wearables"],
    },
    "Tesla": {
        "ceo":         "Elon Musk",
        "industry":    "Automotive",
        "sector":      "Electric Vehicles",
        "competitors": ["Ford", "GM", "BYD", "Rivian"],
        "risks":       ["Production Risk", "Competition Risk", "Regulatory Risk"],
        "products":    ["Model S", "Model 3", "Model X", "Energy Storage"],
    },
    "Microsoft": {
        "ceo":         "Satya Nadella",
        "industry":    "Technology",
        "sector":      "Software & Cloud",
        "competitors": ["Apple", "Google", "Amazon"],
        "risks":       ["Cybersecurity Risk", "Regulatory Risk", "Competition Risk"],
        "products":    ["Azure", "Office 365", "Windows", "LinkedIn"],
    },
    "Berkshire Hathaway": {
        "ceo":         "Warren Buffett",
        "industry":    "Finance",
        "sector":      "Diversified Holdings",
        "competitors": ["BlackRock", "Vanguard", "JPMorgan"],
        "risks":       ["Market Risk", "Interest Rate Risk", "Regulatory Risk"],
        "products":    ["Insurance", "Railroads", "Energy", "Manufacturing"],
    },
    "Google": {
        "ceo":         "Sundar Pichai",
        "industry":    "Technology",
        "sector":      "Internet Services",
        "competitors": ["Microsoft", "Apple", "Meta"],
        "risks":       ["Regulatory Risk", "Competition Risk", "Privacy Risk"],
        "products":    ["Search", "YouTube", "Cloud", "Android"],
    },
}

# ── Node colors by type ──────────────────────────────────────
NODE_COLORS = {
    "company":   "#38bdf8",
    "ceo":       "#a78bfa",
    "industry":  "#34d399",
    "competitor":"#f472b6",
    "risk":      "#ef4444",
    "product":   "#f59e0b",
}

NODE_SIZES = {
    "company":   40,
    "ceo":       25,
    "industry":  30,
    "competitor":20,
    "risk":      20,
    "product":   18,
}


class KnowledgeGraphBuilder:

    def __init__(self):
        self.graph = nx.DiGraph()

    def build(self, companies: list[str] = None) -> nx.DiGraph:
        self.graph.clear()
        targets = companies or list(COMPANY_DATA.keys())

        for company in targets:
            if company not in COMPANY_DATA:
                continue

            data = COMPANY_DATA[company]
            self._add_node(company, "company")

            # CEO
            ceo = data["ceo"]
            self._add_node(ceo, "ceo")
            self.graph.add_edge(company, ceo, label="led by")

            # Industry
            industry = data["industry"]
            self._add_node(industry, "industry")
            self.graph.add_edge(company, industry, label="operates in")

            # Competitors
            for comp in data["competitors"]:
                self._add_node(comp, "competitor")
                self.graph.add_edge(company, comp, label="competes with")

            # Risks
            for risk in data["risks"]:
                self._add_node(risk, "risk")
                self.graph.add_edge(company, risk, label="exposed to")

            # Products
            for product in data["products"]:
                self._add_node(product, "product")
                self.graph.add_edge(company, product, label="offers")

        logger.info(
            f"Graph built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        return self.graph

    def _add_node(self, name: str, node_type: str):
        self.graph.add_node(
            name,
            node_type=node_type,
            color=NODE_COLORS.get(node_type, "#ffffff"),
            size=NODE_SIZES.get(node_type, 20)
        )

    def get_company_subgraph(self, company: str) -> nx.DiGraph:
        if company not in self.graph:
            self.build([company])
        neighbors = list(self.graph.successors(company))
        nodes = [company] + neighbors
        return self.graph.subgraph(nodes).copy()

    def get_stats(self) -> dict:
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "companies":   [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "company"],
            "density":     round(nx.density(self.graph), 4)
        }