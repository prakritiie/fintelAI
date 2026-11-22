import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
from dashboard.ui import apply_theme, page_header, style_fig

st.set_page_config(page_title="Knowledge Graph", page_icon="🕸️", layout="wide")

apply_theme()
page_header("Knowledge Graph", "Explore how companies, industries, competitors, risks and products connect.", eyebrow="Step 5")

# ── Load builder and visualizer ──────────────────────────────
@st.cache_resource
def load_graph_tools():
    from graph.builder    import KnowledgeGraphBuilder
    from graph.visualizer import GraphVisualizer
    return KnowledgeGraphBuilder(), GraphVisualizer()

builder, visualizer = load_graph_tools()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    available = ["Apple", "Tesla", "Microsoft", "Berkshire Hathaway", "Google"]
    selected  = st.multiselect(
        "Select Companies",
        available,
        default=["Apple", "Tesla", "Microsoft"]
    )

    graph_height = st.slider("Graph Height (px)", 400, 900, 600, 50)

    st.markdown("### Node legend")
    legend = [
        ("#38bdf8", "Company"),
        ("#a78bfa", "CEO"),
        ("#34d399", "Industry"),
        ("#f472b6", "Competitor"),
        ("#ef4444", "Risk"),
        ("#f59e0b", "Product"),
    ]
    for color, label in legend:
        st.markdown(
            f"<span style='color:{color}; font-size:1.2rem;'>●</span> {label}",
            unsafe_allow_html=True
        )

    build_btn = st.button("Build graph", type="primary", use_container_width=True)

# ── Build and display graph ───────────────────────────────────
if build_btn or "graph_built" not in st.session_state:

    if not selected:
        st.warning("Please select at least one company.")
        st.stop()

    with st.spinner("Building knowledge graph..."):
        graph = builder.build(selected)
        html  = visualizer.render_html_string(graph)
        st.session_state.graph_html  = html
        st.session_state.graph_built = True
        st.session_state.graph_obj   = graph

# ── Render graph ─────────────────────────────────────────────
if "graph_html" in st.session_state:

    # Stats row
    stats = builder.get_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Nodes",    stats["total_nodes"])
    c2.metric("Total Edges",    stats["total_edges"])
    c3.metric("Companies",      len(stats["companies"]))
    c4.metric("Graph Density",  stats["density"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Graph
    components.html(st.session_state.graph_html, height=graph_height, scrolling=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Relationship table ────────────────────────────────────
    st.markdown("### Relationship explorer")
    graph = st.session_state.graph_obj

    rows = []
    for src, dst, data in graph.edges(data=True):
        src_type = graph.nodes[src].get("node_type", "")
        dst_type = graph.nodes[dst].get("node_type", "")
        rows.append({
            "From":         src,
            "Relationship": data.get("label", ""),
            "To":           dst,
            "From Type":    src_type,
            "To Type":      dst_type
        })

    df = pd.DataFrame(rows)

    # Filter
    rel_filter = st.selectbox(
        "Filter by Relationship",
        ["All"] + sorted(df["Relationship"].unique().tolist())
    )
    if rel_filter != "All":
        df = df[df["Relationship"] == rel_filter]

    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Node type distribution ────────────────────────────────
    st.markdown("### Node type distribution")
    type_counts = {}
    for _, attrs in graph.nodes(data=True):
        t = attrs.get("node_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    color_map = {
        "company":    "#38bdf8",
        "ceo":        "#a78bfa",
        "industry":   "#34d399",
        "competitor": "#f472b6",
        "risk":       "#ef4444",
        "product":    "#f59e0b",
    }

    fig = go.Figure(go.Bar(
        x=list(type_counts.keys()),
        y=list(type_counts.values()),
        marker_color=[color_map.get(k, "#ffffff") for k in type_counts.keys()]
    ))
    style_fig(fig, 300)
    st.plotly_chart(fig, use_container_width=True)