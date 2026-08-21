import streamlit as st
import pandas as pd
import networkx as nx
import graphviz
import re

st.set_page_config(page_title="Developmental Network Explorer", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("nodes.csv"), pd.read_csv("edges.csv")

nodes, edges = load_data()
label_map = dict(zip(nodes["node"], nodes["label"]))
node_meta = nodes.set_index("node").to_dict("index")

def make_graph(edge_df):
    G = nx.DiGraph()
    for _, row in nodes.iterrows():
        G.add_node(row["node"], **row.to_dict())
    for _, row in edge_df.iterrows():
        G.add_edge(row["from"], row["to"],
                   bootstrap_forward=row["bootstrap_forward"],
                   bootstrap_reverse=row["bootstrap_reverse"])
    return G

def upstream_subgraph(G, outcome):
    """Return the full upstream subgraph for an outcome."""
    keep = set(nx.ancestors(G, outcome))
    keep.add(outcome)
    return G.subgraph(keep).copy()

def graphviz_from_nx(H):
    dot = graphviz.Digraph()
    dot.attr(rankdir="LR")
    for node in H.nodes:
        meta = node_meta[node]
        dot.node(node, label=f'{meta["label"]}\n[{meta["period"]}]', shape="box")
    for s, t, data in H.edges(data=True):
        dot.edge(s, t, label=f'{data.get("bootstrap_forward",0):.2f}')
    return dot


def count_paths_to_outcome(H, outcome):
    """
    Count all directed paths from every upstream ancestor to the outcome.
    Uses dynamic programming on a DAG, so it does not need to enumerate
    individual paths or impose an arbitrary cap.
    """
    if outcome not in H:
        return 0

    if not nx.is_directed_acyclic_graph(H):
        raise ValueError(
            "Path-impact analysis requires a DAG, but the retrieved graph contains a cycle."
        )

    # Only nodes that can reach the outcome are relevant.
    relevant = set(nx.ancestors(H, outcome))
    relevant.add(outcome)
    R = H.subgraph(relevant).copy()

    # paths_from[node] = number of distinct directed paths from node to outcome.
    paths_from = {node: 0 for node in R.nodes}
    paths_from[outcome] = 1

    for node in reversed(list(nx.topological_sort(R))):
        if node == outcome:
            continue
        paths_from[node] = sum(paths_from[succ] for succ in R.successors(node))

    # Match the old interpretation: count paths beginning at every ancestor.
    return sum(paths_from[node] for node in R.nodes if node != outcome)


def interpret_question(question):
    q = question.lower()

    outcome = "executive_function_6y"
    explicit = []
    for _, row in nodes.iterrows():
        if str(row["label"]).lower() in q:
            explicit.append(row)
    if explicit:
        outcome = sorted(explicit, key=lambda r: r["age_order"], reverse=True)[0]["node"]
    elif "school readiness" in q:
        outcome = "school_readiness_6y"
    elif "socio-emotional" in q or "socioemotional" in q:
        outcome = "child_socioemotional_6y"
    elif "self-regulation" in q or "self regulation" in q:
        outcome = "child_selfreg_3y"

    period_map = {
        "pregnancy":"Pregnancy","prenatal":"Pregnancy","antenatal":"Pregnancy",
        "infancy":"Infancy","infant":"Infancy",
        "early childhood":"Early childhood","preschool":"Early childhood",
        "middle childhood":"Middle childhood","school age":"Middle childhood"
    }
    periods = []
    for k,v in period_map.items():
        if k in q and v not in periods:
            periods.append(v)
    if "early life" in q or "early-life" in q:
        periods = ["Pregnancy","Infancy","Early childhood"]
    if not periods:
        periods = nodes["period"].unique().tolist()

    domain_map = {
        "built environment":"Built environment","neighbourhood":"Built environment",
        "neighborhood":"Built environment","green space":"Built environment",
        "housing":"Built environment","traffic":"Built environment",
        "services":"Built environment","play space":"Built environment",
        "ses":"SES","socioeconomic":"SES","income":"SES","education":"SES",
        "parenting":"Parenting","caregiving":"Parenting","reading":"Parenting",
        "parent wellbeing":"Parent wellbeing","parenting stress":"Parent wellbeing",
        "maternal stress":"Parent wellbeing","development":"Development",
        "child development":"Development"
    }
    domains = []
    for k,v in domain_map.items():
        if k in q and v not in domains:
            domains.append(v)
    if not domains:
        domains = nodes["domain"].unique().tolist()

    min_stability = 0.65
    if any(x in q for x in ["robust","strong","high confidence","high-confidence"]):
        min_stability = 0.80
    if any(x in q for x in ["exploratory","broad","all possible","include weaker"]):
        min_stability = 0.50

    return {
        "outcome_node": outcome,
        "periods": periods,
        "domains": domains,
        "minimum_stability": min_stability
    }

def run_query(settings):
    q_edges = edges[edges["bootstrap_forward"] >= settings["minimum_stability"]].copy()
    q_G = make_graph(q_edges)
    q_H = upstream_subgraph(q_G, settings["outcome_node"])
    keep = {
        n for n in q_H.nodes
        if n == settings["outcome_node"]
        or (node_meta[n]["period"] in settings["periods"]
            and node_meta[n]["domain"] in settings["domains"])
    }
    return q_H.subgraph(keep).copy()

st.title("Developmental causal-network explorer")
st.caption("Synthetic demonstration for hypothesis generation only.")

with st.sidebar:
    st.header("Manual network query")
    opts = nodes.copy()
    opts["display"] = opts["label"] + " [" + opts["period"] + "]"
    default_i = opts.index[opts["node"]=="executive_function_6y"][0]
    display = st.selectbox("Outcome", opts["display"].tolist(), index=default_i)
    outcome = opts.loc[opts["display"]==display, "node"].iloc[0]
    min_stability = st.slider("Minimum bootstrap edge frequency",0.50,1.00,0.65,0.05)
    all_domains = sorted(nodes["domain"].unique())
    domains = st.multiselect("Keep source domains",all_domains,default=all_domains)

G = make_graph(edges[edges["bootstrap_forward"]>=min_stability])
H = upstream_subgraph(G, outcome)
H = H.subgraph({n for n in H.nodes if n==outcome or node_meta[n]["domain"] in domains}).copy()

c1,c2,c3 = st.columns(3)
c1.metric("Retrieved nodes",H.number_of_nodes())
c2.metric("Retrieved edges",H.number_of_edges())
c3.metric("Outcome ancestors",len(nx.ancestors(H,outcome)) if outcome in H else 0)

st.subheader("Retrieved subgraph")
if H.number_of_edges():
    st.graphviz_chart(graphviz_from_nx(H), use_container_width=True)
else:
    st.warning("No upstream edges meet the current filters.")

st.subheader("Structurally important nodes")
rows=[]
for n in H.nodes:
    if n==outcome:
        continue
    rows.append({
        "node":label_map[n],
        "period":node_meta[n]["period"],
        "domain":node_meta[n]["domain"],
        "descendants_in_subgraph":len(nx.descendants(H,n)),
        "incoming_edges":H.in_degree(n),
        "outgoing_edges":H.out_degree(n),
        "total_edges":H.in_degree(n)+H.out_degree(n)
    })
if rows:
    st.dataframe(pd.DataFrame(rows).sort_values(
        ["descendants_in_subgraph","total_edges"], ascending=False
    ), use_container_width=True, hide_index=True)

st.divider()

# ==========================================
# AUTOMATIC EDGE IMPACT RANKING
# ==========================================

st.subheader("Automatic edge impact ranking")

st.caption(
    "Each edge is removed one at a time to estimate how much "
    "it changes the outcome-specific hypothesis-generating network."
)

if H.number_of_edges() == 0:

    st.info("No edges available to rank.")

else:

    # Original network
    original_ancestors = set(
        nx.ancestors(H, outcome)
    )

    original_path_count = count_paths_to_outcome(
        H,
        outcome
    )

    edge_impact_results = []

    # Remove each edge one at a time
    for source, target in H.edges():

        H_test = H.copy()

        H_test.remove_edge(
            source,
            target
        )

        # Recalculate ancestors
        test_ancestors = set(
            nx.ancestors(
                H_test,
                outcome
            )
        )

        # Recalculate paths
        test_path_count = count_paths_to_outcome(
            H_test,
            outcome
        )

        # Calculate structural impact
        pathways_lost = (
            original_path_count
            - test_path_count
        )

        if original_path_count > 0:

            percent_paths_lost = (
                pathways_lost
                / original_path_count
                * 100
            )

        else:

            percent_paths_lost = 0

        disconnected_nodes = (
            original_ancestors
            - test_ancestors
        )

        source_period = node_meta[source]["period"]
        target_period = node_meta[target]["period"]

        # Store results
        edge_impact_results.append(
            {
                "Edge":
                    f"{label_map[source]} → "
                    f"{label_map[target]}",

                "Timepoints":
                    f"{source_period} → "
                    f"{target_period}",

                "Paths lost":
                    pathways_lost,

                "% paths lost":
                    percent_paths_lost,

                "Nodes disconnected":
                    len(disconnected_nodes),

                "Disconnected nodes":
                    ", ".join(
                        f"{label_map[node]} "
                        f"[{node_meta[node]['period']}]"
                        for node in disconnected_nodes
                    )
            }
        )

    # Create ranked table
    edge_impact_df = pd.DataFrame(
        edge_impact_results
    )

    edge_impact_df = edge_impact_df.sort_values(
        [
            "% paths lost",
            "Nodes disconnected"
        ],
        ascending=False
    )

    edge_impact_df["% paths lost"] = (
        edge_impact_df["% paths lost"]
        .round(1)
    )

    st.dataframe(
        edge_impact_df,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# MANUAL EDGE-REMOVAL SENSITIVITY ANALYSIS
# ==========================================

st.subheader("Edge-removal sensitivity analysis")

st.caption(
    "Select an individual edge to inspect how its removal changes "
    "the upstream structure for the selected outcome."
)

if H.number_of_edges() == 0:

    st.info("No edges available to test.")

else:

    edge_options = []

    for source, target in H.edges():

        source_period = node_meta[source]["period"]
        target_period = node_meta[target]["period"]

        edge_options.append(
            (
                source,
                target,
                f"{label_map[source]} "
                f"[{source_period}] → "
                f"{label_map[target]} "
                f"[{target_period}]"
            )
        )

    edge_labels = [
        item[2]
        for item in edge_options
    ]

    selected_edge_label = st.selectbox(
        "Select an edge to remove",
        edge_labels
    )

    selected_index = edge_labels.index(
        selected_edge_label
    )

    selected_source = edge_options[
        selected_index
    ][0]

    selected_target = edge_options[
        selected_index
    ][1]

    if st.button("Test edge removal"):

        # Original network
        original_ancestors = set(
            nx.ancestors(H, outcome)
        )

        original_path_count = count_paths_to_outcome(
            H,
            outcome
        )

        # Remove selected edge
        H_removed = H.copy()

        H_removed.remove_edge(
            selected_source,
            selected_target
        )

        # Recalculate
        removed_ancestors = set(
            nx.ancestors(
                H_removed,
                outcome
            )
        )

        removed_path_count = count_paths_to_outcome(
            H_removed,
            outcome
        )

        disconnected_nodes = (
            original_ancestors
            - removed_ancestors
        )

        pathways_lost = (
            original_path_count
            - removed_path_count
        )

        if original_path_count > 0:

            percent_paths_lost = (
                pathways_lost
                / original_path_count
                * 100
            )

        else:

            percent_paths_lost = 0

        # Display results
        st.markdown(
            f"### Removing: "
            f"{label_map[selected_source]} "
            f"[{node_meta[selected_source]['period']}] "
            f"→ "
            f"{label_map[selected_target]} "
            f"[{node_meta[selected_target]['period']}]"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Original pathways",
            original_path_count
        )

        col2.metric(
            "Paths lost",
            pathways_lost
        )

        col3.metric(
            "% paths lost",
            f"{percent_paths_lost:.1f}%"
        )

        col4.metric(
            "Nodes disconnected",
            len(disconnected_nodes)
        )

        st.markdown(
            "#### Upstream nodes disconnected"
        )

        if disconnected_nodes:

            disconnected_labels = [
                f"{label_map[node]} "
                f"[{node_meta[node]['period']}]"
                for node in disconnected_nodes
            ]

            st.write(
                ", ".join(
                    disconnected_labels
                )
            )

        else:

            st.write(
                "No upstream nodes were completely disconnected."
            )

        st.markdown(
            "#### Network after edge removal"
        )

        if H_removed.number_of_edges() > 0:

            st.graphviz_chart(
                graphviz_from_nx(
                    H_removed
                ),
                use_container_width=True
            )

        else:

            st.warning(
                "Removing this edge leaves no remaining edges."
            )


st.divider()
st.subheader("Natural-language network query")
st.caption(
    "The parser converts your question into graph filters, "
    "then NetworkX retrieves the matching subgraph."
)

st.markdown("#### Example questions to copy and paste")
st.code("What parenting factors are upstream of school readiness?", language=None)
st.code("What parenting factors are connected to executive function?", language=None)

question = st.text_area(
    "Ask a question",
    placeholder="Type or paste a research question here..."
)
if st.button("Run natural-language query"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        settings=interpret_question(question)
        st.write(
            f'Outcome: {label_map[settings["outcome_node"]]} [{node_meta[settings["outcome_node"]]["period"]}] | '
            f'Periods: {", ".join(settings["periods"])} | '
            f'Domains: {", ".join(settings["domains"])} | '
            f'Minimum stability: {settings["minimum_stability"]:.2f}'
        )
        Q=run_query(settings)
        q1,q2,q3=st.columns(3)
        q1.metric("Retrieved nodes",Q.number_of_nodes())
        q2.metric("Retrieved edges",Q.number_of_edges())
        q3.metric("Outcome ancestors",len(nx.ancestors(Q,settings["outcome_node"])) if settings["outcome_node"] in Q else 0)
        if Q.number_of_edges():
            st.graphviz_chart(graphviz_from_nx(Q), use_container_width=True)
        else:
            st.warning("No relationships met the interpreted query.")
        out=[]
        for n in Q.nodes:
            out.append({"node":label_map[n],"period":node_meta[n]["period"],"domain":node_meta[n]["domain"]})
        if out:
            st.dataframe(pd.DataFrame(out), use_container_width=True, hide_index=True)
