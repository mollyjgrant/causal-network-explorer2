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
    """
    Interpret a research question into:
      - anchor node (construct + timepoint)
      - direction (upstream/downstream)
      - optional filters on returned factors (domain + timepoint)
      - bootstrap threshold

    This is intentionally rule-based for the prototype.
    """
    q = question.lower().strip()

    direction = "downstream" if "downstream" in q else "upstream"

    period_terms = {
        "Pregnancy": ["pregnancy", "prenatal", "antenatal"],
        "Infancy": ["infancy", "infant", "first year", "1 year", "1y"],
        "Early childhood": ["early childhood", "3 year", "3y"],
        "Middle childhood": ["middle childhood", "6 year", "6y"],
    }

    domain_terms = {
        "SES": ["ses", "socioeconomic", "socio-economic"],
        "Environmental": ["environment", "environmental", "air pollution",
                          "pm2.5", "heat stress", "green space", "noise"],
        "Parenting": ["parenting", "parental", "caregiving"],
        "Eating behaviours": ["eating", "eating behaviour", "eating behaviors",
                              "feeding", "diet", "food"],
        "School readiness": ["school readiness", "executive function",
                             "executive functioning", "language", "numeracy"],
    }

    # Split around the directional phrase. For:
    # "What parenting factors in early childhood are downstream of parenting stress in infancy?"
    # left side = requested-factor filters; right side = anchor.
    phrase = "downstream of" if direction == "downstream" else "upstream of"
    if phrase in q:
        left_text, anchor_text = q.split(phrase, 1)
    else:
        left_text, anchor_text = q, q

    def periods_in(s):
        found = []
        for canonical, terms in period_terms.items():
            if any(term in s for term in terms):
                found.append(canonical)
        return found

    def domains_in(s):
        found = []
        for canonical, terms in domain_terms.items():
            if any(term in s for term in terms):
                found.append(canonical)
        return found

    # Anchor period is determined ONLY from the text after "upstream/downstream of".
    anchor_periods = periods_in(anchor_text)
    anchor_period = anchor_periods[0] if anchor_periods else None

    # Match anchor label, preferring the longest label phrase so repeated/overlapping
    # constructs are less likely to be confused.
    candidate_rows = nodes.copy()
    candidate_rows["label_lower"] = candidate_rows["label"].str.lower()

    if anchor_period is not None:
        candidate_rows = candidate_rows[
            candidate_rows["period"] == anchor_period
        ]

    # Aliases for common construct wording.
    anchor_aliases = {
        "executive functioning": "executive function",
        "executive functions": "executive function",
        "ef": "executive function",
    }
    normalized_anchor_text = anchor_text
    for alias, canonical in anchor_aliases.items():
        normalized_anchor_text = normalized_anchor_text.replace(alias, canonical)

    matches = []
    for _, row in candidate_rows.iterrows():
        label = row["label_lower"]
        if label in normalized_anchor_text:
            matches.append((len(label), row["node"]))

    if matches:
        matches.sort(reverse=True)
        anchor_node = matches[0][1]
    else:
        # Fallback: score labels by token overlap with the anchor phrase.
        anchor_tokens = set(re.findall(r"[a-z0-9.]+", normalized_anchor_text))
        best_node = None
        best_score = -1
        for _, row in candidate_rows.iterrows():
            label_tokens = set(re.findall(r"[a-z0-9.]+", row["label_lower"]))
            score = len(anchor_tokens & label_tokens)
            if score > best_score:
                best_score = score
                best_node = row["node"]
        anchor_node = best_node

    if anchor_node is None:
        raise ValueError("Could not identify an anchor variable from the question.")

    # Filters for returned factors come ONLY from the text before the relationship phrase.
    result_periods = periods_in(left_text)
    result_domains = domains_in(left_text)

    # Generic "what factors..." means do not restrict domain/timepoint.
    if not result_periods:
        result_periods = nodes["period"].dropna().unique().tolist()

    if not result_domains:
        result_domains = nodes["domain"].dropna().unique().tolist()

    min_stability = 0.65
    if any(x in q for x in ["robust", "strong", "high confidence", "high-confidence"]):
        min_stability = 0.80
    if any(x in q for x in ["exploratory", "broad", "all possible", "include weaker"]):
        min_stability = 0.50

    return {
        "outcome_node": anchor_node,   # retained for compatibility with existing app code
        "anchor_node": anchor_node,
        "anchor_period": node_meta[anchor_node]["period"],
        "periods": result_periods,
        "domains": result_domains,
        "minimum_stability": min_stability,
        "direction": direction,
    }


def run_query(settings):
    """
    Retrieve matching source nodes in the requested direction and retain the
    graph structure connecting them to the selected anchor/outcome node.
    """
    q_edges = edges[
        edges["bootstrap_forward"] >= settings["minimum_stability"]
    ].copy()

    q_G = make_graph(q_edges)
    anchor = settings["outcome_node"]
    direction = settings.get("direction", "upstream")

    if direction == "downstream":
        candidate_nodes = set(nx.descendants(q_G, anchor))

        matching_nodes = {
            n for n in candidate_nodes
            if (
                node_meta[n]["period"] in settings["periods"]
                and node_meta[n]["domain"] in settings["domains"]
            )
        }

        keep_nodes = {anchor}

        for target in matching_nodes:
            keep_nodes.add(target)

            # Nodes on at least one directed route anchor -> target
            connecting_nodes = (
                set(nx.descendants(q_G, anchor))
                & set(nx.ancestors(q_G, target))
            )
            keep_nodes.update(connecting_nodes)

    else:
        candidate_nodes = set(nx.ancestors(q_G, anchor))

        matching_nodes = {
            n for n in candidate_nodes
            if (
                node_meta[n]["period"] in settings["periods"]
                and node_meta[n]["domain"] in settings["domains"]
            )
        }

        keep_nodes = {anchor}

        for source in matching_nodes:
            keep_nodes.add(source)

            # Nodes on at least one directed route source -> anchor
            connecting_nodes = (
                set(nx.descendants(q_G, source))
                & set(nx.ancestors(q_G, anchor))
            )
            keep_nodes.update(connecting_nodes)

    return q_G.subgraph(keep_nodes).copy()


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
    "Each edge is removed one at a time. An upstream connection is counted as lost "
    "only when that upstream node can no longer reach the selected outcome by any route."
)

if H.number_of_edges() == 0:

    st.info("No edges available to rank.")

else:

    # Original network
    original_ancestors = set(
        nx.ancestors(H, outcome)
    )

    # Treat each upstream node-to-outcome relationship as one connection,
    # regardless of how many alternative directed routes exist between them.
    # A connection is only counted as lost if the upstream node can no longer
    # reach the outcome at all after the edge is removed.
    original_connection_count = len(
        original_ancestors
    )

    edge_impact_results = []

    # Remove each edge one at a time
    for source, target in H.edges():

        H_test = H.copy()

        H_test.remove_edge(
            source,
            target
        )

        # Recalculate which upstream nodes can still reach the outcome
        test_ancestors = set(
            nx.ancestors(
                H_test,
                outcome
            )
        )

        disconnected_nodes = (
            original_ancestors
            - test_ancestors
        )

        connections_lost = len(
            disconnected_nodes
        )

        if original_connection_count > 0:

            percent_connections_lost = (
                connections_lost
                / original_connection_count
                * 100
            )

        else:

            percent_connections_lost = 0

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

                "Upstream connections lost":
                    connections_lost,

                "% upstream connections lost":
                    percent_connections_lost,

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
            "% upstream connections lost",
            "Nodes disconnected"
        ],
        ascending=False
    )

    edge_impact_df["% upstream connections lost"] = (
        edge_impact_df["% upstream connections lost"]
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
    "Select an individual edge to inspect whether its removal completely disconnects "
    "any upstream nodes from the selected outcome."
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

        # Count each upstream node-to-outcome relationship once.
        original_connection_count = len(
            original_ancestors
        )

        # Remove selected edge
        H_removed = H.copy()

        H_removed.remove_edge(
            selected_source,
            selected_target
        )

        # Recalculate which upstream nodes can still reach the outcome
        removed_ancestors = set(
            nx.ancestors(
                H_removed,
                outcome
            )
        )

        disconnected_nodes = (
            original_ancestors
            - removed_ancestors
        )

        connections_lost = len(
            disconnected_nodes
        )

        if original_connection_count > 0:

            percent_connections_lost = (
                connections_lost
                / original_connection_count
                * 100
            )

        else:

            percent_connections_lost = 0

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
            "Original upstream connections",
            original_connection_count
        )

        col2.metric(
            "Upstream connections lost",
            connections_lost
        )

        col3.metric(
            "% upstream connections lost",
            f"{percent_connections_lost:.1f}%"
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
st.subheader("Explore the network")

st.caption(
    "Use a guided prompt or ask your own question. "
    "The interface translates the question into graph filters, "
    "then NetworkX retrieves relationships from the data-derived network."
)

# --------------------------------------------------
# 1. BUILD-YOUR-QUESTION
# --------------------------------------------------

st.markdown("### Build your question")

st.caption(
    "Choose the variable you want to start from, the direction to search, "
    "and optionally restrict the factors returned."
)

anchor_col1, anchor_col2, direction_col = st.columns(3)

available_periods = [
    p for p in [
        "Pregnancy",
        "Infancy",
        "Early childhood",
        "Middle childhood"
    ]
    if p in nodes["period"].unique().tolist()
]

with anchor_col1:
    builder_anchor_period = st.selectbox(
        "Anchor timepoint",
        available_periods,
        key="builder_anchor_period"
    )

anchor_candidates = nodes[
    nodes["period"] == builder_anchor_period
].copy()

anchor_candidates["display"] = anchor_candidates["label"]

with anchor_col2:
    builder_anchor_display = st.selectbox(
        "Anchor variable",
        anchor_candidates["display"].tolist(),
        key="builder_anchor"
    )

builder_anchor_label = anchor_candidates.loc[
    anchor_candidates["display"] == builder_anchor_display,
    "label"
].iloc[0]

with direction_col:
    builder_relation = st.selectbox(
        "Relationship",
        ["Upstream", "Downstream"],
        key="builder_relation"
    )

st.markdown("**Filter factors returned (optional)**")

filter_col1, filter_col2 = st.columns(2)

available_domains = sorted(
    nodes["domain"].dropna().unique().tolist()
)

with filter_col1:
    builder_domain = st.selectbox(
        "Factor domain",
        ["All domains"] + available_domains,
        key="builder_domain"
    )

with filter_col2:
    builder_period = st.selectbox(
        "Factor timepoint",
        ["Any timepoint"] + available_periods,
        key="builder_period"
    )

relation_phrase = (
    "upstream of"
    if builder_relation == "Upstream"
    else "downstream of"
)

factor_bits = []
if builder_period != "Any timepoint":
    factor_bits.append(builder_period.lower())
if builder_domain != "All domains":
    factor_bits.append(builder_domain.lower())

factor_description = " ".join(factor_bits)

if factor_description:
    generated_question = (
        f"What {factor_description} factors are {relation_phrase} "
        f"{builder_anchor_label.lower()} in {builder_anchor_period.lower()}?"
    )
else:
    generated_question = (
        f"What factors are {relation_phrase} "
        f"{builder_anchor_label.lower()} in {builder_anchor_period.lower()}?"
    )

st.markdown("**Suggested question**")
st.info(generated_question)

if st.button("Use this question", key="use_builder_question"):
    st.session_state["nl_question"] = generated_question
    st.rerun()

st.divider()

# --------------------------------------------------
# 2. FREE-TEXT QUERY
# --------------------------------------------------

st.markdown("### Ask your own question")

question = st.text_area(
    "Research question",
    key="nl_question",
    placeholder=(
        "e.g. What infancy parenting factors are upstream of "
        "school readiness in middle childhood?"
    )
)

if st.button("Run natural-language query", key="run_nl_query"):

    if not question.strip():

        st.warning("Enter a question first.")

    else:

        settings = interpret_question(question)

        st.markdown("#### How the question was interpreted")

        st.write(
            f'**Outcome:** {label_map[settings["outcome_node"]]} '
            f'[{node_meta[settings["outcome_node"]]["period"]}]  \n'
            f'**Source timepoints:** {", ".join(settings["periods"])}  \n'
            f'**Source domains:** {", ".join(settings["domains"])}  \n'
            f'**Direction:** {settings["direction"].capitalize()}  \n'
            f'**Minimum bootstrap frequency:** '
            f'{settings["minimum_stability"]:.2f}'
        )

        Q = run_query(settings)

        q1, q2, q3 = st.columns(3)

        q1.metric(
            "Retrieved nodes",
            Q.number_of_nodes()
        )

        q2.metric(
            "Retrieved edges",
            Q.number_of_edges()
        )

        q3.metric(
            "Outcome ancestors",
            len(
                nx.ancestors(
                    Q,
                    settings["outcome_node"]
                )
            )
            if settings["outcome_node"] in Q
            else 0
        )

        st.markdown("#### Retrieved network")

        if Q.number_of_edges():

            st.graphviz_chart(
                graphviz_from_nx(Q),
                use_container_width=True
            )

        else:

            st.warning(
                "No relationships met the interpreted query. "
                "Try broadening the timepoint/domain selection or "
                "lowering the bootstrap threshold in the main panel."
            )

        st.markdown("#### Retrieved nodes")

        out_rows = []

        for n in Q.nodes:

            out_rows.append(
                {
                    "node": label_map[n],
                    "period": node_meta[n]["period"],
                    "domain": node_meta[n]["domain"]
                }
            )

        if out_rows:

            st.dataframe(
                pd.DataFrame(out_rows).sort_values(
                    ["period", "domain", "node"]
                ),
                use_container_width=True,
                hide_index=True
            )
