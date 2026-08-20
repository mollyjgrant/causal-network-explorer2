
import streamlit as st
import pandas as pd
import networkx as nx
import graphviz
from openai import OpenAI
import json

st.set_page_config(
    page_title="Developmental Network Explorer",
    layout="wide"
)

@st.cache_data
def load_data():
    nodes = pd.read_csv("nodes.csv")
    edges = pd.read_csv("edges.csv")
    return nodes, edges

nodes, edges = load_data()

client = OpenAI(
    api_key=st.secrets["Key"]
)

label_map = dict(zip(nodes["node"], nodes["label"]))
node_meta = nodes.set_index("node").to_dict("index")


def make_graph(edge_df):
    G = nx.DiGraph()

    for _, row in nodes.iterrows():
        G.add_node(
            row["node"],
            **row.to_dict()
        )

    for _, row in edge_df.iterrows():
        G.add_edge(
            row["from"],
            row["to"],
            bootstrap_forward=row["bootstrap_forward"],
            bootstrap_reverse=row["bootstrap_reverse"]
        )

    return G


def upstream_subgraph(G, outcome, max_steps):
    keep = {outcome}
    frontier = {outcome}

    for _ in range(max_steps):
        new_frontier = set()

        for node in frontier:
            new_frontier.update(G.predecessors(node))

        new_frontier -= keep
        keep.update(new_frontier)
        frontier = new_frontier

        if not frontier:
            break

    return G.subgraph(keep).copy()


def graphviz_from_nx(H):
    dot = graphviz.Digraph()

    dot.attr(rankdir="LR")

    for node in H.nodes:
        meta = node_meta[node]

        node_label = (
            f'{meta["label"]}\n'
            f'[{meta["period"]}]'
        )

        dot.node(
            node,
            label=node_label,
            shape="box"
        )

    for source, target, data in H.edges(data=True):
        stability = data.get(
            "bootstrap_forward",
            0
        )

        dot.edge(
            source,
            target,
            label=f"{stability:.2f}"
        )

    return dot


def create_path_table(H, outcome, max_paths=200):
    rows = []

    if outcome not in H:
        return pd.DataFrame()

    ancestors = nx.ancestors(H, outcome)

    for source in ancestors:
        try:
            for path in nx.all_simple_paths(
                H,
                source=source,
                target=outcome
            ):

                if len(path) <= 6:

                    edge_stabilities = [
                        H[path[i]][path[i + 1]]["bootstrap_forward"]
                        for i in range(len(path) - 1)
                    ]

                    rows.append(
                        {
                            "source": label_map[source],
                            "path": " → ".join(
                                label_map[node]
                                for node in path
                            ),
                            "steps": len(path) - 1,
                            "weakest_edge": min(edge_stabilities),
                            "mean_stability":
                                sum(edge_stabilities)
                                / len(edge_stabilities)
                        }
                    )

                    if len(rows) >= max_paths:
                        return pd.DataFrame(rows)

        except nx.NetworkXNoPath:
            pass

    return pd.DataFrame(rows)

def interpret_question(question, nodes):

    available_nodes = nodes[
        ["node", "label", "period", "domain"]
    ].to_dict(orient="records")

    available_periods = sorted(
        nodes["period"].unique().tolist()
    )

    available_domains = sorted(
        nodes["domain"].unique().tolist()
    )

    prompt = f"""
You are helping a researcher query a developmental network.

Your ONLY task is to translate the researcher's natural-language
question into graph-query parameters.

Do not answer the scientific question.
Do not invent relationships.
Only select values supported by the available network metadata.

AVAILABLE NODES:
{available_nodes}

AVAILABLE DEVELOPMENTAL PERIODS:
{available_periods}

AVAILABLE DOMAINS:
{available_domains}

RESEARCHER QUESTION:
{question}

Return valid JSON with exactly these fields:

{{
  "outcome_node": "node id",
  "periods": ["period1", "period2"],
  "domains": ["domain1", "domain2"],
  "max_steps": 3,
  "minimum_stability": 0.65,
  "explanation": "brief explanation of how the question was interpreted"
}}

Rules:
- outcome_node must exactly match one available node id.
- periods must contain only available periods.
- domains must contain only available domains.
- max_steps must be between 1 and 5.
- minimum_stability must be between 0.50 and 1.00.
- If the user does not restrict period or domain, include all relevant values.
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    text = response.output_text

    text = (
        text.replace("```json", "")
        .replace("```", "")
        .strip()
    )

    return json.loads(text)
    
st.title(
    "Developmental causal-network explorer"
)

st.caption(
    "Synthetic demonstration. Network relationships are "
    "hypothesis-generating and should not be interpreted "
    "as established causal effects."
)


with st.sidebar:

    st.header("Query the network")

    outcome_labels = sorted(
        nodes["label"].unique()
    )

    default_index = (
        outcome_labels.index("Executive function")
        if "Executive function" in outcome_labels
        else 0
    )

    outcome_label = st.selectbox(
        "Outcome",
        outcome_labels,
        index=default_index
    )

    matching = nodes[
        nodes["label"] == outcome_label
    ].copy()

    matching["display"] = (
        matching["label"]
        + " — "
        + matching["period"]
    )

    chosen_display = st.selectbox(
        "Measurement",
        matching["display"]
    )

    outcome = matching.loc[
        matching["display"] == chosen_display,
        "node"
    ].iloc[0]

    min_stability = st.slider(
        "Minimum bootstrap edge frequency",
        min_value=0.50,
        max_value=1.00,
        value=0.65,
        step=0.05
    )

    max_steps = st.slider(
        "Maximum upstream steps",
        min_value=1,
        max_value=5,
        value=3
    )

    all_domains = sorted(
        nodes["domain"].unique()
    )

    domains = st.multiselect(
        "Keep source domains",
        all_domains,
        default=all_domains
    )


filtered_edges = edges[
    edges["bootstrap_forward"]
    >= min_stability
].copy()

G = make_graph(filtered_edges)

H = upstream_subgraph(
    G,
    outcome,
    max_steps
)

keep_nodes = {
    node
    for node in H.nodes
    if (
        node == outcome
        or node_meta[node]["domain"]
        in domains
    )
}

H = H.subgraph(
    keep_nodes
).copy()


col1, col2, col3 = st.columns(3)

col1.metric(
    "Retrieved nodes",
    H.number_of_nodes()
)

col2.metric(
    "Retrieved edges",
    H.number_of_edges()
)

col3.metric(
    "Outcome ancestors",
    len(nx.ancestors(H, outcome))
    if outcome in H
    else 0
)


st.subheader("Retrieved subgraph")

if H.number_of_edges() == 0:

    st.warning(
        "No upstream edges meet the current filters."
    )

else:

    st.graphviz_chart(
        graphviz_from_nx(H),
        use_container_width=True
    )


st.subheader("Candidate pathways")

paths = create_path_table(
    H,
    outcome
)

if paths.empty:

    st.info(
        "No directed paths found under the "
        "current filters."
    )

else:

    st.dataframe(
        paths.sort_values(
            [
                "weakest_edge",
                "mean_stability"
            ],
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


st.subheader(
    "Structurally important nodes"
)

if H.number_of_nodes() > 1:

    betweenness = nx.betweenness_centrality(H)

    importance = []

    for node in H.nodes:

        if node == outcome:
            continue

        descendants = nx.descendants(
            H,
            node
        )

        importance.append(
            {
                "node": label_map[node],
                "period":
                    node_meta[node]["period"],
                "domain":
                    node_meta[node]["domain"],
                "descendants_in_subgraph":
                    len(descendants),
                "betweenness":
                    betweenness.get(node, 0)
            }
        )

    importance = pd.DataFrame(
        importance
    )

    if not importance.empty:

        st.dataframe(
            importance.sort_values(
                [
                    "descendants_in_subgraph",
                    "betweenness"
                ],
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )


st.divider()

st.subheader(
    "Natural-language research question — prototype"
)

question = st.text_area(
    "Ask a question",
    placeholder=(
        "e.g. What pregnancy and infancy factors "
        "are upstream of executive function in "
        "middle childhood, particularly through "
        "caregiving?"
    )
)

if question:

    if st.button("Query network"):

        try:

            interpretation = interpret_question(
                question,
                nodes
            )

            st.subheader("Question interpretation")

            st.write(
                interpretation["explanation"]
            )

            st.json(interpretation)

            nl_outcome = interpretation[
                "outcome_node"
            ]

            nl_periods = interpretation[
                "periods"
            ]

            nl_domains = interpretation[
                "domains"
            ]

            nl_steps = int(
                interpretation["max_steps"]
            )

            nl_stability = float(
                interpretation[
                    "minimum_stability"
                ]
            )

            nl_edges = edges[
                edges["bootstrap_forward"]
                >= nl_stability
            ].copy()

            nl_G = make_graph(nl_edges)

            nl_H = upstream_subgraph(
                nl_G,
                nl_outcome,
                nl_steps
            )

            keep_nodes = {
                node
                for node in nl_H.nodes
                if (
                    node == nl_outcome
                    or (
                        node_meta[node]["period"]
                        in nl_periods
                        and
                        node_meta[node]["domain"]
                        in nl_domains
                    )
                )
            }

            keep_nodes.add(nl_outcome)

            nl_H = nl_H.subgraph(
                keep_nodes
            ).copy()

            st.subheader(
                "Retrieved network"
            )

            st.write(
                f"Retrieved "
                f"{nl_H.number_of_nodes()} nodes "
                f"and "
                f"{nl_H.number_of_edges()} edges."
            )

            if nl_H.number_of_edges() > 0:

                st.graphviz_chart(
                    graphviz_from_nx(nl_H),
                    use_container_width=True
                )

            else:

                st.warning(
                    "No relationships met the "
                    "interpreted query."
                )

            st.subheader(
                "Candidate pathways"
            )

            nl_paths = create_path_table(
                nl_H,
                nl_outcome
            )

            if not nl_paths.empty:

                st.dataframe(
                    nl_paths.sort_values(
                        [
                            "weakest_edge",
                            "mean_stability"
                        ],
                        ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "No directed pathways were "
                    "retrieved."
                )

        except Exception as e:

            st.error(
                f"Could not interpret the question: {e}"
            )
