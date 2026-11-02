import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from huggingface_hub import InferenceClient

from ontology import validate_class, validate_relation


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="FORGE Knowledge Graph Demonstrator",
    page_icon="⚙️",
    layout="wide"
)

st.title("FORGE Knowledge Graph Demonstrator")

st.caption(
    "Ontology-driven GraphRAG demonstrator using NASA C-MAPSS data"
)


# ============================================================
# CONFIG
# ============================================================

NASA_FILE = "train_FD001.txt"
MAINTENANCE_FILE = "maintenance.csv"

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

ENGINE_ID = 1
SENSOR = "sensor_11"

BASELINE_CYCLES = 30
ANOMALY_THRESHOLD = 2
PERSISTENCE_WINDOW = 3


# ============================================================
# ONTOLOGY FUNCTIONS
# ============================================================

def add_node(graph, name, node_type, **properties):

    validate_class(node_type)

    graph.add_node(
        name,
        type=node_type,
        **properties
    )


def add_edge(graph, source, target, relation):

    source_type = graph.nodes[source]["type"]
    target_type = graph.nodes[target]["type"]

    validate_relation(
        source_type,
        relation,
        target_type
    )

    graph.add_edge(
        source,
        target,
        relation=relation
    )


# ============================================================
# LOAD DATA
# ============================================================

columns = [
    "engine",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
]

columns += [
    f"sensor_{i}"
    for i in range(1, 22)
]


df = pd.read_csv(
    NASA_FILE,
    sep=r"\s+",
    header=None,
    names=columns
)


maintenance = pd.read_csv(
    MAINTENANCE_FILE
)


engine = df[
    df["engine"] == ENGINE_ID
].copy()


# ============================================================
# ANALYTICS
# ============================================================

baseline = engine[
    engine["cycle"] <= BASELINE_CYCLES
][SENSOR]


mean = baseline.mean()
std = baseline.std()


engine["z_score"] = (
    engine[SENSOR] - mean
) / std


engine["anomaly_score"] = (
    engine["z_score"].abs()
)


engine["raw_anomaly"] = (
    (engine["cycle"] > BASELINE_CYCLES)
    &
    (engine["anomaly_score"] >= ANOMALY_THRESHOLD)
)


engine["persistent_anomaly"] = (
    engine["raw_anomaly"]
    .rolling(PERSISTENCE_WINDOW)
    .sum()
    >= PERSISTENCE_WINDOW
)


events = engine[
    engine["persistent_anomaly"]
]


if events.empty:

    st.error(
        "No persistent degradation event detected."
    )

    st.stop()


first_event = events.iloc[0]

event_cycle = int(
    first_event["cycle"]
)

sensor_value = float(
    first_event[SENSOR]
)

z_score = float(
    first_event["z_score"]
)


event_type = (
    "Critical Anomaly"
    if abs(z_score) >= 3
    else "Degradation Signal"
)


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================

G = nx.DiGraph()


engine_node = "Engine 01"

system_node = "Turbofan Engine"

component_node = (
    "High Pressure Compressor"
)

sensor_node = "Sensor 11"

event_node = (
    f"{event_type} - Cycle {event_cycle}"
)

degradation_node = "HPC Degradation"

risk_node = "Engine Failure"


add_node(
    G,
    engine_node,
    "Asset",
    engine_id=ENGINE_ID,
    source="NASA C-MAPSS"
)

add_node(
    G,
    system_node,
    "System"
)

add_node(
    G,
    component_node,
    "Component"
)

add_node(
    G,
    sensor_node,
    "Sensor",
    source="NASA C-MAPSS",
    source_column=SENSOR
)

add_node(
    G,
    event_node,
    "Event",
    cycle=event_cycle,
    sensor_value=round(sensor_value, 4),
    z_score=round(z_score, 2)
)

add_node(
    G,
    degradation_node,
    "FailureMode"
)

add_node(
    G,
    risk_node,
    "Risk"
)


add_edge(
    G,
    engine_node,
    system_node,
    "HAS_SYSTEM"
)

add_edge(
    G,
    system_node,
    component_node,
    "HAS_COMPONENT"
)

add_edge(
    G,
    component_node,
    sensor_node,
    "MONITORED_BY"
)

add_edge(
    G,
    engine_node,
    event_node,
    "HAS_EVENT"
)

add_edge(
    G,
    event_node,
    sensor_node,
    "OBSERVED_BY"
)

add_edge(
    G,
    event_node,
    degradation_node,
    "EVIDENCE_FOR"
)

add_edge(
    G,
    degradation_node,
    component_node,
    "AFFECTS"
)

add_edge(
    G,
    degradation_node,
    risk_node,
    "MAY_LEAD_TO"
)


# ============================================================
# MAINTENANCE DATA
# ============================================================

hpc_maintenance = maintenance[
    maintenance["component"]
    == component_node
]


for _, row in hpc_maintenance.iterrows():

    action = str(
        row["maintenance_action"]
    )

    add_node(
        G,
        action,
        "MaintenanceAction",
        duration_hours=int(
            row["duration_hours"]
        ),
        cost_eur=int(
            row["cost_eur"]
        ),
        operational_impact=str(
            row["operational_impact"]
        )
    )

    add_edge(
        G,
        degradation_node,
        action,
        "MITIGATED_BY"
    )


# ============================================================
# TOP SUMMARY
# ============================================================

st.subheader("Asset overview")


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Asset",
    "Engine 01"
)


c2.metric(
    "Detected event",
    event_type
)


c3.metric(
    "Detected at",
    f"Cycle {event_cycle}"
)


c4.metric(
    "Z-score",
    f"{z_score:.2f}"
)


st.divider()


# ============================================================
# LEFT + RIGHT
# ============================================================

left, right = st.columns(
    [1, 1.4]
)


# ============================================================
# SENSOR EVIDENCE
# ============================================================

with left:

    st.subheader("Detection evidence")

    st.write(
        f"""
        **Sensor:** {SENSOR}

        **Observed value:** {sensor_value:.2f}

        **Baseline:** first {BASELINE_CYCLES} cycles

        **Persistent anomaly:** detected
        """
    )


    fig_sensor, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        engine["cycle"],
        engine[SENSOR]
    )

    ax.axvline(
        event_cycle,
        linestyle="--",
        color="red",
        label="First persistent event"
    )

    ax.set_xlabel(
        "Cycle"
    )

    ax.set_ylabel(
        SENSOR
    )

    ax.legend()

    st.pyplot(
        fig_sensor
    )


# ============================================================
# MAINTENANCE TABLE
# ============================================================

with right:

    st.subheader(
        "Available mitigation actions"
    )


    display_maintenance = (
        hpc_maintenance[
            [
                "maintenance_action",
                "duration_hours",
                "cost_eur",
                "operational_impact"
            ]
        ]
        .copy()
    )


    display_maintenance.columns = [
        "Action",
        "Duration (h)",
        "Cost (EUR)",
        "Operational impact"
    ]


    st.dataframe(
        display_maintenance,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# ============================================================
# GRAPH RETRIEVAL FUNCTIONS
# ============================================================

def find_start_nodes(question):

    q = question.lower()

    matches = []


    if (
        "engine 1" in q
        or "engine 01" in q
        or "motore 1" in q
    ):

        matches.append(
            engine_node
        )


    if (
        "hpc" in q
        or "degradation" in q
        or "degradazione" in q
    ):

        matches.append(
            degradation_node
        )


    maintenance_words = [
        "maintenance",
        "inspection",
        "repair",
        "replacement",
        "cost",
        "duration",
        "manutenzione",
        "ispezione",
        "riparazione",
        "sostituzione",
        "costo",
        "durata"
    ]


    if any(
        word in q
        for word in maintenance_words
    ):

        matches.append(
            degradation_node
        )


    if not matches:

        matches.append(
            engine_node
        )


    return list(
        dict.fromkeys(matches)
    )


def retrieve_graph(
    start_nodes,
    depth=4
):

    selected = set(
        start_nodes
    )


    for start in start_nodes:

        reachable = (
            nx.single_source_shortest_path_length(
                G,
                start,
                cutoff=depth
            )
        )

        selected.update(
            reachable.keys()
        )


    return G.subgraph(
        selected
    ).copy()


def graph_context(graph):

    facts = []


    for source, target, data in graph.edges(
        data=True
    ):

        facts.append(

            f"{source} "
            f"--{data['relation']}--> "
            f"{target}"
        )


    for node, data in graph.nodes(
        data=True
    ):

        properties = [
            f"{key}={value}"
            for key, value in data.items()
            if key != "type"
        ]


        if properties:

            facts.append(

                f"{node} "
                f"[type={data['type']}]: "
                + ", ".join(properties)
            )


    return "\n".join(
        facts
    )


# ============================================================
# ASK FORGE
# ============================================================

st.subheader(
    "Ask FORGE"
)


question = st.text_input(

    "Ask a question about the asset",

    placeholder=(
        "e.g. What is happening to Engine 01 "
        "and what are the available mitigation options?"
    )
)


if question:

    start_nodes = (
        find_start_nodes(
            question
        )
    )


    retrieved_graph = (
        retrieve_graph(
            start_nodes
        )
    )


    context = graph_context(
        retrieved_graph
    )


    prompt = f"""
You are an engineering assistant.

Answer the user's question using ONLY the facts
contained in the retrieved Knowledge Graph.

Do not invent information.

Semantic rules:

EVIDENCE_FOR means evidence for a possible
condition, not a confirmed diagnosis.

MAY_LEAD_TO means a potential consequence,
not a certainty.

Maintenance actions, durations, costs and
operational impacts are synthetic demonstration
data and are not NASA maintenance recommendations.

If the requested information is not available
in the graph, say so explicitly.

KNOWLEDGE GRAPH:

{context}

QUESTION:

{question}

Provide a concise engineering answer.
"""


    with st.spinner(
        "Traversing knowledge graph and asking the LLM..."
    ):

        try:

        client = InferenceClient(
            api_key=st.secrets["HF_TOKEN"],
            provider="auto"
        )
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=500
        )
        
        answer = response.choices[0].message.content

        except Exception as error:

            st.error(
                "Unable to contact the cloud LLM."
            )

            st.code(
                str(error)
            )

            st.stop()


    st.success(
        answer
    )


    with st.expander(
        "Show retrieved knowledge"
    ):

        st.code(
            context
        )


    # ========================================================
    # RETRIEVED GRAPH
    # ========================================================

    st.subheader(
        "Retrieved Knowledge Graph"
    )


    colors = {
        "Asset": "#4472C4",
        "System": "#5B9BD5",
        "Component": "#70AD47",
        "Sensor": "#A5A5A5",
        "Event": "#FFC000",
        "FailureMode": "#ED7D31",
        "Risk": "#C00000",
        "MaintenanceAction": "#7030A0",
    }


    node_colors = [

        colors.get(
            retrieved_graph.nodes[node].get(
                "type"
            ),
            "#D9D9D9"
        )

        for node in retrieved_graph.nodes
    ]


    pos = nx.spring_layout(
        retrieved_graph,
        seed=42,
        k=2
    )


    fig_graph, ax_graph = plt.subplots(
        figsize=(13, 7)
    )


    nx.draw(

        retrieved_graph,

        pos,

        ax=ax_graph,

        with_labels=True,

        node_color=node_colors,

        node_size=3500,

        font_size=7,

        arrows=True,

        arrowsize=16
    )


    edge_labels = (
        nx.get_edge_attributes(
            retrieved_graph,
            "relation"
        )
    )


    nx.draw_networkx_edge_labels(

        retrieved_graph,

        pos,

        ax=ax_graph,

        edge_labels=edge_labels,

        font_size=6
    )


    ax_graph.axis(
        "off"
    )


    st.pyplot(
        fig_graph
    )


# ============================================================
# EXPLANATORY FOOTER
# ============================================================

st.divider()

st.caption(
    "Demonstrator architecture: NASA C-MAPSS → Analytics → "
    "Ontology → Knowledge Graph → Graph Retrieval → Local LLM. "
    "Maintenance data are synthetic."
)
