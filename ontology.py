# ============================================================
# FORGE DEMONSTRATOR - MINI ONTOLOGY
# ============================================================


# ============================================================
# ONTOLOGY CLASSES
# ============================================================

CLASSES = {
    "Asset",
    "System",
    "Component",
    "Sensor",
    "Event",
    "FailureMode",
    "Risk",
    "MaintenanceAction",
}


# ============================================================
# CLASS HIERARCHY
# ============================================================

CLASS_HIERARCHY = {
    "System": "Asset",
}


# ============================================================
# RELATIONS
#
# Each relation defines:
# - domain = allowed source class
# - range  = allowed target class
# ============================================================

RELATIONS = {

    "HAS_SYSTEM": {
        "domain": "Asset",
        "range": "System",
    },

    "HAS_COMPONENT": {
        "domain": "System",
        "range": "Component",
    },

    "MONITORED_BY": {
        "domain": "Component",
        "range": "Sensor",
    },

    "HAS_EVENT": {
        "domain": "Asset",
        "range": "Event",
    },

    "OBSERVED_BY": {
        "domain": "Event",
        "range": "Sensor",
    },

    "EVIDENCE_FOR": {
        "domain": "Event",
        "range": "FailureMode",
    },

    "AFFECTS": {
        "domain": "FailureMode",
        "range": "Component",
    },

    "MAY_LEAD_TO": {
        "domain": "FailureMode",
        "range": "Risk",
    },

    "MITIGATED_BY": {
        "domain": "FailureMode",
        "range": "MaintenanceAction",
    },
}


# ============================================================
# VALIDATION
# ============================================================

def validate_class(class_name):

    if class_name not in CLASSES:

        raise ValueError(
            f"Unknown ontology class: {class_name}"
        )

    return True


def validate_relation(
    source_class,
    relation,
    target_class
):

    validate_class(source_class)
    validate_class(target_class)

    if relation not in RELATIONS:

        raise ValueError(
            f"Unknown ontology relation: {relation}"
        )

    rule = RELATIONS[relation]

    expected_source = rule["domain"]
    expected_target = rule["range"]

    if source_class != expected_source:

        raise ValueError(
            f"Invalid source for {relation}: "
            f"{source_class}. "
            f"Expected {expected_source}."
        )

    if target_class != expected_target:

        raise ValueError(
            f"Invalid target for {relation}: "
            f"{target_class}. "
            f"Expected {expected_target}."
        )

    return True


# =========