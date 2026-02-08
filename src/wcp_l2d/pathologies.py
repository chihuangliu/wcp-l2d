"""Pathology constants and label alignment mappings.

Defines the canonical 7-pathology list shared between CheXpert and NIH,
plus index mappings into the DenseNet model's 18-slot output.
"""

# 7 common pathologies between CheXpert (13) and NIH (14), alphabetically sorted.
# This is the canonical ordering for ALL downstream work.
COMMON_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Pneumonia",
    "Pneumothorax",
]

NUM_PATHOLOGIES = len(COMMON_PATHOLOGIES)  # 7

# The densenet121-res224-chex model's 18-slot output ordering.
# Empty strings indicate unused slots.
CHEX_MODEL_TARGETS = [
    "Atelectasis",       # 0
    "Consolidation",     # 1
    "",                   # 2
    "Pneumothorax",      # 3
    "Edema",             # 4
    "",                   # 5
    "",                   # 6
    "Effusion",          # 7
    "Pneumonia",         # 8
    "",                   # 9
    "Cardiomegaly",      # 10
    "",                   # 11
    "",                   # 12
    "",                   # 13
    "Lung Lesion",       # 14
    "Fracture",          # 15
    "Lung Opacity",      # 16
    "Enlarged Cardiomediastinum",  # 17
]

# Mapping: common pathology name → index in the chex model's 18-slot output
COMMON_TO_MODEL_IDX = {
    "Atelectasis": 0,
    "Cardiomegaly": 10,
    "Consolidation": 1,
    "Edema": 4,
    "Effusion": 7,
    "Pneumonia": 8,
    "Pneumothorax": 3,
}

# Ordered list of model output indices for our 7 common pathologies
COMMON_MODEL_INDICES = [COMMON_TO_MODEL_IDX[p] for p in COMMON_PATHOLOGIES]
