# -*- coding: utf-8 -*-
from dwi_ml.projects.Transformers.transformer_models import AbstractTransformerModel
from dwi_ml.general.tracking.tracker import (
    DWIMLTrackerOneInput, DWIMLTrackerFromWholeStreamline)


# Just combining the two Abstract Classes of interest.
class TransformerTracker(DWIMLTrackerOneInput,
                         DWIMLTrackerFromWholeStreamline):
    """
    For the Transformer, we simply need OneInput + StreamlineMemory.
    """
    model: AbstractTransformerModel

    def __init__(self, **kw):
        super().__init__(verify_opposite_direction=False, **kw)
