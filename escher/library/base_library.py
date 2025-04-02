from copy import deepcopy
from typing import Callable, Dict, List

from escher.cbd_utils import process_descriptions


class Library:
    """
    I don't think there is any case where this base class
    itself will be useful.

    Might be a better idea to:
    (1) TODO: re-attempt refactoring Library and HistoryConditionedLibrary.
    (2) Till then, extend HistoryConditionedLibrary instead of this class.
    """

    def __init__(self):
        self.cls2concepts = None
        self.classes = None
        self.initialized = False

    def initialize(self, initial_descriptors, classes, dedup_func: Callable = None):
        self.cls2concepts = deepcopy(initial_descriptors)
        self.dedup_func = dedup_func if dedup_func else lambda x: x
        self.classes = classes
        self.initialized = True

    def get_library(
        self, should_process_descriptors=True, custom_proc_func=None
    ) -> Dict[str, List[str]]:
        if should_process_descriptors:
            if custom_proc_func:
                return custom_proc_func(self.cls2concepts)
            return process_descriptions(self.cls2concepts)
        return self.cls2concepts

    def get_attributes(self):
        return list(
            set([attr for dlist in self.cls2concepts.values() for attr in dlist])
        )

    def update_class(self, cls, suggested_descriptors):
        for descriptor in suggested_descriptors:
            if descriptor not in self.cls2concepts[cls]:
                self.cls2concepts[cls].append(descriptor)

    # def subselect_class

    # Move deduplicate_descriptors from model.py to here.
    # Modify it so it runs for a single class.
    # Insert into library

    # Deduplicate within class.
    # Deduplicate across classes.
    # Get descriptors for a class.
    # Subselect descriptors in a class.

    def dedup_across_classes(self):
        self.cls2concepts = self.dedup_func(self.cls2concepts)
