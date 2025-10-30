"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from spaceflights.pipelines import data_processing as dp
from spaceflights.pipelines import data_science as ds
from spaceflights.pipelines import reporting as rp

def register_pipelines() -> Dict[str, Pipeline]:
    from spaceflights.pipelines import data_processing as dp
    from spaceflights.pipelines import data_science as ds
    from spaceflights.pipelines import reporting as rp

    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    reporting_pipeline = rp.create_pipeline()

    return {
        "data_processing": data_processing_pipeline,
        "data_science": data_science_pipeline,
        "reporting": reporting_pipeline,
        "__default__": data_processing_pipeline + data_science_pipeline + reporting_pipeline,
    }