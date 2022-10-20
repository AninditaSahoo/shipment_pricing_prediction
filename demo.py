
from supply_chain_.config.configuration import Configuration
from supply_chain_.pipeline.pipeline import Pipeline
def run():
    p=Pipeline(Configuration)
    print(p.run_pipeline())

run()
