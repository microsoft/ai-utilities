from azure_utils.machine_learning.factories.realtime_factory import RealTimeFactory

rts_factory = RealTimeFactory()
init = rts_factory.score_init
run = rts_factory.score_run
