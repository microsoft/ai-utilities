"""
    Author: grecoe

    After deploying your Web Service, you can use this program to load test the endpoint.

    Read the function loadArguments() to determine what parameters to pass in.

    Flow:
        1. Create and start t number of threads where t is identified in the parameters.
            Each thread will run for i iterations (calls to the endpoint) where i is identified
            in the parameters.
        2. Upon completion of all threads collect statistics.
            - All up stats on all calls from all threads
            - Individual thread statistics
            - Each stats bundle contains
                Total Number of Calls
                Succesful Number of Calls
                Average Latency (seconds)
                Min Latency (seconds)
                Max Latency (seconds)
        3. Print the results to the console.
"""

from threading import *
from datetime import datetime
import sys
import time
import collections
import random

# Named tuple to collect from each call
from azure_utils.machine_learning.realtime.load_tester import loadArguments, dumpStats, getThreadStatistics, ThreadRun

test_point = collections.namedtuple("test_point", "thread status elapsed")
# Collection of each call made
test_collection = []
# Global lock to protect collection
test_collection_lock = RLock()

# Counter of running threads
running_threads = 0

# Headers for every call
api_headers = {}

"""
    Program Code:

    The configured number of threads will be executed for the configured number of iterations each 
    hitting the endpoint. 

    This can be used for any number of AMLS endpoints, with the real change being to the payload that is set 
    into the thread class to perform the execution. 
"""

"""
    Using the configuration, fire up as many threads as we need. 
"""
configuration = loadArguments(sys.argv[1:])
names = ["Dave", "Sue", "Dan", "Joe", "Beth"]

# Capture the start time.
start_time = datetime.now()

for i in range(configuration.t):
    payload = {"name": names[random.randint(0, len(names) - 1)]}
    run = ThreadRun(i + 1, configuration.i, configuration.u, api_headers, payload)

    # Increase the thread counter
    test_collection_lock.acquire()
    running_threads += 1
    test_collection_lock.release()

    # Start the worker thread.
    run.start()

"""
    Wait until all threads complete.
"""
counter = 0
while running_threads > 0:
    counter += 1
    if counter % 3 == 0:
        print("Waiting on threads, current count =", running_threads)
    time.sleep(0.5)

# Capture the start time.
end_time = datetime.now()
total_seconds = (end_time - start_time).total_seconds()
print(total_seconds)

"""
    Get and print out the statistics for this run. 
"""
stats = getThreadStatistics()
print("Global Stats:")
print("     Total Time  : ", total_seconds)
print("     Overall RPS : ", stats[0]["calls"] / total_seconds)
dumpStats(stats[0])
for thread_id in stats[1].keys():
    print("Thread", thread_id, "Stats:")
    dumpStats(stats[1][thread_id])
