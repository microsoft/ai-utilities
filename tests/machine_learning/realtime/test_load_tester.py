import collections
import json
import random
import time
from datetime import datetime
from threading import Thread, RLock

# Named tuple to collect from each call
import pytest
import requests
from azureml.core.webservice import AksWebservice

from azure_utils.machine_learning.contexts.workspace_contexts import WorkspaceContext
from azure_utils.machine_learning.realtime.load_tester import dumpStats, getStatistics


@pytest.fixture
def workspace():
    return WorkspaceContext.get_or_create_workspace()


# Collection of each call made
test_collection = []
# Global lock to protect collection
test_collection_lock = RLock()

# Counter of running threads
running_threads = 0


class ThreadRun(Thread):
    """
        Class used as a thread to run the load test against the
        endpoint.
    """

    def __init__(self, id, iterations, url, headers, payload, files=None):
        Thread.__init__(self)
        self.id = id
        self.iterations = iterations
        self.url = url
        self.headers = headers
        self.payload = payload
        self.files = files

    """
        Calling start() runs this as well, but when you queue a thread it will 
        trigger this as well. 
    """

    def run(self):
        global test_collection
        global test_collection_lock
        global running_threads

        test_point = collections.namedtuple("test_point", "thread status elapsed")
        print("Staring thread", self.id)
        for i in range(self.iterations):
            try:
                response = requests.post(
                    url=self.url,
                    headers=self.headers,
                    data=json.dumps(self.payload),
                    files=self.files,
                )
                current_test = test_point(
                    self.id, response.status_code, response.elapsed.total_seconds()
                )
            except Exception as ex:
                print(self.id, ex)
                print(str(ex))
                current_test = test_point(self.id, 500, 1)

            test_collection_lock.acquire()
            test_collection.append(current_test)
            test_collection_lock.release()

        test_collection_lock.acquire()
        running_threads -= 1
        test_collection_lock.release()


def getThreadStatistics():
    """
        Load statistics of the run. Two items are returned as a list

        [0] = Dictionary of global stats
        [1] = Dictionary of dictionaries for each thread.
    """
    global test_collection

    """
        Get stats across threads
    """
    global_stats = getStatistics(test_collection)

    """
        Get individual stats
    """
    thread_stats = {}
    thread_ids = [x.thread for x in test_collection]
    thread_ids = list(set(thread_ids))
    for tid in thread_ids:
        thread_stats[tid] = {}
        thread_collection = [x for x in test_collection if x.thread == tid]
        thread_stats[tid] = getStatistics(thread_collection)

    return [global_stats, thread_stats]


def test_run(workspace, threads, iterations):
    global test_collection
    global test_collection_lock
    global running_threads

    aks_service_name = "deepaksservice"

    print("update")
    assert aks_service_name in workspace.webservices, f"{aks_service_name} not found."
    aks_service = AksWebservice(workspace, name=aks_service_name)
    assert (
        aks_service.state == "Healthy"
    ), f"{aks_service_name} is in state {aks_service.state}."
    scoring_url = aks_service.scoring_uri
    print(scoring_url)
    api_key = aks_service.get_keys()[0]

    headers = {"Authorization": ("Bearer " + api_key)}
    files = {"image": open("snowleopardgaze.jpg", "rb")}
    test_point = collections.namedtuple("test_point", "thread status elapsed")

    # Headers for every call
    api_headers = headers

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
    configuration = {"t": threads, "i": iterations, "u": scoring_url}
    names = ["Dave", "Sue", "Dan", "Joe", "Beth"]

    # Capture the start time.
    start_time = datetime.now()
    print("Starting")

    for i in range(configuration["t"]):
        payload = {"name": names[random.randint(0, len(names) - 1)]}
        run = ThreadRun(
            i + 1,
            configuration["i"],
            configuration["u"],
            api_headers,
            payload,
            files=files,
        )

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
