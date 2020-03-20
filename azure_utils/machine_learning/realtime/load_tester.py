import argparse
import collections
import json
import statistics
from threading import Thread, RLock

import requests
from azureml.contrib.services.aml_response import AMLResponse


def default_response(request) -> AMLResponse:
    """

    :param request:
    :return:
    """
    if request.method == "GET":
        return AMLResponse({"azEnvironment": "Azure"}, 201)
    return AMLResponse("bad request", 500)


# def loadArguments(sys_args):
#     """
#         Load arguments for the program:
#
#         u = URL of the Web Service URL
#         k = API Key for the web service
#         t = Number of threads to run
#         i = Number of calls to make per thread.
#     """
#     global api_headers
#
#     parser = argparse.ArgumentParser(description="Simple model deployment.")
#     parser.add_argument(
#         "-u",
#         required=False,
#         default="http://40.121.6.20:80/api/v1/service/dummycluster/score",
#         type=str,
#         help="Web Service URI",
#     )
#     parser.add_argument(
#         "-k",
#         required=False,
#         default="eGKplWLsq0AKDFx8gb5SyaKU8AeoDqOc",
#         type=str,
#         help="Web Service Key",
#     )
#     parser.add_argument("-t", required=False, default=20, type=int, help="Thread Count")
#     parser.add_argument(
#         "-i", required=False, default=1, type=int, help="Thread Iterations"
#     )
#
#     prog_args = parser.parse_args(sys_args)
#
#     api_headers["Authorization"] = "Bearer " + prog_args.k
#     api_headers["Content-Type"] = "application/json"
#
#     return prog_args


def dumpStats(stats):
    """
        Dump out a dictionary of stats
    """
    for key in stats.keys():
        print("    ", key, "=", stats[key])


def getStatistics(collection):
    """
        From a collection of test_point objects collect the following
        - T0tal Calls
        - Succesful calls
        - Average latency
        - Min latency
        - Maximum Latency
    """
    stats = {}

    success = [x for x in collection if x.status == 200]
    times = [x.elapsed for x in collection]

    stats["calls"] = len(collection)
    stats["success"] = len(success)
    stats["average"] = statistics.mean(times)
    stats["min"] = min(times)
    stats["max"] = max(times)

    return stats


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


test_collection_lock = RLock()
test_collection = []


class ThreadRun(Thread):
    """
        Class used as a thread to run the load test against the
        endpoint.
    """

    def __init__(self, id, running_threads, iterations, url, headers, payload, files=None):
        Thread.__init__(self)
        self.id = id
        self.running_threads = running_threads
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
        self.running_threads -= 1
        test_collection_lock.release()
