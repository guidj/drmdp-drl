import logging

import ray


def wait_till_completion(tasks_refs):
    """
    Waits for every ray task to complete.
    """
    unfinished_tasks = tasks_refs
    while True:
        finished_tasks, unfinished_tasks = ray.wait(unfinished_tasks)
        logging.info(
            "Completed %d task(s). %d left out of %d.",
            len(finished_tasks),
            len(unfinished_tasks),
            len(tasks_refs),
        )

        if len(unfinished_tasks) == 0:
            break
