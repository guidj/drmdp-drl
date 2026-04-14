import pytest
import ray

from drmdp import ray_utils


@pytest.fixture(scope="module", autouse=True)
def ray_cluster():
    with ray.init(num_cpus=1, ignore_reinit_error=True):
        yield


class TestWaitTillCompletion:
    def test_completes_single_task(self):
        @ray.remote
        def identity(x):
            return x

        refs = [identity.remote(42)]
        ray_utils.wait_till_completion(refs)
        assert ray.get(refs[0]) == 42

    def test_completes_multiple_tasks(self):
        @ray.remote
        def square(x):
            return x * x

        refs = [square.remote(idx) for idx in range(5)]
        ray_utils.wait_till_completion(refs)
        results = ray.get(refs)
        assert results == [idx * idx for idx in range(5)]

    def test_empty_task_list_completes_immediately(self):
        ray_utils.wait_till_completion([])
        # No tasks → nothing to get, but function must return
