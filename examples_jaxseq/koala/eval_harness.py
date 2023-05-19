import pprint
import urllib
import requests
import tyro
from lm_eval import evaluator, tasks
from lm_eval.base import LM
from lm_eval.tasks.hendrycks_test import SUBJECTS
from typing import List, Callable
from tqdm.auto import tqdm

class LMEvalHarnessInterface(LM):
    def __init__(
        self, 
        host: str, 
        bsize: int, 
        max_input_length: int=1024, 
        max_output_length: int=1024, 
        n_retries: int=3, 
    ):
        self.host = host
        self.bsize = bsize
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.n_retries = n_retries
        assert self.n_retries > 0, 'n_retries must be positive'

    def loglikelihood(self, inputs):
        in_strs, out_strs = zip(*inputs)
        out_strs = list(out_strs)
        results = []
        for i in tqdm(range(0, len(in_strs), self.bsize)):
            in_strs_batch = in_strs[i:(i+self.bsize)]
            out_strs_batch = out_strs[i:(i+self.bsize)]

            did_succeed = False
            for _ in range(self.n_retries):
                response = requests.post(
                    urllib.parse.urljoin(self.host, 'log_probs'), 
                    json={
                        'in_strs': in_strs_batch, 
                        'out_strs': out_strs_batch, 
                        'max_input_length': self.max_input_length, 
                        'max_output_length': self.max_output_length, 
                    }, 
                ).json()
                if response['status'] == 'success':
                    did_succeed = True
                    break
            if not did_succeed:
                raise Exception('Failed to get logprobs after {} retries'.format(self.n_retries))

            is_greedy = [False]*len(response['data']) # is_greedy is not used in the multiple choice evaluation code.
            results.extend(list(zip(response['data'], is_greedy)))
        return results
    
    def greedy_until(self, inputs):
        raise NotImplementedError
    
    def loglikelihood_rolling(self, inputs):
        raise NotImplementedError


def main(
    host: str, 
    k_shot: int, 
    task_names: List[str], 
    bsize: int=8, 
    n_retries: int=3, 
    max_input_length: int=1024, 
    max_output_length: int=1024, 
):
    model = LMEvalHarnessInterface(host, bsize, max_input_length, max_output_length, n_retries)
    results = evaluator.evaluate(
        model, tasks.get_task_dict(task_names), False, k_shot, None, 
    )
    pprint.pprint(results)

if __name__ == "__main__":
    tyro.cli(main)
