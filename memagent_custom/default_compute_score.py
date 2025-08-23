# Original path in memagent: memagent/verl/utils/reward_score/__init__.py

import os
import sys

# Add the VERL package to Python path so we can import from verl modules
current_dir = os.path.dirname(os.path.abspath(__file__))
verl_path = os.path.join(current_dir, "..", "verl")
if verl_path not in sys.path:
    sys.path.insert(0, verl_path)

# Add the parent directory to Python path so we can import from memagent_custom
parent_dir = os.path.join(current_dir, "..")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from verl.utils.import_utils import deprecated


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, eval_method='embed', **kwargs):
    if data_source == "implicit_persona":
        from . import reward_score
        res = reward_score.compute_score(solution_str, ground_truth, method=eval_method)

    elif data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)

    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval', 'AIME', 'AMC', 'MINERVA', "MATH", 'math_dapo'] or \
            'MATH' in data_source:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)

    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)

    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)

    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)

    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)

    elif data_source in ['hotpotqa']:
        from . import hotpotqa
        res = hotpotqa.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    
__all__ = ["default_compute_score"]
