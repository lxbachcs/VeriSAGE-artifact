"""LLM proposer for VeriSAGE. Local models via Ollama (:11434); OpenAI models
(model id starting 'gpt-') via the Chat Completions API, key from $OPENAI_API_KEY.
The key is read from the environment only and never logged."""
from __future__ import annotations
import urllib.request, json, re, os
from typing import List

OLLAMA = "http://localhost:11434/api/generate"
OPENAI = "https://api.openai.com/v1/chat/completions"

_CALL_N = 0   # monotone per-process LLM-call counter (for per-call seeds)


def _call(model: str, prompt: str, temperature: float, timeout: int = 120) -> str:
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        body = {"model": model, "temperature": temperature, "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(OPENAI, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"})
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        return r["choices"][0]["message"]["content"]
    opts = {"temperature": temperature, "num_predict": 300}
    _base = os.environ.get("VERISAGE_LLM_SEED")
    if _base is not None:
        # Unique, reproducible seed PER CALL (base + monotone counter): makes
        # blind-resampling draw genuinely independent samples and gives each
        # task/repair-round its own draw, while a fixed base keeps the whole
        # replica reproducible.
        global _CALL_N
        opts["seed"] = int(_base) + _CALL_N
        _CALL_N += 1
    body = {"model": model, "prompt": prompt, "stream": False, "options": opts}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read()).get("response", "")


def _extract_names(text: str) -> List[str]:
    """Pull the first JSON array of strings from the model output; robust to keys."""
    # try any [...] array of quoted strings
    for m in re.finditer(r"\[[^\[\]]*\]", text, re.S):
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr) and arr:
                return arr
        except Exception:
            pass
    # try a JSON object with a list value
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        for v in obj.values():
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return v
    except Exception:
        pass
    # last resort: quoted tokens
    toks = re.findall(r'"([A-Za-z][A-Za-z0-9_]+)"', text)
    return toks


def propose(model: str, catalogue_desc: str, goal_nl: str,
            repairs: List[str], temperature: float = 0.7,
            prev_plan: List[str] = None) -> List[str]:
    if prev_plan:
        # REPAIR: patch the previous plan given the specific counterexample(s),
        # keeping everything that already worked (CEGIS-style incremental fix).
        r = "\n- ".join(repairs)
        prompt = (
            "You are repairing a service composition. Below is your previous plan "
            "and the verifier's specific counterexample(s). Return a CORRECTED JSON "
            "array of service names that FIXES the reported problem(s) while keeping "
            "the rest of the plan intact (add missing producers, reorder, or insert "
            "the required step; do not drop steps that already satisfy the goal).\n\n"
            f"CATALOGUE (name: inputs/outputs):\n{catalogue_desc}\n\n"
            f"GOAL: {goal_nl}\n\n"
            f"PREVIOUS PLAN: {json.dumps(prev_plan)}\n\n"
            f"VERIFIER COUNTEREXAMPLES (fix these):\n- {r}\n\n"
            'Answer with ONLY the corrected JSON array, e.g. ["svcA","svcB"].'
        )
    else:
        prompt = (
            "You are a service-composition planner. Given a catalogue and a goal, "
            "output ONLY a JSON array of service names (from the catalogue) in the "
            "exact execution order. No prose.\n\n"
            f"CATALOGUE (name: inputs/outputs):\n{catalogue_desc}\n\n"
            f"GOAL: {goal_nl}\n\n"
            'Answer with a JSON array, e.g. ["svcA","svcB"].'
        )
    try:
        out = _call(model, prompt, temperature)
    except Exception:
        return []
    return _extract_names(out)


# --------------------------------------------------------------------------
# SOTA self-repair baseline: Self-Refine (Madaan et al. 2023) / Reflexion
# (Shinn et al. 2023). The model critiques and revises its OWN plan with NO
# formal verifier in the loop -- the honest "why not just let the LLM reflect?"
# comparison to VeriSAGE's typed-counterexample repair.
# --------------------------------------------------------------------------
def self_critique(model: str, catalogue_desc: str, goal_nl: str,
                  cur_plan: List[str], temperature: float = 0.5) -> str:
    """Ask the model to critique its own plan (no ground-truth verifier). Returns
    free-text feedback; 'OK' (verbatim) means the model believes it is correct."""
    prompt = (
        "You are reviewing YOUR OWN service-composition plan before running it. "
        "Critique it: list any problems that would make it unsafe or invalid to "
        "execute -- missing prerequisites, wrong ordering, data-privacy leaks "
        "(sensitive fields reaching a public sink), policy violations, or exceeding "
        "the latency budget. Be specific and brief. If the plan is fully correct, "
        "reply with exactly OK.\n\n"
        f"CATALOGUE (name: inputs/outputs):\n{catalogue_desc}\n\n"
        f"GOAL: {goal_nl}\n\n"
        f"YOUR PLAN: {json.dumps(cur_plan)}\n\n"
        "Feedback:"
    )
    try:
        return _call(model, prompt, temperature)
    except Exception:
        return "OK"


def self_refine(model: str, catalogue_desc: str, goal_nl: str,
                cur_plan: List[str], feedback: str,
                temperature: float = 0.5) -> List[str]:
    """Revise the plan using the model's OWN feedback. No verifier witness."""
    prompt = (
        "Revise your service-composition plan using your own feedback below. "
        "Return ONLY a corrected JSON array of service names (from the catalogue) "
        "in execution order.\n\n"
        f"CATALOGUE (name: inputs/outputs):\n{catalogue_desc}\n\n"
        f"GOAL: {goal_nl}\n\n"
        f"CURRENT PLAN: {json.dumps(cur_plan)}\n\n"
        f"YOUR FEEDBACK:\n{feedback}\n\n"
        'Answer with ONLY the JSON array, e.g. ["svcA","svcB"].'
    )
    try:
        out = _call(model, prompt, temperature)
    except Exception:
        return cur_plan
    names = _extract_names(out)
    return names if names else cur_plan


# --------------------------------------------------------------------------
# LLM value function for LATS (Language Agent Tree Search, Zhou et al. 2024):
# score how likely a (partial) plan is to complete the task correctly. This is
# an LLM self-evaluation, NOT a formal verifier -- LATS uses it to guide search.
# --------------------------------------------------------------------------
def value_estimate(model: str, catalogue_desc: str, goal_nl: str,
                   plan: List[str], temperature: float = 0.2) -> float:
    """Return an LLM value in [0,1] estimating whether the plan will correctly
    and safely complete the goal (0 = certainly not, 1 = certainly yes)."""
    prompt = (
        "You are scoring a candidate service-composition plan. Given the catalogue "
        "and goal, estimate how likely this plan is to complete the goal correctly "
        "and safely (right order, prerequisites met, no data leak, within budget). "
        "Answer with ONLY a single integer from 0 to 10 (10 = certainly correct).\n\n"
        f"CATALOGUE (name: inputs/outputs):\n{catalogue_desc}\n\n"
        f"GOAL: {goal_nl}\n\n"
        f"PLAN: {json.dumps(plan)}\n\n"
        "Score (0-10):"
    )
    try:
        out = _call(model, prompt, temperature)
    except Exception:
        return 0.0
    m = re.search(r"\d+", out)
    if not m:
        return 0.0
    return max(0.0, min(1.0, int(m.group(0)) / 10.0))


def propose_next(model: str, catalogue_desc: str, goal_nl: str,
                 prefix: List[str], k: int, temperature: float = 0.8) -> List[str]:
    """For LATS expansion: given a plan prefix, propose up to k distinct next
    service names to append (or an empty string to signal 'stop / plan done')."""
    prompt = (
        "You are extending a service-composition plan one step at a time. Given the "
        "catalogue, goal, and the plan so far, list up to "
        f"{k} good candidate NEXT service names to append (from the catalogue), most "
        "promising first. If the plan already achieves the goal, return [\"STOP\"].\n\n"
        f"CATALOGUE (name: inputs/outputs):\n{catalogue_desc}\n\n"
        f"GOAL: {goal_nl}\n\n"
        f"PLAN SO FAR: {json.dumps(prefix)}\n\n"
        'Answer with ONLY a JSON array of service names, e.g. ["svcA","svcB"].'
    )
    try:
        out = _call(model, prompt, temperature)
    except Exception:
        return []
    return _extract_names(out)[:k]
