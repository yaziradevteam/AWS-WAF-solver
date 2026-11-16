import base64
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

from colorama import Fore, Style, init
from wreq.blocking import Client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset
from solver import HybridSolver

init()

base_website = "www.officeworks.com.au"
home_url = "https://www.officeworks.com.au/"
get_captcha = (
    "https://db02010d94bb.8093b5d4.ap-south-1.captcha.awswaf.com/db02010d94bb/b4e7dcbb99d9/7b880e4c169a/problem?kind=visual&domain="
    + base_website
    + "&locale=en-us"
)
verify_url = (
    "https://db02010d94bb.8093b5d4.ap-south-1.captcha.awswaf.com/db02010d94bb/b4e7dcbb99d9/7b880e4c169a/verify?region=ap-south-1"
)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36"

HOME_HEADERS = {
    "user-agent": UA,
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "sec-ch-ua": '"Google Chrome";v="153", "Not_A Brand";v="8", "Chromium";v="153"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
}

PROBLEM_HEADERS = {
    "user-agent": UA,
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "referer": "https://www.officeworks.com.au/",
    "origin": "https://www.officeworks.com.au",
    "sec-ch-ua": '"Google Chrome";v="153", "Not_A Brand";v="8", "Chromium";v="153"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "cross-site",
}

VERIFY_HEADERS = {
    "user-agent": UA,
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "content-type": "text/plain;charset=UTF-8",
    "origin": "https://www.officeworks.com.au",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "referer": "https://www.officeworks.com.au/",
    "sec-ch-ua": '"Google Chrome";v="153", "Not_A Brand";v="8", "Chromium";v="153"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "cross-site",
}

GOKU_RE = re.compile(r"window\.gokuProps\s*=\s*(\{.*?\})\s*;", re.S)


def extract_goku_props(html):
    m = GOKU_RE.search(html)
    if not m:
        raise RuntimeError("gokuProps not found in page")
    return json.loads(m.group(1))


def normalize_target(problem):
    target_raw = problem["assets"]["target"]
    words = json.loads(target_raw) if isinstance(target_raw, str) else target_raw
    word = words[0].strip().lower()
    if word.endswith("s") and word[:-1] in dataset.TARGETS:
        word = word[:-1]
    return word


def decode_images(problem, folder):
    images_raw = problem["assets"]["images"]
    images = json.loads(images_raw) if isinstance(images_raw, str) else images_raw
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, image_base64 in enumerate(images):
        decoded = base64.b64decode(image_base64)
        filename = os.path.join(folder, f"captcha_{i}.jpg")
        with open(filename, "wb") as f:
            f.write(decoded)
        paths.append(filename)
    return paths


def solve_once(solver, index):
    start = time.time()
    folder = None
    diag = ""
    try:
        with Client(timeout=timedelta(seconds=30), cookie_store=True) as client:
            with client.get(home_url, headers=HOME_HEADERS) as home_resp:
                goku_props = extract_goku_props(home_resp.text())

            with client.get(get_captcha, headers=PROBLEM_HEADERS) as response:
                problem = response.json()

            target = normalize_target(problem)
            folder = tempfile.mkdtemp(prefix="captcha_solve_")
            image_paths = decode_images(problem, folder)
            solution, hashes, used = solver.solve(image_paths, target)
            diag = f"target={target} picked={solution} [{used}]"

            body = {
                "state": problem["state"],
                "key": problem["key"],
                "hmac_tag": problem["hmac_tag"],
                "client_solution": solution,
                "metrics": {"solve_time_millis": random.randint(4000, 9000)},
                "goku_props": goku_props,
                "locale": "en-us",
            }
            payload = json.dumps(body, separators=(",", ":"))

            with client.post(verify_url, body=payload, headers=VERIFY_HEADERS) as verify_resp:
                text = verify_resp.text()
                try:
                    vj = json.loads(text)
                except Exception:
                    vj = {}

        elapsed = time.time() - start
        dataset.record_solve(image_paths, target, solution, bool(vj.get("success")))
        if vj.get("success"):
            return index, True, elapsed, vj.get("captcha_voucher")
        reason = vj.get("reason")
        detail = f"{reason} | {diag}" if reason else f"no reason | {diag}"
        if vj.get("problem"):
            detail += " | server returned new problem (wrong answer)"
        return index, False, elapsed, detail
    except Exception as e:
        return index, False, time.time() - start, f"{type(e).__name__}: {e}"
    finally:
        if folder:
            shutil.rmtree(folder, ignore_errors=True)


def main():
    print("loading hybrid solver (CLIP + DB)...")
    solver = HybridSolver()
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(solve_once, solver, i) for i in range(100)]
        for fut in as_completed(futures):
            index, ok, elapsed, extra = fut.result()
            if ok:
                voucher = extra or ""
                short = voucher if len(voucher) <= 60 else voucher[:40] + "..." + voucher[-15:]
                print(f"{Fore.GREEN}solved!{Style.RESET_ALL} [#{index}] in {elapsed:.2f}s  voucher: {short}")
            else:
                print(f"{Fore.RED}failed{Style.RESET_ALL}  [#{index}] in {elapsed:.2f}s ({extra})")

if __name__ == "__main__":
    main()
