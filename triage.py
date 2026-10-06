import argparse
import json
import urllib.request
from pathlib import Path

API_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:1b"
CATEGORIES = {"billing", "account", "technical", "general", "security"}
PRIORITIES = {"low", "medium", "high", "critical"}

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "priority": {"type": "string"},
        "summary": {"type": "string"},
        "escalate": {"type": "boolean"}
    },
    "required": ["category", "priority", "summary", "escalate"]
}

SYSTEM_PROMPT = """You are a support ticket triage engine.
Return values that follow the supplied JSON schema.

Category rules:
- billing: charges, payments, subscriptions, refunds, or invoices
- account: sign-in, password, profile, or account settings without a security concern
- technical: errors, outages, broken features, data imports, or data loss
- security: account compromise, credential exposure, unauthorized access, or malicious activity
- general: questions or requests that do not fit the other categories

Priority rules:
- critical: widespread outage, confirmed data loss, confirmed unauthorized access, or an active security incident
- high: one user is blocked, failures repeat, or action is needed within one day
- medium: billing dispute, single-user error with a workaround, or degraded service
- low: informational question or routine request with no current impact

Set escalate to true for every security ticket, every critical ticket, and explicit legal threats.
Use exactly one category and one priority value from the rules.
Write summary as one short sentence.
"""

def call_ollama(messages, output_format=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0}
    }
    if output_format is not None:
        payload["format"] = output_format

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except OSError as error:
        raise RuntimeError(
            "Could not reach Ollama at {}: {}".format(API_URL, error)
        ) from error

    return body["message"]["content"]


def triage_baseline(ticket):
    prompt = (
        "Classify this support ticket and explain your decision in two sentences.\n\n"
        "Ticket: " + ticket
    )
    return call_ollama([{"role": "user", "content": prompt}])


def validate_result(result):
    required_fields = {"category", "priority", "summary", "escalate"}

    if not isinstance(result, dict):
        raise ValueError("The model result is not a JSON object.")
    if set(result.keys()) != required_fields:
        raise ValueError("The result must contain exactly: {}".format(
            ", ".join(sorted(required_fields))
        ))
    if result["category"] not in CATEGORIES:
        raise ValueError("Unknown category: {}".format(result["category"]))
    if result["priority"] not in PRIORITIES:
        raise ValueError("Unknown priority: {}".format(result["priority"]))
    if not isinstance(result["summary"], str) or not result["summary"].strip():
        raise ValueError("Summary must be a non-empty string.")
    if type(result["escalate"]) is not bool:
        raise ValueError("Escalate must be true or false.")


def triage_ticket(ticket):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Ticket:\n" + ticket}
    ]
    content = call_ollama(messages, TRIAGE_SCHEMA)
    result = json.loads(content)
    validate_result(result)
    return result


def run_evaluation(path):
    cases = json.loads(path.read_text(encoding="utf-8"))
    fields = ("category", "priority", "escalate")
    correct = 0
    total = len(cases) * len(fields)

    for number, case in enumerate(cases, start=1):
        try:
            actual = triage_ticket(case["ticket"])
            expected = case["expected"]
            matched = [field for field in fields if actual[field] == expected[field]]
            correct += len(matched)
            status = "PASS" if len(matched) == len(fields) else "FAIL"
            print("{} {}: {}".format(status, number, case["ticket"]))
            if status == "FAIL":
                print("  expected: {}".format(expected))
                print("  actual:   {}".format(
                    {field: actual[field] for field in fields}
                ))
        except (KeyError, RuntimeError, ValueError) as error:
            print("ERROR {}: {}".format(number, error))

    score = correct / total if total else 0
    print("\nField accuracy: {}/{} ({:.0%})".format(correct, total, score))

    
def main():
    parser = argparse.ArgumentParser(
        description="Triage support tickets with a local Ollama model."
    )
    parser.add_argument("ticket", nargs="?", help="support ticket text")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="show the unconstrained baseline response"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="run the labelled evaluation set"
    )
    args = parser.parse_args()

    try:
        if args.eval:
            run_evaluation(Path("tickets.json"))
        elif args.ticket:
            if args.baseline:
                print("BASELINE OUTPUT (unvalidated):")
                print(triage_baseline(args.ticket))
            else:
                print(json.dumps(triage_ticket(args.ticket), indent=2))
        else:
            parser.error("provide a ticket or use --eval")
    except (KeyError, RuntimeError, ValueError) as error:
        parser.exit(1, "Error: {}\n".format(error))

if __name__ == "__main__":
    main()