import json
import os
import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get("MODEL_ID", "amazon.nova-micro-v1:0")
MAX_ROWS = int(os.environ.get("MAX_ROWS", "5000"))
MAX_COLUMNS = int(os.environ.get("MAX_COLUMNS", "100"))
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", "900000"))
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def response(status_code: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": os.environ.get("CORS_ORIGIN", "*"),
            "Access-Control-Allow-Headers": "content-type",
            "Access-Control-Allow-Methods": "POST,OPTIONS",
        },
        "body": json.dumps(payload, ensure_ascii=False),
    }


def profile_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic checks. The model never calculates these metrics."""
    if not isinstance(records, list) or not records:
        raise ValueError("Provide at least one row of CSV data.")
    if len(records) > MAX_ROWS:
        raise ValueError(f"Dataset exceeds the {MAX_ROWS}-row demo limit.")
    if not all(isinstance(row, dict) for row in records):
        raise ValueError("Rows must be JSON objects.")

    columns = list(records[0].keys())
    if not columns:
        raise ValueError("The dataset has no columns.")
    if len(columns) > MAX_COLUMNS:
        raise ValueError(f"Dataset exceeds the {MAX_COLUMNS}-column demo limit.")
    if any(list(row.keys()) != columns for row in records):
        # Normalize differing key order while rejecting inconsistent schemas.
        if any(set(row.keys()) != set(columns) for row in records):
            raise ValueError("Rows have inconsistent columns.")

    row_count = len(records)
    column_report = []
    for col in columns:
        values = [row.get(col) for row in records]
        missing = sum(
            v is None or (isinstance(v, str) and not v.strip())
            for v in values
        )
        nonempty = [v for v in values if not (
            v is None or (isinstance(v, str) and not v.strip())
        )]
        distinct = len({str(v) for v in nonempty})
        numeric = []
        for v in nonempty:
            try:
                numeric.append(float(v))
            except (TypeError, ValueError):
                numeric = []
                break

        numeric_summary = None
        if numeric:
            ordered = sorted(numeric)
            n = len(ordered)
            median = ordered[n // 2] if n % 2 else (ordered[n//2 - 1] + ordered[n//2]) / 2
            numeric_summary = {
                "min": round(min(numeric), 4),
                "max": round(max(numeric), 4),
                "mean": round(sum(numeric) / len(numeric), 4),
                "median": round(median, 4),
            }

        column_report.append({
            "column": col,
            "missing_count": missing,
            "missing_percent": round(100 * missing / row_count, 2),
            "distinct_nonempty": distinct,
            "duplicate_values_present": distinct < len(nonempty),
            "numeric_summary": numeric_summary,
        })

    serialized = [json.dumps(r, sort_keys=True, default=str) for r in records]
    duplicate_rows = len(serialized) - len(set(serialized))
    issues = []
    for col in column_report:
        if col["missing_count"]:
            issues.append({
                "severity": "high" if col["missing_percent"] >= 20 else "medium",
                "type": "missing_values",
                "column": col["column"],
                "evidence": f'{col["missing_count"]}/{row_count} rows missing ({col["missing_percent"]}%).',
            })
        if col["duplicate_values_present"]:
            issues.append({
                "severity": "info",
                "type": "repeated_values",
                "column": col["column"],
                "evidence": f'{col["distinct_nonempty"]} distinct non-empty values among {row_count - col["missing_count"]} non-empty rows.',
            })
    if duplicate_rows:
        issues.append({
            "severity": "medium",
            "type": "duplicate_rows",
            "column": None,
            "evidence": f"{duplicate_rows} fully duplicated rows detected.",
        })

    high_count = sum(i["severity"] == "high" for i in issues)
    medium_count = sum(i["severity"] == "medium" for i in issues)
    return {
        "row_count": row_count,
        "column_count": len(columns),
        "duplicate_row_count": duplicate_rows,
        "columns": column_report,
        "issues": issues,
        "summary": {
            "high": high_count,
            "medium": medium_count,
            "info": sum(i["severity"] == "info" for i in issues),
        },
    }


def ask_bedrock(profile: dict[str, Any]) -> str:
    prompt = (
        "You are DataTrust AI, a careful data quality assistant. "
        "Use only the supplied computed profile as evidence. Do not invent metrics, "
        "claim you inspected raw data beyond this profile, or assert causation. "
        "Give: (1) a one-sentence assessment, (2) the top three actions in priority "
        "order, (3) one caution before model training. Keep it clear and encouraging. "
        "If evidence is insufficient, say so. Profile JSON:\n"
        + json.dumps(profile, ensure_ascii=False)
    )
    result = bedrock.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 700, "temperature": 0.2, "topP": 0.9},
    )
    return "\n".join(
        item["text"]
        for item in result["output"]["message"]["content"]
        if "text" in item
    ).strip()


def lambda_handler(event, context):
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod", "POST")
    )
    if method == "OPTIONS":
        return response(200, {"ok": True})
    if method != "POST":
        return response(405, {"error": "Use POST for dataset analysis."})

    raw_body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        return response(400, {"error": "Base64-encoded request bodies are not supported in this demo."})
    if len(raw_body.encode("utf-8")) > MAX_BODY_BYTES:
        return response(413, {"error": "Request too large. Use a smaller sample dataset."})

    try:
        payload = json.loads(raw_body)
        records = payload.get("records")
        profile = profile_records(records)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return response(400, {"error": str(exc)})
    except Exception:
        logger.exception("Unexpected profiling failure")
        return response(500, {"error": "Dataset profiling failed. Check the request format."})

    try:
        explanation = ask_bedrock(profile)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "BedrockError")
        logger.exception("Amazon Bedrock request failed")
        return response(502, {
            "error": "The data profile was calculated, but the AI explanation failed.",
            "detail": f"AWS returned {code}. Check model access, region, IAM permissions, and quotas.",
            "profile": profile,
        })
    except Exception:
        logger.exception("Unexpected model invocation failure")
        return response(502, {
            "error": "The data profile was calculated, but the AI explanation failed.",
            "profile": profile,
        })

    logger.info("Analysis complete: rows=%s columns=%s issues=%s",
                profile["row_count"], profile["column_count"], len(profile["issues"]))
    return response(200, {"profile": profile, "explanation": explanation})
