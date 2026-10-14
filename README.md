# DataTrust AI

**An evidence-first data quality and ML-readiness agent built with Python, AWS Lambda, and Amazon Bedrock.**

DataTrust AI computes dataset quality metrics deterministically, then asks Amazon Bedrock to explain the evidence and recommend prioritized next steps. The model does not calculate or invent the metrics.

## Current MVP features

- CSV upload and sample dataset in a Streamlit UI.
- Missing-value counts and percentages.
- Distinct-value counts and repeated-value signals.
- Numeric min, max, mean, and median summaries.
- Exact duplicate-row detection.
- Evidence-grounded Bedrock explanation.
- Downloadable JSON profile.
- Input size limits and explicit error responses.
- Basic unit tests for the profiler.

## Architecture

Streamlit UI → HTTPS Lambda Function URL → Python profiler → Amazon Bedrock Converse API → evidence + recommendations.

The first version is deliberately small. It is not yet a production data-governance product. It does not infer schema semantics, detect every form of anomaly, or guarantee model correctness.

## Prerequisites

- AWS account and permission to update a Lambda function.
- Amazon Bedrock model access in the selected region.
- A Lambda execution role with `bedrock:InvokeModel` permission for the selected model.
- Python 3.11 or newer locally.

## Local setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

## Deploy the Lambda backend

1. In the AWS Console, select the region in which your model is available.
2. In Amazon Bedrock, verify access to the chosen model. The default in this code is `amazon.nova-micro-v1:0`; change `MODEL_ID` if that model is not available to your account/region.
3. Create a new Lambda function named `DataTrustAI` using Python 3.11, or use a separate function rather than overwriting your existing AI-Job-Agent.
4. Set the handler to `lambda_function.lambda_handler`.
5. Set environment variables:
   - `MODEL_ID` = a model ID you have access to
   - `CORS_ORIGIN` = `*` for a temporary demo only, or your exact deployed UI origin
   - `MAX_ROWS` = `5000`
   - `MAX_COLUMNS` = `100`
6. Give the Lambda execution role only the permissions it needs, including `bedrock:InvokeModel` for the chosen model. Do not put access keys in code.
7. Upload `lambda_function.py` and any required deployment dependencies. Boto3 is included in the standard Lambda Python runtime, but package a tested version if your deployment needs one.
8. Configure a Function URL with `AuthType: NONE` only for a public demo with no sensitive data. This is a public endpoint and can incur costs. Add throttling/protection or use API Gateway and authentication before real-world use.
9. Set Function URL CORS to allow only the origin you use. The handler includes basic CORS headers; align Function URL CORS configuration with them.
10. Copy the HTTPS Function URL into the Streamlit sidebar and run the UI.

## Test the backend

Example event body:

```json
{
  "records": [
    {"customer_id": "1", "age": "25", "spend": "120"},
    {"customer_id": "2", "age": "", "spend": "200"},
    {"customer_id": "2", "age": "", "spend": "200"}
  ]
}
```

## Security and limitations

- Do not upload personal, confidential, regulated, or production customer data to the demo.
- This demo accepts JSON records, not arbitrary file uploads, at the Lambda endpoint.
- Public unauthenticated endpoints can be abused. Use a protected API for production.
- The model's suggestions require human review.
- Review CloudWatch logs, model access, quotas, and AWS billing before sharing the endpoint publicly.
- Do not claim production readiness or performance results without evidence.

## Next steps

1. Add tests for malformed input, large payloads, and Bedrock failures.
2. Add a schema-aware profile and configurable rules.
3. Add a safe preview of suggested cleaning operations before applying them.
4. Add an evaluation set of representative datasets and track correctness.
5. Add infrastructure-as-code and a deployment walkthrough.
6. Capture screenshots and a short demo video of the actual deployed application.
