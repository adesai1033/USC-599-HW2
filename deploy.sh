#!/usr/bin/env bash
# gcloud builds submit + gcloud run deploy
# Usage: set -a; source .env; set +a; ./deploy.sh
set -euo pipefail

PROJECT_ID="usc-599-hw2"
REGION="us-west1"
SERVICE="paper-assistant"
IMAGE="us-west1-docker.pkg.dev/${PROJECT_ID}/images/paper-assistant"
SERVICE_ACCOUNT="paper-assistant-run@${PROJECT_ID}.iam.gserviceaccount.com"

for var in GOOGLE_API_KEY GEMINI_MODEL; do
  if [[ -z "${!var:-}" ]]; then
    echo "error: ${var} is not set. Export it first, e.g.: set -a; source .env; set +a" >&2
    exit 1
  fi
done

echo "Building ${IMAGE} ..."
gcloud builds submit --project "${PROJECT_ID}" --tag "${IMAGE}" .

echo "Deploying ${SERVICE} to Cloud Run (${REGION}) ..."
gcloud run deploy "${SERVICE}" \
  --project "${PROJECT_ID}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --service-account "${SERVICE_ACCOUNT}" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --concurrency 10 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars "GOOGLE_API_KEY=${GOOGLE_API_KEY},GEMINI_MODEL=${GEMINI_MODEL}"

URL="$(gcloud run services describe "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" --format 'value(status.url)')"
echo "Service URL: ${URL}"
