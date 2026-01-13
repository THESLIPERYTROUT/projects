import httpx



httpx.post(
    "https://api.dev.oursky.ai/v1/survey-instruction",
    headers={
      "Content-Type": "application/json",
      "Authorization": "Bearer YOUR_SECRET_TOKEN"
    },
    json={
  "firstStepAt": "2025-11-11T13:40:00Z",
  "steps": [
    {
      "az": 266.03,
      "el": 31.00,
      "ra": 195.252,
      "dec": -4.6868,
      "durationSeconds": 180,
      "nodeId": "sro-cdk24",
      "time": "2025-11-11T13:40:00Z"
    }
  ],
  "targetId": "3I-ATLAS"
}

      
)
``
