Review the "Final temporal caption under evaluation (audio + visual alignment)"
against the uploaded full video file. Produce strict JSON with the shape, no "strengths" or "issues" should be mentioned in your output:
{
  "score": <number from 0 to 5>,
  "verdict": "pass" | "warn" | "fail",
  "summary": "ONLY one concise sentence",
}
please output as raw text of JSON insted of markdown. If there is no extremely obvious error, a higher score is recommended.