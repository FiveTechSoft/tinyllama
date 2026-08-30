import json, subprocess
out = subprocess.run(["gh", "run", "view", "33280863749", "--json", "jobs"],
                     capture_output=True, text=True)
data = json.loads(out.stdout)
for job in data["jobs"]:
    print("JOB:", job["name"], job["status"], job.get("conclusion"))
    for s in job["steps"]:
        print("  step:", s["name"], s["status"], s.get("conclusion"))