# Project 2: do models know their own preferences?
#
#   .\run_project2.ps1
#
# Requires run_all.ps1 to have completed first, the privileged-access test
# scores predictions against the revealed preference matrices it produces.
#
# Reuses the same harness. Ratings are 80 forward passes per model; each
# prediction condition is 1560. Roughly 6 minutes per model on a 24GB card.

$py = ".venv\Scripts\python.exe"
$QI = "Qwen/Qwen2.5-7B-Instruct"
$MI = "mistralai/Mistral-7B-Instruct-v0.3"

function Stage($n, $label) { "`n=== [$n] $label ===`n" }

Stage "1/3" "Stated ratings + predicted choices - Qwen2.5-7B-Instruct"
& $py scripts\08_stated.py --model $QI --batch-size 16

Stage "2/3" "Stated ratings + predicted choices - Mistral-7B-Instruct-v0.3"
& $py scripts\08_stated.py --model $MI --batch-size 16

Stage "3/3" "Stated vs revealed, and the privileged-access test"
& $py scripts\09_selfknowledge.py --models $QI $MI

"`n=== run_project2 complete ===`n"
