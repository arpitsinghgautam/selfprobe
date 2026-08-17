# Concept-injection introspection benchmark across scale and family.
#
#   .\run_injection.ps1
#
# Project 2 currently measures self-knowledge about BEHAVIOUR (can a model predict
# its own choices?). This measures self-knowledge about INTERNAL STATE, which is
# the question Track 3 of the sprint actually asks, and the one with genuine
# ground truth: we know exactly what we injected.
#
# Every cell carries a false-positive reading (same question, nothing injected)
# and an A/B mass reading. Detection without a false-alarm baseline is
# uninterpretable, and a "yes" from a model that has stopped answering is damage
# rather than introspection.
#
# All checkpoints are already cached from the persona sweeps, so this is compute
# only -- no downloads.

$py = ".venv\Scripts\python.exe"

$models = @(
  @{ id = "Qwen/Qwen2.5-1.5B-Instruct";              quant = $null  },
  @{ id = "Qwen/Qwen2.5-3B-Instruct";                quant = $null  },
  @{ id = "Qwen/Qwen2.5-7B-Instruct";                quant = $null  },
  @{ id = "mistralai/Mistral-7B-Instruct-v0.3";      quant = $null  },
  @{ id = "microsoft/Phi-3.5-mini-instruct";         quant = $null  },
  @{ id = "tiiuae/Falcon3-7B-Instruct";              quant = $null  },
  @{ id = "allenai/OLMo-2-1124-7B-Instruct";         quant = $null  },
  @{ id = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit";   quant = $null  }
)

foreach ($m in $models) {
  "`n=== injection benchmark  $($m.id) ===`n"
  if ($m.quant) {
    & $py scripts\20_injection.py --model $m.id --quant $m.quant --batch-size 16
  } else {
    & $py scripts\20_injection.py --model $m.id --batch-size 16
  }
}

"`n=== run_injection complete ===`n"
