#!/bin/bash
# Re-run every configuration with the assignment's own CLI form (run from the repository root).
run() {  # $1 = label, rest = CLI args
  label=$1; shift
  res=$(python -W ignore run_recbole.py --dataset=ml-100k "$@" 2>&1 | grep "test result" | sed 's/.*test result: //')
  echo "$label|$*|$res"
}
for m in item user; do for k in 1 5 10 100 500; do run q1 --model=ItemKNN --k=$k --knn_method=$m; done; done
run literal --model=ItemKNN --k=10 --method=user   # the flag from the handout: ignored, same as item-based k=10
for e in 32 64 128 256; do run q2emb --model=BPR --embedding_size=$e --reg_lambda=0.1; done
for r in 0.01 0.1 0.5; do run q2reg --model=BPR --embedding_size=64 --reg_lambda=$r; done
run q3 --model=ItemKNN
run q3 --model=SLIMElastic
run q3 --model=EASE
