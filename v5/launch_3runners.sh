#!/usr/bin/env bash
# EWAT v5 — lance la collecte 3 runners (tt/tt-b/tt-c) proprement.
#
# Fait tout : préflight bloquant (contexte, 3×64/64, login, RAM) → démarrage
# DÉCALÉ des 3 runners (stagger 0/240/480 s pour désynchroniser leurs cycles et
# éviter la collision simultanée sur le Prometheus partagé) → build continu.
# tmux si dispo (re-attachable), sinon nohup. Idempotent : refuse si une
# campagne tourne déjà.
#
# Usage :
#   bash v5/launch_3runners.sh          # lance
#   bash v5/launch_3runners.sh stop     # arrête tout (runners + build)
#
# Surcharges (env) : V5_KUBE_CONTEXT NODE_IP V5_USERS V5_REPS V5_RESET_EVERY
#                    V5_HELDOUT_CAP V5_RAM_CEILING
set -uo pipefail

V5="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$V5")"
SRC="$REPO/src"
OUT="$REPO/data/raw_v5"

KCTX="${V5_KUBE_CONTEXT:-observit-cluster1}"
NODE_IP="${NODE_IP:-172.16.203.12}"
USERS="${V5_USERS:-12}"
REPS="${V5_REPS:-30}"
RESET="${V5_RESET_EVERY:-10}"
HELDCAP="${V5_HELDOUT_CAP:-28}"
RAMCEIL="${V5_RAM_CEILING:-90}"

# ns  port  rep_start rep_end pf_offset stagger_s  window
RUNNERS=(
  "tt   32677 0  10 0  0   runA"
  "tt-b 32679 10 20 10 240 runB"
  "tt-c 32681 20 30 20 480 runC"
)

# ---------- stop ----------
if [ "${1:-}" = "stop" ]; then
  echo "== arrêt collecte =="
  command -v tmux >/dev/null 2>&1 && tmux kill-session -t ewat 2>/dev/null && echo "  session tmux 'ewat' tuée"
  pkill -f "collect.run_campaign" 2>/dev/null && echo "  run_campaign tués"
  pkill -f "collect.build_features_v5" 2>/dev/null && echo "  build tué"
  pkill -f "loadgen.runner" 2>/dev/null
  # chaos orphelins : un runner tué en pleine injection laisse le chaos APPLIQUÉ
  # (vu 2026-06-03 : container-kill actif >1h sur tt-b/tt-c) → dégrade le namespace
  # et affame les épisodes suivants. On nettoie tout chaos v5 résiduel.
  # NB : les CR chaos-mesh ont des finalizers qui BLOQUENT `delete` si le
  # contrôleur ne peut pas réconcilier (pod cible disparu, nœud qui flappe — vu
  # 2026-06-08, delete coincé plusieurs minutes). On retire le finalizer AVANT,
  # puis delete --wait=false (non bloquant). La faute est déjà expirée de toute façon.
  for ns in tt tt-b tt-c; do
    for kind in stresschaos networkchaos podchaos dnschaos timechaos iochaos; do
      for name in $(kubectl --context "$KCTX" get "$kind" -n "$ns" -o name 2>/dev/null); do
        kubectl --context "$KCTX" patch "$name" -n "$ns" --type=merge \
          -p '{"metadata":{"finalizers":null}}' 2>/dev/null
      done
      kubectl --context "$KCTX" delete "$kind" -n "$ns" --all --ignore-not-found --wait=false 2>/dev/null
    done
  done
  echo "  chaos orphelins nettoyés (tt/tt-b/tt-c, finalizers forcés)"
  echo "fait."
  exit 0
fi

echo "== EWAT v5 — collecte 3 runners =="
echo "repo=$REPO  contexte=$KCTX  node=$NODE_IP  users=$USERS  ram_ceiling=$RAMCEIL"

# ---------- préflight ----------
echo "-- préflight --"
kubectl --context "$KCTX" get ns tt tt-b tt-c >/dev/null 2>&1 \
  || { echo "ABORT: contexte '$KCTX' ne voit pas tt/tt-b/tt-c (kubectl config get-contexts ?)"; exit 1; }
fail=0
for r in "${RUNNERS[@]}"; do
  set -- $r; ns=$1; port=$2
  ready=$(kubectl --context "$KCTX" get pods -n "$ns" --no-headers 2>/dev/null | grep -c "1/1")
  if [ "$ready" -ge 60 ]; then echo "  $ns: $ready/64 prêts ✓"; else echo "  $ns: $ready/64 ✗"; fail=1; fi
  login=$(curl -s -m10 -XPOST "http://$NODE_IP:$port/api/v1/users/login" \
            -H 'Content-Type: application/json' \
            -d '{"username":"fdse_microservice","password":"111111"}' 2>/dev/null | head -c 80)
  if echo "$login" | grep -q "success"; then echo "  $ns login :$port ✓"; else echo "  $ns login :$port ✗"; fail=1; fi
done
# NodePort Prometheus + Loki (collecte sans port-forward) — doivent répondre
PROM_NP="${V5_PROM_NODEPORT:-32700}"; LOKI_NP="${V5_LOKI_NODEPORT:-32701}"
if curl -s -m10 "http://$NODE_IP:$PROM_NP/-/ready" | grep -qi "ready"; then echo "  Prometheus NodePort :$PROM_NP ✓"; else echo "  Prometheus NodePort :$PROM_NP ✗ (kubectl apply -f v5/deploy/monitoring_nodeports.yaml ?)"; fail=1; fi
if curl -s -m10 "http://$NODE_IP:$LOKI_NP/ready" | grep -qi "ready"; then echo "  Loki NodePort :$LOKI_NP ✓"; else echo "  Loki NodePort :$LOKI_NP ✗ (kubectl apply -f v5/deploy/monitoring_nodeports.yaml ?)"; fail=1; fi
echo "  RAM workers :"
kubectl --context "$KCTX" top nodes --no-headers 2>/dev/null | awk '/workers/{printf "    %s %s\n",$1,$5}'
[ "$fail" -eq 0 ] || { echo "ABORT: préflight échoué (voir ✗ ci-dessus)"; exit 1; }
mkdir -p "$OUT"

# ---------- refuse si déjà en cours ----------
if pgrep -f "collect.run_campaign" >/dev/null 2>&1; then
  echo "ABORT: une campagne run_campaign tourne déjà. Arrête-la d'abord :"
  echo "  bash $0 stop"
  exit 1
fi

# construit la commande d'un runner (ns port rep_start rep_end offset)
run_cmd() {
  echo "cd '$V5' && export PYTHONPATH='$SRC' V5_KUBE_CONTEXT='$KCTX' && python -m collect.run_campaign --namespace $1 --address http://$NODE_IP:$2 --rep-start $3 --rep-end $4 --reps $REPS --pf-offset $5 --out-root '$OUT' --users $USERS --reset-every $RESET --held-out-cap $HELDCAP --ram-ceiling $RAMCEIL 2>&1 | tee -a '$OUT/_campaign_$1.log'"
}
build_cmd="cd '$V5' && export PYTHONPATH='$SRC' V5_KUBE_CONTEXT='$KCTX' && while true; do python -m collect.build_features_v5 --raw-root '$OUT' --workers 2 2>&1 | tee -a '$OUT/_build.log'; sleep 1800; done"

# ---------- lancement ----------
if command -v tmux >/dev/null 2>&1; then
  tmux has-session -t ewat 2>/dev/null && { echo "ABORT: session tmux 'ewat' existe déjà → bash $0 stop"; exit 1; }
  echo "-- lancement tmux (session 'ewat') --"
  tmux new-session -d -s ewat -n runA
  i=0
  for r in "${RUNNERS[@]}"; do
    set -- $r; ns=$1; port=$2; rs=$3; re=$4; off=$5; stag=$6; w=$7
    [ "$i" -gt 0 ] && tmux new-window -t ewat -n "$w"
    cmd="$(run_cmd "$ns" "$port" "$rs" "$re" "$off")"
    [ "$stag" -gt 0 ] && cmd="sleep $stag && $cmd"
    tmux send-keys -t "ewat:$w" "$cmd" Enter
    echo "  $ns → fenêtre $w (démarrage +${stag}s)"
    i=$((i+1))
  done
  if [ "${V5_NO_BUILD:-0}" = "1" ]; then
    echo "  build → DÉSACTIVÉ (V5_NO_BUILD=1) : collecte seule, build+validate à la fin"
  else
    tmux new-window -t ewat -n build
    tmux send-keys -t ewat:build "$build_cmd" Enter
    echo "  build → fenêtre build"
  fi
  echo "== lancé. Attache : tmux attach -t ewat  (Ctrl-b n = fenêtre suivante, Ctrl-b d = détacher) =="
else
  echo "-- tmux absent → nohup --"
  for r in "${RUNNERS[@]}"; do
    set -- $r; ns=$1; port=$2; rs=$3; re=$4; off=$5; stag=$6
    cmd="$(run_cmd "$ns" "$port" "$rs" "$re" "$off")"
    [ "$stag" -gt 0 ] && cmd="sleep $stag && $cmd"
    nohup bash -c "$cmd" >/dev/null 2>&1 &
    echo "  $ns lancé (nohup, démarrage +${stag}s)"
  done
  if [ "${V5_NO_BUILD:-0}" = "1" ]; then
    echo "  build → DÉSACTIVÉ (V5_NO_BUILD=1) : collecte seule, build+validate à la fin"
  else
    nohup bash -c "$build_cmd" >/dev/null 2>&1 &
  fi
  disown -a
  echo "== lancé en nohup. Logs : $OUT/_campaign_*.log =="
fi

echo
echo "Suivi : cd $REPO && watch -n60 'echo collectés=\$(ls data/raw_v5|grep -c episode_) buildés=\$(find data/raw_v5 -name signal.npz|wc -l) échecs=\$(find data/raw_v5 -name .raw_failed|wc -l)'"
