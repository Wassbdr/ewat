# Nommage sémantique des clusters EWAT

_Généré le 2026-05-20_

| Cluster | Nom (FR) | Scénario dominant | Pureté | N ép. | Top-3 features |
|---------|----------|-------------------|--------|-------|----------------|
| C0 | Pression mémoire (net_sat, latency_p99) | `memory_pressure` | 0.322 | 59 | net_sat, latency_p99, disk_io |
| C1 | Rampe de trafic (drift) (latency_p99, disk_io) | `drift_traffic_ramp` | 0.679 | 28 | latency_p99, disk_io, net_sat |
| C2 | Fuite de ressources (disk_io, latency_p99) | `resource_leak` | 0.633 | 30 | disk_io, latency_p99, ram_util |
| C3 | Voisin bruyant (cpu_util, ram_util) | `noisy_neighbor` | 0.952 | 21 | cpu_util, ram_util, trace_depth |
| C4 | Crash pod (net_sat, latency_p99) | `crash` | 0.388 | 49 | net_sat, latency_p99, disk_io |
| C5 | Déploiement progressif (drift) (disk_io, net_sat) | `drift_rolling_deploy` | 0.778 | 18 | disk_io, net_sat, lexical_entropy |
| C6 | Changement config (drift) (net_sat, latency_p99) | `drift_config_change` | 1.000 | 12 | net_sat, latency_p99, disk_io |
| C7 | Contention CPU (net_sat, disk_io) | `cpu_starvation` | 0.319 | 47 | net_sat, disk_io, latency_cv |
| C8 | Déploiement défectueux (drift ∩ anomalie) (net_sat, latency_p99) | `faulty_deploy_overlap` | 0.769 | 26 | net_sat, latency_p99, disk_io |
| C9 | Autoscaling (drift) (net_sat, latency_p99) | `drift_scale_up` | 0.889 | 9 | net_sat, latency_p99, ram_util |
