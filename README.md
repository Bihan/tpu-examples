```pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html ```
```pip install accelerate```
```accelerate config```
```compute_environment: LOCAL_MACHINE                                                                                                                              
debug: false
distributed_type: XLA
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false```
