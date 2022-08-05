python ../util/plot.py \
-t avg_mbs_with_bn_512_cifar10.json \
--target_name mbs_with_bn \
-b avg_baseline_512_cifar10.json \
--base_name baseline \
-n cifar10_with_bn

python ../util/plot.py \
-t avg_mbs_with_bn_512_cifar100.json \
--target_name mbs_with_bn \
-b avg_baseline_512_cifar100.json \
--base_name baseline \
-n cifar100_with_bn

python ../util/plot.py \
-t avg_mbs_without_bn_512_cifar10.json \
--target_name mbs_without_bn \
-b avg_baseline_512_cifar10.json \
--base_name baseline \
-n cifar10_without_bn

python ../util/plot.py \
-t avg_mbs_without_bn_512_cifar100.json \
--target_name mbs_without_bn \
-b avg_baseline_512_cifar100.json \
--base_name baseline \
-n cifar100_without_bn

python ../util/plot.py \
-t avg_mbs_with_bn_512_cifar10.json \
--target_name mbs_with_bn \
-b avg_mbs_without_bn_512_cifar10.json \
--base_name mbs_without_bn \
-n cifar10_compare

python ../util/plot.py \
-t avg_mbs_with_bn_512_cifar100.json \
--target_name mbs_with_bn \
-b avg_mbs_without_bn_512_cifar100.json \
--base_name mbs_without_bn \
-n cifar100_compare