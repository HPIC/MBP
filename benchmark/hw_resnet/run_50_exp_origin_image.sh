TOML_FILE='./toml/flower_50.toml'
BATCH_SIZE_WO='2'
BATCH_SIZE_W='2 128'

# resnet-50 up to 2 mini-batch (92% utilization)

# Baseline
for i in $BATCH_SIZE_WO; do
    sed -i 's/train_batch=1/train_batch='"$i"'/' $TOML_FILE;
    python main.py -m train -c $TOML_FILE
    sed -i 's/train_batch='"$i"'/train_batch=1/' $TOML_FILE;
done

# MBS
sed -i 's/enable=false/enable=true/' $TOML_FILE;

## ResNet-50 with 2 mini-batch and 1 micro-batch
sed -i 's/train_batch=1/train_batch=2/' $TOML_FILE;
python main.py -m train -c $TOML_FILE
sed -i 's/train_batch=2/train_batch=1/' $TOML_FILE;

## ResNet-50 with 128 mini-batch and 2 micro-batch
sed -i 's/micro_batch_size=1/micro_batch_size=2/' $TOML_FILE;
sed -i 's/train_batch=1/train_batch=128/' $TOML_FILE;
python main.py -m train -c $TOML_FILE
sed -i 's/train_batch=128/train_batch=1/' $TOML_FILE;
sed -i 's/micro_batch_size=2/micro_batch_size=1/' $TOML_FILE;

sed -i 's/enable=true/enable=false/' $TOML_FILE;
