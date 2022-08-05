SEED='10000 1000000 500000 5000000 900000000'
TOML_FILE='./toml/flower.toml'
BATCH_SIZE='16'
MICRO_BATCH='4'
MODEL_VERSION='50'

sed -i 's/version=1/version='"$MODEL_VERSION"'/' $TOML_FILE;
sed -i 's/train_batch=1/train_batch='"$BATCH_SIZE"'/' $TOML_FILE;

# Baseline
for i in $SEED; do
    sed -i 's/seed=1/seed='"$i"'/' $TOML_FILE;
    python main.py -m train -c $TOML_FILE
    sed -i 's/seed='"$i"'/seed=1/' $TOML_FILE;
done


# MBS
sed -i 's/enable=false/enable=true/' $TOML_FILE;
for m in $MICRO_BATCH; do
    sed -i 's/micro_batch_size=1/micro_batch_size='"$m"'/' $TOML_FILE;
    for i in $SEED; do
        sed -i 's/seed=1/seed='"$i"'/' $TOML_FILE;
        python main.py -m train -c $TOML_FILE
        sed -i 's/seed='"$i"'/seed=1/' $TOML_FILE;
    done
    sed -i 's/micro_batch_size='"$m"'/micro_batch_size=1/' $TOML_FILE;
done
sed -i 's/enable=true/enable=false/' $TOML_FILE;

sed -i 's/train_batch='"$BATCH_SIZE"'/train_batch=1/' $TOML_FILE;
sed -i 's/version='"$MODEL_VERSION"'/version=1/' $TOML_FILE;
