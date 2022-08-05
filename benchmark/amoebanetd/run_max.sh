SEED='100000 500000 5000000'
SEED2='100000 5000000'
SEED3='1000000 100000'
TOML_FILE='./toml/flower.toml'
BATCH_SIZE='128 1024'
BATCH_SIZE2='64 256 512'
BATCH_SIZE3='1024'
MICRO_BATCH='32'

sed -i 's/micro_batch_size=1/micro_batch_size='"$MICRO_BATCH"'/' $TOML_FILE;
sed -i 's/enable=false/enable=true/' $TOML_FILE;
# MBS
## 128, 1024 batch (6)
for i in $SEED3; do
    sed -i 's/seed=1/seed='"$i"'/' $TOML_FILE;
    for b in $BATCH_SIZE3; do
        sed -i 's/train_batch=1/train_batch='"$b"'/' $TOML_FILE;
        python main.py -m train -c $TOML_FILE
        sed -i 's/train_batch='"$b"'/train_batch=1/' $TOML_FILE;
    done
    sed -i 's/seed='"$i"'/seed=1/' $TOML_FILE;
done

# sed -i 's/seed=1/seed=100000/' $TOML_FILE;
# for b in $BATCH_SIZE3; do
#     sed -i 's/train_batch=1/train_batch='"$b"'/' $TOML_FILE;
#     python main.py -m train -c $TOML_FILE
#     sed -i 's/train_batch='"$b"'/train_batch=1/' $TOML_FILE;
# done
# sed -i 's/seed=100000/seed=1/' $TOML_FILE;

# ## 128, 1024 batch with 3 seed (18)
# for b in $BATCH_SIZE; do
#     sed -i 's/train_batch=1/train_batch='"$b"'/' $TOML_FILE;
#     for i in $SEED; do
#         sed -i 's/seed=1/seed='"$i"'/' $TOML_FILE;
#         python main.py -m train -c $TOML_FILE
#         sed -i 's/seed='"$i"'/seed=1/' $TOML_FILE;
#     done
#     sed -i 's/train_batch='"$b"'/train_batch=1/' $TOML_FILE;
# done

# sed -i 's/seed=1/seed=100000/' $TOML_FILE;
# sed -i 's/train_batch=1/train_batch=1024/' $TOML_FILE;
# python main.py -m train -c $TOML_FILE
# sed -i 's/train_batch=1024/train_batch=1/' $TOML_FILE;
# sed -i 's/seed=100000/seed=1/' $TOML_FILE;

# ## 64, 256, 512 batch with 2 seed (18)
# for i in $SEED2; do
#     sed -i 's/seed=1/seed='"$i"'/' $TOML_FILE;
#     for b in $BATCH_SIZE2; do
#         sed -i 's/train_batch=1/train_batch='"$b"'/' $TOML_FILE;
#         python main.py -m train -c $TOML_FILE
#         sed -i 's/train_batch='"$b"'/train_batch=1/' $TOML_FILE;
#     done
#     sed -i 's/seed='"$i"'/seed=1/' $TOML_FILE;
# done

sed -i 's/enable=true/enable=false/' $TOML_FILE;
sed -i 's/micro_batch_size='"$MICRO_BATCH"'/micro_batch_size=1/' $TOML_FILE;
