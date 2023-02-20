for model in FE_NGCF
do
for dataset in azuki bayc
do
    python main.py \
        --model $model \
        --feature 'txt' \
        --dataset $dataset \
        --config 'FE_NGCF' &

done
done