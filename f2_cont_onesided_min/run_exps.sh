LOG_FILE="grad_ascent.log"
OUTPUT_DIR="results"
CUTOFF=10800

rm -r $LOG_FILE

for subgame_size in 20 30 40
do
    for seed in {1..10}
    do
    /usr/bin/time timeout $CUTOFF python3 quadr_subgrad_ascent_exp.py $subgame_size $seed $OUTPUT_DIR >> $LOG_FILE

    /usr/bin/time timeout $CUTOFF python3 linear_subgrad_ascent_exp.py $subgame_size $seed $OUTPUT_DIR >> $LOG_FILE

    done
    
done