#!/bin/bash

DATADIR=dataverse_files/data_

SOLVER="liblinear"
PENALTY="l2"
CW='balanced'
C=0.2
options="--show-top-features --solver $SOLVER --penalty $PENALTY --class-weight $CW --format_wise_models --C $C " 
learner="logReg"

NAME=3.frmtw_dep_final

mkdir -p runs

for features in "user:client:session:base:pos:morph:form:dep:len:position:time:uvocab:country"
do
    for lang in en_es es_en fr_en
    do
	JOBNAME=$NAME.$SOLVER.$PENALTY.cw$CW.c$C.$lang.$features
	echo "#!/bin/bash"  > $$tmp
	mkdir -p outputs/$NAME

	# use training from d1 onwards
	echo "python src/baseline_system.py --train $DATADIR$lang/$lang.slam.20171218.train.from_d1+dev \
                       --test $DATADIR$lang/test.$lang \
                       --pred outputs/$NAME/$lang.train.from_d1+dev.test.$JOBNAME.pred $options \
                       --lang $lang --show-top-features --feats $features" >> $$tmp
	
	## also check dev results
	echo "python src/baseline_system.py --train $DATADIR$lang/$lang.slam.20171218.train \
                       --test $DATADIR$lang/$lang.slam.20171218.dev \
                       --pred outputs/$NAME/$lang.dev.$JOBNAME.pred $options \
                       --lang $lang --feats $features" >> $$tmp
	
	cat $$tmp
	rm $$tmp
    done
done

