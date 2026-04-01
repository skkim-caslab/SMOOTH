# SMOOTH
[ISCA 26] SMOOTH

##1. setup LLMCompass

##2. make data
cd $SMOOTH_HOME
cd src/policies/smooth
run_all.sh
cd $SMOOTH_HOME
cd src/policies/smooth-er
run_all.sh
cd $SMOOTH_HOME
cd src/policies/capuchin
run_all.sh
cd $SMOOTH_HOME
cd src/policies/compiler-ideal
run_all.sh
cd $SMOOTH_HOME
cd src/policies/gemmini
run_all.sh
cd $SMOOTH_HOME

##3. plot figure
cd $SMOOTH_HOME
cd src/ae/figure14
python plot_ttft.py ../../../data/seq_1/8MB
-> src/ae/figure14/TTFT_8MB.eps

cd $SMOOTH_HOME
cd src/ae/figure16
python plot_latency.py ../../../data/seq_32K/8MB
-> src/ae/figure16/latency_8MB.png


