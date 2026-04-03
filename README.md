# SMOOTH
[ISCA 26] SMOOTH

##1. setup LLMCompass & verilog(yosys, sta)

##2. make data
cd $SMOOTH_HOME/src/policies
run_all_policies.sh


##3. plot figure 14, 16
cd $SMOOTH_HOME
cd src/ae/figure14
python plot_ttft.py ../../../data/seq_1/8MB
-> src/ae/figure14/TTFT_8MB.eps

cd $SMOOTH_HOME
cd src/ae/figure16
python plot_latency.py ../../../data/seq_32K/8MB
-> src/ae/figure16/latency_8MB.png


##4. get verilog result
cd $SMOOTH_HOME/src/verilog/
run_all.sh


##5. plot figure 20
cd $SMOOTH_HOME
cd src/ae/figure20
python plot_energy.py
-> src/ae/figure20/energy_8MB.eps

##6. get table: table 1, table 2
cd $SMOOTH_HOME
cd src/ae/table1
python get_area.py

cd $SMOOTH_HOME
cd src/ae/table2
python get_power.py
