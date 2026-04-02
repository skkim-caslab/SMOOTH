read_liberty ../libs/asap7.lib
read_verilog synthesized_find_longest_zero.v
link_design find_logest_zero_run
read_sdc ../config/constraints_940MHz.sdc
report_checks -path_delay max >> ../../../data/energy/verilog/find_zero.out
report_power >> ../../../data/energy/verilog/find_zero.out
