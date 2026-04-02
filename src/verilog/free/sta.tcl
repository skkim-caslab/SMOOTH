read_liberty ../libs/asap7.lib
read_verilog synthesized_free.v
link_design free
read_sdc ../config/constraints_940MHz.sdc
report_checks -path_delay max >> ../../../data/energy/verilog/free.out
report_power >> ../../../data/energy/verilog/free.out
